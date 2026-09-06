"""Recurrent PPO training loop, shared by the baseline and assisted variants.

Ties together :class:`RolloutCollector`, :func:`compute_gae`, and
:func:`optimize` into repeated rollout-collect/PPO-update cycles. Native
``async def`` throughout: the assisted variant's per-step consultation is a
real ``await`` on the Gatekeeper round-trip, and running natively on the
event loop (rather than in a background thread) is what lets other
local-mode agents' behaviours keep being serviced during training. The
SPADE RL Orchestrator (Milestone 5) awaits this directly; this module is
also directly usable standalone for the baseline variant and for tests.

Memory/persistence architecture (spec sections 1-2): each rollout's
heavyweight ``StepRecord`` list (holding graph tensors, logits, ...) is used
only to compute GAE and run the PPO update, then to build one rollout's
worth of compact decisions.csv rows -- after that, ``records`` goes out of
scope and is never retained. When ``run_dir`` is given, those compact rows
(plus that rollout's episode/update/rollout-level rows) are flushed to disk
immediately (``metrics.writer.append_rollout_metrics``), which is what makes
an interrupted long run's results resilient to how the process ends.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from marla.config.models import Config
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.learning.checkpoint import save_checkpoint
from marla.learning.gae import compute_gae
from marla.learning.lr_schedule import compute_learning_rate
from marla.learning.ppo import optimize
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import (
    EVAL_SEED_OFFSET,
    ConsultFn,
    EpisodeSummary,
    RolloutCollector,
    run_evaluation_episodes,
)
from marla.metrics.accumulators import ConsultationStats

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    policy: RecurrentPolicy
    optimizer: torch.optim.Optimizer
    episode_summaries: list[EpisodeSummary] = field(default_factory=list)
    # One dict per PPO minibatch update (spec section 7): update/rollout/
    # epoch/minibatch/environment_steps/learning_rate plus ppo.optimize()'s
    # own loss/diagnostic scalars. Never a per-step or per-rollout aggregate
    # -- those live in rollout_rows below.
    update_metrics: list[dict[str, Any]] = field(default_factory=list)
    # One dict per completed rollout (spec section 6's rollouts.csv).
    rollout_rows: list[dict[str, Any]] = field(default_factory=list)
    # One dict per environment step (spec section 3's decisions.csv), built
    # from that rollout's StepRecords right before they're discarded. Each
    # row is a handful of scalars/strings -- proportional to step count, not
    # to any retained tensor -- which is what replaces the old
    # ``all_records: list[StepRecord]`` (spec section 1).
    decision_rows: list[dict[str, Any]] = field(default_factory=list)
    environment_steps: int = 0
    stopped_by_user: bool = False
    eval_episode_summaries: list[EpisodeSummary] = field(default_factory=list)
    # Resume support (research/aamas2027's staged training): how many
    # updates/environment steps this TrainingResult's own numbering starts
    # counting from (0 for a fresh run), the seed the *next* rollout
    # collector should continue from, and a snapshot of torch's global RNG
    # state at the end of this run -- all threaded into the next
    # save_checkpoint() call by write_run_artifacts, not used by ordinary
    # (non-resumed) runs.
    update_count_offset: int = 0
    next_episode_seed: int | None = None
    final_rng_state: torch.Tensor | None = None
    # Compact accumulators replacing all_records for summary.json (spec
    # sections 1/9): consultation/advice statistics and action-type
    # frequencies, updated once per rollout from that rollout's
    # decision_rows before the underlying StepRecords are discarded.
    consultation_stats: ConsultationStats = field(default_factory=ConsultationStats)
    action_type_counts: Counter = field(default_factory=Counter)
    total_training_seconds: float = 0.0
    total_collection_seconds: float = 0.0
    total_optimization_seconds: float = 0.0
    total_evaluation_seconds: float = 0.0
    # checkpoint_best.pt's running criterion (spec section 10): goal success
    # rate primary, mean eval return as tie-breaker. Only meaningful when
    # periodic evaluation is enabled; -1.0 sorts below any real rate so the
    # first eval always counts as "best so far".
    best_eval_goal_success_rate: float = -1.0
    best_eval_mean_return: float = float("-inf")


def build_policy_and_optimizer(
    config: Config, device: torch.device, consultation_enabled: bool = False
) -> tuple[RecurrentPolicy, torch.optim.Optimizer]:
    """Adam only (never AdamW), with an explicit ``eps`` from
    ``config.policy.ppo.optimizer`` -- spec section 11. The config model
    defaults ``eps`` to ``1e-5`` and rejects any ``optimizer.type`` other
    than ``"adam"`` at validation time, so there is nothing to branch on
    here.
    """
    policy = RecurrentPolicy(config.policy, consultation_enabled=consultation_enabled).to(device)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=config.policy.ppo.learning_rate, eps=config.policy.ppo.optimizer.eps
    )
    return policy, optimizer


def _is_better_checkpoint(
    goal_success_rate: float, mean_return: float | None, best_goal_success_rate: float, best_mean_return: float
) -> bool:
    """checkpoint_best.pt's selection criterion (spec section 10): goal
    success rate is primary, mean (benchmark) return only breaks a tie --
    a checkpoint that solves the objective more often is "better" even if
    its return happens to be lower, and return is otherwise not a
    comparable quantity to success rate.
    """
    if goal_success_rate != best_goal_success_rate:
        return goal_success_rate > best_goal_success_rate
    return (mean_return or float("-inf")) > best_mean_return


async def run_training_loop(
    policy: RecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    adapter: NasimEmuAdapter,
    run_id: str,
    ppo_config,
    sequence_length: int,
    num_rollouts: int,
    device: torch.device,
    seed: int,
    consultation_enabled: bool = False,
    consultation_cost: float = 0.0,
    consult_fn: ConsultFn | None = None,
    stop_event: asyncio.Event | None = None,
    eval_episodes: int = 0,
    eval_every_rollouts: int = 1,
    initial_environment_steps: int = 0,
    initial_update_count: int = 0,
    config: Config | None = None,
    run_dir: Path | None = None,
) -> TrainingResult:
    """Repeated rollout-collect/PPO-update cycles over pre-built components.

    Used both by :func:`run_baseline_training` (which builds everything
    itself, for standalone use and tests) and by the SPADE RL Orchestrator,
    which resolves the device and constructs the policy *before* connecting
    to XMPP -- a local-model failure must prevent the agent from ever
    reporting itself ready (spec section 18).

    ``initial_environment_steps``/``initial_update_count`` are nonzero only
    when resuming from a checkpoint (research/aamas2027): ``seed`` is then
    the checkpoint's saved ``next_episode_seed`` (continuing the episode-seed
    sequence, not repeating it), and every ``environment_steps`` value
    reported in ``updates.csv`` reflects the true cumulative total across
    both the original and resumed runs, not just this invocation's own new
    steps -- so it stays a meaningful x-axis for a stitched learning curve.

    ``run_dir``/``config``: when both are given, this rollout's results are
    flushed to disk incrementally (spec section 2) via
    ``metrics.writer.append_rollout_metrics``, and ``checkpoint_last.pt``
    (plus ``checkpoint_best.pt`` when periodic evaluation is enabled) is
    kept up to date at every rollout boundary. When either is ``None``
    (e.g. most unit tests, and the assisted-variant test fixtures that
    build a ``TrainingResult`` directly), nothing is written during the
    run -- the caller is expected to persist the returned ``TrainingResult``
    itself afterward (see ``metrics.writer.write_run_artifacts``).
    """
    # Imported lazily: metrics.writer imports TrainingResult from this
    # module, so importing it at module scope would be circular.
    from marla.metrics.writer import append_rollout_metrics, build_decision_rows, build_rollout_row

    collector = RolloutCollector(
        adapter=adapter,
        policy=policy,
        run_id=run_id,
        base_seed=seed,
        consultation_enabled=consultation_enabled,
        consultation_cost=consultation_cost,
        consult_fn=consult_fn,
        stop_event=stop_event,
    )
    rng = random.Random(seed)

    result = TrainingResult(
        policy=policy, optimizer=optimizer,
        environment_steps=initial_environment_steps,
        update_count_offset=initial_update_count,
    )
    training_start = time.monotonic()
    incremental = run_dir is not None and config is not None

    # A *separate* NasimEmuAdapter for periodic evaluation, never the
    # training `adapter` above. NASimEmuEnv is genuinely mutable live state
    # (the current scenario instance, step index, ...), and for a scenario
    # whose host count varies across resets (a "ranges of hosts" scenario
    # file -- most bundled ones, including the default), an evaluation
    # pass's reset()/step() calls on a *shared* adapter would silently
    # overwrite the training collector's live environment out from under
    # its own paused, resumed-next-rollout mid-episode state (observed:
    # an IndexError inside compute_state_delta on the very next training
    # step, from a host-row array that had shrunk between two supposedly
    # consecutive steps of the *same* episode). Cheap to construct (no
    # heavyweight setup -- see NasimEmuAdapter.__init__).
    eval_adapter = (
        NasimEmuAdapter(
            scenario=adapter.scenario,
            max_episode_steps=adapter.max_episode_steps,
            completion_reward=adapter.completion_reward,
            premature_finish_penalty=adapter.premature_finish_penalty,
            premature_finish_penalty_per_remaining_target=adapter.premature_finish_penalty_per_remaining_target,
        )
        if eval_episodes > 0
        else None
    )

    for rollout_index in range(1, num_rollouts + 1):
        logger.info("Rollout %d/%d: collecting %d environment steps...", rollout_index, num_rollouts, ppo_config.rollout_steps)
        environment_steps_before = result.environment_steps

        collection_start = time.monotonic()
        records, summaries = await collector.collect(ppo_config.rollout_steps)
        collection_seconds = time.monotonic() - collection_start

        for summary in summaries:
            summary.rollout = rollout_index
        for offset, record in enumerate(records):
            record.rollout = rollout_index
            record.global_environment_step = environment_steps_before + offset
        result.episode_summaries.extend(summaries)
        result.environment_steps += len(records)
        result.total_collection_seconds += collection_seconds
        logger.info(
            "Rollout %d/%d collected: %d steps this rollout, %d total, %d episode(s) finished so far",
            rollout_index, num_rollouts, len(records), result.environment_steps, len(result.episode_summaries),
        )

        optimization_seconds = 0.0
        decision_rows: list[dict[str, Any]] = []
        update_metrics: list[dict[str, Any]] = []
        rollout_row: dict[str, Any] | None = None

        if records:
            rewards = [r.training_reward for r in records]
            values = [r.critic_value for r in records]
            terminated = [r.terminated for r in records]
            truncated = [r.truncated for r in records]
            bootstrap_values = [r.bootstrap_value for r in records]

            advantages, returns = compute_gae(
                rewards, values, terminated, truncated, bootstrap_values, ppo_config.gamma, ppo_config.gae_lambda
            )

            # Linear LR schedule (spec section 12): progress is measured in
            # environment steps completed *before* this rollout, against the
            # config's total budget -- evaluated once per rollout, not once
            # per minibatch, so it never depends on how many sequence chunks
            # a rollout's episode boundaries happen to produce.
            current_lr = compute_learning_rate(
                ppo_config.learning_rate_schedule, ppo_config.learning_rate,
                environment_steps_before, ppo_config.total_environment_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            logger.info("Rollout %d/%d: running %d PPO epoch(s)...", rollout_index, num_rollouts, ppo_config.epochs)
            optimization_start = time.monotonic()
            metrics = optimize(
                policy=policy,
                optimizer=optimizer,
                records=records,
                advantages=advantages,
                returns=returns,
                ppo_config=ppo_config,
                sequence_length=sequence_length,
                minibatch_sequences=ppo_config.minibatch_sequences,
                device=device,
                rng=rng,
                consultation_cost=consultation_cost,
            )
            optimization_seconds = time.monotonic() - optimization_start
            result.total_optimization_seconds += optimization_seconds
            logger.info("Rollout %d/%d: PPO update complete", rollout_index, num_rollouts)

            for update_metric in metrics:
                update_metric["update"] = result.update_count_offset + len(result.update_metrics) + len(update_metrics) + 1
                update_metric["rollout"] = rollout_index
                update_metric["environment_steps"] = result.environment_steps
                update_metric["learning_rate"] = optimizer.param_groups[0]["lr"]
                update_metrics.append(update_metric)
            result.update_metrics.extend(update_metrics)

            # Extract this rollout's compact metrics from `records` *before*
            # it goes out of scope at the end of this loop body (spec
            # sections 1/3/6) -- nothing below this point touches a
            # StepRecord's tensors again.
            decision_rows = build_decision_rows(records, advantages, returns)
            result.decision_rows.extend(decision_rows)
            result.consultation_stats.update(decision_rows)
            result.action_type_counts.update(r.selected_action_type for r in records)

            rollout_row = build_rollout_row(
                run_id=run_id,
                rollout=rollout_index,
                environment_steps_total=result.environment_steps,
                records=records,
                episodes_finished=len(summaries),
                advantages=advantages,
                returns=returns,
                collection_seconds=collection_seconds,
                optimization_seconds=optimization_seconds,
                evaluation_seconds=None,  # filled in below if evaluation runs this rollout
            )

        eval_summaries: list[EpisodeSummary] = []
        evaluation_seconds: float | None = None
        if eval_episodes > 0 and rollout_index % eval_every_rollouts == 0:
            logger.info(
                "Rollout %d/%d: running %d deterministic evaluation episode(s)...",
                rollout_index, num_rollouts, eval_episodes,
            )
            evaluation_start = time.monotonic()
            assert eval_adapter is not None
            eval_summaries = await run_evaluation_episodes(
                policy=policy,
                adapter=eval_adapter,
                run_id=run_id,
                num_episodes=eval_episodes,
                seed_start=EVAL_SEED_OFFSET,  # same fixed episodes every checkpoint -- an apples-to-apples test set
                consultation_enabled=consultation_enabled,
                consultation_cost=consultation_cost,
                consult_fn=consult_fn,
            )
            evaluation_seconds = time.monotonic() - evaluation_start
            result.total_evaluation_seconds += evaluation_seconds
            for summary in eval_summaries:
                summary.rollout = rollout_index
            result.eval_episode_summaries.extend(eval_summaries)
            if eval_summaries:
                logger.info(
                    "Rollout %d/%d: evaluation mean return %.2f over %d episode(s)",
                    rollout_index, num_rollouts,
                    sum(s.nasimemu_return for s in eval_summaries) / len(eval_summaries),
                    len(eval_summaries),
                )

        if rollout_row is not None:
            rollout_row["evaluation_seconds"] = evaluation_seconds
            result.rollout_rows.append(rollout_row)

        result.total_training_seconds = time.monotonic() - training_start

        if incremental:
            append_rollout_metrics(
                run_dir, config,
                [*summaries, *eval_summaries],
                decision_rows,
                rollout_row or _empty_rollout_row(run_id, rollout_index, result.environment_steps),
                update_metrics,
            )
            _update_checkpoints(run_dir, config, result, eval_summaries, collector.next_episode_seed, torch.get_rng_state())

        # Nothing below this point needs `records`/`decision_rows` again --
        # both go out of scope at the top of the next iteration (or at
        # function return), which is what keeps this loop's memory
        # footprint proportional to one rollout's worth of steps rather
        # than the whole training horizon (spec section 1).

        if stop_event is not None and stop_event.is_set():
            logger.info("Stop requested; ending training after %d environment step(s)", result.environment_steps)
            result.stopped_by_user = True
            break

    # Captured regardless of how the loop ended (ran to completion or
    # stopped early) so a checkpoint saved from either state can resume
    # correctly -- see write_run_artifacts' save_checkpoint call.
    result.next_episode_seed = collector.next_episode_seed
    result.final_rng_state = torch.get_rng_state()
    return result


def _empty_rollout_row(run_id: str, rollout: int, environment_steps_total: int) -> dict[str, Any]:
    """A rollout that collected zero records (e.g. stopped before its first
    step) still needs one rollouts.csv row so ``rollout`` numbering has no
    silent gaps; every aggregate is simply absent.
    """
    return {
        "run_id": run_id,
        "rollout": rollout,
        "environment_steps_total": environment_steps_total,
        "steps_collected": 0,
        "episodes_finished": 0,
        "mean_nasimemu_reward": None,
        "std_nasimemu_reward": None,
        "mean_training_reward": None,
        "std_training_reward": None,
        "mean_critic_value": None,
        "mean_return_target": None,
        "mean_advantage": None,
        "std_advantage": None,
        "mean_abs_advantage": None,
        "mean_query_probability": None,
        "actual_query_rate": None,
        "mean_beta": None,
        "collection_seconds": 0.0,
        "optimization_seconds": 0.0,
        "evaluation_seconds": None,
    }


def _update_checkpoints(
    run_dir: Path,
    config: Config,
    result: TrainingResult,
    eval_summaries: list[EpisodeSummary],
    next_episode_seed: int,
    rng_state: torch.Tensor,
) -> None:
    """Spec section 10: ``checkpoint_last.pt`` overwritten at every rollout
    boundary; ``checkpoint_best.pt`` overwritten only when this rollout's
    periodic evaluation (if any) beats the running best. No per-minibatch
    checkpointing -- this is called at most once per rollout.

    Includes the same resume fields (``next_episode_seed``/``rng_state``) a
    normal end-of-run checkpoint does, not just policy/optimizer weights --
    a process killed hard (no graceful shutdown) between rollouts still
    leaves a checkpoint that ``marla run --resume`` can continue from.
    """
    from marla.config.loader import config_hash

    update_count = result.update_count_offset + len(result.update_metrics)
    save_checkpoint(
        run_dir / "checkpoint_last.pt",
        result.policy,
        result.optimizer,
        update_count=update_count,
        environment_steps=result.environment_steps,
        config_hash=config_hash(config),
        next_episode_seed=next_episode_seed,
        rng_state=rng_state,
    )

    if not eval_summaries:
        return
    goal_success_rate = sum(1.0 for s in eval_summaries if s.goal_success) / len(eval_summaries)
    mean_return = sum(s.nasimemu_return for s in eval_summaries) / len(eval_summaries)
    if _is_better_checkpoint(goal_success_rate, mean_return, result.best_eval_goal_success_rate, result.best_eval_mean_return):
        result.best_eval_goal_success_rate = goal_success_rate
        result.best_eval_mean_return = mean_return
        save_checkpoint(
            run_dir / "checkpoint_best.pt",
            result.policy,
            result.optimizer,
            update_count=update_count,
            environment_steps=result.environment_steps,
            config_hash=config_hash(config),
        )


async def run_baseline_training(
    config: Config,
    scenario_path: str,
    num_rollouts: int,
    device: torch.device | None = None,
    seed: int | None = None,
) -> TrainingResult:
    """Run ``num_rollouts`` collect/optimize cycles of baseline recurrent PPO."""
    if config.consultation.mode != "disabled":
        raise ValueError("run_baseline_training requires consultation.mode == 'disabled'")

    resolved_device = device or torch.device("cpu")
    resolved_seed = seed if seed is not None else config.experiment.seed
    torch.manual_seed(resolved_seed)

    policy, optimizer = build_policy_and_optimizer(config, resolved_device, consultation_enabled=False)
    adapter = NasimEmuAdapter(
        scenario=scenario_path,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
        premature_finish_penalty_per_remaining_target=config.objective.premature_finish_penalty_per_remaining_target,
    )

    return await run_training_loop(
        policy=policy,
        optimizer=optimizer,
        adapter=adapter,
        run_id=config.experiment.run_id or config.experiment.name,
        ppo_config=config.policy.ppo,
        sequence_length=config.policy.recurrent.sequence_length,
        num_rollouts=num_rollouts,
        device=resolved_device,
        seed=resolved_seed,
        eval_episodes=config.metrics.eval_episodes,
        eval_every_rollouts=config.metrics.eval_every_rollouts,
    )
