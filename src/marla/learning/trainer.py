"""Recurrent PPO training loop, shared by the baseline and assisted variants.

Ties together :class:`RolloutCollector`, :func:`compute_gae`, and
:func:`optimize` into repeated rollout-collect/PPO-update cycles. Native
``async def`` throughout: the assisted variant's per-step consultation is a
real ``await`` on the Gatekeeper round-trip, and running natively on the
event loop (rather than in a background thread) is what lets other
local-mode agents' behaviours keep being serviced during training. The
SPADE RL Orchestrator (Milestone 5) awaits this directly; this module is
also directly usable standalone for the baseline variant and for tests.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field

import torch

from marla.config.models import Config
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.learning.gae import compute_gae
from marla.learning.ppo import optimize
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import (
    EVAL_SEED_OFFSET,
    ConsultFn,
    EpisodeSummary,
    RolloutCollector,
    StepRecord,
    run_evaluation_episodes,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    policy: RecurrentPolicy
    optimizer: torch.optim.Optimizer
    episode_summaries: list[EpisodeSummary] = field(default_factory=list)
    update_metrics: list[dict[str, float]] = field(default_factory=list)
    environment_steps: int = 0
    all_records: list[StepRecord] = field(default_factory=list)
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


def build_policy_and_optimizer(
    config: Config, device: torch.device, consultation_enabled: bool = False
) -> tuple[RecurrentPolicy, torch.optim.Optimizer]:
    policy = RecurrentPolicy(config.policy, consultation_enabled=consultation_enabled).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy.ppo.learning_rate)
    return policy, optimizer


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
    """
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

    for rollout_index in range(1, num_rollouts + 1):
        logger.info("Rollout %d/%d: collecting %d environment steps...", rollout_index, num_rollouts, ppo_config.rollout_steps)
        records, summaries = await collector.collect(ppo_config.rollout_steps)
        for summary in summaries:
            summary.rollout = rollout_index
        result.episode_summaries.extend(summaries)
        result.environment_steps += len(records)
        result.all_records.extend(records)
        logger.info(
            "Rollout %d/%d collected: %d steps this rollout, %d total, %d episode(s) finished so far",
            rollout_index, num_rollouts, len(records), result.environment_steps, len(result.episode_summaries),
        )

        if records:
            rewards = [r.training_reward for r in records]
            values = [r.critic_value for r in records]
            terminated = [r.terminated for r in records]
            truncated = [r.truncated for r in records]
            bootstrap_values = [r.bootstrap_value for r in records]

            advantages, returns = compute_gae(
                rewards, values, terminated, truncated, bootstrap_values, ppo_config.gamma, ppo_config.gae_lambda
            )

            logger.info("Rollout %d/%d: running %d PPO epoch(s)...", rollout_index, num_rollouts, ppo_config.epochs)
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
            logger.info("Rollout %d/%d: PPO update complete", rollout_index, num_rollouts)

            # Per-rollout aggregates (spec section 21's updates.csv): the
            # same values for every ppo_update() call within this rollout,
            # since they all trained on the same collected batch.
            betas = [r.beta for r in records if r.beta is not None]
            query_probabilities = [r.old_query_probability for r in records if r.old_query_probability is not None]
            mean_nasimemu_reward = sum(r.nasimemu_reward for r in records) / len(records)
            mean_training_reward = sum(r.training_reward for r in records) / len(records)
            elapsed = time.monotonic() - training_start
            for update_index, update_metrics in enumerate(metrics, start=1):
                update_metrics["update"] = result.update_count_offset + len(result.update_metrics) + update_index
                update_metrics["environment_steps"] = result.environment_steps
                update_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
                update_metrics["mean_beta"] = sum(betas) / len(betas) if betas else None
                update_metrics["mean_query_probability"] = (
                    sum(query_probabilities) / len(query_probabilities) if query_probabilities else None
                )
                update_metrics["actual_query_rate"] = sum(r.sampled_query for r in records) / len(records)
                # The raw per-step reward signal this rollout actually trained
                # on -- distinct from episodes.csv's per-episode *return*
                # (the sum of these over a whole episode). mean_nasimemu_reward
                # is the unmodified NASimEmu reward (comparable across variants);
                # mean_training_reward additionally reflects consultation-cost
                # deductions, so the two only diverge for the assisted variant.
                update_metrics["mean_nasimemu_reward"] = mean_nasimemu_reward
                update_metrics["mean_training_reward"] = mean_training_reward
                update_metrics["elapsed_training_seconds"] = elapsed
                update_metrics["checkpoint_id"] = None  # no checkpointing wired into this run loop yet
            result.update_metrics.extend(metrics)

        if eval_episodes > 0 and rollout_index % eval_every_rollouts == 0:
            logger.info(
                "Rollout %d/%d: running %d deterministic evaluation episode(s)...",
                rollout_index, num_rollouts, eval_episodes,
            )
            eval_summaries = await run_evaluation_episodes(
                policy=policy,
                adapter=adapter,
                run_id=run_id,
                num_episodes=eval_episodes,
                seed_start=EVAL_SEED_OFFSET,  # same fixed episodes every checkpoint -- an apples-to-apples test set
                consultation_enabled=consultation_enabled,
                consultation_cost=consultation_cost,
                consult_fn=consult_fn,
            )
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
