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
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from marla.config.models import CarbonConfig, Config
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.learning.checkpoint import save_checkpoint
from marla.learning.gae import compute_gae
from marla.learning.lr_scheduler import build_scheduler, current_learning_rate
from marla.learning.ppo import compute_policy_probe, critic_refinement, optimize
from marla.monitoring.carbon import CarbonSummary, CarbonTracker, write_carbon_summary
from marla.monitoring.resources import ResourceMonitor
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import (
    EVAL_SEED_OFFSET,
    ConsultFn,
    EpisodeSummary,
    RolloutCollector,
    env_base_seed,
    env_episode_id_offset,
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
    # The LR scheduler this run actually used (built fresh inside
    # run_training_loop from ppo_config.optimizer.scheduler -- see
    # learning/lr_scheduler.py). None only for a TrainingResult built
    # directly by a test/fixture that never called run_training_loop.
    # Exposed here so save_checkpoint (metrics/writer.py,
    # learning/trainer.py's own periodic-checkpoint helper) can persist
    # scheduler.state_dict() without threading a separate parameter
    # through every checkpoint call site.
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    # CPU/RAM/GPU telemetry for this run (monitoring/resources.py). None
    # only for a TrainingResult built directly by a test/fixture that
    # never called run_training_loop (same status as `scheduler` above).
    resource_monitor: "ResourceMonitor | None" = None
    # Per-agent energy/CO2eq tracking for this run (monitoring/carbon.py).
    # Same None-only-for-bare-test-fixtures status as resource_monitor.
    carbon_tracker: "CarbonTracker | None" = None
    # Set once, at the very end of run_training_loop (carbon_tracker.stop()
    # can only be called once -- unlike resource_monitor, which can be
    # queried live via .samples at any time).
    carbon_summary: "CarbonSummary | None" = None


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
        policy.parameters(), lr=config.policy.ppo.optimizer.learning_rate, eps=config.policy.ppo.optimizer.eps
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


_PROBE_FIELDS = (
    "action_entropy", "mean_finish_probability", "max_finish_probability",
    "mean_finish_probability_objective_reached", "mean_finish_probability_objective_not_reached",
    "mean_max_action_probability", "max_max_action_probability",
    "mean_abs_logit", "max_abs_logit",
    "value_prediction_mean", "value_prediction_std", "value_prediction_min", "value_prediction_max",
    # Visible-state-conditioned (State N/I/C -- the no-visible-target-FINISH
    # collapse investigation, spec sections 9/23): policy-observable, a
    # different axis from the objective_reached/not_reached split above
    # (hidden evaluator truth). *_count fields let a reader tell "0.0
    # because FINISH probability was truly low" apart from "no decisions in
    # this state occurred this rollout" (mean would be None in the latter).
    "mean_finish_probability_state_N", "finish_probability_count_state_N",
    "mean_finish_probability_state_I", "finish_probability_count_state_I",
    "mean_finish_probability_state_C", "finish_probability_count_state_C",
    "mean_max_action_probability_state_N", "mean_max_action_probability_state_I", "mean_max_action_probability_state_C",
    "mean_entropy_state_N", "mean_entropy_state_I", "mean_entropy_state_C",
    "mean_value_state_N", "mean_value_state_I", "mean_value_state_C",
    "mean_abs_logit_state_N", "mean_abs_logit_state_I", "mean_abs_logit_state_C",
    "decision_count_state_N", "decision_count_state_I", "decision_count_state_C",
)


def _diff_policy_probe(before: dict[str, float | None], after: dict[str, float | None]) -> dict[str, float | None]:
    """Renames :func:`marla.learning.ppo.compute_policy_probe`'s two raw
    dicts into ``rollouts.csv``'s ``*_before_update``/``*_after_update``
    columns and adds the one derived quantity spec section 11 asks for --
    ``action_entropy_change`` (and, symmetrically, ``finish_probability_change``)
    -- ``None`` whenever either side is ``None`` (an empty-denominator
    diagnostic, e.g. no FINISH action was legal this rollout, is not a
    numeric 0 change).
    """

    def _delta(key: str) -> float | None:
        b, a = before.get(key), after.get(key)
        return (a - b) if (b is not None and a is not None) else None

    row: dict[str, float | None] = {}
    for field_name in _PROBE_FIELDS:
        row[f"{field_name}_before_update"] = before.get(field_name)
        row[f"{field_name}_after_update"] = after.get(field_name)
    row["action_entropy_change"] = _delta("action_entropy")
    row["finish_probability_change"] = _delta("mean_finish_probability")
    row["finish_probability_state_N_change"] = _delta("mean_finish_probability_state_N")
    row["finish_probability_state_I_change"] = _delta("mean_finish_probability_state_I")
    row["finish_probability_state_C_change"] = _delta("mean_finish_probability_state_C")
    return row


def compute_gae_per_stream(
    per_env_records: list[list], gamma: float, gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """GAE computed independently PER STREAM (spec section 20: never
    concatenate raw environment timelines and compute GAE as though one
    followed another), then concatenated in the same stream order as
    ``per_env_records``. For the single-stream case
    (``per_env_records == [records]``) this reduces to exactly one
    ``compute_gae`` call on the full list -- mathematically identical to
    the pre-multi-env code (spec section 6).
    """
    advantages: list[float] = []
    returns: list[float] = []
    for env_records in per_env_records:
        if not env_records:
            continue
        rewards = [r.training_reward for r in env_records]
        values = [r.critic_value for r in env_records]
        terminated = [r.terminated for r in env_records]
        truncated = [r.truncated for r in env_records]
        bootstrap_values = [r.bootstrap_value for r in env_records]
        env_advantages, env_returns = compute_gae(
            rewards, values, terminated, truncated, bootstrap_values, gamma, gae_lambda
        )
        advantages.extend(env_advantages)
        returns.extend(env_returns)
    return advantages, returns


async def run_training_loop(
    policy: RecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    adapter: NasimEmuAdapter | list[NasimEmuAdapter],
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
    initial_scheduler_state: dict[str, Any] | None = None,
    config: Config | None = None,
    run_dir: Path | None = None,
    collapse_diagnostics: bool = False,
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
    ``initial_scheduler_state`` (``scheduler.state_dict()`` from that same
    checkpoint) restores the LR scheduler's exact trajectory the same way --
    a fresh scheduler is always built here from ``ppo_config`` (never
    persisted/reconstructed some other way), then this state is loaded into
    it before the first rollout of THIS invocation runs, so a resumed run's
    LR continues exactly where the original left off rather than
    restarting the schedule's horizon from this invocation's own rollout 1.

    ``run_dir``/``config``: when both are given, this rollout's results are
    flushed to disk incrementally (spec section 2) via
    ``metrics.writer.append_rollout_metrics``, and ``checkpoint_last.pt``
    (plus ``checkpoint_best.pt`` when periodic evaluation is enabled) is
    kept up to date at every rollout boundary. When either is ``None``
    (e.g. most unit tests, and the assisted-variant test fixtures that
    build a ``TrainingResult`` directly), nothing is written during the
    run -- the caller is expected to persist the returned ``TrainingResult``
    itself afterward (see ``metrics.writer.write_run_artifacts``).

    ``collapse_diagnostics`` (the seq32-vs-64 PPO collapse investigation,
    opt-in, default ``False`` -- production ``marla run`` behavior and
    performance are unchanged unless a caller explicitly asks for this):
    when ``True``, runs :func:`marla.learning.ppo.compute_policy_probe`
    once immediately before this rollout's ``optimize()`` call and once
    immediately after, over the exact same stored records -- an
    apples-to-apples "how much did this one PPO update change the
    policy" measurement, attached to this rollout's row via
    ``*_before_update``/``*_after_update``/``*_change`` fields (``None``
    when disabled). Never calls the environment or the Plan Maker again;
    an additional cost of two no-grad forward passes over the rollout's
    own already-collected observations.

    ``adapter``: a single :class:`NasimEmuAdapter` (every existing call
    site -- exact unchanged single-environment behavior), OR a
    ``list[NasimEmuAdapter]`` of ``ppo_config.num_envs`` independently-
    constructed adapters (spec: the 2048-transition, 4-independent-
    environment-stream ablation). Each list entry gets its own
    :class:`RolloutCollector` (own recurrent-hidden-state chain, own
    deterministically-derived seed range via
    :func:`marla.learning.rollout.env_base_seed`, own globally-unique
    episode-ID range via
    :func:`marla.learning.rollout.env_episode_id_offset`) -- collected
    SEQUENTIALLY (wall-clock parallelism is not required for scientific
    correctness -- spec section 4), all under the SAME frozen ``policy``
    parameters, since nothing here takes an optimizer step until every
    stream has finished collecting this rollout's share. GAE is computed
    independently per stream (never across a stream/episode boundary --
    spec section 20) before the resulting advantages/returns are
    concatenated, in the same stream order as ``records``, for one shared
    PPO update over the combined batch. For ``len(adapter) == 1`` (or a
    bare single adapter), this reduces to exactly one stream's own GAE
    call on the full ``records`` list -- mathematically and
    computationally identical to the pre-multi-env code path (spec
    section 6's byte-identical-behavior requirement).
    """
    # Imported lazily: metrics.writer imports TrainingResult from this
    # module, so importing it at module scope would be circular.
    from marla.metrics.writer import append_rollout_metrics, build_decision_rows, build_rollout_row, write_resource_artifacts

    adapters = adapter if isinstance(adapter, list) else [adapter]
    # Every collector shares run_id verbatim (never a per-stream suffix):
    # decisions.csv's own run_id column is documented elsewhere as a
    # per-RUN constant, not a per-row-varying value -- StepRecord.env_index
    # (spec section 41) is the correct, additive field for distinguishing
    # streams, so run_id's existing meaning is never disturbed.
    collectors = [
        RolloutCollector(
            adapter=env_adapter,
            policy=policy,
            run_id=run_id,
            base_seed=env_base_seed(seed, i),
            consultation_enabled=consultation_enabled,
            consultation_cost=consultation_cost,
            consult_fn=consult_fn,
            stop_event=stop_event,
            episode_id_offset=env_episode_id_offset(i),
            env_index=i if len(adapters) > 1 else None,
        )
        for i, env_adapter in enumerate(adapters)
    ]
    rng = random.Random(seed)

    # Built fresh every invocation, from this run's own ppo_config (never
    # persisted/reconstructed by any other means) -- see
    # learning/lr_scheduler.py for stepping/horizon semantics. On resume,
    # `initial_scheduler_state` restores its exact trajectory (last_epoch,
    # base_lrs, ...); the optimizer's LR is synced explicitly afterward
    # since load_state_dict alone does not do this until the next .step().
    scheduler = build_scheduler(optimizer, ppo_config)
    if initial_scheduler_state is not None:
        scheduler.load_state_dict(initial_scheduler_state)
        optimizer.param_groups[0]["lr"] = scheduler.get_last_lr()[0]

    # CPU/RAM/GPU telemetry (spec: resource accounting is a first-class
    # research metric). Config-driven when a real Config is available;
    # disabled for the many bare unit-test call sites that pass no
    # `config` at all, to avoid spinning up a background thread those
    # tests never asked for and never read the output of.
    resource_monitor = ResourceMonitor(
        sampling_interval_seconds=config.metrics.resource_monitoring.sampling_interval_seconds if config else 1.0,
        enabled=config.metrics.resource_monitoring.enabled if config else False,
    )
    resource_monitor.start()

    # Per-agent CodeCarbon energy/CO2eq tracking (spec: "one agent = one
    # MARLA run, one config, one seed"). Config-driven, same pattern as
    # resource_monitor above -- disabled for bare unit-test call sites
    # with no real Config. Its own output_dir defaults to a throwaway
    # temp directory when no run_dir is given (carbon tracking still
    # works in-memory; there is just nowhere persistent to put
    # CodeCarbon's own raw CSV housekeeping files).
    carbon_output_dir = (run_dir / "carbon") if run_dir is not None else Path(tempfile.mkdtemp(prefix="marla_carbon_"))
    carbon_tracker = CarbonTracker(
        config=config.carbon if config else CarbonConfig(enabled=False),
        output_dir=carbon_output_dir,
        project_name=run_id,
    )
    carbon_tracker.start()
    carbon_tracker.start_task("training")

    result = TrainingResult(
        policy=policy, optimizer=optimizer, scheduler=scheduler, resource_monitor=resource_monitor,
        carbon_tracker=carbon_tracker,
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
            scenario=adapters[0].scenario,
            max_episode_steps=adapters[0].max_episode_steps,
            completion_reward=adapters[0].completion_reward,
            premature_finish_penalty=adapters[0].premature_finish_penalty,
            premature_finish_penalty_per_remaining_target=adapters[0].premature_finish_penalty_per_remaining_target,
        )
        if eval_episodes > 0
        else None
    )

    try:
        for rollout_index in range(1, num_rollouts + 1):
            logger.info(
                "Rollout %d/%d: collecting %d environment steps (%d stream(s) x %d)...",
                rollout_index, num_rollouts, ppo_config.steps_per_env * len(collectors), len(collectors), ppo_config.steps_per_env,
            )
            environment_steps_before = result.environment_steps

            resource_monitor.set_context(
                phase="rollout_collection", global_env_step=environment_steps_before, ppo_update=rollout_index
            )
            collection_start = time.monotonic()
            # Sequential, not wall-clock-parallel (spec section 4: scientific
            # correctness does not require actual parallel execution) -- every
            # stream collects under the exact same, still-unmodified `policy`
            # parameters, since no optimizer step happens until every stream
            # here has finished and `optimize()` runs once, below, over the
            # combined batch. `per_env_records` always has exactly
            # `len(collectors)` entries -- length 1 for the ordinary single-
            # environment path, reducing every loop below to one iteration
            # over the full `records` list (spec section 6: byte-identical to
            # the pre-multi-env code for that case).
            per_env_records: list[list] = []
            summaries: list[EpisodeSummary] = []
            for env_collector in collectors:
                env_records, env_summaries = await env_collector.collect(ppo_config.steps_per_env)
                per_env_records.append(env_records)
                summaries.extend(env_summaries)
            records = [r for env_records in per_env_records for r in env_records]
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
                advantages, returns = compute_gae_per_stream(per_env_records, ppo_config.gamma, ppo_config.gae_lambda)

                # The LR this update actually uses -- and the scheduler's own
                # step count at that moment -- captured BEFORE optimize() and
                # BEFORE scheduler.step() below, so update_metric rows record
                # exactly what this update used, not what the NEXT update will
                # use (learning/lr_scheduler.py's stepping-semantics docstring:
                # one .step() per COMPLETED update, called after optimize()).
                lr_this_update = current_learning_rate(optimizer)
                scheduler_step_this_update = scheduler.last_epoch

                probe_before = compute_policy_probe(policy, records, device) if collapse_diagnostics else None

                logger.info("Rollout %d/%d: running %d PPO epoch(s)...", rollout_index, num_rollouts, ppo_config.epochs)
                resource_monitor.set_context(phase="ppo_update")
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

                # Same stored `records` as probe_before -- exactly the "before
                # vs. after this rollout's PPO update" comparison spec sections
                # 11-14 ask for, never a fresh environment/simulator query.
                probe_after_actor = compute_policy_probe(policy, records, device) if collapse_diagnostics else None
                probe_diagnostics = _diff_policy_probe(probe_before, probe_after_actor) if collapse_diagnostics else None

                for update_metric in metrics:
                    update_metric["update"] = result.update_count_offset + len(result.update_metrics) + len(update_metrics) + 1
                    update_metric["rollout"] = rollout_index
                    update_metric["environment_steps"] = result.environment_steps
                    update_metric["learning_rate"] = lr_this_update
                    update_metric["scheduler_type"] = ppo_config.optimizer.scheduler.type
                    update_metric["scheduler_step"] = scheduler_step_this_update
                    update_metrics.append(update_metric)
                result.update_metrics.extend(update_metrics)

                # Exactly one scheduler.step() per COMPLETED PPO update (spec:
                # never once per minibatch/epoch) -- after optimize() so this
                # update itself used lr_this_update above, and before
                # critic_refinement below (an experimental, additive, off-by-
                # default extra pass that is not part of the actor+critic
                # update the schedule paces against).
                scheduler.step()

                # Critic-only refinement (spec: State-N critic-calibration
                # investigation) -- runs strictly AFTER the normal actor+critic
                # PPO update above, never interleaved with it, using the exact
                # same records/advantages/returns (no new GAE computation, no
                # Monte Carlo/oracle data -- see ppo.critic_refinement's own
                # docstring). A no-op when ppo_config.critic_refinement_epochs
                # is 0 (the default), which is what keeps every existing
                # config's behavior byte-identical to before this feature
                # existed.
                refinement_metrics = critic_refinement(
                    policy=policy, optimizer=optimizer, records=records, advantages=advantages, returns=returns,
                    sequence_length=sequence_length, minibatch_sequences=ppo_config.minibatch_sequences,
                    critic_refinement_epochs=ppo_config.critic_refinement_epochs, max_grad_norm=ppo_config.max_grad_norm,
                    device=device, rng=rng,
                )
                probe_after_refinement = (
                    compute_policy_probe(policy, records, device)
                    if collapse_diagnostics and refinement_metrics else None
                )
                refinement_probe_diagnostics = (
                    _diff_policy_probe(probe_after_actor, probe_after_refinement)
                    if probe_after_refinement is not None else None
                )
                if refinement_metrics:
                    # `result.update_metrics` was already extended with this
                    # rollout's actor rows just above (line ~373), so their
                    # count is NOT added again here (that would double-count
                    # and skip numbers) -- only this loop's own running index
                    # advances the sequence, mirroring the actor loop's own
                    # pattern above but against the post-actor-extend baseline.
                    for index, update_metric in enumerate(refinement_metrics):
                        update_metric["update"] = result.update_count_offset + len(result.update_metrics) + index + 1
                        update_metric["rollout"] = rollout_index
                        update_metric["environment_steps"] = result.environment_steps
                        update_metric["learning_rate"] = optimizer.param_groups[0]["lr"]
                    result.update_metrics.extend(refinement_metrics)
                    # Bugfix: `append_rollout_metrics` below is given the LOCAL
                    # `update_metrics` list (this rollout's rows only, for
                    # incremental per-rollout CSV flushing) -- without also
                    # extending it here, critic-refinement update rows were
                    # computed and numbered correctly in `result.update_metrics`
                    # but silently never reached updates.csv on disk for any
                    # run using incremental persistence (i.e. every real
                    # training run; spec section 25 requires these to actually
                    # be persisted).
                    update_metrics.extend(refinement_metrics)

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
                    decision_rows=decision_rows,
                )
                if probe_diagnostics is not None:
                    rollout_row.update(probe_diagnostics)
                if refinement_probe_diagnostics is not None:
                    # Re-keyed with a "refinement_" prefix -- _diff_policy_probe's
                    # generic *_before_update/*_after_update/*_change names
                    # would otherwise collide with the actor-update diff
                    # merged in just above (spec section 26: keep the
                    # within-critic-refinement-phase probe distinct from the
                    # within-actor-update one).
                    rollout_row.update({f"refinement_{k}": v for k, v in refinement_probe_diagnostics.items()})

            eval_summaries: list[EpisodeSummary] = []
            evaluation_seconds: float | None = None
            if eval_episodes > 0 and rollout_index % eval_every_rollouts == 0:
                logger.info(
                    "Rollout %d/%d: running %d deterministic evaluation episode(s)...",
                    rollout_index, num_rollouts, eval_episodes,
                )
                resource_monitor.set_context(phase="evaluation")
                # CodeCarbon: close the currently-open "training" segment and
                # open an "evaluation" one for the duration of this eval pass
                # only, then reopen "training" immediately after -- MARLA's
                # periodic evaluation is interleaved with training, not a
                # single trailing phase, so this happens possibly many times
                # across one agent's run (see CarbonTracker.start_task's own
                # docstring for how repeated same-named tasks are handled).
                carbon_tracker.stop_task("training")
                carbon_tracker.start_task("evaluation")
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
                carbon_tracker.stop_task("evaluation")
                carbon_tracker.start_task("training")
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
                _update_checkpoints(run_dir, config, result, eval_summaries, collectors[0].next_episode_seed, torch.get_rng_state())
                # Full overwrite each rollout (small file, cheap) -- same
                # resilience contract as checkpoint_last.pt: a process killed
                # hard between rollouts still leaves resources.csv/
                # resource_summary.json reflecting everything sampled so far.
                write_resource_artifacts(run_dir, resource_monitor)

            # Nothing below this point needs `records`/`decision_rows` again --
            # both go out of scope at the top of the next iteration (or at
            # function return), which is what keeps this loop's memory
            # footprint proportional to one rollout's worth of steps rather
            # than the whole training horizon (spec section 1).

            if stop_event is not None and stop_event.is_set():
                logger.info("Stop requested; ending training after %d environment step(s)", result.environment_steps)
                result.stopped_by_user = True
                break

    except Exception:
        # Exception-safety net: a mid-training exception (e.g. the
        # NaN/Inf fail-fast assertion in optimize(), or an unrecoverable
        # environment error) must still stop both background trackers
        # and, where possible, persist whatever real partial carbon/
        # resource data CodeCarbon/ResourceMonitor already collected up
        # to the point of failure -- never silently discarded, and never
        # left running past this function's return. Each cleanup step is
        # independently best-effort: a SECONDARY failure here must never
        # mask the ORIGINAL exception this block exists to propagate.
        try:
            resource_monitor.stop()
            if incremental:
                write_resource_artifacts(run_dir, resource_monitor)
        except Exception:
            logger.exception("resource_monitor cleanup failed during exception handling")
        try:
            result.carbon_summary = carbon_tracker.stop()
            if incremental and result.carbon_summary is not None:
                write_carbon_summary(run_dir, result.carbon_summary, agent_id=run_id)
        except Exception:
            logger.exception("carbon_tracker cleanup failed during exception handling")
        raise
    # Captured regardless of how the loop ended (ran to completion or
    # stopped early) so a checkpoint saved from either state can resume
    # correctly -- see write_run_artifacts' save_checkpoint call.
    result.next_episode_seed = collectors[0].next_episode_seed  # multi-env resume is not a supported path in this phase -- only stream 0's seed is tracked
    result.final_rng_state = torch.get_rng_state()
    resource_monitor.stop()
    # Closes the currently-open "training" task (there always is one,
    # opened above and reopened after every evaluation pass) before the
    # tracker itself stops -- CarbonTracker.stop() would do this anyway,
    # but doing it explicitly here keeps the "one open task at a time"
    # invariant visible at the call site too.
    carbon_summary = carbon_tracker.stop()
    result.carbon_summary = carbon_summary
    if incremental:
        write_resource_artifacts(run_dir, resource_monitor)
        # No derived efficiency metrics here (root_auc/successful_finish
        # counts are a post-hoc analysis-layer concept -- see
        # research/optuna's own aggregation step, which augments this
        # file after computing ROOT AUC from episodes.csv/decisions.csv).
        write_carbon_summary(run_dir, carbon_summary, agent_id=run_id)
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
        critic_refinement_epochs=config.policy.ppo.critic_refinement_epochs,
        num_envs=config.policy.ppo.num_envs,
        steps_per_env=config.policy.ppo.steps_per_env,
        scheduler=result.scheduler,
        scheduler_config=config.policy.ppo.optimizer.scheduler.model_dump(),
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
            critic_refinement_epochs=config.policy.ppo.critic_refinement_epochs,
            num_envs=config.policy.ppo.num_envs,
            steps_per_env=config.policy.ppo.steps_per_env,
            scheduler=result.scheduler,
            scheduler_config=config.policy.ppo.optimizer.scheduler.model_dump(),
        )


async def run_baseline_training(
    config: Config,
    scenario_path: str,
    num_rollouts: int,
    device: torch.device | None = None,
    seed: int | None = None,
    collapse_diagnostics: bool = False,
    run_dir: Path | None = None,
) -> TrainingResult:
    """Run ``num_rollouts`` collect/optimize cycles of baseline recurrent PPO.

    ``collapse_diagnostics``/``run_dir`` are passed straight through to
    :func:`run_training_loop` (see its own docstring) -- both default to
    off/``None``, so every existing caller of this test/diagnostic helper
    is unaffected.
    """
    if config.consultation.mode != "disabled":
        raise ValueError("run_baseline_training requires consultation.mode == 'disabled'")

    resolved_device = device or torch.device("cpu")
    resolved_seed = seed if seed is not None else config.experiment.seed
    torch.manual_seed(resolved_seed)

    policy, optimizer = build_policy_and_optimizer(config, resolved_device, consultation_enabled=False)
    # One independent NasimEmuAdapter per environment stream (spec: the
    # 2048-transition, 4-independent-environment-stream ablation) --
    # num_envs=1 (every existing config) builds exactly one, matching the
    # pre-multi-env code exactly. Never a slice of one shared adapter.
    adapters = [
        NasimEmuAdapter(
            scenario=scenario_path,
            max_episode_steps=config.environment.max_episode_steps,
            completion_reward=config.objective.completion_reward,
            premature_finish_penalty=config.objective.premature_finish_penalty,
            premature_finish_penalty_per_remaining_target=config.objective.premature_finish_penalty_per_remaining_target,
        )
        for _ in range(config.policy.ppo.num_envs)
    ]

    return await run_training_loop(
        policy=policy,
        optimizer=optimizer,
        adapter=adapters if len(adapters) > 1 else adapters[0],
        run_id=config.experiment.run_id or config.experiment.name,
        ppo_config=config.policy.ppo,
        sequence_length=config.policy.recurrent.sequence_length,
        num_rollouts=num_rollouts,
        device=resolved_device,
        seed=resolved_seed,
        eval_episodes=config.metrics.eval_episodes,
        eval_every_rollouts=config.metrics.eval_every_rollouts,
        config=config,
        run_dir=run_dir,
        collapse_diagnostics=collapse_diagnostics,
    )
