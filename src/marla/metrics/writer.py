"""Writes the run directory (spec section 21): metadata.json, episodes.csv,
decisions.csv, rollouts.csv, updates.csv, summary.json, config.yaml,
checkpoint_last.pt / checkpoint_best.pt.

Two write paths:

- **Incremental** (used by the real ``marla run`` path, via
  ``learning.trainer.run_training_loop``): :func:`initialize_run_directory`
  runs once at the very start of a run (config.yaml, a provisional
  metadata.json with ``status: "running"``, and every CSV's header row);
  :func:`append_rollout_metrics` runs once after every completed rollout,
  appending that rollout's episodes/decisions/rollout/update rows;
  :func:`finalize_run_directory` runs once at shutdown (final metadata.json,
  summary.json, checkpoint). This is what makes an interrupted long run's
  results resilient (spec section 2): everything up to the last completed
  rollout is already safely on disk no matter how the process ends.
- **Single-shot** (:func:`write_run_artifacts`): writes every artifact at
  once from a fully in-memory :class:`~marla.learning.trainer.TrainingResult`
  -- used directly by tests and any caller that never wired up incremental
  writing (e.g. ``run_training_loop`` called without a ``run_dir``).

Some spec-listed decisions.csv columns are never populated in this release
and are intentionally omitted rather than always-empty: ``schema_revision_count``
(the Gatekeeper's per-request revision count is never reported back to the
RL Orchestrator -- doing so would need a wire-protocol change to the
ADVISORY_RESPONSE artifact), ``action_success`` (NASimEmu's reward doesn't
expose a distinct success/failure signal separate from the reward value
itself, and inventing a generic boolean was explicitly rejected), and
``artifact_path`` (spec's ``artifacts/plan_maker/<request-id>.json`` full
request/response dump isn't implemented -- ``marla run --debug`` writes
comparable query/response text files to ``debug/<run_id>/`` instead).
``updates.csv``'s old ``checkpoint_id`` column is dropped for the same
reason: this module no longer checkpoints per-update at all (see
:mod:`marla.learning.checkpoint`'s ``checkpoint_last.pt``/``checkpoint_best.pt``).
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from marla.config.loader import config_hash, redacted_config_dict
from marla.config.models import Config
from marla.learning.checkpoint import save_checkpoint
from marla.learning.masking import masked_entropy
from marla.learning.query_gate import compute_top_two_margin
from marla.learning.rollout import EpisodeSummary, StepRecord
from marla.learning.trainer import TrainingResult
from marla.metrics.accumulators import ConsultationStats
from marla.runtime.device import ResolvedDevice
from marla.utils.versions import collect_dependency_versions, git_commit, python_version

_EPISODES_FIELDS = [
    "run_id", "variant", "episode_id", "seed", "scenario", "goal_success", "nasimemu_return",
    "training_return", "benchmark_return", "environment_steps", "rl_decisions", "steps_to_goal",
    "finish_step", "finish_delay_steps", "episode_seconds", "consultation_count", "consultation_cost",
    "schema_rejection_count", "finish_reason", "rollout", "is_eval",
    # objective_reached is independent of FINISH (spec: "objective_reached
    # no longer requires FINISH"); successful_finish/episode_success both
    # require an explicit FINISH while the objective holds (identical to
    # each other and to the deprecated goal_success above -- see
    # EpisodeSummary's own docstring for why goal_success is kept, not
    # removed). sensitive_targets_*_final are the simulator's real state at
    # episode end (whichever way it ended), not a running/cached value.
    "objective_reached", "successful_finish", "episode_success",
    "sensitive_targets_total", "sensitive_targets_with_root_final", "sensitive_targets_remaining_final",
]

_DECISIONS_FIELDS = [
    "run_id", "global_environment_step", "rollout", "episode_id", "environment_step", "observation_id",
    "legal_action_count", "selected_action_id", "selected_action_type",
    "base_policy_entropy", "base_top_two_margin", "selected_action_base_probability",
    "selected_action_final_probability", "selected_action_base_rank", "selected_action_final_rank",
    "final_policy_entropy", "base_top_action_id", "final_top_action_id",
    "critic_value", "gae_advantage", "return_target",
    "nasimemu_reward", "training_reward", "consultation_cost",
    "objective_satisfied", "objective_became_satisfied", "terminated", "truncated",
    "new_hosts_discovered", "new_subnets_discovered", "new_services_confirmed", "new_processes_confirmed",
    "access_gain",
    "sensitive_targets_total", "sensitive_targets_with_root", "sensitive_targets_remaining",
    "query_probability", "queried", "request_id", "response_status", "response_latency_ms",
    "plan_maker_top_action_id", "selected_action_plan_maker_rank", "beta", "alpha",
    "advice_changed_top_action",
    "plan_maker_input_tokens", "plan_maker_output_tokens", "plan_maker_total_tokens",
]

_ROLLOUTS_FIELDS = [
    "run_id", "rollout", "environment_steps_total", "steps_collected", "episodes_finished",
    "mean_nasimemu_reward", "std_nasimemu_reward", "mean_training_reward", "std_training_reward",
    "mean_critic_value", "mean_return_target", "mean_advantage", "std_advantage", "mean_abs_advantage",
    "mean_query_probability", "actual_query_rate", "mean_beta",
    "collection_seconds", "optimization_seconds", "evaluation_seconds",
]

_UPDATES_FIELDS = [
    "run_id", "update", "rollout", "epoch", "minibatch", "environment_steps",
    "policy_loss", "value_loss", "query_entropy", "action_entropy", "approximate_kl", "clip_fraction",
    "explained_variance", "gradient_norm", "learning_rate",
]


def _variant(config: Config) -> str:
    return "assisted" if config.consultation.mode == "learned" else "baseline"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ensure_csv_header(path: Path, fieldnames: list[str]) -> None:
    """Creates ``path`` with just a header row, if it doesn't already exist.

    Called once at run start (spec section 2) so even a run that fails
    before completing a single rollout leaves a well-formed (if empty)
    CSV, and so :func:`_append_csv` below never has to re-decide whether a
    header is needed.
    """
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writeheader()


def _append_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Appends ``rows`` to an existing (header-only or partially written) CSV."""
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)


def _rank_descending(scores: list[float], index: int) -> int:
    """1-based rank of ``scores[index]`` among ``scores``, ties broken by index order."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order.index(index) + 1


def _decision_row(record: StepRecord, advantage: float, return_target: float) -> dict[str, Any]:
    """Builds one decisions.csv row from a still-live ``StepRecord``.

    Must run before the record's tensors (``base_logits``/``final_logits``/
    ``graph_data``, ...) are discarded -- this is the one place those
    tensors are read at all after collection, and nothing here retains a
    full probability vector: every value extracted is a single scalar
    (entropy, a rank, one indexed probability), computed once while the
    logits are available, per spec section 3.
    """
    action_ids = [a.action_id for a in record.legal_action_descriptors]
    base_scores = record.base_logits.tolist()
    final_scores = record.final_logits.tolist()
    base_probs = record.base_logits.softmax(dim=-1)
    final_probs = record.final_logits.softmax(dim=-1)
    mask = record.base_logits.new_ones(record.base_logits.shape)

    base_top_index = max(range(len(base_scores)), key=lambda i: base_scores[i])
    final_top_index = max(range(len(final_scores)), key=lambda i: final_scores[i])
    selected_id = action_ids[record.selected_action_index]
    base_rank = _rank_descending(base_scores, record.selected_action_index)
    final_rank = _rank_descending(final_scores, record.selected_action_index)

    plan_maker_top_id = None
    plan_maker_rank = None
    if record.plan_maker_scores_in_action_order is not None:
        pm_scores = record.plan_maker_scores_in_action_order
        pm_top_index = max(range(len(pm_scores)), key=lambda i: pm_scores[i])
        plan_maker_top_id = action_ids[pm_top_index]
        plan_maker_rank = _rank_descending(pm_scores, record.selected_action_index)

    return {
        "run_id": record.run_id,
        "global_environment_step": record.global_environment_step,
        "rollout": record.rollout,
        "episode_id": record.episode_id,
        "environment_step": record.environment_step,
        "observation_id": record.observation_id,
        "legal_action_count": len(action_ids),
        "selected_action_id": selected_id,
        "selected_action_type": record.selected_action_type,
        "base_policy_entropy": float(masked_entropy(base_probs.clamp_min(1e-12).log(), mask)),
        "base_top_two_margin": float(compute_top_two_margin(record.base_logits)),
        "selected_action_base_probability": float(base_probs[record.selected_action_index]),
        "selected_action_final_probability": float(final_probs[record.selected_action_index]),
        "selected_action_base_rank": base_rank,
        "selected_action_final_rank": final_rank,
        "final_policy_entropy": float(masked_entropy(final_probs.clamp_min(1e-12).log(), mask)),
        "base_top_action_id": action_ids[base_top_index],
        "final_top_action_id": action_ids[final_top_index],
        "critic_value": record.critic_value,
        "gae_advantage": advantage,
        "return_target": return_target,
        "nasimemu_reward": record.nasimemu_reward,
        "training_reward": record.training_reward,
        "consultation_cost": record.consultation_cost,
        "objective_satisfied": record.objective_satisfied,
        "objective_became_satisfied": record.objective_became_satisfied,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "new_hosts_discovered": record.state_delta.new_hosts_discovered,
        "new_subnets_discovered": record.state_delta.new_subnets_discovered,
        "new_services_confirmed": record.state_delta.new_services_confirmed,
        "new_processes_confirmed": record.state_delta.new_processes_confirmed,
        "access_gain": int(record.state_delta.access_gain),
        "sensitive_targets_total": record.sensitive_targets_total,
        "sensitive_targets_with_root": record.sensitive_targets_with_root,
        "sensitive_targets_remaining": record.sensitive_targets_remaining,
        "query_probability": record.old_query_probability,
        "queried": record.sampled_query,
        "request_id": record.plan_maker_request_id,
        "response_status": record.plan_maker_response_status,
        "response_latency_ms": record.plan_maker_latency_ms,
        "plan_maker_top_action_id": plan_maker_top_id,
        "selected_action_plan_maker_rank": plan_maker_rank,
        "beta": record.beta,
        "alpha": record.alpha,
        "advice_changed_top_action": (
            action_ids[base_top_index] != action_ids[final_top_index] if record.sampled_query else None
        ),
        "plan_maker_input_tokens": record.plan_maker_input_tokens,
        "plan_maker_output_tokens": record.plan_maker_output_tokens,
        "plan_maker_total_tokens": record.plan_maker_total_tokens,
    }


def build_decision_rows(records: list[StepRecord], advantages: list[float], returns: list[float]) -> list[dict[str, Any]]:
    """One decisions.csv row per record -- the single point where a rollout's
    ``StepRecord`` list is turned into compact, storable metrics (spec
    section 1/3). Called exactly once per rollout, right after GAE and the
    PPO update, before ``records`` goes out of scope.
    """
    return [_decision_row(record, advantage, return_target) for record, advantage, return_target in zip(records, advantages, returns)]


def _episode_row(config: Config, summary: EpisodeSummary) -> dict[str, Any]:
    return {
        "run_id": summary.run_id,
        "variant": _variant(config),
        "episode_id": summary.episode_id,
        "seed": summary.seed,
        "scenario": config.environment.scenario,
        "goal_success": summary.goal_success,
        "nasimemu_return": summary.nasimemu_return,
        "training_return": summary.training_return,
        # NASimEmu's own reward, unaffected by consultation-cost deductions
        # -- the one comparable figure across baseline and assisted
        # variants (spec doesn't define this field further; see module
        # docstring for other interpreted gaps).
        "benchmark_return": summary.nasimemu_return,
        "environment_steps": summary.environment_steps,
        "rl_decisions": summary.environment_steps - summary.consultation_count,
        "steps_to_goal": summary.steps_to_goal,
        "finish_step": summary.finish_step,
        "finish_delay_steps": summary.finish_delay_steps,
        "episode_seconds": summary.episode_seconds,
        "consultation_count": summary.consultation_count,
        "consultation_cost": summary.consultation_cost,
        "schema_rejection_count": summary.schema_rejection_count,
        "finish_reason": summary.finish_reason,
        "rollout": summary.rollout,
        "is_eval": summary.is_eval,
        "objective_reached": summary.objective_reached,
        "successful_finish": summary.successful_finish,
        "episode_success": summary.episode_success,
        "sensitive_targets_total": summary.sensitive_targets_total,
        "sensitive_targets_with_root_final": summary.sensitive_targets_with_root_final,
        "sensitive_targets_remaining_final": summary.sensitive_targets_remaining_final,
    }


def build_episode_rows(config: Config, summaries: list[EpisodeSummary]) -> list[dict[str, Any]]:
    return [_episode_row(config, summary) for summary in summaries]


def build_rollout_row(
    run_id: str,
    rollout: int,
    environment_steps_total: int,
    records: list[StepRecord],
    episodes_finished: int,
    advantages: list[float],
    returns: list[float],
    collection_seconds: float,
    optimization_seconds: float,
    evaluation_seconds: float | None,
) -> dict[str, Any]:
    """The one rollouts.csv row for a completed rollout (spec section 6).

    Every rollout-level aggregate (mean reward, query rate, mean beta, ...)
    lives here exactly once, computed straight from that rollout's
    ``records``/``advantages``/``returns`` -- never repeated per PPO
    minibatch in updates.csv (see :func:`build_update_rows`).
    """

    def _mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def _std(values: list[float]) -> float | None:
        if not values:
            return None
        m = sum(values) / len(values)
        return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5

    def _mean_abs(values: list[float]) -> float | None:
        return _mean([abs(v) for v in values])

    nasimemu_rewards = [r.nasimemu_reward for r in records]
    training_rewards = [r.training_reward for r in records]
    critic_values = [r.critic_value for r in records]
    betas = [r.beta for r in records if r.beta is not None]
    query_probabilities = [r.old_query_probability for r in records if r.old_query_probability is not None]

    return {
        "run_id": run_id,
        "rollout": rollout,
        "environment_steps_total": environment_steps_total,
        "steps_collected": len(records),
        "episodes_finished": episodes_finished,
        "mean_nasimemu_reward": _mean(nasimemu_rewards),
        "std_nasimemu_reward": _std(nasimemu_rewards),
        "mean_training_reward": _mean(training_rewards),
        "std_training_reward": _std(training_rewards),
        "mean_critic_value": _mean(critic_values),
        "mean_return_target": _mean(list(returns)),
        "mean_advantage": _mean(list(advantages)),
        "std_advantage": _std(list(advantages)),
        "mean_abs_advantage": _mean_abs(list(advantages)),
        "mean_query_probability": _mean(query_probabilities),
        "actual_query_rate": sum(r.sampled_query for r in records) / len(records) if records else None,
        "mean_beta": _mean(betas),
        "collection_seconds": collection_seconds,
        "optimization_seconds": optimization_seconds,
        "evaluation_seconds": evaluation_seconds,
    }


def write_episodes_csv(
    run_dir: Path,
    config: Config,
    episode_summaries: list[EpisodeSummary],
    eval_episode_summaries: list[EpisodeSummary] = (),
) -> None:
    rows = build_episode_rows(config, [*episode_summaries, *eval_episode_summaries])
    _write_csv(run_dir / "episodes.csv", _EPISODES_FIELDS, rows)


def write_decisions_csv(run_dir: Path, records: list[StepRecord], advantages: list[float], returns: list[float]) -> None:
    rows = build_decision_rows(records, advantages, returns)
    _write_csv(run_dir / "decisions.csv", _DECISIONS_FIELDS, rows)


def write_rollouts_csv(run_dir: Path, rollout_rows: list[dict[str, Any]]) -> None:
    _write_csv(run_dir / "rollouts.csv", _ROLLOUTS_FIELDS, rollout_rows)


def build_update_rows(run_id: str, update_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepends ``run_id`` to each of ``ppo.optimize()``'s already-built
    per-update metrics dicts (each already carries ``update``/``rollout``/
    ``epoch``/``minibatch``/``environment_steps``/``learning_rate``, set by
    ``learning.trainer.run_training_loop``). No rollout-level aggregate is
    repeated here; those live once per rollout in rollouts.csv (spec
    section 7).
    """
    return [{"run_id": run_id, **metrics} for metrics in update_metrics]


def write_updates_csv(run_dir: Path, config: Config, update_metrics: list[dict[str, Any]]) -> None:
    run_id = config.experiment.run_id or config.experiment.name
    _write_csv(run_dir / "updates.csv", _UPDATES_FIELDS, build_update_rows(run_id, update_metrics))


def _percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (0 <= p <= 1); ``None`` if empty."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = p * (len(ordered) - 1)
    low, high = int(rank), min(int(rank) + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    return _percentile(values, 0.5)


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | tuple[None, None]:
    """95%-by-default Wilson score interval for a binomial proportion.

    More reliable than a normal approximation for small ``n`` or a
    proportion near 0 or 1 (goal_success_rate is exactly one or the other
    for a short run), which is exactly the regime a MARLA run's episode
    count often falls into.
    """
    if n == 0:
        return None, None
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * ((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5)
    return max(0.0, center - margin), min(1.0, center + margin)


def write_summary_json(
    run_dir: Path,
    episode_summaries: list[EpisodeSummary],
    consultation_stats: ConsultationStats,
    action_type_counts: Counter,
    environment_steps: int,
    total_training_seconds: float,
    timing_totals: dict[str, float],
    eval_episode_summaries: list[EpisodeSummary] = (),
) -> None:
    """Compact run-level summary, built entirely from already-aggregated
    inputs -- no ``all_records``/full-run ``StepRecord`` list involved
    (spec sections 1/9).
    """
    successful_episodes = [s for s in episode_summaries if s.goal_success]
    is_premature_finish = lambda s: s.finish_reason == "finish" and not s.goal_success  # noqa: E731
    is_timeout = lambda s: s.finish_reason == "truncated"  # noqa: E731
    goal_success_ci_low, goal_success_ci_high = _wilson_interval(len(successful_episodes), len(episode_summaries))

    finish_delays = [s.finish_delay_steps for s in successful_episodes if s.finish_delay_steps is not None]
    successful_steps_to_goal = [s.steps_to_goal for s in successful_episodes if s.steps_to_goal is not None]

    action_type_total = sum(action_type_counts.values())
    action_type_frequencies = (
        {action_type: count / action_type_total for action_type, count in action_type_counts.items()}
        if action_type_total
        else {}
    )

    timing_total = sum(v for v in timing_totals.values() if v)
    timing_proportions = (
        {f"{key}_fraction": value / timing_total for key, value in timing_totals.items()} if timing_total else {}
    )

    objective_reached_ci_low, objective_reached_ci_high = _wilson_interval(
        len([s for s in episode_summaries if s.objective_reached]), len(episode_summaries)
    )

    summary_dict = {
        "episode_count": len(episode_summaries),
        # DEPRECATED: identical to successful_finish_rate below -- kept for
        # backward compatibility with existing analysis code. NEVER a
        # synonym for objective_reached_rate (spec: "do not use it as a
        # synonym for objective_reached").
        "goal_success_rate": _mean([1.0 if s.goal_success else 0.0 for s in episode_summaries]),
        "goal_success_rate_ci_low": goal_success_ci_low,
        "goal_success_rate_ci_high": goal_success_ci_high,
        # objective_reached: true the instant the simulator state satisfies
        # the objective, independent of FINISH (spec section 25: report
        # this SEPARATELY from the FINISH-gated rate below, never collapsed
        # into one number).
        "objective_reached_rate": _mean([1.0 if s.objective_reached else 0.0 for s in episode_summaries]),
        "objective_reached_rate_ci_low": objective_reached_ci_low,
        "objective_reached_rate_ci_high": objective_reached_ci_high,
        # successful_finish: objective reached AND an explicit FINISH was
        # then selected -- identical value to goal_success_rate above,
        # under its clear, non-deprecated name.
        "successful_finish_rate": _mean([1.0 if s.successful_finish else 0.0 for s in episode_summaries]),
        "successful_finish_rate_ci_low": goal_success_ci_low,
        "successful_finish_rate_ci_high": goal_success_ci_high,
        "premature_finish_rate": _mean([1.0 if is_premature_finish(s) else 0.0 for s in episode_summaries]),
        "timeout_rate": _mean([1.0 if is_timeout(s) else 0.0 for s in episode_summaries]),
        "mean_benchmark_return": _mean([s.nasimemu_return for s in episode_summaries]),
        "median_benchmark_return": _median([s.nasimemu_return for s in episode_summaries]),
        "mean_environment_steps": _mean([float(s.environment_steps) for s in episode_summaries]),
        "mean_steps_to_goal": _mean([float(s.steps_to_goal) for s in episode_summaries if s.steps_to_goal is not None]),
        "mean_steps_to_goal_successful_episodes": _mean([float(v) for v in successful_steps_to_goal]),
        "mean_finish_delay_steps": _mean([float(v) for v in finish_delays]),
        "median_finish_delay_steps": _median([float(v) for v in finish_delays]),
        "mean_episode_duration_seconds": _mean([s.episode_seconds for s in episode_summaries]),
        "action_type_counts": dict(action_type_counts),
        "action_type_frequencies": action_type_frequencies,
        "total_consultations": consultation_stats.consultations_total,
        "mean_consultations_per_episode": _mean([float(s.consultation_count) for s in episode_summaries]),
        "queries_per_successful_episode": (
            consultation_stats.consultations_total / len(successful_episodes)
            if successful_episodes and consultation_stats.consultations_total > 0
            else None
        ),
        "mean_consultation_cost": _mean([s.consultation_cost for s in episode_summaries if s.consultation_count > 0]),
        "mean_plan_maker_latency_ms": _mean(consultation_stats.latencies_ms),
        "median_plan_maker_latency_ms": _median(consultation_stats.latencies_ms),
        "p95_plan_maker_latency_ms": _percentile(consultation_stats.latencies_ms, 0.95),
        "max_plan_maker_latency_ms": max(consultation_stats.latencies_ms) if consultation_stats.latencies_ms else None,
        "mean_beta": consultation_stats.mean_beta,
        "advice_changed_top_action_rate": consultation_stats.advice_changed_top_action_rate,
        "schema_rejection_rate": consultation_stats.schema_rejection_rate,
        "advice_acceptance_rate": consultation_stats.advice_acceptance_rate,
        "total_training_environment_steps": environment_steps,
        "total_training_seconds": total_training_seconds,
        "total_collection_seconds": timing_totals.get("collection_seconds"),
        "total_optimization_seconds": timing_totals.get("optimization_seconds"),
        "total_evaluation_seconds": timing_totals.get("evaluation_seconds"),
        **timing_proportions,
        # Deterministic (greedy) periodic evaluation episodes -- see
        # config.metrics.eval_episodes / learning/rollout.py's
        # run_evaluation_episodes. Empty/None when eval_episodes == 0.
        "eval_episode_count": len(eval_episode_summaries),
        # DEPRECATED: identical to eval_successful_finish_rate.
        "eval_goal_success_rate": _mean([1.0 if s.goal_success else 0.0 for s in eval_episode_summaries]),
        "eval_objective_reached_rate": _mean([1.0 if s.objective_reached else 0.0 for s in eval_episode_summaries]),
        "eval_successful_finish_rate": _mean([1.0 if s.successful_finish else 0.0 for s in eval_episode_summaries]),
        "mean_eval_return": _mean([s.nasimemu_return for s in eval_episode_summaries]),
        "median_eval_return": _median([s.nasimemu_return for s in eval_episode_summaries]),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")


def _metadata_dict(
    config: Config,
    resolved_device: ResolvedDevice,
    start_time: datetime,
    status: str,
    scenario_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan_maker_cfg = config.agents[0] if config.agents else None
    return {
        "run_id": config.experiment.run_id,
        "experiment_name": config.experiment.name,
        "variant": _variant(config),
        "phase": config.experiment.phase,
        "seed": config.experiment.seed,
        "marla_version": _marla_version(),
        "python_version": python_version(),
        "dependency_versions": collect_dependency_versions(),
        "os": _os_info(),
        "execution_mode": config.execution.mode,
        "device_requested": resolved_device.requested,
        "device_resolved": resolved_device.resolved,
        "scenario": config.environment.scenario,
        "objective": {"type": config.objective.type, "description": config.objective.description},
        "rl_orchestrator": {"alias": config.rl_orchestrator.alias, "jid": config.rl_orchestrator.jid},
        "gatekeeper": (
            {"alias": config.gatekeeper.alias, "jid": config.gatekeeper.jid} if config.gatekeeper else None
        ),
        "plan_maker": (
            {
                "alias": plan_maker_cfg.alias,
                "jid": plan_maker_cfg.jid,
                "model_name": plan_maker_cfg.model.name,
                "prompt_version": plan_maker_cfg.prompt_version,
                "knowledge_version": plan_maker_cfg.knowledge.version,
            }
            if plan_maker_cfg is not None
            else None
        ),
        "config_hash": config_hash(config),
        "git_commit": git_commit(),
        "start_time": start_time.isoformat(),
        "end_time": None,
        "status": status,
        # Compact only (spec section 31): proof that this run's scenario
        # passed the preflight solvability check, without repeating any of
        # its (potentially large) diagnostic detail here. None for a run
        # started before this field existed, or if the caller never ran
        # the check (e.g. tests constructing metadata directly).
        "scenario_validation": scenario_validation,
    }


def write_metadata_json(
    run_dir: Path,
    config: Config,
    resolved_device: ResolvedDevice,
    start_time: datetime,
    end_time: datetime | None,
    status: str,
    scenario_validation: dict[str, Any] | None = None,
) -> None:
    metadata = _metadata_dict(config, resolved_device, start_time, status, scenario_validation)
    metadata["end_time"] = end_time.isoformat() if end_time is not None else None
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _marla_version() -> str:
    from marla import __version__

    return __version__


def _os_info() -> str:
    import platform

    return f"{platform.system()} {platform.release()}"


def initialize_run_directory(
    run_dir: Path,
    config: Config,
    resolved_device: ResolvedDevice,
    start_time: datetime,
    scenario_validation: dict[str, Any] | None = None,
) -> None:
    """Everything that can be written before a single environment step runs
    (spec section 2): the resolved config, a provisional ``status:
    "running"`` metadata.json, and every metrics CSV's header row -- so a
    run that never completes a rollout still leaves well-formed (if empty)
    artifacts behind, and every later append only ever adds rows.

    ``scenario_validation``: a compact record (status/validator_version/
    scenario_hash) that this run's scenario passed the ``marla run``
    preflight solvability check (spec section 31) -- proof, later, that a
    paper experiment used a validated scenario. ``None`` for any caller
    that didn't run the check (e.g. most direct-construction tests).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(redacted_config_dict(config)), encoding="utf-8")
    write_metadata_json(run_dir, config, resolved_device, start_time, None, "running", scenario_validation)
    _ensure_csv_header(run_dir / "episodes.csv", _EPISODES_FIELDS)
    if config.metrics.record_decisions:
        _ensure_csv_header(run_dir / "decisions.csv", _DECISIONS_FIELDS)
    _ensure_csv_header(run_dir / "rollouts.csv", _ROLLOUTS_FIELDS)
    _ensure_csv_header(run_dir / "updates.csv", _UPDATES_FIELDS)


def append_rollout_metrics(
    run_dir: Path,
    config: Config,
    episode_summaries: list[EpisodeSummary],
    decision_rows: list[dict[str, Any]],
    rollout_row: dict[str, Any],
    update_metrics: list[dict[str, Any]],
) -> None:
    """Flushes one completed rollout's results to disk (spec section 2).

    Called once per rollout from ``learning.trainer.run_training_loop``,
    right after that rollout's GAE + PPO update -- by the time this
    returns, the rollout's heavyweight ``StepRecord``s (and the
    ``decision_rows``/``episode_summaries`` derived from them) can be
    safely dropped by the caller; nothing here needs them again.
    """
    run_id = config.experiment.run_id or config.experiment.name
    _append_csv(run_dir / "episodes.csv", _EPISODES_FIELDS, build_episode_rows(config, episode_summaries))
    if config.metrics.record_decisions:
        _append_csv(run_dir / "decisions.csv", _DECISIONS_FIELDS, decision_rows)
    _append_csv(run_dir / "rollouts.csv", _ROLLOUTS_FIELDS, [rollout_row])
    _append_csv(run_dir / "updates.csv", _UPDATES_FIELDS, build_update_rows(run_id, update_metrics))


def finalize_run_directory(
    run_dir: Path,
    config: Config,
    result: TrainingResult | None,
    resolved_device: ResolvedDevice,
    start_time: datetime,
    end_time: datetime,
    status: str,
    scenario_validation: dict[str, Any] | None = None,
) -> None:
    """The shutdown step (spec section 2): final metadata.json, summary.json,
    and checkpoint(s). Does **not** rewrite episodes/decisions/rollouts/
    updates CSVs -- those were already flushed incrementally by
    :func:`append_rollout_metrics` during training (or are already
    header-only from :func:`initialize_run_directory`, if the run never
    completed a rollout).

    ``scenario_validation`` is passed by the caller (the same dict given to
    :func:`initialize_run_directory`) because this function replaces
    metadata.json wholesale rather than merging into it -- omitting it here
    would silently drop the field the provisional metadata.json had set.
    """
    write_metadata_json(run_dir, config, resolved_device, start_time, end_time, status, scenario_validation)
    if result is None:
        write_summary_json(run_dir, [], ConsultationStats(), Counter(), 0, 0.0, {})
        return

    write_summary_json(
        run_dir,
        result.episode_summaries,
        result.consultation_stats,
        result.action_type_counts,
        result.environment_steps,
        result.total_training_seconds,
        {
            "collection_seconds": result.total_collection_seconds,
            "optimization_seconds": result.total_optimization_seconds,
            "evaluation_seconds": result.total_evaluation_seconds,
        },
        result.eval_episode_summaries,
    )
    _save_final_checkpoints(run_dir, config, result)


def _save_final_checkpoints(run_dir: Path, config: Config, result: TrainingResult) -> None:
    """``checkpoint_last.pt`` always; ``checkpoint.pt`` kept as an identical
    copy for backward compatibility with tooling that looks for the old,
    single-checkpoint name (spec section 10). ``checkpoint_best.pt`` is
    left as whatever the training loop's periodic-eval callback last wrote
    (see ``learning.trainer.run_training_loop``) -- there is nothing left
    to decide about it here.
    """
    update_count = result.update_count_offset + len(result.update_metrics)
    for name in ("checkpoint_last.pt", "checkpoint.pt"):
        save_checkpoint(
            run_dir / name,
            result.policy,
            result.optimizer,
            update_count=update_count,
            environment_steps=result.environment_steps,
            config_hash=config_hash(config),
            next_episode_seed=result.next_episode_seed,
            rng_state=result.final_rng_state,
        )


def write_run_artifacts(
    run_dir: Path,
    config: Config,
    result: TrainingResult | None,
    resolved_device: ResolvedDevice,
    start_time: datetime,
    end_time: datetime,
    status: str,
) -> None:
    """Single-shot path: writes every spec section 21 artifact at once from
    a fully in-memory ``TrainingResult`` -- used by tests, and by any
    caller (e.g. distributed mode, or ``run_training_loop`` invoked without
    a ``run_dir``) that didn't wire up :func:`initialize_run_directory` /
    :func:`append_rollout_metrics` incrementally during the run.

    Safe to call even when ``result`` is ``None`` (construction failed
    before any training happened) -- episodes/decisions/rollouts/updates
    are then written empty, and metadata.json still records the failure.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(redacted_config_dict(config)), encoding="utf-8")
    write_metadata_json(run_dir, config, resolved_device, start_time, end_time, status)

    episode_summaries = result.episode_summaries if result is not None else []
    eval_episode_summaries = result.eval_episode_summaries if result is not None else []
    update_metrics = result.update_metrics if result is not None else []
    rollout_rows = result.rollout_rows if result is not None else []
    environment_steps = result.environment_steps if result is not None else 0

    write_episodes_csv(run_dir, config, episode_summaries, eval_episode_summaries)
    write_rollouts_csv(run_dir, rollout_rows)
    write_updates_csv(run_dir, config, update_metrics)
    if config.metrics.record_decisions:
        decision_rows = result.decision_rows if result is not None else []
        _write_csv(run_dir / "decisions.csv", _DECISIONS_FIELDS, decision_rows)

    write_summary_json(
        run_dir,
        episode_summaries,
        result.consultation_stats if result is not None else ConsultationStats(),
        result.action_type_counts if result is not None else Counter(),
        environment_steps,
        result.total_training_seconds if result is not None else 0.0,
        {
            "collection_seconds": result.total_collection_seconds if result is not None else 0.0,
            "optimization_seconds": result.total_optimization_seconds if result is not None else 0.0,
            "evaluation_seconds": result.total_evaluation_seconds if result is not None else 0.0,
        },
        eval_episode_summaries,
    )

    if result is not None:
        # Spec section 10's normal-shutdown checkpointing: checkpoint_last.pt
        # (plus a checkpoint.pt copy for backward compatibility) always;
        # checkpoint_best.pt is whatever the training loop's periodic-eval
        # callback last wrote directly to run_dir, nothing to redo here.
        _save_final_checkpoints(run_dir, config, result)
