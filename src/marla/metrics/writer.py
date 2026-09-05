"""Writes the run directory (spec section 21): metadata.json, episodes.csv,
decisions.csv, updates.csv, summary.json, config.yaml, checkpoint.pt.

The RL Orchestrator is the only central metrics writer (spec section 21);
this module is called once, after a run ends (whether it completed,
stopped by user, or failed), from the CLI's ``run`` command.

Some spec-listed decisions.csv columns are not populated in this release
and are always written empty: ``schema_revision_count`` (the Gatekeeper's
per-request revision count is never reported back to the RL Orchestrator --
doing so would need a wire-protocol change to the ADVISORY_RESPONSE
artifact), ``action_success`` (NASimEmu's reward doesn't expose a distinct
success/failure signal separate from the reward value itself), and
``artifact_path`` (spec's ``artifacts/plan_maker/<request-id>.json`` full
request/response dump isn't implemented -- ``marla run --debug`` writes
comparable query/response text files to ``debug/<run_id>/`` instead).
``updates.csv``'s ``checkpoint_id`` is likewise always empty: this module
saves exactly one checkpoint per run, at the very end (spec section 5's
normal-shutdown step 1), not one per PPO update -- there is nothing to put
in a per-update column. Load the final checkpoint with
``marla.learning.checkpoint.load_checkpoint``.
"""

from __future__ import annotations

import csv
import json
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
from marla.runtime.device import ResolvedDevice
from marla.utils.versions import collect_dependency_versions, git_commit, python_version

_EPISODES_FIELDS = [
    "run_id", "variant", "episode_id", "seed", "scenario", "goal_success", "nasimemu_return",
    "training_return", "benchmark_return", "environment_steps", "rl_decisions", "steps_to_goal",
    "episode_seconds", "consultation_count", "consultation_cost", "schema_rejection_count", "finish_reason",
    "rollout", "is_eval",
]

_DECISIONS_FIELDS = [
    "run_id", "episode_id", "environment_step", "observation_id", "legal_action_count", "query_probability",
    "queried", "consultation_cost", "request_id", "response_status", "response_latency_ms",
    "schema_revision_count", "base_policy_entropy", "base_top_two_margin", "base_top_action_id",
    "plan_maker_top_action_id", "final_top_action_id", "selected_action_id", "selected_action_base_rank",
    "selected_action_plan_maker_rank", "beta", "alpha", "advice_changed_top_action", "action_success",
    "nasimemu_reward", "training_reward", "terminated", "truncated", "artifact_path",
    "plan_maker_input_tokens", "plan_maker_output_tokens", "plan_maker_total_tokens",
]

_UPDATES_FIELDS = [
    "run_id", "update", "environment_steps", "policy_loss", "value_loss", "query_entropy", "action_entropy",
    "approximate_kl", "clip_fraction", "explained_variance", "gradient_norm", "learning_rate", "mean_beta",
    "mean_query_probability", "actual_query_rate", "mean_nasimemu_reward", "mean_training_reward",
    "elapsed_training_seconds", "checkpoint_id",
]


def _variant(config: Config) -> str:
    return "assisted" if config.consultation.mode == "learned" else "baseline"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rank_descending(scores: list[float], index: int) -> int:
    """1-based rank of ``scores[index]`` among ``scores``, ties broken by index order."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order.index(index) + 1


def _decision_row(config: Config, record: StepRecord) -> dict[str, Any]:
    action_ids = [a.action_id for a in record.legal_action_descriptors]
    base_scores = record.base_logits.tolist()
    base_probs = record.base_logits.softmax(dim=-1)
    mask = record.base_logits.new_ones(record.base_logits.shape)

    base_top_index = max(range(len(base_scores)), key=lambda i: base_scores[i])
    final_top_index = max(range(len(record.final_logits.tolist())), key=lambda i: record.final_logits.tolist()[i])
    selected_id = action_ids[record.selected_action_index]
    base_rank = _rank_descending(base_scores, record.selected_action_index)

    plan_maker_top_id = None
    plan_maker_rank = None
    if record.plan_maker_scores_in_action_order is not None:
        pm_scores = record.plan_maker_scores_in_action_order
        pm_top_index = max(range(len(pm_scores)), key=lambda i: pm_scores[i])
        plan_maker_top_id = action_ids[pm_top_index]
        plan_maker_rank = _rank_descending(pm_scores, record.selected_action_index)

    return {
        "run_id": record.run_id,
        "episode_id": record.episode_id,
        "environment_step": record.environment_step,
        "observation_id": record.observation_id,
        "legal_action_count": len(action_ids),
        "query_probability": record.old_query_probability,
        "queried": record.sampled_query,
        "consultation_cost": record.consultation_cost,
        "request_id": record.plan_maker_request_id,
        "response_status": record.plan_maker_response_status,
        "response_latency_ms": record.plan_maker_latency_ms,
        "schema_revision_count": None,  # see module docstring
        "base_policy_entropy": float(masked_entropy(base_probs.clamp_min(1e-12).log(), mask)),
        "base_top_two_margin": float(compute_top_two_margin(record.base_logits)),
        "base_top_action_id": action_ids[base_top_index],
        "plan_maker_top_action_id": plan_maker_top_id,
        "final_top_action_id": action_ids[final_top_index],
        "selected_action_id": selected_id,
        "selected_action_base_rank": base_rank,
        "selected_action_plan_maker_rank": plan_maker_rank,
        "beta": record.beta,
        "alpha": record.alpha,
        "advice_changed_top_action": (
            action_ids[base_top_index] != action_ids[final_top_index] if record.sampled_query else None
        ),
        "action_success": None,  # see module docstring
        "nasimemu_reward": record.nasimemu_reward,
        "training_reward": record.training_reward,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "artifact_path": None,  # see module docstring
        "plan_maker_input_tokens": record.plan_maker_input_tokens,
        "plan_maker_output_tokens": record.plan_maker_output_tokens,
        "plan_maker_total_tokens": record.plan_maker_total_tokens,
    }


def write_episodes_csv(
    run_dir: Path,
    config: Config,
    episode_summaries: list[EpisodeSummary],
    eval_episode_summaries: list[EpisodeSummary] = (),
) -> None:
    variant = _variant(config)
    scenario = config.environment.scenario
    rows = []
    for summary in [*episode_summaries, *eval_episode_summaries]:
        rows.append(
            {
                "run_id": summary.run_id,
                "variant": variant,
                "episode_id": summary.episode_id,
                "seed": summary.seed,
                "scenario": scenario,
                "goal_success": summary.goal_success,
                "nasimemu_return": summary.nasimemu_return,
                "training_return": summary.training_return,
                # NASimEmu's own reward, unaffected by consultation-cost
                # deductions -- the one comparable figure across baseline
                # and assisted variants (spec doesn't define this field
                # further; see module docstring for other interpreted gaps).
                "benchmark_return": summary.nasimemu_return,
                "environment_steps": summary.environment_steps,
                "rl_decisions": summary.environment_steps - summary.consultation_count,
                "steps_to_goal": summary.steps_to_goal,
                "episode_seconds": summary.episode_seconds,
                "consultation_count": summary.consultation_count,
                "consultation_cost": summary.consultation_cost,
                "schema_rejection_count": summary.schema_rejection_count,
                "finish_reason": summary.finish_reason,
                "rollout": summary.rollout,
                "is_eval": summary.is_eval,
            }
        )
    _write_csv(run_dir / "episodes.csv", _EPISODES_FIELDS, rows)


def write_decisions_csv(run_dir: Path, config: Config, records: list[StepRecord]) -> None:
    rows = [_decision_row(config, record) for record in records]
    _write_csv(run_dir / "decisions.csv", _DECISIONS_FIELDS, rows)


def write_updates_csv(run_dir: Path, config: Config, update_metrics: list[dict[str, Any]]) -> None:
    run_id = config.experiment.run_id or config.experiment.name
    rows = [{"run_id": run_id, **metrics} for metrics in update_metrics]
    _write_csv(run_dir / "updates.csv", _UPDATES_FIELDS, rows)


def write_summary_json(
    run_dir: Path,
    episode_summaries: list[EpisodeSummary],
    update_metrics: list[dict[str, Any]],
    all_records: list[StepRecord],
    environment_steps: int,
    eval_episode_summaries: list[EpisodeSummary] = (),
) -> None:
    def _mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def _median(values: list[float]) -> float | None:
        return _percentile(values, 0.5)

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

    def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | tuple[None, None]:
        """95%-by-default Wilson score interval for a binomial proportion.

        More reliable than a normal approximation for small ``n`` or a
        proportion near 0 or 1 (goal_success_rate is exactly one or the
        other for a short run), which is exactly the regime a MARLA run's
        episode count often falls into.
        """
        if n == 0:
            return None, None
        p_hat = successes / n
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = (z / denom) * ((p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) ** 0.5)
        return max(0.0, center - margin), min(1.0, center + margin)

    # Derived from all_records (every decision made), not episode_summaries
    # (only *completed* episodes) -- a short or interrupted run can have
    # real consultations with no episode having finished yet, and
    # episode_summaries would then silently under-report them as zero.
    queried_records = [r for r in all_records if r.sampled_query]
    accepted_records = [r for r in queried_records if r.plan_maker_validation_status == "accepted"]
    consultations_total = len(queried_records)
    schema_rejections_total = sum(
        1 for r in queried_records if r.plan_maker_response_status == "schema_rejected"
    )

    advice_changed_count = 0
    for record in accepted_records:
        base_top = int(record.base_logits.argmax())
        final_top = int(record.final_logits.argmax())
        if base_top != final_top:
            advice_changed_count += 1

    total_training_seconds = update_metrics[-1]["elapsed_training_seconds"] if update_metrics else 0.0

    latencies = [r.plan_maker_latency_ms for r in queried_records if r.plan_maker_latency_ms is not None]

    # Episode outcome breakdown (spec's finish_reason plus goal_success):
    # every completed episode is exactly one of these three, so the three
    # rates always sum to 1 -- there is no fourth "environment/protocol
    # failure" category to report because NASimEmu never internally
    # terminates a non-FINISH action (see nasimemu_adapter.py's step()) and
    # a NASimEmu action-space precondition failure is absorbed as a normal
    # failed attempt, not an episode-ending error (see
    # NasimEmuAdapter._absorb_invalid_action).
    successful_episodes = [s for s in episode_summaries if s.goal_success]
    is_premature_finish = lambda s: s.finish_reason == "finish" and not s.goal_success  # noqa: E731
    is_timeout = lambda s: s.finish_reason == "truncated"  # noqa: E731
    goal_success_ci_low, goal_success_ci_high = _wilson_interval(
        len(successful_episodes), len(episode_summaries)
    )

    summary_dict = {
        "episode_count": len(episode_summaries),
        "goal_success_rate": _mean([1.0 if s.goal_success else 0.0 for s in episode_summaries]),
        "goal_success_rate_ci_low": goal_success_ci_low,
        "goal_success_rate_ci_high": goal_success_ci_high,
        "premature_finish_rate": _mean([1.0 if is_premature_finish(s) else 0.0 for s in episode_summaries]),
        "timeout_rate": _mean([1.0 if is_timeout(s) else 0.0 for s in episode_summaries]),
        "mean_benchmark_return": _mean([s.nasimemu_return for s in episode_summaries]),
        "median_benchmark_return": _median([s.nasimemu_return for s in episode_summaries]),
        "mean_environment_steps": _mean([float(s.environment_steps) for s in episode_summaries]),
        "mean_steps_to_goal": _mean([float(s.steps_to_goal) for s in episode_summaries if s.steps_to_goal is not None]),
        "mean_episode_duration_seconds": _mean([s.episode_seconds for s in episode_summaries]),
        "total_consultations": consultations_total,
        "mean_consultations_per_episode": _mean([float(s.consultation_count) for s in episode_summaries]),
        "queries_per_successful_episode": (
            consultations_total / len(successful_episodes)
            if successful_episodes and consultations_total > 0
            else None
        ),
        "mean_consultation_cost": _mean([s.consultation_cost for s in episode_summaries if s.consultation_count > 0]),
        "mean_plan_maker_latency_ms": _mean(latencies),
        "median_plan_maker_latency_ms": _median(latencies),
        "p95_plan_maker_latency_ms": _percentile(latencies, 0.95),
        "max_plan_maker_latency_ms": max(latencies) if latencies else None,
        "mean_beta": _mean([r.beta for r in accepted_records if r.beta is not None]),
        "advice_changed_top_action_rate": (
            advice_changed_count / len(accepted_records) if accepted_records else None
        ),
        "schema_rejection_rate": (
            schema_rejections_total / consultations_total if consultations_total > 0 else None
        ),
        "advice_acceptance_rate": (
            len(accepted_records) / consultations_total if consultations_total > 0 else None
        ),
        "total_training_environment_steps": environment_steps,
        "total_training_seconds": total_training_seconds,
        # Deterministic (greedy) periodic evaluation episodes -- see
        # config.metrics.eval_episodes / learning/rollout.py's
        # run_evaluation_episodes. Empty/None when eval_episodes == 0.
        "eval_episode_count": len(eval_episode_summaries),
        "eval_goal_success_rate": _mean([1.0 if s.goal_success else 0.0 for s in eval_episode_summaries]),
        "mean_eval_return": _mean([s.nasimemu_return for s in eval_episode_summaries]),
        "median_eval_return": _median([s.nasimemu_return for s in eval_episode_summaries]),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")


def write_metadata_json(
    run_dir: Path,
    config: Config,
    resolved_device: ResolvedDevice,
    start_time: datetime,
    end_time: datetime,
    status: str,
) -> None:
    plan_maker_cfg = config.agents[0] if config.agents else None
    metadata = {
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
        "end_time": end_time.isoformat(),
        "status": status,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _marla_version() -> str:
    from marla import __version__

    return __version__


def _os_info() -> str:
    import platform

    return f"{platform.system()} {platform.release()}"


def write_run_artifacts(
    run_dir: Path,
    config: Config,
    result: TrainingResult | None,
    resolved_device: ResolvedDevice,
    start_time: datetime,
    end_time: datetime,
    status: str,
) -> None:
    """Write every spec section 21 artifact this release produces for one run.

    Safe to call even when ``result`` is ``None`` (construction failed
    before any training happened) -- episodes/decisions/updates are then
    written empty, and metadata.json still records the failure.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(redacted_config_dict(config)), encoding="utf-8")
    write_metadata_json(run_dir, config, resolved_device, start_time, end_time, status)

    episode_summaries = result.episode_summaries if result is not None else []
    eval_episode_summaries = result.eval_episode_summaries if result is not None else []
    update_metrics = result.update_metrics if result is not None else []
    all_records = result.all_records if result is not None else []
    environment_steps = result.environment_steps if result is not None else 0

    write_episodes_csv(run_dir, config, episode_summaries, eval_episode_summaries)
    write_updates_csv(run_dir, config, update_metrics)
    write_summary_json(
        run_dir, episode_summaries, update_metrics, all_records, environment_steps, eval_episode_summaries
    )
    if config.metrics.record_decisions:
        write_decisions_csv(run_dir, config, all_records)

    if result is not None:
        # Spec section 5's normal-shutdown step 1 ("save final checkpoint"):
        # one checkpoint of the final policy/optimizer state, regardless of
        # whether training ran to completion or was stopped early. This is
        # distinct from (and does not populate) updates.csv's per-update
        # checkpoint_id column -- no per-update checkpointing is wired into
        # the training loop, only this single end-of-run save.
        # update_count/environment_steps are cumulative (resume-aware) --
        # result.environment_steps already starts from the resumed
        # checkpoint's own value (learning/trainer.run_training_loop), and
        # update_count_offset carries the equivalent for update numbering.
        save_checkpoint(
            run_dir / "checkpoint.pt",
            result.policy,
            result.optimizer,
            update_count=result.update_count_offset + len(result.update_metrics),
            environment_steps=result.environment_steps,
            config_hash=config_hash(config),
            next_episode_seed=result.next_episode_seed,
            rng_state=result.final_rng_state,
        )
