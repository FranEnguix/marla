"""Executes ONE agent (one config, one seed) and ONE trial (two tuning
seeds under the same sampled hyperparameters) -- the actual MARLA
training invocation behind the Optuna study, using the exact same
production ``run_baseline_training`` path every other MARLA research
phase has used (spec: "MARLA should expose Optuna as a supported research
workflow", not a research-only hacked trainer).
"""

from __future__ import annotations

import asyncio
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from marla.config.loader import parse_config
from marla.config.models import compute_num_rollouts
from marla.learning.checkpoint import PolicyRepresentationMismatchError
from marla.learning.ppo import TrainingDivergedError
from marla.learning.trainer import run_baseline_training
from marla.metrics.analysis import detect_collapse, root_auc
from marla.metrics.csv_schema import read_decisions_csv
from marla.metrics.writer import finalize_run_directory, initialize_run_directory
from marla.monitoring.carbon import CarbonTrackerUnavailableError
from marla.optuna_study.objective import SeedResult
from marla.runtime.device import resolve_device
from marla.scenarios.uri import resolve_scenario_reference


class InfrastructureFailure(Exception):
    """Raised (never silently absorbed into a numerical objective) when
    an agent fails for a reason that is NOT scientific collapse: NaN/Inf
    fail-fast, CUDA OOM, an unhandled training exception, or corrupted
    metrics output (spec section 53). Carries the exception type name and
    message, plus whatever run directory/partial carbon data exists.
    """

    def __init__(self, exception_type: str, message: str, run_dir: Path | None):
        super().__init__(f"{exception_type}: {message}")
        self.exception_type = exception_type
        self.message = message
        self.run_dir = run_dir


def _behavioral_rates(episodes: pd.DataFrame) -> dict[str, float | None]:
    train_ep = episodes[episodes["is_eval"] == False] if "is_eval" in episodes.columns else episodes  # noqa: E712
    if train_ep.empty:
        return {"objective_reached_rate": None, "successful_finish_rate": None, "premature_finish_rate": None}
    return {
        "objective_reached_rate": float(train_ep["objective_reached"].mean()) if "objective_reached" in train_ep else None,
        "successful_finish_rate": float(train_ep["successful_finish"].mean()) if "successful_finish" in train_ep else None,
        "premature_finish_rate": (
            float((~train_ep["successful_finish"].astype(bool) & (train_ep["finish_reason"] == "finish")).mean())
            if "finish_reason" in train_ep.columns and "successful_finish" in train_ep.columns
            else None
        ),
    }


def _late_ppo_health(updates: pd.DataFrame) -> dict[str, float | None]:
    if updates.empty:
        return {"action_entropy_final": None, "approx_kl_final": None, "clip_fraction_final": None, "explained_variance_final": None}
    last_rollout = updates["rollout"].max() if "rollout" in updates.columns else None
    late = updates[updates["rollout"] == last_rollout] if last_rollout is not None else updates
    return {
        "action_entropy_final": float(late["action_entropy"].mean()) if "action_entropy" in late.columns and late["action_entropy"].notna().any() else None,
        "approx_kl_final": float(late["approximate_kl"].mean()) if "approximate_kl" in late.columns and late["approximate_kl"].notna().any() else None,
        "clip_fraction_final": float(late["clip_fraction"].mean()) if "clip_fraction" in late.columns and late["clip_fraction"].notna().any() else None,
        "explained_variance_final": float(late["explained_variance"].mean()) if "explained_variance" in late.columns and late["explained_variance"].notna().any() else None,
    }


async def run_agent(config_dict: dict[str, Any], scenario_examples_dir: Path, run_dir: Path) -> SeedResult:
    """Runs exactly one MARLA agent (one config dict, already carrying its
    own seed/run_id/step budget) through the production
    ``run_baseline_training`` path, with resource + carbon telemetry
    forced on, and returns its :class:`SeedResult`. Raises
    :class:`InfrastructureFailure` for anything that is not scientific
    collapse.
    """
    config_dict = dict(config_dict)
    config_dict.setdefault("carbon", {})
    config_dict["carbon"] = {**config_dict.get("carbon", {}), "enabled": True}
    # Resource + carbon telemetry are ALWAYS on for HPO agents (spec:
    # per-agent sustainability accounting is a required output of the
    # study, not an opt-in), regardless of what the base config says.
    existing_metrics = config_dict.get("metrics", {})
    config_dict["metrics"] = {
        **existing_metrics,
        "resource_monitoring": {**existing_metrics.get("resource_monitoring", {}), "enabled": True},
    }

    if run_dir.exists():
        shutil.rmtree(run_dir)

    exception_type: str | None = None
    exception_message: str | None = None
    result = None
    try:
        config = parse_config(config_dict)
        scenario_path = resolve_scenario_reference(config.environment.scenario, scenario_examples_dir)
        resolved_device = resolve_device(config.device)
        start_time = datetime.now(timezone.utc)
        initialize_run_directory(run_dir, config, resolved_device, start_time, None)
        num_rollouts = compute_num_rollouts(config.policy.ppo)

        wall_clock_start = time.monotonic()
        result = await run_baseline_training(
            config, scenario_path, num_rollouts=num_rollouts, seed=config.experiment.seed,
            device=resolved_device.torch_device, run_dir=run_dir, collapse_diagnostics=False,
        )
        wall_clock_seconds = time.monotonic() - wall_clock_start
        finalize_run_directory(run_dir, config, result, resolved_device, start_time, datetime.now(timezone.utc), "completed")
    except (TrainingDivergedError, PolicyRepresentationMismatchError, CarbonTrackerUnavailableError) as exc:
        raise InfrastructureFailure(type(exc).__name__, str(exc), run_dir) from exc
    except Exception as exc:  # noqa: BLE001 -- any other unhandled exception is ALSO an infrastructure failure, never a silent numerical penalty
        raise InfrastructureFailure(type(exc).__name__, f"{exc}\n{traceback.format_exc()}", run_dir) from exc

    episodes = pd.read_csv(run_dir / "episodes.csv")
    rollouts = pd.read_csv(run_dir / "rollouts.csv")
    updates = pd.read_csv(run_dir / "updates.csv", low_memory=False)
    decisions = read_decisions_csv(run_dir / "decisions.csv") if (run_dir / "decisions.csv").exists() else pd.DataFrame()

    auc = root_auc(episodes, rollouts)
    collapse = detect_collapse(episodes, decisions, rollouts, updates)
    rates = _behavioral_rates(episodes)
    ppo_health = _late_ppo_health(updates)

    train_ep = episodes[episodes["is_eval"] == False] if "is_eval" in episodes.columns else episodes  # noqa: E712
    mean_targets_rooted = float(train_ep["sensitive_targets_with_root_final"].mean()) if not train_ep.empty else None
    fraction_with_any_root = float((train_ep["sensitive_targets_with_root_final"] >= 1).mean()) if not train_ep.empty else None

    from marla.metrics.analysis import per_rollout_root_curve, trapz_auc

    root_curve = per_rollout_root_curve(episodes, rollouts)
    root_auc_fraction = trapz_auc(root_curve, "fraction_with_any_root")

    carbon_summary = result.carbon_summary
    energy_kwh = carbon_summary.energy_consumed_kwh if carbon_summary is not None else None
    co2eq_kg = carbon_summary.emissions_kg_co2eq if carbon_summary is not None else None

    return SeedResult(
        seed=config.experiment.seed,
        run_id=config.experiment.run_id or config.experiment.name,
        run_dir=str(run_dir),
        root_auc=auc,
        collapse=collapse,
        mean_targets_rooted=mean_targets_rooted,
        fraction_with_any_root=fraction_with_any_root,
        root_auc_fraction_with_any_root=root_auc_fraction,
        late_p_finish_state_N=collapse.get("late_p_finish_state_N"),
        late_occupancy_state_N=collapse.get("late_occupancy_state_N"),
        wall_clock_seconds=wall_clock_seconds,
        energy_kwh=energy_kwh,
        co2eq_kg=co2eq_kg,
        environment_steps=result.environment_steps,
        infrastructure_failure=None,
        **rates,
        **ppo_health,
    )


async def run_trial_seeds(
    trial_config_dicts: dict[int, dict[str, Any]], scenario_examples_dir: Path, run_dirs: dict[int, Path]
) -> list[SeedResult]:
    """Runs every seed of one trial STRICTLY SEQUENTIALLY (spec: n_jobs=1,
    clean per-agent CodeCarbon/resource attribution, no GPU contention --
    never asyncio.gather'd, never run concurrently), in ascending seed
    order for determinism.
    """
    results: list[SeedResult] = []
    for seed in sorted(trial_config_dicts):
        results.append(await run_agent(trial_config_dicts[seed], scenario_examples_dir, run_dirs[seed]))
    return results
