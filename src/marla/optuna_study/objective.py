"""The Optuna objective: worst-seed ROOT AUC with a hard collapse
constraint (spec sections 40-43).

    objective = min(ROOT_AUC_909, ROOT_AUC_919)   if neither seed collapsed
    objective = -1.0 - max(0, -min_root_auc)      if either seed collapsed

Collapsed trials get a NEGATIVE objective, strictly worse than every
valid non-collapsed trial (whose ROOT AUC is always >= 0 by construction
-- ``mean_targets_rooted`` cannot be negative) -- this is the "a simple
convention... provided valid ROOT AUC is non-negative" from spec section
42, tested explicitly in tests/test_optuna_objective.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SeedResult:
    """Everything computed for ONE seed of ONE trial -- also used
    verbatim for finalist/holdout agents (spec sections 26-27), not just
    ordinary Optuna trials.
    """

    seed: int
    run_id: str
    run_dir: str
    root_auc: float | None
    collapse: dict[str, Any]
    mean_targets_rooted: float | None
    fraction_with_any_root: float | None
    root_auc_fraction_with_any_root: float | None
    objective_reached_rate: float | None
    successful_finish_rate: float | None
    premature_finish_rate: float | None
    late_p_finish_state_N: float | None
    late_occupancy_state_N: float | None
    action_entropy_final: float | None
    approx_kl_final: float | None
    clip_fraction_final: float | None
    explained_variance_final: float | None
    wall_clock_seconds: float
    energy_kwh: float | None
    co2eq_kg: float | None
    environment_steps: int
    infrastructure_failure: str | None = None  # exception type name, or None if the agent completed normally


COLLAPSE_PENALTY_BASE = -1.0


def trial_objective(seed_results: list[SeedResult]) -> tuple[float, bool]:
    """Returns ``(objective_value, trial_stable)``. Raises ValueError if
    any seed had an infrastructure failure -- callers must handle that
    BEFORE calling this (spec section 53: infrastructure failures are a
    FAILED trial, never transformed into a numerical objective value).
    """
    for r in seed_results:
        if r.infrastructure_failure is not None:
            raise ValueError(
                f"trial_objective called with an infrastructure failure present (seed {r.seed}: "
                f"{r.infrastructure_failure}) -- this trial must be marked FAIL, not scored."
            )

    collapsed = any(r.collapse.get("collapsed", False) for r in seed_results)
    root_aucs = [r.root_auc if r.root_auc is not None else 0.0 for r in seed_results]
    worst_root_auc = min(root_aucs)

    if collapsed:
        # Strictly worse than every possible valid (non-negative) ROOT
        # AUC objective -- the further below zero, the fewer sensitive
        # targets a collapsed trial still managed to root before/while
        # collapsing, so a "less bad" collapse is still ranked below any
        # stable trial, but distinguishable from a "total" collapse.
        return COLLAPSE_PENALTY_BASE - max(0.0, -worst_root_auc), False
    return worst_root_auc, True


def build_user_attrs(seed_results: list[SeedResult], objective_value: float, trial_stable: bool) -> dict[str, Any]:
    """Secondary metrics persisted as Optuna user attributes (spec
    section 43) -- per-seed AND aggregated, so a completed trial's full
    scientific record survives independent of this module's own
    objective-scoring convention.
    """
    root_aucs = [r.root_auc for r in seed_results]
    attrs: dict[str, Any] = {
        "objective_used_by_optuna": objective_value,
        "trial_stable": trial_stable,
        "mean_root_auc": (sum(a for a in root_aucs if a is not None) / len(root_aucs)) if all(a is not None for a in root_aucs) else None,
        "worst_seed_root_auc": min((a for a in root_aucs if a is not None), default=None),
        "total_wall_clock_seconds": sum(r.wall_clock_seconds for r in seed_results),
        "total_energy_kwh": sum((r.energy_kwh or 0.0) for r in seed_results) if any(r.energy_kwh is not None for r in seed_results) else None,
        "total_co2eq_kg": sum((r.co2eq_kg or 0.0) for r in seed_results) if any(r.co2eq_kg is not None for r in seed_results) else None,
    }
    for r in seed_results:
        prefix = f"seed{r.seed}_"
        attrs.update(
            {
                prefix + "root_auc": r.root_auc,
                prefix + "collapsed": r.collapse.get("collapsed", False),
                prefix + "mean_targets_rooted": r.mean_targets_rooted,
                prefix + "fraction_with_any_root": r.fraction_with_any_root,
                prefix + "objective_reached_rate": r.objective_reached_rate,
                prefix + "successful_finish_rate": r.successful_finish_rate,
                prefix + "premature_finish_rate": r.premature_finish_rate,
                prefix + "late_p_finish_state_N": r.late_p_finish_state_N,
                prefix + "late_occupancy_state_N": r.late_occupancy_state_N,
                prefix + "action_entropy_final": r.action_entropy_final,
                prefix + "approx_kl_final": r.approx_kl_final,
                prefix + "clip_fraction_final": r.clip_fraction_final,
                prefix + "explained_variance_final": r.explained_variance_final,
                prefix + "wall_clock_seconds": r.wall_clock_seconds,
                prefix + "energy_kwh": r.energy_kwh,
                prefix + "co2eq_kg": r.co2eq_kg,
                prefix + "run_id": r.run_id,
                prefix + "run_dir": r.run_dir,
            }
        )
    return attrs
