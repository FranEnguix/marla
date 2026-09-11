"""Finalist confirmation runs (fresh, full-horizon, spec section 26),
winner selection (spec section 46's lexicographic rule), and holdout
seed929 validation (spec section 27).
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import optuna

from marla.optuna_study.config import StudyConfig, resolve_path
from marla.optuna_study.objective import SeedResult
from marla.optuna_study.runner import run_trial_seeds
from marla.optuna_study.search_space import apply_to_config_dict, hyperparameters_from_params_dict
from marla.optuna_study.study import examples_dir, load_base_config_dict

logger = logging.getLogger(__name__)


async def run_finalists(
    config: StudyConfig, base_dir: Path, base_config_dict: dict[str, Any], candidates: list[optuna.trial.FrozenTrial]
) -> dict[int, list[SeedResult]]:
    """Runs each candidate trial's hyperparameters FRESH (never reusing a
    tuning-horizon checkpoint) at the finalist horizon, for both tuning
    seeds, in a SEPARATE ``<runs_root>-finalists/`` namespace so trial
    directories are never overwritten (spec sections 26, 51).
    """
    finalists_root = Path(str(resolve_path(config.runs_root, base_dir)) + "-finalists")
    results: dict[int, list[SeedResult]] = {}
    for candidate in candidates:
        sampled = hyperparameters_from_params_dict(candidate.params)
        candidate_dir = finalists_root / f"candidate_trial{candidate.number:04d}"
        trial_config_dicts = {}
        run_dirs = {}
        for seed in config.tuning_seeds:
            run_id = f"finalist-trial{candidate.number:04d}-seed{seed}"
            trial_config_dicts[seed] = apply_to_config_dict(
                base_config_dict, sampled, seed=seed, run_id=run_id,
                total_environment_steps=config.finalist_total_environment_steps,
            )
            run_dirs[seed] = candidate_dir / f"seed{seed}"
        logger.info("Running finalist candidate (source trial %d) at the full horizon...", candidate.number)
        results[candidate.number] = await run_trial_seeds(trial_config_dicts, examples_dir(), run_dirs)
    return results


def select_winner(
    candidates: list[optuna.trial.FrozenTrial], finalist_results: dict[int, list[SeedResult]]
) -> tuple[optuna.trial.FrozenTrial | None, dict[str, Any]]:
    """Spec section 46's lexicographic rule. Returns
    ``(winner_trial_or_None, selection_report)``. ``None`` winner means
    every finalist collapsed on at least one seed -- report says so
    explicitly, never silently picks a collapsed configuration.
    """
    scored = []
    for candidate in candidates:
        seed_results = finalist_results[candidate.number]
        collapsed_any = any(r.collapse.get("collapsed", False) for r in seed_results)
        root_aucs = [r.root_auc if r.root_auc is not None else 0.0 for r in seed_results]
        worst_root_auc = min(root_aucs)
        mean_root_auc = sum(root_aucs) / len(root_aucs)
        successful_finish_rate = sum((r.successful_finish_rate or 0.0) for r in seed_results) / len(seed_results)
        premature_finish_rate = sum((r.premature_finish_rate or 0.0) for r in seed_results) / len(seed_results)
        total_co2eq_kg = sum((r.co2eq_kg or 0.0) for r in seed_results)
        scored.append(
            {
                "trial_number": candidate.number,
                "collapsed_any": collapsed_any,
                "worst_root_auc": worst_root_auc,
                "mean_root_auc": mean_root_auc,
                "successful_finish_rate": successful_finish_rate,
                "premature_finish_rate": premature_finish_rate,
                "total_co2eq_kg": total_co2eq_kg,
            }
        )

    # First: reject any configuration that collapses on either seed.
    non_collapsed = [s for s in scored if not s["collapsed_any"]]
    report: dict[str, Any] = {"all_candidates": scored, "eligible_after_collapse_filter": [s["trial_number"] for s in non_collapsed]}
    if not non_collapsed:
        report["winner"] = None
        report["reason"] = "every finalist candidate collapsed on at least one seed"
        return None, report

    # Second: highest min(ROOT_AUC_909, ROOT_AUC_919). "Practically tied"
    # (spec section 47): within 2% relative of the best worst-seed ROOT
    # AUC is treated as indistinguishable, deferring to the later
    # tie-break criteria rather than a spurious tiny numeric difference.
    best_worst_auc = max(s["worst_root_auc"] for s in non_collapsed)
    tie_threshold = abs(best_worst_auc) * 0.02
    tied_on_worst_auc = [s for s in non_collapsed if best_worst_auc - s["worst_root_auc"] <= tie_threshold]

    def _break_tie(pool: list[dict]) -> list[dict]:
        if len(pool) <= 1:
            return pool
        best_mean = max(s["mean_root_auc"] for s in pool)
        mean_tie_threshold = abs(best_mean) * 0.02
        pool = [s for s in pool if best_mean - s["mean_root_auc"] <= mean_tie_threshold]
        if len(pool) <= 1:
            return pool
        best_success = max(s["successful_finish_rate"] for s in pool)
        pool = [s for s in pool if s["successful_finish_rate"] >= best_success - 1e-9]
        if len(pool) <= 1:
            return pool
        best_premature = min(s["premature_finish_rate"] for s in pool)
        pool = [s for s in pool if s["premature_finish_rate"] <= best_premature + 1e-9]
        if len(pool) <= 1:
            return pool
        best_co2 = min(s["total_co2eq_kg"] for s in pool)
        return [s for s in pool if s["total_co2eq_kg"] <= best_co2 + 1e-12]

    final_pool = _break_tie(tied_on_worst_auc)
    winner_summary = final_pool[0] if final_pool else max(non_collapsed, key=lambda s: s["worst_root_auc"])
    winner_trial = next(c for c in candidates if c.number == winner_summary["trial_number"])

    report["practical_tie_among"] = [s["trial_number"] for s in tied_on_worst_auc]
    report["winner"] = winner_summary["trial_number"]
    report["winner_summary"] = winner_summary
    return winner_trial, report


async def run_holdout(
    config: StudyConfig, base_dir: Path, base_config_dict: dict[str, Any], winner_trial: optuna.trial.FrozenTrial
) -> SeedResult:
    """Runs the selected winner ONCE on the held-out tuning seed (spec
    section 27) -- never used to pick a different winner in this task.
    """
    sampled = hyperparameters_from_params_dict(winner_trial.params)
    holdout_root = Path(str(resolve_path(config.runs_root, base_dir)) + "-holdout")
    run_id = f"holdout-trial{winner_trial.number:04d}-seed{config.holdout_seed}"
    config_dict = apply_to_config_dict(
        base_config_dict, sampled, seed=config.holdout_seed, run_id=run_id,
        total_environment_steps=config.finalist_total_environment_steps,
    )
    run_dir = holdout_root / f"seed{config.holdout_seed}"
    results = await run_trial_seeds({config.holdout_seed: config_dict}, examples_dir(), {config.holdout_seed: run_dir})
    return results[0]
