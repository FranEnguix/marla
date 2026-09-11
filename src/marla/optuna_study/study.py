"""The Optuna study driver: creates/resumes a persistent study, runs
trials strictly sequentially (n_jobs=1, spec section 21) until the
COMPLETED-trial target is met, and drives the finalist/holdout stages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import yaml
from dataclasses import asdict
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState

from marla.optuna_study.config import StudyConfig, resolve_path, resolve_storage_url
from marla.optuna_study.objective import SeedResult, build_user_attrs, trial_objective
from marla.optuna_study.runner import InfrastructureFailure, run_trial_seeds
from marla.optuna_study.search_space import apply_to_config_dict, hyperparameters_from_params_dict, sample_hyperparameters

logger = logging.getLogger(__name__)

EXAMPLES_DIR_NAME = "examples"  # resolve_scenario_reference's own second-arg convention


def create_or_load_study(config: StudyConfig, base_dir: Path) -> optuna.Study:
    storage_url = resolve_storage_url(config, base_dir)
    sampler = optuna.samplers.TPESampler(seed=config.sampler_seed, n_startup_trials=config.n_startup_trials)
    return optuna.create_study(
        study_name=config.study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),  # spec section 20: no pruning in this first study
        direction="maximize",
        load_if_exists=True,
    )


def _repo_root() -> Path:
    # src/marla/optuna_study/study.py -> repo root is 4 parents up.
    return Path(__file__).resolve().parents[3]


def examples_dir() -> Path:
    return _repo_root() / EXAMPLES_DIR_NAME


def load_base_config_dict(config: StudyConfig, base_dir: Path) -> dict[str, Any]:
    base_config_path = resolve_path(config.base_config, base_dir)
    return yaml.safe_load(base_config_path.read_text(encoding="utf-8"))


def _trial_dir(config: StudyConfig, base_dir: Path, trial_number: int) -> Path:
    return resolve_path(config.runs_root, base_dir) / f"trial_{trial_number:04d}"


def _write_trial_artifacts(
    trial_dir: Path, sampled: dict[str, Any], seed_results: list[SeedResult], objective_value: float, trial_stable: bool
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "params.yaml").write_text(yaml.safe_dump(sampled, sort_keys=False), encoding="utf-8")
    aggregate = {
        "objective": objective_value,
        "trial_stable": trial_stable,
        "seeds": {str(r.seed): {**asdict(r)} for r in seed_results},
        **build_user_attrs(seed_results, objective_value, trial_stable),
    }
    (trial_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")


async def run_one_trial(
    trial: "optuna.trial.Trial", config: StudyConfig, base_dir: Path, base_config_dict: dict[str, Any]
) -> tuple[float, bool, list[SeedResult]]:
    """Runs both tuning seeds under the trial's sampled hyperparameters,
    sequentially. Raises InfrastructureFailure (propagated from
    run_trial_seeds) if either seed's agent fails for a non-scientific
    reason -- the caller (run_study) is responsible for telling Optuna
    FAIL in that case, never a numerical objective.
    """
    sampled = sample_hyperparameters(trial)
    trial_dir = _trial_dir(config, base_dir, trial.number)

    trial_config_dicts = {}
    run_dirs = {}
    for seed in config.tuning_seeds:
        run_id = f"trial{trial.number:04d}-seed{seed}"
        trial_config_dicts[seed] = apply_to_config_dict(
            base_config_dict, sampled, seed=seed, run_id=run_id, total_environment_steps=config.tuning_total_environment_steps
        )
        run_dirs[seed] = trial_dir / f"seed{seed}"

    seed_results = await run_trial_seeds(trial_config_dicts, examples_dir(), run_dirs)
    objective_value, trial_stable = trial_objective(seed_results)
    _write_trial_artifacts(trial_dir, sampled, seed_results, objective_value, trial_stable)
    return objective_value, trial_stable, seed_results


async def run_study(config: StudyConfig, base_dir: Path) -> optuna.Study:
    study = create_or_load_study(config, base_dir)
    base_config_dict = load_base_config_dict(config, base_dir)

    def _completed_count() -> int:
        return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))

    consecutive_infra_failures = 0
    while _completed_count() < config.n_completed_trials_target:
        trial = study.ask()
        logger.info("Starting Optuna trial %d (completed so far: %d/%d)", trial.number, _completed_count(), config.n_completed_trials_target)
        try:
            objective_value, trial_stable, seed_results = await run_one_trial(trial, config, base_dir, base_config_dict)
        except InfrastructureFailure as exc:
            consecutive_infra_failures += 1
            logger.error("Trial %d FAILED (infrastructure): %s: %s", trial.number, exc.exception_type, exc.message)
            trial.set_user_attr("infrastructure_failure_type", exc.exception_type)
            trial.set_user_attr("infrastructure_failure_message", exc.message)
            trial.set_user_attr("infrastructure_failure_run_dir", str(exc.run_dir) if exc.run_dir else None)
            study.tell(trial, state=TrialState.FAIL)
            if consecutive_infra_failures > config.max_consecutive_infrastructure_failures:
                raise RuntimeError(
                    f"{consecutive_infra_failures} consecutive infrastructure failures "
                    f"(> max_consecutive_infrastructure_failures={config.max_consecutive_infrastructure_failures}) "
                    "-- stopping the study rather than continuing to retry a systemic issue. "
                    f"Last failure: {exc.exception_type}: {exc.message}"
                ) from exc
            continue

        consecutive_infra_failures = 0
        for key, value in build_user_attrs(seed_results, objective_value, trial_stable).items():
            trial.set_user_attr(key, value)
        study.tell(trial, objective_value, state=TrialState.COMPLETE)
        logger.info("Trial %d COMPLETE: objective=%.4f stable=%s", trial.number, objective_value, trial_stable)

    return study


def select_finalist_candidates(study: optuna.Study, n: int = 3) -> list[optuna.trial.FrozenTrial]:
    """Top-N COMPLETE trials by objective, restricted to trial_stable=True
    (spec section 45), deduplicated by hyperparameter values (spec: "do
    not select three duplicate/near-identical aliases from resumed
    trials") -- two trials with EXACTLY the same params dict count as one
    candidate; the higher-objective one is kept.
    """
    complete = [t for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)) if t.user_attrs.get("trial_stable", False)]
    complete.sort(key=lambda t: t.value, reverse=True)

    seen_param_signatures: set[tuple] = set()
    candidates: list[optuna.trial.FrozenTrial] = []
    for t in complete:
        signature = tuple(sorted(t.params.items()))
        if signature in seen_param_signatures:
            continue
        seen_param_signatures.add(signature)
        candidates.append(t)
        if len(candidates) >= n:
            break
    return candidates
