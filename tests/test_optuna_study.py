"""marla.optuna_study.study/finalists -- study lifecycle (resume, failed-
trial handling, paper-seed exclusion), finalist selection, and holdout
isolation. Uses a FAKE run_trial_seeds (no real PPO training) throughout,
per spec section 54.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from optuna.trial import TrialState

from marla.optuna_study import study as study_module
from marla.optuna_study.config import StudyConfig, load_study_config
from marla.optuna_study.finalists import select_winner
from marla.optuna_study.objective import SeedResult
from marla.optuna_study.runner import InfrastructureFailure


def _base_config_dict() -> dict:
    return {
        "schema_version": "1.0",
        "experiment": {"name": "x", "seed": 1},
        "execution": {"mode": "local"},
        "device": "cpu",
        "xmpp": {"server": "localhost"},
        "environment": {"mode": "simulation", "scenario": "marla://x.yaml", "max_episode_steps": 100},
        "objective": {"type": "capture_target", "description": "d"},
        "policy": {
            "algorithm": "recurrent_ppo",
            "graph_encoder": {"type": "graphsage", "hidden_size": 128, "layers": 2},
            "action_encoder": {"hidden_size": 128, "action_type_embedding_size": 32},
            "recurrent": {"hidden_size": 128, "sequence_length": 64},
            "ppo": {
                "total_environment_steps": 12288, "num_envs": 4, "steps_per_env": 512, "epochs": 4,
                "minibatch_sequences": 8, "gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.2,
                "value_coefficient": 0.5, "action_entropy_coefficient": 0.01, "query_entropy_coefficient": 0.01,
                "max_grad_norm": 0.5, "critic_refinement_epochs": 0,
                "optimizer": {"type": "adam", "learning_rate": 3e-4, "eps": 1e-5, "scheduler": {"type": "constant"}},
            },
        },
        "consultation": {"mode": "disabled"},
        "rl_orchestrator": {"alias": "rl_orchestrator", "jid": "rl@localhost"},
    }


def _write_study_yaml(tmp_path: Path, **overrides) -> Path:
    base_config_path = tmp_path / "base_config.yaml"
    base_config_path.write_text(yaml.safe_dump(_base_config_dict()), encoding="utf-8")
    config = {
        "study_name": "test-study",
        "storage": "sqlite:///study.db",
        "sampler_seed": 4242,
        "n_startup_trials": 1,
        "n_completed_trials_target": 3,
        "base_config": "base_config.yaml",
        "tuning_seeds": [909, 919],
        "holdout_seed": 929,
        "tuning_total_environment_steps": 12288,
        "finalist_total_environment_steps": 20480,
        "runs_root": "runs",
        "max_consecutive_infrastructure_failures": 5,
        **overrides,
    }
    study_path = tmp_path / "study.yaml"
    study_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return study_path


def _fake_seed_result(seed: int, root_auc: float, collapsed: bool = False, **overrides) -> SeedResult:
    defaults = dict(
        seed=seed, run_id=f"x-seed{seed}", run_dir="/tmp/x", root_auc=root_auc,
        collapse={"collapsed": collapsed}, mean_targets_rooted=0.5, fraction_with_any_root=0.3,
        root_auc_fraction_with_any_root=0.3, objective_reached_rate=0.1, successful_finish_rate=0.0,
        premature_finish_rate=0.5, late_p_finish_state_N=0.1, late_occupancy_state_N=0.5,
        action_entropy_final=3.0, approx_kl_final=0.001, clip_fraction_final=0.0,
        explained_variance_final=0.2, wall_clock_seconds=10.0, energy_kwh=0.001, co2eq_kg=0.0001,
        environment_steps=12288, infrastructure_failure=None,
    )
    defaults.update(overrides)
    return SeedResult(**defaults)


def test_config_rejects_paper_seeds_anywhere():
    with pytest.raises(Exception):
        StudyConfig(
            study_name="s", storage="sqlite:///x.db", sampler_seed=4242, n_completed_trials_target=1,
            base_config="b.yaml", tuning_seeds=[101, 919], holdout_seed=929,
            tuning_total_environment_steps=12288, finalist_total_environment_steps=20480, runs_root="r",
        )
    with pytest.raises(Exception):
        StudyConfig(
            study_name="s", storage="sqlite:///x.db", sampler_seed=101, n_completed_trials_target=1,
            base_config="b.yaml", tuning_seeds=[909, 919], holdout_seed=929,
            tuning_total_environment_steps=12288, finalist_total_environment_steps=20480, runs_root="r",
        )
    with pytest.raises(Exception):
        StudyConfig(
            study_name="s", storage="sqlite:///x.db", sampler_seed=4242, n_completed_trials_target=1,
            base_config="b.yaml", tuning_seeds=[909, 919], holdout_seed=303,
            tuning_total_environment_steps=12288, finalist_total_environment_steps=20480, runs_root="r",
        )


def test_config_rejects_holdout_seed_overlapping_tuning_seeds():
    with pytest.raises(Exception):
        StudyConfig(
            study_name="s", storage="sqlite:///x.db", sampler_seed=4242, n_completed_trials_target=1,
            base_config="b.yaml", tuning_seeds=[909, 919], holdout_seed=919,
            tuning_total_environment_steps=12288, finalist_total_environment_steps=20480, runs_root="r",
        )


@pytest.mark.asyncio
async def test_run_study_runs_two_seeds_per_trial_and_reaches_the_completed_target(tmp_path, monkeypatch):
    study_path = _write_study_yaml(tmp_path, n_completed_trials_target=2)
    config, base_dir = load_study_config(study_path)

    calls = []

    async def fake_run_trial_seeds(trial_config_dicts, scenario_examples_dir, run_dirs):
        calls.append(sorted(trial_config_dicts.keys()))
        return [_fake_seed_result(seed, root_auc=0.5) for seed in sorted(trial_config_dicts)]

    monkeypatch.setattr(study_module, "run_trial_seeds", fake_run_trial_seeds)
    study = await study_module.run_study(config, base_dir)

    assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 2
    assert all(c == [909, 919] for c in calls)  # every trial evaluates EXACTLY seeds 909 and 919


@pytest.mark.asyncio
async def test_resume_does_not_rerun_already_completed_trials(tmp_path, monkeypatch):
    study_path = _write_study_yaml(tmp_path, n_completed_trials_target=2)
    config, base_dir = load_study_config(study_path)

    run_count = {"n": 0}

    async def fake_run_trial_seeds(trial_config_dicts, scenario_examples_dir, run_dirs):
        run_count["n"] += 1
        return [_fake_seed_result(seed, root_auc=0.5) for seed in sorted(trial_config_dicts)]

    monkeypatch.setattr(study_module, "run_trial_seeds", fake_run_trial_seeds)
    await study_module.run_study(config, base_dir)
    assert run_count["n"] == 2

    # "Resume": load the same study again with a HIGHER target -- must
    # only run the DELTA (1 more trial), never re-run trials 0/1.
    config2, base_dir2 = load_study_config(study_path)
    object.__setattr__(config2, "n_completed_trials_target", 3)
    await study_module.run_study(config2, base_dir2)
    assert run_count["n"] == 3  # exactly one new trial ran


@pytest.mark.asyncio
async def test_infrastructure_failure_marks_trial_fail_not_a_numerical_penalty(tmp_path, monkeypatch):
    study_path = _write_study_yaml(tmp_path, n_completed_trials_target=1)
    config, base_dir = load_study_config(study_path)

    call_count = {"n": 0}

    async def flaky_run_trial_seeds(trial_config_dicts, scenario_examples_dir, run_dirs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise InfrastructureFailure("RuntimeError", "simulated crash", None)
        return [_fake_seed_result(seed, root_auc=0.5) for seed in sorted(trial_config_dicts)]

    monkeypatch.setattr(study_module, "run_trial_seeds", flaky_run_trial_seeds)
    study = await study_module.run_study(config, base_dir)

    trials = study.get_trials(deepcopy=False)
    assert any(t.state == TrialState.FAIL for t in trials)
    assert any(t.state == TrialState.COMPLETE for t in trials)
    failed = next(t for t in trials if t.state == TrialState.FAIL)
    assert failed.user_attrs["infrastructure_failure_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_too_many_consecutive_infrastructure_failures_stops_the_study(tmp_path, monkeypatch):
    study_path = _write_study_yaml(tmp_path, n_completed_trials_target=5, max_consecutive_infrastructure_failures=2)
    config, base_dir = load_study_config(study_path)

    async def always_fails(trial_config_dicts, scenario_examples_dir, run_dirs):
        raise InfrastructureFailure("RuntimeError", "systemic failure", None)

    monkeypatch.setattr(study_module, "run_trial_seeds", always_fails)
    with pytest.raises(RuntimeError, match="consecutive infrastructure failures"):
        await study_module.run_study(config, base_dir)


@pytest.mark.asyncio
async def test_collapsed_trial_never_counts_toward_completed_target_incorrectly(tmp_path, monkeypatch):
    """A collapsed trial IS still COMPLETE in Optuna's own state machine
    (it is not an infrastructure failure) -- this just confirms that
    path doesn't crash and produces a negative objective as expected.
    """
    study_path = _write_study_yaml(tmp_path, n_completed_trials_target=1)
    config, base_dir = load_study_config(study_path)

    async def fake_run_trial_seeds(trial_config_dicts, scenario_examples_dir, run_dirs):
        return [_fake_seed_result(909, root_auc=0.9, collapsed=True), _fake_seed_result(919, root_auc=0.9)]

    monkeypatch.setattr(study_module, "run_trial_seeds", fake_run_trial_seeds)
    study = await study_module.run_study(config, base_dir)
    trial = study.get_trials(states=(TrialState.COMPLETE,))[0]
    assert trial.value < 0
    assert trial.user_attrs["trial_stable"] is False


def _fake_trial(number: int, params: dict, value: float, stable: bool = True):
    import optuna

    study = optuna.create_study(direction="maximize")
    t = study.ask()
    for name, v in params.items():
        if isinstance(v, str):
            study.sampler = optuna.samplers.RandomSampler()
    # Simplest reliable approach: build a FrozenTrial directly.
    from optuna.trial import FrozenTrial, TrialState
    import datetime

    return FrozenTrial(
        number=number, state=TrialState.COMPLETE, value=value, values=None,
        datetime_start=datetime.datetime.now(), datetime_complete=datetime.datetime.now(),
        params=params, distributions={}, user_attrs={"trial_stable": stable}, system_attrs={},
        intermediate_values={}, trial_id=number,
    )


def test_select_winner_rejects_configs_that_collapse_on_either_seed():
    candidate = _fake_trial(0, {"learning_rate": 1e-4}, value=0.9)
    finalist_results = {0: [_fake_seed_result(909, root_auc=0.9, collapsed=True), _fake_seed_result(919, root_auc=0.9)]}
    winner, report = select_winner([candidate], finalist_results)
    assert winner is None
    assert report["reason"].startswith("every finalist")


def test_select_winner_picks_highest_worst_seed_root_auc():
    c0 = _fake_trial(0, {"learning_rate": 1e-4}, value=0.3)
    c1 = _fake_trial(1, {"learning_rate": 2e-4}, value=0.6)
    finalist_results = {
        0: [_fake_seed_result(909, root_auc=0.3), _fake_seed_result(919, root_auc=0.35)],
        1: [_fake_seed_result(909, root_auc=0.6), _fake_seed_result(919, root_auc=0.65)],
    }
    winner, report = select_winner([c0, c1], finalist_results)
    assert winner.number == 1


def test_select_winner_uses_carbon_as_final_tiebreak_only_when_practically_tied():
    c0 = _fake_trial(0, {"learning_rate": 1e-4}, value=0.5)
    c1 = _fake_trial(1, {"learning_rate": 2e-4}, value=0.5)
    finalist_results = {
        0: [
            _fake_seed_result(909, root_auc=0.500, successful_finish_rate=0.1, premature_finish_rate=0.3, co2eq_kg=0.01),
            _fake_seed_result(919, root_auc=0.501, successful_finish_rate=0.1, premature_finish_rate=0.3, co2eq_kg=0.01),
        ],
        1: [
            _fake_seed_result(909, root_auc=0.500, successful_finish_rate=0.1, premature_finish_rate=0.3, co2eq_kg=0.001),
            _fake_seed_result(919, root_auc=0.501, successful_finish_rate=0.1, premature_finish_rate=0.3, co2eq_kg=0.001),
        ],
    }
    winner, report = select_winner([c0, c1], finalist_results)
    assert winner.number == 1  # lower total CO2eq, everything else practically tied
    assert 0 in report["practical_tie_among"] and 1 in report["practical_tie_among"]


@pytest.mark.asyncio
async def test_holdout_seed_never_appears_in_a_normal_trial(tmp_path, monkeypatch):
    study_path = _write_study_yaml(tmp_path, n_completed_trials_target=1, holdout_seed=929, tuning_seeds=[909, 919])
    config, base_dir = load_study_config(study_path)

    seen_seeds = set()

    async def fake_run_trial_seeds(trial_config_dicts, scenario_examples_dir, run_dirs):
        seen_seeds.update(trial_config_dicts.keys())
        return [_fake_seed_result(seed, root_auc=0.5) for seed in sorted(trial_config_dicts)]

    monkeypatch.setattr(study_module, "run_trial_seeds", fake_run_trial_seeds)
    await study_module.run_study(config, base_dir)
    assert 929 not in seen_seeds
    assert seen_seeds == {909, 919}
