"""marla.optuna_study.objective -- the worst-seed ROOT AUC objective and
the collapse hard constraint.
"""

from __future__ import annotations

import pytest

from marla.optuna_study.objective import SeedResult, build_user_attrs, trial_objective


def _seed_result(seed: int, root_auc: float | None, collapsed: bool, **overrides) -> SeedResult:
    defaults = dict(
        seed=seed, run_id=f"x-seed{seed}", run_dir="/tmp/x", root_auc=root_auc,
        collapse={"collapsed": collapsed}, mean_targets_rooted=0.5, fraction_with_any_root=0.3,
        root_auc_fraction_with_any_root=0.3, objective_reached_rate=0.1, successful_finish_rate=0.0,
        premature_finish_rate=0.5, late_p_finish_state_N=0.1, late_occupancy_state_N=0.5,
        action_entropy_final=3.0, approx_kl_final=0.001, clip_fraction_final=0.0,
        explained_variance_final=0.2, wall_clock_seconds=100.0, energy_kwh=0.001, co2eq_kg=0.0001,
        environment_steps=12288, infrastructure_failure=None,
    )
    defaults.update(overrides)
    return SeedResult(**defaults)


def test_objective_is_the_worst_seed_root_auc_when_both_stable():
    results = [_seed_result(909, root_auc=0.6, collapsed=False), _seed_result(919, root_auc=0.4, collapsed=False)]
    objective, stable = trial_objective(results)
    assert objective == pytest.approx(0.4)
    assert stable is True


def test_objective_is_symmetric_in_seed_order():
    results_a = [_seed_result(909, root_auc=0.4, collapsed=False), _seed_result(919, root_auc=0.6, collapsed=False)]
    results_b = list(reversed(results_a))
    assert trial_objective(results_a)[0] == trial_objective(results_b)[0]


def test_collapsed_trial_gets_a_negative_objective_strictly_worse_than_any_valid_trial():
    stable_results = [_seed_result(909, root_auc=0.01, collapsed=False), _seed_result(919, root_auc=0.01, collapsed=False)]
    collapsed_results = [_seed_result(909, root_auc=0.9, collapsed=True), _seed_result(919, root_auc=0.9, collapsed=False)]
    stable_objective, stable_flag = trial_objective(stable_results)
    collapsed_objective, collapsed_flag = trial_objective(collapsed_results)
    assert collapsed_flag is False
    assert stable_flag is True
    assert collapsed_objective < 0.0
    assert collapsed_objective < stable_objective  # worse than even a barely-positive stable trial


def test_collapse_in_either_seed_marks_the_whole_trial_unstable():
    results = [_seed_result(909, root_auc=0.8, collapsed=True), _seed_result(919, root_auc=0.8, collapsed=False)]
    _, stable = trial_objective(results)
    assert stable is False


def test_none_root_auc_treated_as_zero_not_a_crash():
    results = [_seed_result(909, root_auc=None, collapsed=False), _seed_result(919, root_auc=0.5, collapsed=False)]
    objective, stable = trial_objective(results)
    assert objective == pytest.approx(0.0)
    assert stable is True


def test_infrastructure_failure_present_raises_rather_than_scoring():
    results = [_seed_result(909, root_auc=0.5, collapsed=False, infrastructure_failure="CUDA OOM")]
    with pytest.raises(ValueError):
        trial_objective(results)


def test_build_user_attrs_has_per_seed_and_aggregate_keys():
    results = [_seed_result(909, root_auc=0.6, collapsed=False), _seed_result(919, root_auc=0.4, collapsed=False)]
    objective, stable = trial_objective(results)
    attrs = build_user_attrs(results, objective, stable)
    assert attrs["worst_seed_root_auc"] == pytest.approx(0.4)
    assert attrs["mean_root_auc"] == pytest.approx(0.5)
    assert attrs["seed909_root_auc"] == pytest.approx(0.6)
    assert attrs["seed919_root_auc"] == pytest.approx(0.4)
    assert attrs["seed909_collapsed"] is False
    assert attrs["trial_stable"] is stable
    assert attrs["total_wall_clock_seconds"] == pytest.approx(200.0)
    assert attrs["total_energy_kwh"] == pytest.approx(0.002)
    assert attrs["total_co2eq_kg"] == pytest.approx(0.0002)
