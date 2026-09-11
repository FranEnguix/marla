"""FINISH decision-time diagnostics: does the policy assign high FINISH
probability once the true objective is already satisfied? Measured
directly (per-decision FINISH probability/rank/argmax), never inferred
from successful_finish_rate alone.

Two separate concerns get their own test groups:
- decision-time correctness: objective_satisfied_before_action (and the
  paired sensitive_targets_*_before_action fields) must reflect the state
  that existed WHEN THE POLICY CHOSE the action, never the post-action
  state -- verified via a real rollout invariant, not a hand-picked
  scenario.
- FINISH field correctness/aggregation: unit-level, against synthetic
  StepRecord fixtures.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch

from marla.config.loader import load_config, parse_config
from marla.environment.action_compatibility import COMPATIBILITY_FEATURE_DIM
from marla.environment.actions import ActionDescriptor
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.environment.visible_facts import VISIBLE_PROGRESS_DIM
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import RolloutCollector, StepRecord
from marla.metrics.writer import _DECISIONS_FIELDS, _ROLLOUTS_FIELDS, build_decision_rows, build_rollout_row
from marla.scenarios import diagnostic_scenario_path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _tiny_config():
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    data["policy"]["ppo"]["steps_per_env"] = 24
    data["policy"]["recurrent"]["sequence_length"] = 4
    return parse_config(data)


# --- Decision-time correctness (real rollout invariant) --------------------


@pytest.mark.asyncio
async def test_objective_satisfied_before_action_reflects_state_at_decision_time_not_after():
    """For consecutive steps within the same episode, record[i+1]'s
    decision-time truth must equal record[i]'s post-action truth (that IS
    the state the policy saw when choosing action i+1) -- this holds
    regardless of which actions get selected, so it's checked as a general
    invariant over a real rollout rather than by hand-picking a sequence.
    The very first decision of an episode must also match a fresh
    objective_satisfied() query (both false, since a fresh episode never
    starts with the objective already satisfied in these scenarios).
    """
    config = _tiny_config()
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy)
    adapter = NasimEmuAdapter(
        scenario=str(diagnostic_scenario_path("micro_c_mixed_os_discrimination.v2.yaml")),
        max_episode_steps=20,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    collector = RolloutCollector(adapter, policy, run_id="finish-diag-test", base_seed=4)
    records, _summaries = await collector.collect(30)

    assert records, "need at least one collected step"
    # First record of the first episode: nothing has happened yet.
    first_episode_id = records[0].episode_id
    assert records[0].objective_satisfied_before_action is False

    for prev, curr in zip(records, records[1:]):
        if curr.episode_id != prev.episode_id:
            # New episode: same "starts unsatisfied" invariant as above.
            assert curr.objective_satisfied_before_action is False
            continue
        assert curr.objective_satisfied_before_action == prev.objective_satisfied
        assert curr.sensitive_targets_remaining_before_action == prev.sensitive_targets_remaining
        assert curr.sensitive_targets_with_root_before_action == prev.sensitive_targets_with_root


@pytest.mark.asyncio
async def test_objective_satisfied_before_action_is_true_exactly_one_step_after_the_rooting_action():
    """The concrete case the spec calls out: the step that obtains the
    final ROOT has objective_satisfied_before_action=False (nothing rooted
    yet when the action was chosen) and objective_satisfied=True (the
    action just achieved it); the NEXT decision then sees
    objective_satisfied_before_action=True.
    """
    config = _tiny_config()
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy)
    adapter = NasimEmuAdapter(
        scenario=str(diagnostic_scenario_path("micro_a_direct_root.v2.yaml")),
        max_episode_steps=20,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    collector = RolloutCollector(adapter, policy, run_id="finish-diag-test-2", base_seed=1)
    records, _summaries = await collector.collect(400)

    became_satisfied_steps = [r for r in records if r.objective_became_satisfied]
    assert became_satisfied_steps, "expected at least one objective-became-satisfied transition"
    root_step = became_satisfied_steps[0]
    assert root_step.objective_satisfied_before_action is False
    assert root_step.objective_satisfied is True

    following = [r for r in records if r.episode_id == root_step.episode_id and r.environment_step == root_step.environment_step + 1]
    if following:
        assert following[0].objective_satisfied_before_action is True


# --- FINISH field correctness (unit-level) ----------------------------------


def _record_with_actions(descriptors, base_logits, final_logits, selected_index):
    return StepRecord(
        run_id="r", episode_id=1, environment_step=0, observation_id="o",
        graph_data=None, node_key_to_index={}, legal_action_descriptors=descriptors,
        compatibility_features=torch.zeros((len(descriptors), COMPATIBILITY_FEATURE_DIM)),
        visible_progress=torch.zeros(VISIBLE_PROGRESS_DIM),
        initial_gru_hidden_state=torch.zeros(2), previous_action_embedding=torch.zeros(2),
        previous_training_reward=0.0, previous_query=False,
        base_logits=torch.tensor(base_logits, dtype=torch.float32),
        final_logits=torch.tensor(final_logits, dtype=torch.float32),
        selected_action_index=selected_index,
        old_action_log_probability=0.0, old_joint_log_probability=0.0, critic_value=0.0,
        nasimemu_reward=0.0, consultation_cost=0.0, training_reward=0.0,
        terminated=False, truncated=False,
        selected_action_type=descriptors[selected_index].action_type,
        objective_satisfied_before_action=False,
    )


def _mixed_actions():
    return [
        ActionDescriptor(action_id="os-scan:h", action_type="os_scan", target_key="h", parameters={}),
        ActionDescriptor(action_id="exploit:h:e", action_type="exploit", target_key="h", parameters={"service": "s", "os": "linux"}),
        ActionDescriptor(action_id="finish", action_type="finish", target_key=None, is_finish=True),
    ]


def test_exactly_one_finish_candidate_is_always_found_and_fields_are_well_formed():
    descriptors = _mixed_actions()
    record = _record_with_actions(descriptors, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], selected_index=2)
    [row] = build_decision_rows([record], advantages=[0.0], returns=[0.0])

    assert row["base_finish_probability"] is not None
    assert 0.0 <= row["base_finish_probability"] <= 1.0
    assert 0.0 <= row["final_finish_probability"] <= 1.0
    assert row["base_finish_rank"] >= 1
    assert row["final_finish_rank"] >= 1
    assert row["base_finish_is_argmax"] is True  # logit 3.0 is the max
    assert row["final_finish_is_argmax"] is True
    assert row["finish_selected"] is True


def test_finish_not_selected_and_not_argmax():
    descriptors = _mixed_actions()
    record = _record_with_actions(descriptors, [5.0, 2.0, 0.1], [5.0, 2.0, 0.1], selected_index=0)
    [row] = build_decision_rows([record], advantages=[0.0], returns=[0.0])

    assert row["base_finish_is_argmax"] is False
    assert row["final_finish_is_argmax"] is False
    assert row["finish_selected"] is False
    assert row["base_finish_rank"] == 3  # lowest of the three logits


def test_finish_rank_matches_documented_convention_one_is_highest():
    descriptors = _mixed_actions()
    # FINISH has the highest logit -> rank 1.
    record = _record_with_actions(descriptors, [0.0, 1.0, 9.0], [0.0, 1.0, 9.0], selected_index=2)
    [row] = build_decision_rows([record], advantages=[0.0], returns=[0.0])
    assert row["base_finish_rank"] == 1
    assert row["final_finish_rank"] == 1


def test_ppo_only_base_equals_final_finish_fields():
    """PPO_ONLY (baseline) never perturbs final_logits away from
    base_logits -- so every base_finish_* must equal its final_finish_*
    counterpart exactly."""
    descriptors = _mixed_actions()
    logits = [0.3, -1.2, 2.1]
    record = _record_with_actions(descriptors, logits, logits, selected_index=2)
    [row] = build_decision_rows([record], advantages=[0.0], returns=[0.0])

    assert row["base_finish_probability"] == pytest.approx(row["final_finish_probability"])
    assert row["base_finish_rank"] == row["final_finish_rank"]
    assert row["base_finish_is_argmax"] == row["final_finish_is_argmax"]


def test_finish_probability_sums_correctly_and_is_never_nan():
    descriptors = _mixed_actions()
    record = _record_with_actions(descriptors, [10.0, 10.0, 10.0], [10.0, 10.0, 10.0], selected_index=0)
    [row] = build_decision_rows([record], advantages=[0.0], returns=[0.0])
    assert not math.isnan(row["base_finish_probability"])
    assert row["base_finish_probability"] == pytest.approx(1.0 / 3.0)


# --- Conditional rollout aggregates -----------------------------------------


def _finish_decision_row(
    objective_satisfied_before, base_finish_prob, base_finish_argmax, finish_selected,
    known_subnets_total=1, known_subnets_scanned=0,
    has_visible_sensitive_target=False, all_visible_sensitive_targets_rooted=None,
):
    return {
        "objective_satisfied_before_action": objective_satisfied_before,
        "base_finish_probability": base_finish_prob,
        "final_finish_probability": base_finish_prob,
        "base_finish_is_argmax": base_finish_argmax,
        "final_finish_is_argmax": base_finish_argmax,
        "finish_selected": finish_selected,
        "base_compatible_action_probability_mass": 0.5,
        "final_compatible_action_probability_mass": 0.5,
        "selected_action_visible_preconditions_status": "unknown",
        "known_subnets_total_before_action": known_subnets_total,
        "known_subnets_scanned_before_action": known_subnets_scanned,
        "known_subnets_unscanned_before_action": known_subnets_total - known_subnets_scanned,
        "fraction_known_subnets_scanned_before_action": (
            known_subnets_scanned / known_subnets_total if known_subnets_total else 0.0
        ),
        "known_exploration_frontier_remaining_before_action": known_subnets_scanned < known_subnets_total,
        "has_visible_sensitive_target_before_action": has_visible_sensitive_target,
        "all_visible_sensitive_targets_rooted_before_action": all_visible_sensitive_targets_rooted,
    }


def _fake_step_records(n: int) -> list[StepRecord]:
    descriptors = _mixed_actions()
    return [_record_with_actions(descriptors, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], selected_index=0) for _ in range(n)]


def test_rollout_finish_aggregates_use_decision_time_objective_split_with_correct_denominators():
    rows = [
        _finish_decision_row(False, 0.1, False, False),
        _finish_decision_row(False, 0.2, False, False),
        _finish_decision_row(True, 0.8, True, True),
        _finish_decision_row(True, 0.6, True, False),
    ]
    row = build_rollout_row(
        run_id="r", rollout=1, environment_steps_total=100, records=_fake_step_records(len(rows)),
        episodes_finished=0, advantages=[0.0] * len(rows), returns=[0.0] * len(rows),
        collection_seconds=1.0, optimization_seconds=1.0, evaluation_seconds=None, decision_rows=rows,
    )

    assert row["mean_base_finish_probability_objective_not_reached"] == pytest.approx(0.15)
    assert row["mean_base_finish_probability_objective_reached"] == pytest.approx(0.7)
    assert row["base_finish_argmax_rate_objective_not_reached"] == pytest.approx(0.0)
    assert row["base_finish_argmax_rate_objective_reached"] == pytest.approx(1.0)
    assert row["finish_selected_rate_objective_not_reached"] == pytest.approx(0.0)
    assert row["finish_selected_rate_objective_reached"] == pytest.approx(0.5)


def test_rollout_finish_aggregates_are_null_not_zero_when_a_partition_is_empty():
    rows = [_finish_decision_row(False, 0.1, False, False)]  # never reaches the objective this rollout
    row = build_rollout_row(
        run_id="r", rollout=1, environment_steps_total=100, records=_fake_step_records(len(rows)),
        episodes_finished=0, advantages=[0.0], returns=[0.0],
        collection_seconds=1.0, optimization_seconds=1.0, evaluation_seconds=None, decision_rows=rows,
    )
    assert row["mean_base_finish_probability_objective_reached"] is None
    assert row["base_finish_argmax_rate_objective_reached"] is None
    assert row["finish_selected_rate_objective_reached"] is None
    # The populated partition must still have a real value, not also None.
    assert row["mean_base_finish_probability_objective_not_reached"] == pytest.approx(0.1)


def test_rollout_row_finish_fields_are_none_when_no_decision_rows_available():
    row = build_rollout_row(
        run_id="r", rollout=1, environment_steps_total=100, records=[],
        episodes_finished=0, advantages=[], returns=[],
        collection_seconds=1.0, optimization_seconds=1.0, evaluation_seconds=None, decision_rows=None,
    )
    for field in (
        "mean_base_finish_probability_objective_not_reached", "mean_base_finish_probability_objective_reached",
        "base_finish_argmax_rate_objective_not_reached", "finish_selected_rate_objective_reached",
        "selected_confirmed_compatible_rate_applicable_actions",
    ):
        assert row[field] is None


# --- Applicable-actions-only compatibility denominator ----------------------


def test_applicable_actions_compatibility_rate_excludes_scans_and_finish_from_denominator():
    from marla.environment.action_compatibility import CompatibilityStatus

    rows = [
        {**_finish_decision_row(False, 0.1, False, False), "selected_action_visible_preconditions_status": CompatibilityStatus.NOT_APPLICABLE.value},
        {**_finish_decision_row(False, 0.1, False, False), "selected_action_visible_preconditions_status": CompatibilityStatus.NOT_APPLICABLE.value},
        {**_finish_decision_row(False, 0.1, False, False), "selected_action_visible_preconditions_status": CompatibilityStatus.CONFIRMED_COMPATIBLE.value},
        {**_finish_decision_row(False, 0.1, False, False), "selected_action_visible_preconditions_status": CompatibilityStatus.UNKNOWN.value},
    ]
    row = build_rollout_row(
        run_id="r", rollout=1, environment_steps_total=100, records=_fake_step_records(len(rows)),
        episodes_finished=0, advantages=[0.0] * len(rows), returns=[0.0] * len(rows),
        collection_seconds=1.0, optimization_seconds=1.0, evaluation_seconds=None, decision_rows=rows,
    )

    # Overall (unchanged) denominator: every decision, including the two
    # NOT_APPLICABLE ones.
    assert row["selected_confirmed_compatible_rate"] == pytest.approx(1 / 4)
    assert row["selected_not_applicable_rate"] == pytest.approx(2 / 4)
    # Applicable-only denominator: just the 2 applicable decisions.
    assert row["selected_confirmed_compatible_rate_applicable_actions"] == pytest.approx(1 / 2)
    assert row["selected_unknown_rate_applicable_actions"] == pytest.approx(1 / 2)
    assert row["selected_contradicted_rate_applicable_actions"] == pytest.approx(0.0)


# --- Plot compatibility (new columns present / absent) ----------------------


@pytest.mark.integration
def test_generate_plots_includes_finish_diagnostics_plots_for_a_real_run(tmp_path):
    """A real (tiny) training run's own rollouts.csv/decisions.csv already
    carry the new FINISH columns end to end -- both new plots must render
    from them without any hand-built fixture."""
    import asyncio
    from datetime import datetime, timezone

    from marla.config.loader import load_config, parse_config
    from marla.learning.trainer import run_baseline_training
    from marla.metrics.plots import generate_plots
    from marla.metrics.writer import write_run_artifacts
    from marla.runtime.device import resolve_device

    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    data["policy"]["ppo"]["steps_per_env"] = 16
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["device"] = "cpu"
    data["metrics"]["eval_episodes"] = 0
    config = parse_config(data)

    scenario_path = str(diagnostic_scenario_path("micro_a_direct_root.v2.yaml"))
    result = asyncio.run(run_baseline_training(config, scenario_path, num_rollouts=2, seed=1))
    resolved_device = resolve_device(config.device)
    run_dir = tmp_path / "run"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    write_run_artifacts(run_dir, config, result, resolved_device, start, end, status="completed")

    written = generate_plots(run_dir, run_dir / "plots")
    names = {p.name for p in written}
    assert "finish_probability_by_objective_state.png" in names


def test_finish_plots_skip_cleanly_for_a_run_written_before_the_columns_existed(tmp_path):
    """marla summarize must still work on an old run directory that
    predates this task's new rollouts.csv/decisions.csv columns -- one
    absent column must not break any other plot."""
    from marla.metrics.plots import generate_plots

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    old_rollout_fields = [f for f in _ROLLOUTS_FIELDS if "finish" not in f]
    with (run_dir / "rollouts.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=old_rollout_fields)
        writer.writeheader()
        writer.writerow(
            {field: "" for field in old_rollout_fields}
            | {"run_id": "x", "rollout": 1, "environment_steps_total": 100, "collection_seconds": 1.0, "optimization_seconds": 1.0}
        )

    old_decision_fields = [f for f in _DECISIONS_FIELDS if "finish" not in f]
    with (run_dir / "decisions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=old_decision_fields)
        writer.writeheader()
        writer.writerow(
            {field: "" for field in old_decision_fields}
            | {"run_id": "x", "environment_step": 0, "critic_value": 0.0, "gae_advantage": 0.0, "return_target": 0.0}
        )

    written = generate_plots(run_dir, run_dir / "plots")
    names = {p.name for p in written}
    assert "finish_probability_by_objective_state.png" not in names
    assert "finish_probability_by_remaining_targets.png" not in names
    # rollouts.csv/decisions.csv still parse and drive every other plot
    # that doesn't need the new columns -- one absent column must not
    # break the whole file (empty here since most fields are blank
    # strings, but the call itself must not raise).
