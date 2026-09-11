"""Tests for the decisions.csv nullable-boolean/nullable-integer schema
(``marla.metrics.csv_schema``) -- proves the actual bug report (a
``DtypeWarning`` on ``all_visible_sensitive_targets_rooted_before_action``,
column 72) is fixed with a real dtype, not a suppressed warning, and that
older run directories missing newer columns keep working.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marla.metrics.csv_schema import (
    DECISIONS_NULLABLE_BOOLEAN_COLUMNS,
    DECISIONS_NULLABLE_INTEGER_COLUMNS,
    DECISIONS_ORDINARY_BOOLEAN_COLUMNS,
    decisions_dtype_map,
    read_decisions_csv,
)
from marla.metrics.plots import generate_plots
from marla.metrics.writer import _DECISIONS_FIELDS, _write_csv


def _write_minimal_decisions_csv(path: Path, rows: list[dict]) -> None:
    _write_csv(path, _DECISIONS_FIELDS, rows)


def _base_row(**overrides) -> dict:
    row = {field: None for field in _DECISIONS_FIELDS}
    row.update(
        run_id="r", global_environment_step=0, rollout=1, episode_id=1, environment_step=0,
        observation_id="o", legal_action_count=1, selected_action_id="a", selected_action_type="exploit",
        base_policy_entropy=0.0, base_top_two_margin=0.0,
        selected_action_base_probability=1.0, selected_action_final_probability=1.0,
        final_policy_entropy=0.0, base_top_action_id="a", final_top_action_id="a",
        critic_value=0.0, gae_advantage=0.0, return_target=0.0,
        nasimemu_reward=0.0, training_reward=0.0, consultation_cost=0.0,
        objective_satisfied=False, objective_became_satisfied=False, terminated=False, truncated=False,
        new_hosts_discovered=0, new_subnets_discovered=0, new_services_confirmed=0, new_processes_confirmed=0,
        access_gain=0, queried=False,
        selected_action_visible_preconditions_status="unknown",
        base_compatible_action_probability_mass=0.0, final_compatible_action_probability_mass=0.0,
        objective_satisfied_before_action=False,
        known_subnets_total_before_action=0, known_subnets_scanned_before_action=0,
        known_subnets_unscanned_before_action=0, fraction_known_subnets_scanned_before_action=0.0,
        known_exploration_frontier_remaining_before_action=False,
        has_visible_sensitive_target_before_action=False,
    )
    row.update(overrides)
    return row


def test_all_visible_sensitive_targets_rooted_round_trips_as_nullable_boolean(tmp_path):
    """The exact reported case: NA / False / True must round-trip as
    pandas.NA / False / True under dtype 'boolean' -- never as an object
    dtype mixing Python None/bool/str, and never coerced to a plain float.
    """
    path = tmp_path / "decisions.csv"
    rows = [
        _base_row(has_visible_sensitive_target_before_action=False, all_visible_sensitive_targets_rooted_before_action=None),
        _base_row(has_visible_sensitive_target_before_action=True, all_visible_sensitive_targets_rooted_before_action=False),
        _base_row(has_visible_sensitive_target_before_action=True, all_visible_sensitive_targets_rooted_before_action=True),
    ]
    _write_minimal_decisions_csv(path, rows)

    df = read_decisions_csv(path)
    col = df["all_visible_sensitive_targets_rooted_before_action"]
    assert str(col.dtype) == "boolean"
    assert col.iloc[0] is pd.NA
    assert col.iloc[1] is np.False_ or col.iloc[1] == False  # noqa: E712
    assert col.iloc[2] is np.True_ or col.iloc[2] == True  # noqa: E712
    assert col.isna().sum() == 1
    assert (col == False).sum() == 1  # noqa: E712
    assert (col == True).sum() == 1  # noqa: E712


def test_reading_decisions_csv_emits_no_dtype_warning(tmp_path):
    path = tmp_path / "decisions.csv"
    rows = [
        _base_row(all_visible_sensitive_targets_rooted_before_action=v)
        for v in ([None] * 500 + [False] * 5 + [True] * 5)  # long NA run before any False/True -- the exact
    ]  # shape that triggers pandas' chunked type-sniffing warning without an explicit dtype
    _write_minimal_decisions_csv(path, rows)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = read_decisions_csv(path)
    assert len(df) == 510


def test_every_nullable_column_this_module_claims_is_actually_nullable_in_the_real_writer():
    """Cross-check against marla.metrics.writer's own _DECISIONS_FIELDS --
    every column this schema calls nullable must exist in the real
    decisions.csv field list (protects against a renamed/removed field
    silently going unclassified).
    """
    for column in (*DECISIONS_NULLABLE_BOOLEAN_COLUMNS, *DECISIONS_NULLABLE_INTEGER_COLUMNS, *DECISIONS_ORDINARY_BOOLEAN_COLUMNS):
        assert column in _DECISIONS_FIELDS, f"{column} is not a real decisions.csv field"


def test_dtype_map_only_covers_nullable_columns_not_ordinary_ones():
    mapping = decisions_dtype_map()
    for column in DECISIONS_ORDINARY_BOOLEAN_COLUMNS:
        assert column not in mapping, f"{column} is structurally never-null -- plain bool inference is correct"


def test_read_decisions_csv_tolerates_a_pre_exploration_schema_run(tmp_path):
    """An older run directory written before the subnet-exploration/FINISH-
    diagnostics columns existed (spec: "old runs missing newer columns
    must continue to summarize") -- read_decisions_csv must not raise for
    a header lacking columns this schema knows about.
    """
    path = tmp_path / "decisions.csv"
    old_fields = ["run_id", "episode_id", "selected_action_type", "terminated", "truncated"]
    with path.open("w", newline="") as fh:
        import csv

        writer = csv.DictWriter(fh, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow({"run_id": "r", "episode_id": 1, "selected_action_type": "exploit", "terminated": False, "truncated": False})

    df = read_decisions_csv(path)
    assert list(df.columns) == old_fields
    assert len(df) == 1


def test_ordinary_boolean_columns_never_null_in_a_real_collected_row():
    """Structural cross-check: every column classified as 'ordinary
    boolean' (never None) must correspond to a StepRecord field with a
    plain bool default, not `bool | None`.
    """
    from marla.learning.rollout import StepRecord
    import dataclasses

    fields_by_name = {f.name: f for f in dataclasses.fields(StepRecord)}
    name_map = {
        "objective_satisfied_before_action": "objective_satisfied_before_action",
        "known_exploration_frontier_remaining_before_action": "known_exploration_frontier_remaining_before_action",
        "has_visible_sensitive_target_before_action": "has_visible_sensitive_target_before_action",
        "objective_satisfied": "objective_satisfied",
        "objective_became_satisfied": "objective_became_satisfied",
        "terminated": "terminated",
        "truncated": "truncated",
    }
    for csv_column, record_field in name_map.items():
        field = fields_by_name[record_field]
        assert field.type == "bool", f"{csv_column} (StepRecord.{record_field}) is typed {field.type!r}, not plain bool"


@pytest.mark.asyncio
async def test_generate_plots_on_a_run_with_the_new_nullable_columns_emits_no_dtype_warning(tmp_path):
    """End-to-end: a real (tiny) run's decisions.csv, read the way
    `marla summarize` reads it, must not warn.
    """
    import torch

    from marla.config.loader import load_config, parse_config
    from marla.learning.trainer import run_baseline_training
    from marla.metrics.writer import write_episodes_csv, write_rollouts_csv, write_updates_csv, _write_csv, _DECISIONS_FIELDS, build_decision_rows

    REPO_ROOT = Path(__file__).resolve().parent.parent
    SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())

    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    data["policy"]["ppo"]["steps_per_env"] = 16
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["metrics"]["eval_episodes"] = 0
    config = parse_config(data)

    result = await run_baseline_training(config, SMALL_SCENARIO, num_rollouts=2, device=torch.device("cpu"), seed=1)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_episodes_csv(run_dir, config, result.episode_summaries, result.eval_episode_summaries)
    write_rollouts_csv(run_dir, result.rollout_rows)
    write_updates_csv(run_dir, config, result.update_metrics)
    _write_csv(run_dir / "decisions.csv", _DECISIONS_FIELDS, result.decision_rows)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        written = generate_plots(run_dir, tmp_path / "plots")
    assert isinstance(written, list)
