"""``marla scenario check``/``marla scenario repair`` CLI (spec section 22 J/K)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from marla.cli import app

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parent.parent
UNSOLVABLE_SCENARIO = REPO_ROOT / "NASimEmu" / "scenarios" / "sm_entry_user_three_subnets.v2.yaml"
SOLVABLE_SCENARIO = REPO_ROOT / "NASimEmu" / "scenarios" / "sm_entry_dmz_one_subnet.v2.yaml"


# --- J. `marla scenario check` ---------------------------------------------


def test_scenario_check_passes_on_solvable_scenario():
    result = runner.invoke(app, ["scenario", "check", str(SOLVABLE_SCENARIO)])
    assert result.exit_code == 0
    assert "PASSED" in result.output


def test_scenario_check_fails_on_unsolvable_scenario():
    result = runner.invoke(app, ["scenario", "check", str(UNSOLVABLE_SCENARIO)])
    assert result.exit_code == 1
    assert "FAILED" in result.output
    assert "windows" in result.output
    assert "marla scenario repair" in result.output


def test_scenario_check_missing_file_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["scenario", "check", str(tmp_path / "does_not_exist.v2.yaml")])
    assert result.exit_code == 1


def test_scenario_check_json_output_is_machine_readable():
    result = runner.invoke(app, ["scenario", "check", str(UNSOLVABLE_SCENARIO), "--json"])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "proven_unsolvable"
    assert data["universally_solvable"] is False
    assert len(data["failure_classes"]) == 1
    assert data["failure_classes"][0]["os"] == "windows"


def test_scenario_check_json_output_on_solvable_scenario():
    result = runner.invoke(app, ["scenario", "check", str(SOLVABLE_SCENARIO), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "proven_solvable"
    assert data["universally_solvable"] is True
    assert data["failure_classes"] == []


# --- K. `marla scenario repair` --------------------------------------------


def test_scenario_repair_creates_new_file_and_revalidates(tmp_path):
    src = tmp_path / "sm_entry_user_three_subnets.v2.yaml"
    shutil.copy(UNSOLVABLE_SCENARIO, src)

    result = runner.invoke(app, ["scenario", "repair", str(src)])

    assert result.exit_code == 0
    expected_output = tmp_path / "sm_entry_user_three_subnets.solvable.v2.yaml"
    assert expected_output.is_file()
    assert "PROVEN_SOLVABLE" in result.output
    # Original left untouched.
    assert src.read_text(encoding="utf-8") == UNSOLVABLE_SCENARIO.read_text(encoding="utf-8")

    # Independently re-check the generated file through the CLI too.
    check_result = runner.invoke(app, ["scenario", "check", str(expected_output)])
    assert check_result.exit_code == 0
    assert "PASSED" in check_result.output


def test_scenario_repair_refuses_to_overwrite_existing_output_by_default(tmp_path):
    src = tmp_path / "sm_entry_user_three_subnets.v2.yaml"
    shutil.copy(UNSOLVABLE_SCENARIO, src)
    output = tmp_path / "sm_entry_user_three_subnets.solvable.v2.yaml"
    output.write_text("sentinel", encoding="utf-8")

    result = runner.invoke(app, ["scenario", "repair", str(src)])

    assert result.exit_code == 1
    assert output.read_text(encoding="utf-8") == "sentinel"  # untouched


def test_scenario_repair_overwrite_flag_allows_replacing_output(tmp_path):
    src = tmp_path / "sm_entry_user_three_subnets.v2.yaml"
    shutil.copy(UNSOLVABLE_SCENARIO, src)
    output = tmp_path / "sm_entry_user_three_subnets.solvable.v2.yaml"
    output.write_text("sentinel", encoding="utf-8")

    result = runner.invoke(app, ["scenario", "repair", str(src), "--overwrite"])

    assert result.exit_code == 0
    assert "sentinel" not in output.read_text(encoding="utf-8")


def test_scenario_repair_reports_no_repair_needed_for_already_solvable_scenario():
    result = runner.invoke(app, ["scenario", "repair", str(SOLVABLE_SCENARIO)])
    assert result.exit_code == 0
    assert "no repair needed" in result.output.lower()


def test_scenario_repair_custom_output_path(tmp_path):
    src = tmp_path / "sm_entry_user_three_subnets.v2.yaml"
    shutil.copy(UNSOLVABLE_SCENARIO, src)
    custom_output = tmp_path / "custom_name.v2.yaml"

    result = runner.invoke(app, ["scenario", "repair", str(src), "--output", str(custom_output)])

    assert result.exit_code == 0
    assert custom_output.is_file()
