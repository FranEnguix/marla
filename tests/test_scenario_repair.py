"""Repair minimality and provenance (spec section 22 H/I)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent, indent

import pytest
import yaml

from marla.scenario.repair import RepairNotSupported, repair_scenario
from marla.scenario.solvability import check_solvability
from marla.scenario.spec import load_scenario_spec

REPO_ROOT = Path(__file__).resolve().parent.parent
UNSOLVABLE_SCENARIO = REPO_ROOT / "NASimEmu" / "scenarios" / "sm_entry_user_three_subnets.v2.yaml"


# --- H. Repair produces a NASimEmu-loadable, proven-solvable file ---------


def test_repair_output_loads_in_real_nasimemu_and_is_proven_solvable(tmp_path):
    import shutil

    import nasimemu.nasim as nasim

    src = tmp_path / "sm_entry_user_three_subnets.v2.yaml"
    shutil.copy(UNSOLVABLE_SCENARIO, src)

    outcome = repair_scenario(src)

    assert outcome.repaired is True
    # 1. loads in NASimEmu (would raise if malformed).
    nasim.load_scenario(str(outcome.output_path))
    # 2. proven universally solvable.
    assert outcome.after.status.value == "proven_solvable"
    spec = load_scenario_spec(outcome.output_path)
    assert check_solvability(spec).universally_solvable is True


def test_repair_preserves_unrelated_scenario_fields(tmp_path):
    import shutil

    src = tmp_path / "sm_entry_user_three_subnets.v2.yaml"
    shutil.copy(UNSOLVABLE_SCENARIO, src)

    outcome = repair_scenario(src)

    before = yaml.safe_load(UNSOLVABLE_SCENARIO.read_text(encoding="utf-8"))
    after = yaml.safe_load(outcome.output_path.read_text(encoding="utf-8"))

    for key in ("subnets", "topology", "sensitive_hosts", "os", "services", "sensitive_services",
                "exploits", "firewall", "host_configurations", "address_space_bounds"):
        assert after[key] == before[key], f"unrelated field {key!r} changed by repair"

    # Only privilege_escalation gained the one new artificial entry.
    assert set(after["privilege_escalation"]) - set(before["privilege_escalation"]) == {
        "marla_repair_windows_root_privesc"
    }


# --- I. Minimality: no unnecessary changes when one vulnerability suffices


def test_repair_does_not_touch_topology_probabilities_or_host_counts(tmp_path):
    import shutil

    src = tmp_path / "sm_entry_user_three_subnets.v2.yaml"
    shutil.copy(UNSOLVABLE_SCENARIO, src)

    outcome = repair_scenario(src)

    before = yaml.safe_load(UNSOLVABLE_SCENARIO.read_text(encoding="utf-8"))
    after = yaml.safe_load(outcome.output_path.read_text(encoding="utf-8"))

    assert after["subnets"] == before["subnets"]
    assert after["topology"] == before["topology"]
    assert after["sensitive_hosts"] == before["sensitive_hosts"]  # probabilities unchanged
    assert after["exploits"] == before["exploits"]  # no unrelated exploit touched
    assert len(outcome.plan.changes) == 1  # exactly one minimal change


def test_repair_reuses_existing_service_and_os_never_inventing_new_ones(tmp_path):
    import shutil

    src = tmp_path / "sm_entry_user_three_subnets.v2.yaml"
    shutil.copy(UNSOLVABLE_SCENARIO, src)

    outcome = repair_scenario(src)

    change = outcome.plan.changes[0]
    assert change.section == "privilege_escalation"
    assert change.definition["os"] == "windows"  # reuses the existing, failing OS
    assert change.definition["process"] is None  # unconstrained -- matches spec 14: never
    # invents a process requirement that couldn't occur.


def test_repair_change_names_are_clearly_artificial(tmp_path):
    import shutil

    src = tmp_path / "sm_entry_user_three_subnets.v2.yaml"
    shutil.copy(UNSOLVABLE_SCENARIO, src)

    outcome = repair_scenario(src)

    assert outcome.plan.changes[0].name.startswith("marla_repair_")


def test_repair_already_solvable_scenario_reports_no_repair_needed(tmp_path):
    solvable = REPO_ROOT / "NASimEmu" / "scenarios" / "sm_entry_dmz_one_subnet.v2.yaml"

    outcome = repair_scenario(solvable, output_path=tmp_path / "out.v2.yaml")

    assert outcome.repaired is False
    assert not (tmp_path / "out.v2.yaml").exists()
    assert "no repair needed" in outcome.message.lower()


def test_repair_raises_file_exists_error_without_overwrite(tmp_path):
    import shutil

    src = tmp_path / "sm_entry_user_three_subnets.v2.yaml"
    shutil.copy(UNSOLVABLE_SCENARIO, src)
    existing = tmp_path / "sm_entry_user_three_subnets.solvable.v2.yaml"
    existing.write_text("do not touch", encoding="utf-8")

    with pytest.raises(FileExistsError):
        repair_scenario(src)

    assert existing.read_text(encoding="utf-8") == "do not touch"
