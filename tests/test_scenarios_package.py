"""Regression test for every MARLA-owned scenario under
``src/marla/scenarios/solvable/`` (spec section 12): this must keep
failing loudly if a future edit ever makes a committed ``.solvable``
scenario non-solvable again, or if it stops loading/parsing correctly.
"""

from __future__ import annotations

from pathlib import Path

import nasimemu.nasim as nasim
import pytest
import yaml

from marla.scenario.solvability import check_solvability
from marla.scenario.spec import load_scenario_spec
from marla.scenarios import solvable_scenario_path, solvable_scenarios_dir

COMMITTED_SOLVABLE_SCENARIOS = [
    p.name for p in sorted(solvable_scenarios_dir().glob("*.yaml")) if ".solvable" in p.name
]


def test_at_least_one_solvable_scenario_is_committed():
    assert COMMITTED_SOLVABLE_SCENARIOS, "no *.solvable*.yaml scenario found under src/marla/scenarios/solvable/"


@pytest.mark.parametrize("filename", COMMITTED_SOLVABLE_SCENARIOS)
def test_committed_scenario_filename_contains_solvable_marker(filename):
    assert ".solvable.v2.yaml" in filename or ".solvable.yaml" in filename


@pytest.mark.parametrize("filename", COMMITTED_SOLVABLE_SCENARIOS)
def test_committed_scenario_parses_as_yaml(filename):
    path = solvable_scenario_path(filename)
    yaml.safe_load(path.read_text(encoding="utf-8"))  # must not raise


@pytest.mark.parametrize("filename", COMMITTED_SOLVABLE_SCENARIOS)
def test_committed_scenario_loads_through_real_nasimemu(filename):
    path = solvable_scenario_path(filename)
    scenario = nasim.load_scenario(str(path))  # must not raise
    assert len(scenario.hosts) > 0


@pytest.mark.parametrize("filename", COMMITTED_SOLVABLE_SCENARIOS)
def test_committed_scenario_is_proven_universally_solvable(filename):
    """The core regression guard: if a future edit to this file (or to the
    checker itself) ever makes this scenario non-universally-solvable,
    this test must fail -- it re-runs the full adversarially-validated
    checker (see test_scenario_adversarial.py / test_scenario_differential.py),
    not a cached/hardcoded result.
    """
    path = solvable_scenario_path(filename)
    spec = load_scenario_spec(path)
    result = check_solvability(spec)

    assert result.status.value == "proven_solvable", (
        f"{filename} is no longer PROVEN_SOLVABLE: {result.to_dict()}"
    )
    assert result.universally_solvable is True
    assert not result.host_rootability_failures
    assert not result.network_reachability_failures


def test_solvable_scenario_path_resolves_correctly():
    path = solvable_scenario_path("sm_entry_user_three_subnets.solvable.v2.yaml")
    assert path.is_file()
    assert path.name == "sm_entry_user_three_subnets.solvable.v2.yaml"


def test_solvable_scenario_path_raises_for_unknown_name():
    with pytest.raises(FileNotFoundError):
        solvable_scenario_path("does_not_exist.solvable.v2.yaml")


def test_original_nasimemu_scenario_is_untouched_and_remains_unsolvable():
    """Provenance check (spec section 2): the original scenario is
    deliberately left in place and unmodified -- it remains evidence of
    *why* the repair exists. If this ever starts passing on its own, the
    repaired copy may no longer be needed (or something changed upstream)
    and this should be investigated, not silently ignored.
    """
    repo_root = Path(__file__).resolve().parent.parent
    original = repo_root / "NASimEmu" / "scenarios" / "sm_entry_user_three_subnets.v2.yaml"
    assert original.is_file()

    spec = load_scenario_spec(original)
    result = check_solvability(spec)
    assert result.status.value == "proven_unsolvable"


def test_ood_scenario_path_resolves_correctly():
    path = solvable_scenario_path("md_entry_user_three_subnets.solvable.v2.yaml")
    assert path.is_file()
    assert path.name == "md_entry_user_three_subnets.solvable.v2.yaml"


def test_original_ood_nasimemu_scenario_is_untouched_and_remains_unsolvable():
    """Same provenance check as above, for the medium-network OOD scenario
    used for generalization evaluation (never for training)."""
    repo_root = Path(__file__).resolve().parent.parent
    original = repo_root / "NASimEmu" / "scenarios" / "md_entry_user_three_subnets.v2.yaml"
    assert original.is_file()

    spec = load_scenario_spec(original)
    result = check_solvability(spec)
    assert result.status.value == "proven_unsolvable"
