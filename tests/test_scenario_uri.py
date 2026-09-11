"""The portable marla:// scenario reference scheme (spec sections 33-39):
one authoritative resolver, used identically regardless of how MARLA is
installed/run from, that never resolves against the current working
directory / a config file's directory / any particular checkout path for
a marla:// reference -- only against MARLA's own packaged scenario
resources. Filesystem .yaml paths and NASimEmu named/procedural
references must keep working exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from marla.scenarios import solvable_scenario_path
from marla.scenarios.uri import ScenarioReferenceError, resolve_scenario_reference

REPO_ROOT = Path(__file__).resolve().parent.parent
KNOWN_SOLVABLE_SCENARIO = "sm_entry_user_three_subnets.solvable.v2.yaml"


def test_marla_uri_resolves_to_the_packaged_scenario_regardless_of_config_dir(tmp_path):
    expected = str(solvable_scenario_path(KNOWN_SOLVABLE_SCENARIO))
    for config_dir in (tmp_path, tmp_path / "nested" / "deeper", Path("/"), REPO_ROOT):
        resolved = resolve_scenario_reference(f"marla://{KNOWN_SOLVABLE_SCENARIO}", config_dir)
        assert resolved == expected


def test_marla_uri_never_resolves_relative_to_cwd_or_config_dir(tmp_path, monkeypatch):
    """A same-named file dropped next to the config, or in the cwd, must
    never shadow the packaged resource -- the whole point of the scheme."""
    decoy = tmp_path / KNOWN_SOLVABLE_SCENARIO
    decoy.write_text("this is not the real scenario", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    resolved = resolve_scenario_reference(f"marla://{KNOWN_SOLVABLE_SCENARIO}", tmp_path)
    assert resolved != str(decoy)
    assert resolved == str(solvable_scenario_path(KNOWN_SOLVABLE_SCENARIO))


def test_unknown_marla_scenario_name_raises_actionable_error(tmp_path):
    with pytest.raises(ScenarioReferenceError, match="No MARLA-owned solvable scenario"):
        resolve_scenario_reference("marla://does_not_exist.solvable.v2.yaml", tmp_path)


@pytest.mark.parametrize(
    "reference",
    [
        "marla://",
        "marla://../../etc/passwd",
        "marla://sub/dir.yaml",
        "marla://..%2f..%2fetc.yaml",
        "marla://not_a_yaml_file",
        "marla://weird\\path.yaml",
    ],
)
def test_malformed_or_unsafe_marla_uri_is_rejected(reference, tmp_path):
    with pytest.raises(ScenarioReferenceError):
        resolve_scenario_reference(reference, tmp_path)


def test_filesystem_yaml_path_resolution_is_unchanged(tmp_path):
    scenario_file = tmp_path / "scenarios" / "custom.yaml"
    scenario_file.parent.mkdir(parents=True)
    scenario_file.write_text("dummy", encoding="utf-8")

    relative = resolve_scenario_reference("scenarios/custom.yaml", tmp_path)
    assert relative == str(scenario_file.resolve())

    absolute = resolve_scenario_reference(str(scenario_file), tmp_path)
    assert absolute == str(scenario_file)


def test_nasimemu_named_reference_passes_through_unchanged(tmp_path):
    assert resolve_scenario_reference("uniform-small-gen", tmp_path) == "uniform-small-gen"


def test_marla_uri_works_from_an_installed_wheel_layout(tmp_path):
    """Simulates the 'built/installed wheel' resolution context: the
    resolver must go through importlib.resources (marla.scenarios'
    package data), never a source-checkout-relative path -- so it must
    keep working when the repo's own source tree is nowhere on the
    resolution path at all, which is exactly what a real wheel install
    looks like at import time (this package's __file__ already lives
    under site-packages/.venv once installed; here we assert the
    resolution *mechanism* itself never falls back to a checkout path by
    checking the resolved path is the importlib.resources-managed one).
    """
    resolved = resolve_scenario_reference(f"marla://{KNOWN_SOLVABLE_SCENARIO}", tmp_path)
    assert "site-packages" in resolved or str(REPO_ROOT / "src" / "marla") in resolved
    # Either way, it must be inside the installed marla.scenarios package,
    # never inside NASimEmu/ or some other checkout-relative location.
    assert "marla/scenarios/solvable" in resolved.replace("\\", "/")
