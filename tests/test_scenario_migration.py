"""Migration to the repaired scenario (spec sections 16-18): every example,
``marla init``-generated config, and future AAMAS config must resolve to a
scenario that passes MARLA's universal solvability preflight.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from marla.config.loader import load_config
from marla.runtime.local import resolve_scenario_path
from marla.scenario.preflight import preflight_check
from marla.scenario.models import SolvabilityStatus

REPO_ROOT = Path(__file__).resolve().parent.parent


def _assert_preflight_passes(config_path: Path):
    config = load_config(config_path)
    resolved_scenario = resolve_scenario_path(config, config_path.parent)
    result = preflight_check(config, config_path.parent, resolved_scenario)
    assert result.status == SolvabilityStatus.PROVEN_SOLVABLE, (
        f"{config_path} resolves to a scenario that fails preflight: {result.to_dict()}"
    )
    return result


# --- Section 16: examples/ -------------------------------------------------


def test_example_baseline_config_passes_preflight():
    _assert_preflight_passes(REPO_ROOT / "examples" / "baseline.yaml")


def test_example_assisted_config_passes_preflight():
    _assert_preflight_passes(REPO_ROOT / "examples" / "assisted.yaml")


def test_examples_use_the_marla_owned_solvable_scenario():
    for name in ("baseline.yaml", "assisted.yaml"):
        data = yaml.safe_load((REPO_ROOT / "examples" / name).read_text(encoding="utf-8"))
        scenario = data["environment"]["scenario"]
        assert "scenarios/solvable/sm_entry_user_three_subnets.solvable.v2.yaml" in scenario
        assert scenario != "../NASimEmu/scenarios/sm_entry_user_three_subnets.v2.yaml"


# --- Section 17: `marla init` ----------------------------------------------


def test_marla_init_generated_configs_pass_preflight(tmp_path):
    from marla.config.templates import render_templates
    from marla.scenarios import solvable_scenario_path

    scenario_value = str(solvable_scenario_path("sm_entry_user_three_subnets.solvable.v2.yaml"))
    for name, content in render_templates(scenario_value).items():
        config_path = tmp_path / name
        config_path.write_text(content, encoding="utf-8")
        _assert_preflight_passes(config_path)


def test_marla_init_cli_generated_configs_pass_preflight(tmp_path):
    """End-to-end through the actual CLI command, not just the template
    rendering helper -- proves `marla init`'s own scenario-resolution
    logic (not a hand-rolled equivalent) produces a working config.
    """
    from typer.testing import CliRunner

    from marla.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["init", str(tmp_path / "generated")])
    assert result.exit_code == 0, result.output

    _assert_preflight_passes(tmp_path / "generated" / "baseline.yaml")
    _assert_preflight_passes(tmp_path / "generated" / "assisted.yaml")


# --- Section 18: future AAMAS configs --------------------------------------


def test_future_aamas_configs_use_the_solvable_scenario_and_pass_preflight():
    """PPO_ONLY and MARLA_FULL future configs (spec: "Both experiment
    variants must use exactly the same scenario") -- checked-in
    research/aamas2027/configs/*.yaml, which describe *future,
    not-yet-executed* runs (see research/aamas2027/manifest.yaml's own
    provenance notes for which runs have actually been executed under the
    old scenario -- those are handled separately as pilot/invalid, not
    silently rewritten).
    """
    configs_dir = REPO_ROOT / "research" / "aamas2027" / "configs"
    config_paths = sorted(configs_dir.glob("*.yaml"))
    assert config_paths, "no AAMAS configs found"

    resolved_scenarios = set()
    for config_path in config_paths:
        result = _assert_preflight_passes(config_path)
        resolved_scenarios.add(result.scenario_path)

    assert len(resolved_scenarios) == 1, (
        f"PPO_ONLY and MARLA_FULL configs must use exactly the same scenario: {resolved_scenarios}"
    )


def test_future_aamas_configs_reference_marla_owned_scenario_file():
    configs_dir = REPO_ROOT / "research" / "aamas2027" / "configs"
    for config_path in sorted(configs_dir.glob("*.yaml")):
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        scenario = data["environment"]["scenario"]
        assert "scenarios/solvable/sm_entry_user_three_subnets.solvable.v2.yaml" in scenario, (
            f"{config_path} still references a non-repaired scenario: {scenario}"
        )
