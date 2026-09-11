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
        # Named via the portable marla:// scheme (marla.scenarios.uri), not
        # a machine-specific filesystem path -- see that module's docstring
        # for why this must never be a materialized path in a checked-in
        # config.
        assert scenario == "marla://sm_entry_user_three_subnets.solvable.v2.yaml"
        assert scenario != "../NASimEmu/scenarios/sm_entry_user_three_subnets.v2.yaml"


# --- Section 17: `marla init` ----------------------------------------------


def test_marla_init_generated_configs_pass_preflight(tmp_path):
    from marla.config.templates import render_templates

    scenario_value = "marla://sm_entry_user_three_subnets.solvable.v2.yaml"
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


# Section 18 ("future AAMAS configs") intentionally removed from the
# published platform test suite: those two tests globbed
# research/aamas2027/configs/*.yaml -- private AAMAS research content, not
# part of the installable marla package -- so they would fail (or find zero
# files) in a clean clone/install without a local research/ checkout. The
# platform's own test suite must never depend on research/ (see the
# platform-publication audit); research/aamas2027's own configs are
# exercised by the research project's own local workflow instead.
