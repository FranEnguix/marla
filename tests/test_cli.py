from pathlib import Path

import pytest
from typer.testing import CliRunner

from marla.cli import app
from marla.config.loader import load_config

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "validate" in result.output
    assert "summarize" in result.output
    assert "version" in result.output


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "marla 0.2.0" in result.output


def test_validate_baseline_ok(baseline_config_path):
    result = runner.invoke(app, ["validate", str(baseline_config_path)])
    assert result.exit_code == 0
    assert "OK" in result.output


def test_validate_assisted_ok(assisted_config_path):
    result = runner.invoke(app, ["validate", str(assisted_config_path)])
    assert result.exit_code == 0


def test_validate_missing_file_fails(tmp_path):
    result = runner.invoke(app, ["validate", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 1


def test_run_assisted_local_fails_cleanly_without_password_env(assisted_config_path, monkeypatch):
    # Assisted local execution is real as of Milestones 6-9 (Gatekeeper +
    # Plan Maker agents); a real end-to-end run is covered by
    # subprocess-isolated integration tests instead, since SPADE's Container
    # is a process-wide singleton (see runtime/local.py) and the example
    # config's model would trigger a real (large) download. Here we only
    # check that a missing password_env fails fast and cleanly -- before any
    # heavy model loading -- rather than leaking an unhandled exception.
    #
    # examples/assisted.yaml's bundled scenario is not (yet) universally
    # solvable (see marla.scenario.solvability's analysis of
    # sm_entry_user_three_subnets.v2.yaml -- a real, separate finding this
    # test is not about), so the new preflight check would otherwise be the
    # first thing to fail here instead of the password check this test
    # exists to cover -- stub it out to keep this test focused.
    from marla.scenario.models import ScenarioSolvabilityResult, SolvabilityStatus

    monkeypatch.setattr(
        "marla.cli.preflight_check",
        lambda config, config_dir, scenario_path: ScenarioSolvabilityResult(
            status=SolvabilityStatus.PROVEN_SOLVABLE,
            universally_solvable=True,
            scenario_path=scenario_path,
            scenario_format="v2",
            objective="capture_target",
            randomized=True,
        ),
    )
    for var in ("MARLA_RL_ORCHESTRATOR_PASSWORD", "MARLA_GATEKEEPER_PASSWORD", "MARLA_PLAN_MAKER_1_PASSWORD"):
        monkeypatch.delenv(var, raising=False)

    result = runner.invoke(app, ["run", str(assisted_config_path)])
    assert result.exit_code == 1
    assert "Run failed" in result.output
    assert "mode=local" in result.output


def test_run_local_rejects_agent_option(baseline_config_path):
    result = runner.invoke(app, ["run", str(baseline_config_path), "--agent", "rl_orchestrator"])
    assert result.exit_code == 1
    assert "does not accept --agent" in result.output


def test_run_distributed_requires_agent(tmp_path, assisted_config_path):
    import yaml

    data = yaml.safe_load(assisted_config_path.read_text())
    data["execution"]["mode"] = "distributed"
    data["environment"]["scenario"] = str(
        (assisted_config_path.parent / data["environment"]["scenario"]).resolve()
    )
    config_path = tmp_path / "distributed.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = runner.invoke(app, ["run", str(config_path)])
    assert result.exit_code == 1
    assert "requires at least one --agent" in result.output


def test_resolve_agent_selectors_accepts_known_aliases(assisted_config_path):
    # Distributed mode is real as of Milestone 10: a full end-to-end run is
    # covered by subprocess-isolated tests instead (SPADE's Container is a
    # process-wide singleton -- its event loop closes after the first
    # spade.run() call in a process, so at most one CliRunner test per
    # process may reach real execution; --agent resolution is plain
    # string-matching logic and is tested directly here instead).
    from marla.cli import _resolve_agent_selectors

    config = load_config(assisted_config_path)
    resolved = _resolve_agent_selectors(config, ["rl_orchestrator", "gatekeeper"])
    assert resolved == ["rl_orchestrator", "gatekeeper"]


def test_resolve_agent_selectors_accepts_jids(assisted_config_path):
    from marla.cli import _resolve_agent_selectors

    config = load_config(assisted_config_path)
    resolved = _resolve_agent_selectors(config, [config.rl_orchestrator.jid])
    assert resolved == ["rl_orchestrator"]


def test_resolve_agent_selectors_rejects_duplicates(assisted_config_path):
    import typer

    from marla.cli import _resolve_agent_selectors

    config = load_config(assisted_config_path)
    with pytest.raises(typer.BadParameter):
        _resolve_agent_selectors(config, ["rl_orchestrator", "rl_orchestrator"])


def test_run_distributed_unknown_agent_selector_fails(tmp_path, assisted_config_path):
    import yaml

    data = yaml.safe_load(assisted_config_path.read_text())
    data["execution"]["mode"] = "distributed"
    data["environment"]["scenario"] = str(
        (assisted_config_path.parent / data["environment"]["scenario"]).resolve()
    )
    config_path = tmp_path / "distributed.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = runner.invoke(app, ["run", str(config_path), "--agent", "not_a_real_agent"])
    assert result.exit_code != 0


def test_summarize_requires_existing_directory(tmp_path):
    result = runner.invoke(app, ["summarize", str(tmp_path / "does-not-exist")])
    assert result.exit_code == 1


def test_summarize_requires_summary_json(tmp_path):
    empty_dir = tmp_path / "not-a-run-dir"
    empty_dir.mkdir()
    result = runner.invoke(app, ["summarize", str(empty_dir)])
    assert result.exit_code == 1
    assert "summary.json" in result.output


def _write_real_run(tmp_path):
    """A real tiny baseline TrainingResult, written via write_run_artifacts."""
    import asyncio
    from datetime import datetime, timezone

    from marla.config.loader import load_config, parse_config
    from marla.learning.trainer import run_baseline_training
    from marla.metrics.writer import write_run_artifacts
    from marla.runtime.device import resolve_device

    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    data["policy"]["ppo"]["rollout_steps"] = 12
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["device"] = "cpu"
    config = parse_config(data)
    scenario = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())

    result = asyncio.run(run_baseline_training(config, scenario, num_rollouts=2, seed=1))
    run_dir = tmp_path / "run"
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    write_run_artifacts(run_dir, config, result, resolve_device(config.device), now, now, status="completed")
    return run_dir


@pytest.mark.integration
def test_summarize_prints_stats_and_generates_plots(tmp_path):
    run_dir = _write_real_run(tmp_path)

    result = runner.invoke(app, ["summarize", str(run_dir)])

    assert result.exit_code == 0
    assert "episodes:" in result.output
    # "goal success rate" was renamed to "successful finish rate", reported
    # alongside the new, distinct "objective reached rate" (spec sections
    # 20/25: objective_reached must not be conflated with successful_finish).
    assert "objective reached rate:" in result.output
    assert "successful finish rate:" in result.output
    plots_dir = run_dir / "plots"
    assert plots_dir.is_dir()
    written_files = {p.name for p in plots_dir.glob("*.png")}
    assert "episode_returns.png" in written_files
    assert "ppo_losses.png" in written_files
    assert "episode_efficiency.png" in written_files
    assert "training_dynamics.png" in written_files
    assert "gradient_and_clipping.png" in written_files
    assert "reward_over_training.png" in written_files
    assert "episode_outcomes.png" in written_files
    # Baseline: no consultations, so consultation/query/advice plots are skipped.
    assert "consultation_activity.png" not in written_files
    assert "query_behavior.png" not in written_files
    assert "advice_influence.png" not in written_files
    assert "gatekeeper_reliability.png" not in written_files
    assert "plan_maker_latency.png" not in written_files
    assert "query_decision_analysis.png" not in written_files


def test_init_creates_valid_baseline_and_assisted_templates(tmp_path, monkeypatch):
    # marla init always resolves MARLA's own packaged, pre-validated
    # solvable scenario (marla.scenarios.solvable_scenario_path) -- it
    # genuinely exists on disk regardless of CWD or any nearby NASimEmu/
    # checkout, so this exercises full load_config (schema + on-disk
    # scenario check), not just parse_config.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "could not find NASimEmu" not in result.output

    target = tmp_path / "experiment_templates"
    baseline_path = target / "baseline.yaml"
    assisted_path = target / "assisted.yaml"
    assert baseline_path.exists()
    assert assisted_path.exists()

    baseline_config = load_config(baseline_path)
    assert baseline_config.execution.mode == "local"
    assert baseline_config.consultation.mode == "disabled"
    assert baseline_config.policy.recurrent.sequence_length == 64
    assert baseline_config.policy.ppo.optimizer.eps == pytest.approx(1.0e-5)
    assert baseline_config.policy.ppo.learning_rate_schedule == "linear"
    assert "sm_entry_user_three_subnets.solvable.v2.yaml" in baseline_config.environment.scenario
    assert "scenarios/solvable/" in baseline_config.environment.scenario

    assisted_config = load_config(assisted_path)
    assert assisted_config.consultation.mode == "learned"
    assert assisted_config.policy.recurrent.sequence_length == 64
    assert assisted_config.gatekeeper is not None
    assert len(assisted_config.agents) == 1


def test_init_always_uses_marla_owned_scenario_regardless_of_a_nearby_nasimemu_checkout(tmp_path, monkeypatch):
    """marla init must resolve the MARLA-owned, pre-validated scenario
    (marla.scenarios.solvable_scenario_path) even from a directory that
    happens to have its own (irrelevant, and in this fixture also
    unsolvable-shaped) NASimEmu/scenarios/ checkout nearby -- unlike the
    old filesystem-search behavior, which would have found and used that
    local (and, for the real bundled scenario, NOT universally solvable)
    file instead. See marla.scenarios' own docstring for why this
    guarantee only holds for a MARLA-owned scenario, never a filesystem
    search.
    """
    from marla.scenarios import solvable_scenario_path

    nasim_scenario = tmp_path / "NASimEmu" / "scenarios" / "sm_entry_user_three_subnets.v2.yaml"
    nasim_scenario.parent.mkdir(parents=True)
    nasim_scenario.write_text("placeholder", encoding="utf-8")

    workdir = tmp_path / "some" / "nested" / "cwd"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "could not find NASimEmu" not in result.output

    expected_scenario = solvable_scenario_path("sm_entry_user_three_subnets.solvable.v2.yaml")
    baseline_path = workdir / "experiment_templates" / "baseline.yaml"
    content = baseline_path.read_text(encoding="utf-8")
    assert f"scenario: {expected_scenario}" in content
    assert str(nasim_scenario) not in content

    config = load_config(baseline_path)
    assert str(expected_scenario) == config.environment.scenario


def test_init_refuses_to_overwrite_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init"]).exit_code == 0

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 1
    assert "baseline.yaml" in result.output
    assert "assisted.yaml" in result.output


def test_init_force_overwrites_existing_templates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init"]).exit_code == 0

    result = runner.invoke(app, ["init", "--force"])
    assert result.exit_code == 0


def test_init_accepts_custom_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "my_configs"])
    assert result.exit_code == 0
    assert (tmp_path / "my_configs" / "baseline.yaml").exists()
    assert (tmp_path / "my_configs" / "assisted.yaml").exists()
