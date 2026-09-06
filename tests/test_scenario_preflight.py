"""`marla run` preflight solvability gate (spec section 22 L/M): a
proven-unsolvable scenario must exit non-zero, explain the problem, print
the exact repair command, and -- crucially -- never reach agent startup.
``run_local``/``run_distributed`` are monkeypatched to prove they are never
called, rather than merely asserting on printed text.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from marla.cli import app

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parent.parent
UNSOLVABLE_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_user_three_subnets.v2.yaml").resolve())
SOLVABLE_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


def _write_config(tmp_path: Path, scenario: str, run_id: str) -> Path:
    data = yaml.safe_load((REPO_ROOT / "examples" / "baseline.yaml").read_text())
    data["experiment"]["run_id"] = run_id
    data["environment"]["scenario"] = scenario
    config_path = tmp_path / f"{run_id}.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def test_preflight_failure_never_starts_local_agents(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, UNSOLVABLE_SCENARIO, "preflight-fail-local")
    monkeypatch.setenv("MARLA_RL_ORCHESTRATOR_PASSWORD", "testpass")

    called = []
    monkeypatch.setattr("marla.runtime.local.run_local", lambda *a, **k: called.append(True))

    result = runner.invoke(app, ["run", str(config_path)])

    assert result.exit_code == 1
    assert called == []  # run_local (which starts every local agent) was never reached
    assert "Scenario solvability check FAILED" in result.output
    assert "No MARLA agents were started." in result.output
    assert f"marla scenario repair {UNSOLVABLE_SCENARIO}" in result.output
    assert "windows" in result.output


def test_preflight_failure_never_starts_distributed_agents(tmp_path, monkeypatch):
    data = yaml.safe_load((REPO_ROOT / "examples" / "assisted.yaml").read_text())
    data["experiment"]["run_id"] = "preflight-fail-distributed"
    data["execution"]["mode"] = "distributed"
    data["environment"]["scenario"] = UNSOLVABLE_SCENARIO
    config_path = tmp_path / "preflight-fail-distributed.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("MARLA_RL_ORCHESTRATOR_PASSWORD", "testpass")

    called = []
    monkeypatch.setattr("marla.runtime.distributed.run_distributed", lambda *a, **k: called.append(True))

    result = runner.invoke(app, ["run", str(config_path), "--agent", "rl_orchestrator"])

    assert result.exit_code == 1
    assert called == []
    assert "Scenario solvability check FAILED" in result.output
    assert "No MARLA agents were started." in result.output


def test_preflight_does_not_create_run_directory_on_failure(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, UNSOLVABLE_SCENARIO, "preflight-fail-no-rundir")
    monkeypatch.setenv("MARLA_RL_ORCHESTRATOR_PASSWORD", "testpass")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["run", str(config_path)])

    assert result.exit_code == 1
    assert not (tmp_path / "runs").exists()


def test_preflight_success_proceeds_to_agent_startup(tmp_path, monkeypatch):
    """A solvable scenario must pass preflight quietly and reach the normal
    startup path (spec 22.M) -- proven by observing run_local IS called,
    not by letting a real (slow) run complete.
    """
    config_path = _write_config(tmp_path, SOLVABLE_SCENARIO, "preflight-pass-local")
    monkeypatch.setenv("MARLA_RL_ORCHESTRATOR_PASSWORD", "testpass")

    called = []

    def _fake_run_local(*args, **kwargs):
        called.append(True)
        raise RuntimeError("stop here -- this test only checks preflight let us reach agent startup")

    monkeypatch.setattr("marla.runtime.local.run_local", _fake_run_local)
    # run_local raises a bare RuntimeError, not LocalRunError -- let it
    # propagate as an unhandled exception (CliRunner captures it) rather
    # than trying to match cli.py's specific error-handling branch.
    result = runner.invoke(app, ["run", str(config_path)])

    assert called == [True]
    assert "Scenario solvability: PASSED" in result.output
    assert "Validating scenario solvability..." in result.output


def test_preflight_output_is_quiet_on_success(tmp_path, monkeypatch):
    """Normal successful preflight prints only a couple of lines (spec
    section 25) -- not the full diagnostic report."""
    config_path = _write_config(tmp_path, SOLVABLE_SCENARIO, "preflight-quiet")
    monkeypatch.setenv("MARLA_RL_ORCHESTRATOR_PASSWORD", "testpass")
    monkeypatch.setattr("marla.runtime.local.run_local", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))

    result = runner.invoke(app, ["run", str(config_path)])

    assert "Scenario solvability: PASSED" in result.output
    assert "Failure class" not in result.output
    assert "Possible sensitive-host locations" not in result.output
