"""SPADE's Container is a process-wide singleton (see runtime/local.py), so
every test here that calls run_local()/`marla run` spawns its own subprocess
-- calling it twice in the same interpreter reuses a closed event loop and
fails. This is a real constraint of the library, not a workaround for a bug.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


def _write_tiny_baseline_config(tmp_path: Path, run_id: str) -> Path:
    data = yaml.safe_load((REPO_ROOT / "examples" / "baseline.yaml").read_text())
    data["experiment"]["run_id"] = run_id
    data["environment"]["scenario"] = SMALL_SCENARIO
    # Small enough that an untrained random policy is guaranteed to hit at
    # least one episode boundary (termination or truncation) within the
    # 12-step rollout window collected below.
    data["environment"]["max_episode_steps"] = 5
    data["policy"]["ppo"]["total_environment_steps"] = 12
    data["policy"]["ppo"]["steps_per_env"] = 12
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4

    config_path = tmp_path / "tiny_baseline.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


@pytest.mark.integration
def test_marla_run_baseline_local_completes_via_subprocess(tmp_path):
    config_path = _write_tiny_baseline_config(tmp_path, run_id="cli-baseline-1")

    env = {**os.environ, "MARLA_RL_ORCHESTRATOR_PASSWORD": "testpass"}
    result = subprocess.run(
        [sys.executable, "-m", "marla", "run", str(config_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Run complete" in result.stdout
    assert "12 environment steps" in result.stdout


@pytest.mark.integration
def test_marla_run_baseline_local_fails_without_password_env(tmp_path):
    config_path = _write_tiny_baseline_config(tmp_path, run_id="cli-baseline-2")

    env = {k: v for k, v in os.environ.items() if k != "MARLA_RL_ORCHESTRATOR_PASSWORD"}
    result = subprocess.run(
        [sys.executable, "-m", "marla", "run", str(config_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    assert "MARLA_RL_ORCHESTRATOR_PASSWORD" in result.stdout + result.stderr


@pytest.mark.integration
def test_run_local_python_api_returns_training_result(tmp_path):
    config_path = _write_tiny_baseline_config(tmp_path, run_id="api-baseline-1")

    script = textwrap.dedent(
        f"""
        from pathlib import Path
        from marla.config.loader import load_config
        from marla.runtime.local import run_local

        config = load_config({str(config_path)!r})
        orchestrator = run_local(config, Path({str(config_path.parent)!r}), num_rollouts=1)
        assert orchestrator.failure is None, orchestrator.failure
        assert orchestrator.training_result.environment_steps == 12
        assert len(orchestrator.training_result.episode_summaries) >= 1
        print("OK")
        """
    )
    env = {**os.environ, "MARLA_RL_ORCHESTRATOR_PASSWORD": "testpass"}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_resolve_debug_dir_creates_and_returns_a_top_level_debug_directory(tmp_path, monkeypatch):
    from marla.config.loader import load_config
    from marla.runtime.local import resolve_debug_dir

    monkeypatch.chdir(tmp_path)
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")

    debug_dir = resolve_debug_dir(config, "my-run-id")

    # Deliberately not nested under metrics.output_directory ("runs/") -- see
    # resolve_debug_dir's docstring.
    assert debug_dir == Path("debug") / "my-run-id"
    assert (tmp_path / debug_dir).is_dir()
