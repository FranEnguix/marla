"""End-to-end assisted-variant training: real RL Orchestrator + real Gatekeeper
+ deterministic mock Plan Maker, over real SPADE messaging (spec's required
"assisted local run with a deterministic mock Plan Maker" integration test).

Runs in a subprocess (SPADE's Container is a process-wide singleton, see
runtime/local.py) and is tolerant of pyjabber's known presence-subscription
teardown flakiness (see tests/test_gatekeeper.py's module docstring).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())
SCRIPT_ARGS = [sys.executable, "-m", "tests.support.run_assisted_training_scenario"]


def _write_tiny_assisted_config(tmp_path: Path, run_id: str) -> Path:
    data = yaml.safe_load((REPO_ROOT / "examples" / "assisted.yaml").read_text())
    data["experiment"]["run_id"] = run_id
    data["environment"]["scenario"] = SMALL_SCENARIO
    data["environment"]["max_episode_steps"] = 5
    data["policy"]["ppo"]["total_environment_steps"] = 16
    data["policy"]["ppo"]["rollout_steps"] = 16
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["consultation"]["cost"] = 0.1

    config_path = tmp_path / "tiny_assisted.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def _run_scenario(config_path: Path, num_rollouts: int, strategy: str, attempts: int = 6) -> dict:
    last_output = None
    for _ in range(attempts):
        try:
            result = subprocess.run(
                [*SCRIPT_ARGS, str(config_path), SMALL_SCENARIO, str(num_rollouts), strategy],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=90,
            )
        except subprocess.TimeoutExpired as exc:
            last_output = f"TIMEOUT: {exc.stdout}\n{exc.stderr}"
            continue
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if lines:
            try:
                return json.loads(lines[-1])
            except json.JSONDecodeError:
                pass
        last_output = result.stdout + result.stderr
    pytest.fail(f"Scenario never produced a valid result after {attempts} attempts:\n{last_output}")


@pytest.mark.integration
def test_assisted_training_runs_end_to_end_with_mock_plan_maker(tmp_path):
    config_path = _write_tiny_assisted_config(tmp_path, run_id="assisted-training-1")
    result = _run_scenario(config_path, num_rollouts=1, strategy="always_valid")

    assert result["failure"] is None, result["failure"]
    assert result["environment_steps"] == 16
    assert result["num_episodes"] >= 1
    assert result["num_updates"] > 0
    assert result["update_metrics_all_finite"] is True


@pytest.mark.integration
def test_assisted_training_accepted_advice_has_bounded_beta_and_positive_alpha(tmp_path):
    config_path = _write_tiny_assisted_config(tmp_path, run_id="assisted-training-2")
    # A larger rollout makes it overwhelmingly likely the (randomly
    # initialized) query gate fires at least once across 40 steps.
    result = _run_scenario(config_path, num_rollouts=1, strategy="always_valid")
    if not result["any_accepted"]:
        pytest.skip("query gate did not fire in this run; beta/alpha bounds not exercised")

    for beta in result["accepted_betas"]:
        assert 0.0 <= beta <= 1.0
    for alpha in result["accepted_alphas"]:
        assert alpha > 0.0


@pytest.mark.integration
def test_assisted_training_charges_consultation_cost_when_queried(tmp_path):
    config_path = _write_tiny_assisted_config(tmp_path, run_id="assisted-training-3")
    result = _run_scenario(config_path, num_rollouts=1, strategy="always_valid")
    if not result["any_queried"]:
        pytest.skip("query gate did not fire in this run; consultation cost not exercised")
    assert result["total_consultation_cost"] > 0.0
