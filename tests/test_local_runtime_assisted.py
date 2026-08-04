"""marla run <assisted.yaml> end-to-end via runtime/local.py: real
RLOrchestratorAgent + real GatekeeperAgent + real PlanMakerAgent (tiny HF
model), fully wired the way the CLI actually constructs them. Complements
test_local_runtime.py's baseline coverage.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


def _write_tiny_assisted_config(tmp_path: Path, run_id: str) -> Path:
    data = yaml.safe_load((REPO_ROOT / "examples" / "assisted.yaml").read_text())
    data["experiment"]["run_id"] = run_id
    data["environment"]["scenario"] = SMALL_SCENARIO
    data["environment"]["max_episode_steps"] = 5
    data["policy"]["ppo"]["total_environment_steps"] = 8
    data["policy"]["ppo"]["rollout_steps"] = 8
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["consultation"]["cost"] = 0.1
    data["agents"][0]["model"]["name"] = "sshleifer/tiny-gpt2"  # fast, no download surprises
    # tiny-gpt2's 1024-token context window is smaller than a real prompt
    # (spec-realistic scenario data + knowledge rules); on CUDA, the
    # resulting IndexError poisons the whole process's CUDA context
    # (including the RL policy's own tensors) rather than staying scoped to
    # the one failed call, unlike the clean, isolated Python exception on
    # CPU. A production-sized model's much larger context window avoids
    # this in practice; forcing CPU here keeps the test about
    # runtime/local.py's wiring, not CUDA context-poisoning recovery.
    data["device"] = "cpu"

    config_path = tmp_path / "tiny_assisted_runtime.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


@pytest.mark.integration
def test_marla_run_assisted_local_completes_via_subprocess(tmp_path):
    import os

    config_path = _write_tiny_assisted_config(tmp_path, run_id="cli-assisted-1")
    env = {
        **os.environ,
        "MARLA_RL_ORCHESTRATOR_PASSWORD": "orchestrator-pass",
        "MARLA_GATEKEEPER_PASSWORD": "gatekeeper-pass",
        "MARLA_PLAN_MAKER_1_PASSWORD": "planmaker-pass",
    }

    # A known, pre-existing pyjabber presence-subscription flakiness (see
    # README / test_gatekeeper.py) can occasionally stall a run entirely;
    # retrying, with TimeoutExpired treated the same as a failed attempt, is
    # the same pattern used elsewhere for this documented infra issue.
    last_output = None
    for _ in range(6):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "marla", "run", str(config_path)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            last_output = f"TIMEOUT: {exc.stdout}\n{exc.stderr}"
            continue
        if result.returncode == 0 and "Run complete" in result.stdout:
            assert "8 environment steps" in result.stdout
            return
        last_output = result.stdout + result.stderr
    pytest.fail(f"marla run never completed successfully after retries:\n{last_output}")
