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


def _write_tiny_assisted_config(tmp_path: Path, run_id: str, output_directory: Path | None = None) -> Path:
    data = yaml.safe_load((REPO_ROOT / "examples" / "assisted.yaml").read_text())
    data["experiment"]["run_id"] = run_id
    if output_directory is not None:
        data["metrics"]["output_directory"] = str(output_directory)
    data["environment"]["scenario"] = SMALL_SCENARIO
    data["environment"]["max_episode_steps"] = 5
    data["policy"]["ppo"]["total_environment_steps"] = 8
    data["policy"]["ppo"]["steps_per_env"] = 8
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


@pytest.mark.integration
def test_marla_run_assisted_local_produces_consistent_sent_handled_message_telemetry(tmp_path):
    """Real end-to-end smoke: a real RLOrchestratorAgent + GatekeeperAgent +
    PlanMakerAgent, real SPADE/XMPP message traffic, non-paper seed/scenario
    (the tiny CPU config shared with the test above). Manually verifies
    messages.csv/message_summary.json show the full lifecycle -- readiness,
    START_EXPERIMENT, the advisory request/response chain, STOP_EXPERIMENT
    -- with sent/handled linkage that is internally consistent, and that a
    real, currently-unhandled message type (START_EXPERIMENT -- neither the
    Gatekeeper nor the Plan Maker register a behaviour for it) is visible
    as sent-but-not-handled rather than silently implied as received.
    """
    import csv
    import json
    import os

    run_id = "cli-assisted-telemetry-1"
    output_directory = tmp_path / "runs"
    config_path = _write_tiny_assisted_config(tmp_path, run_id=run_id, output_directory=output_directory)
    env = {
        **os.environ,
        "MARLA_RL_ORCHESTRATOR_PASSWORD": "orchestrator-pass",
        "MARLA_GATEKEEPER_PASSWORD": "gatekeeper-pass",
        "MARLA_PLAN_MAKER_1_PASSWORD": "planmaker-pass",
    }

    last_output = None
    run_dir = output_directory / "ppo_plan_maker" / run_id
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
            break
        last_output = result.stdout + result.stderr
    else:
        pytest.fail(f"marla run never completed successfully after retries:\n{last_output}")

    messages_csv = run_dir / "messages.csv"
    summary_json = run_dir / "message_summary.json"
    assert messages_csv.is_file(), f"messages.csv missing from {run_dir}"
    assert summary_json.is_file(), f"message_summary.json missing from {run_dir}"

    with messages_csv.open() as f:
        rows = list(csv.DictReader(f))
    summary = json.loads(summary_json.read_text())

    sent_rows = [r for r in rows if r["event_type"] == "sent"]
    handled_rows = [r for r in rows if r["event_type"] == "handled"]
    assert sent_rows and handled_rows

    sent_ids = {r["message_id"] for r in sent_rows}
    handled_ids = {r["message_id"] for r in handled_rows}
    # Sent/handled linkage consistency (invariant D/C): every handled event
    # must correlate to a real sent event, and message_ids are never
    # reused across distinct logical messages.
    assert handled_ids <= sent_ids
    assert len(sent_ids) == len(sent_rows), "sent message_ids must be unique per logical message"
    assert len(handled_ids) == len(handled_rows), "handled events must not be double-recorded"

    sent_types = {r["message_type"] for r in sent_rows}
    handled_types_by_id = {r["message_id"]: r["message_type"] for r in handled_rows}

    # Readiness handshake: sent and actually handled by both participants.
    assert "READY_CHECK" in sent_types
    assert "READY" in sent_types
    assert "READY_CHECK" in handled_types_by_id.values()
    assert "READY" in handled_types_by_id.values()

    # START_EXPERIMENT: sent to every participant, by design never handled
    # by either (no behaviour is registered for it) -- exactly the kind of
    # discrepancy sent_not_handled_count exists to make visible rather than
    # implying "received" the way the old single-event design did.
    assert "START_EXPERIMENT" in sent_types
    assert "START_EXPERIMENT" not in handled_types_by_id.values()
    assert summary["sent_not_handled_count"] >= 1

    # Advisory request/response chain: two ADVISORY_REQUEST hops
    # (Orchestrator->Gatekeeper, Gatekeeper->Plan Maker) and at least one
    # ADVISORY_RESPONSE hop, all sent AND handled.
    assert "ADVISORY_REQUEST" in sent_types
    assert "ADVISORY_RESPONSE" in sent_types
    assert "ADVISORY_REQUEST" in handled_types_by_id.values()
    assert "ADVISORY_RESPONSE" in handled_types_by_id.values()
    assert summary["consultation_request_count"] >= 2  # both hops, at least one consultation

    # STOP_EXPERIMENT: sent to every participant, and (unlike
    # START_EXPERIMENT) actually handled -- both Gatekeeper and Plan Maker
    # register a StopExperimentBehaviour for it.
    assert "STOP_EXPERIMENT" in sent_types
    assert "STOP_EXPERIMENT" in handled_types_by_id.values()

    # summary.json's own aggregates must be consistent with the raw rows.
    assert summary["total_message_count"] == len(sent_rows)
    assert summary["total_event_count"] == len(rows)
    assert sum(summary["sent_count_by_agent"].values()) == len(sent_rows)
    assert sum(summary["handled_count_by_agent"].values()) == len(handled_rows)
    assert "received_count_by_agent" not in summary
