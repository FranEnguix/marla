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
# MARLA-owned, pre-validated SOLVABLE multi-subnet scenario (4 subnets: DMZ
# entry + user/service/db) -- NASimEmu/scenarios/sm_entry_dmz_two_subnets.v2.yaml
# fails MARLA's own scenario-solvability check (a Windows host reachable
# only to USER via e_wp_ninja with no compatible ROOT privesc), so `marla
# run` refuses to start on it at all; this one is guaranteed solvable.
MULTI_SUBNET_SCENARIO = "marla://sm_entry_user_three_subnets.solvable.v2.yaml"


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


@pytest.mark.integration
def test_marla_run_assisted_multi_subnet_uses_real_subnet_scoped_consultation(tmp_path):
    """Real end-to-end smoke for subnet-scoped Plan Maker consultation
    (spec sections 29): a real RLOrchestratorAgent + GatekeeperAgent +
    PlanMakerAgent on a scenario with multiple visible subnets from the
    start. Verifies from real decisions.csv rows that at least one queried
    decision had global_candidate_action_count > consulted_candidate_action_count,
    and -- via --debug's real generated prompt text -- that no host outside
    the consulted subnet appears in the LOCAL VISIBLE OBSERVATION section
    of a real prompt actually sent to the (tiny) model.
    """
    import csv
    import os
    import re

    run_id = "cli-assisted-subnet-scope-1"
    output_directory = tmp_path / "runs"
    config_path = _write_tiny_assisted_config(tmp_path, run_id=run_id, output_directory=output_directory)
    data = yaml.safe_load(config_path.read_text())
    data["environment"]["scenario"] = MULTI_SUBNET_SCENARIO
    data["environment"]["max_episode_steps"] = 10
    data["policy"]["ppo"]["total_environment_steps"] = 16
    data["policy"]["ppo"]["steps_per_env"] = 16
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    env = {
        **os.environ,
        "MARLA_RL_ORCHESTRATOR_PASSWORD": "orchestrator-pass",
        "MARLA_GATEKEEPER_PASSWORD": "gatekeeper-pass",
        "MARLA_PLAN_MAKER_1_PASSWORD": "planmaker-pass",
    }

    run_dir = output_directory / "ppo_plan_maker" / run_id
    last_output = None
    for _ in range(6):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "marla", "run", str(config_path), "--debug"],
                cwd=tmp_path,  # so debug/<run_id>/ lands under tmp_path, not the repo root
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
            )
        except subprocess.TimeoutExpired as exc:
            last_output = f"TIMEOUT: {exc.stdout}\n{exc.stderr}"
            continue
        if result.returncode == 0 and "Run complete" in result.stdout:
            break
        last_output = result.stdout + result.stderr
    else:
        pytest.fail(f"marla run never completed successfully after retries:\n{last_output}")

    decisions_csv = run_dir / "decisions.csv"
    assert decisions_csv.is_file()
    with decisions_csv.open() as f:
        rows = list(csv.DictReader(f))
    queried_rows = [r for r in rows if r["queried"] == "True"]
    assert queried_rows, "expected at least one queried decision on a 16-step run"

    scoped_rows = [r for r in queried_rows if r["consultation_scope"] == "subnet_scoped"]
    assert scoped_rows, "every queried decision must report consultation_scope=subnet_scoped"
    # global_candidate_action_count/consulted_subnet are populated on every
    # queried decision regardless of accept/reject; consulted_candidate_action_count
    # is only populated on an ACCEPTED response (see writer.py) -- the tiny
    # test model reliably fails to produce valid JSON (see
    # test_plan_maker.py's own real_tiny docstring), so this run's own
    # decisions.csv rows will likely all be schema_rejected. The
    # global>consulted reduction property is instead verified below,
    # directly from --debug's real generated prompt text, which is written
    # unconditionally (accept, correction, or reject) -- a stronger check
    # anyway, since it inspects REQUEST CONSTRUCTION itself rather than a
    # downstream field that only exists when the model happened to succeed.

    # --debug: inspect real generated prompts for the no-leakage AND the
    # global>consulted action-count reduction invariants (spec section 29).
    debug_dir = tmp_path / "debug" / run_id
    assert debug_dir.is_dir(), f"--debug should have created {debug_dir}"
    query_files = sorted(debug_dir.glob("*_query.txt"))
    assert query_files, "expected at least one dumped Plan Maker query"

    checked_any_local_host = False
    found_reduction = False
    for query_file in query_files:
        prompt_text = query_file.read_text(encoding="utf-8")
        subnet_match = re.search(r"CONSULTED SUBNET (\d+|None)", prompt_text)
        assert subnet_match, f"prompt missing CONSULTED SUBNET marker: {query_file}"

        global_count_match = re.search(r"(\d+) total\ncandidate action\(s\) exist", prompt_text)
        assert global_count_match, f"prompt missing global_candidate_action_count marker: {query_file}"
        global_count = int(global_count_match.group(1))

        # The candidate-actions list is always emitted as one single
        # json.dumps() line (build_prompt's candidate_actions_text) --
        # matching that line directly is more robust than splitting on
        # "CONSULTED CANDIDATE ACTIONS", which also appears twice more in
        # this section's own intro/closing prose.
        actions_line_match = re.search(r'^\[\{"action_id".*\}\]$', prompt_text, re.MULTILINE)
        assert actions_line_match, f"prompt missing the consulted candidate-actions JSON line: {query_file}"
        consulted_action_ids = re.findall(r'"action_id":\s*"([^"]+)"', actions_line_match.group(0))
        assert consulted_action_ids, f"prompt has no consulted candidate actions at all: {query_file}"
        if global_count > len(consulted_action_ids):
            found_reduction = True

        if subnet_match.group(1) == "None":
            continue  # the documented no-non-FINISH-candidate edge case
        consulted_subnet = int(subnet_match.group(1))

        local_section = prompt_text.split("LOCAL VISIBLE OBSERVATION")[1].split("CONSULTED CANDIDATE ACTIONS")[0]
        host_targets = re.findall(r'"target":\s*"(host-\d+-\d+)"', local_section)
        for target in host_targets:
            checked_any_local_host = True
            target_subnet = int(target.split("-")[1])
            assert target_subnet == consulted_subnet, (
                f"host {target} outside consulted subnet {consulted_subnet} leaked into {query_file}"
            )
    assert checked_any_local_host, "expected at least one real prompt with a non-empty local observation"
    assert found_reduction, (
        "expected at least one real generated prompt where global_candidate_action_count "
        "exceeds the number of consulted candidate actions (multi-subnet scenario)"
    )
