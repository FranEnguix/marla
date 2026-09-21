"""marla.messaging.telemetry -- messages.csv raw event log and the
exactly-once counting rule build_message() enforces by construction.
"""

from __future__ import annotations

import pytest

from marla.messaging import telemetry
from marla.messaging.builders import build_message
from marla.messaging.schemas import MessageMetadata, MessageType


def _metadata(sender="rl_orchestrator", receiver="gatekeeper", message_type=MessageType.READY_CHECK, request_id="req-default"):
    return MessageMetadata(
        performative="request",
        message_type=message_type,
        schema_version="1.0",
        run_id="test-run",
        conversation_id="conv-1",
        request_id=request_id,
        sender_alias=sender,
        receiver_alias=receiver,
    )


@pytest.fixture(autouse=True)
def _reset_telemetry():
    # Never let one test's active log leak into the next.
    telemetry.stop_run()
    yield
    telemetry.stop_run()


def test_build_message_is_a_noop_for_telemetry_when_no_run_is_active():
    build_message("gatekeeper@localhost", _metadata())
    assert telemetry.get_active() is None  # never implicitly started


def test_build_message_records_exactly_one_event_per_call():
    log = telemetry.start_run("test-run")
    build_message("gatekeeper@localhost", _metadata())
    build_message("gatekeeper@localhost", _metadata())
    assert len(log.events) == 2


def test_recorded_event_carries_the_full_envelope():
    log = telemetry.start_run("test-run")
    build_message("gatekeeper@localhost", _metadata(sender="rl_orchestrator", receiver="gatekeeper", request_id="req-1"))
    event = log.events[0]
    assert event.sender_alias == "rl_orchestrator"
    assert event.receiver_alias == "gatekeeper"
    assert event.message_type == "READY_CHECK"
    assert event.performative == "request"
    assert event.conversation_id == "conv-1"
    assert event.request_id == "req-1"
    assert event.run_id == "test-run"


def test_sent_and_received_counts_derive_from_the_same_event_stream_not_two_counters():
    """The canonical counting rule: sent-by/received-by are two groupings
    of ONE event list, so they can never drift apart or double-count."""
    log = telemetry.start_run("test-run")
    build_message("gatekeeper@localhost", _metadata(sender="rl_orchestrator", receiver="gatekeeper"))
    build_message("plan-maker@localhost", _metadata(sender="rl_orchestrator", receiver="plan_maker_1"))
    build_message("rl-orchestrator@localhost", _metadata(sender="gatekeeper", receiver="rl_orchestrator"))

    summary = log.summarize()
    assert summary["sent_count_by_agent"]["rl_orchestrator"] == 2
    assert summary["sent_count_by_agent"]["gatekeeper"] == 1
    assert summary["received_count_by_agent"]["gatekeeper"] == 1
    assert summary["received_count_by_agent"]["plan_maker_1"] == 1
    assert summary["received_count_by_agent"]["rl_orchestrator"] == 1
    assert summary["total_message_count"] == 3
    # Every message counted exactly once in aggregate too.
    assert sum(summary["sent_count_by_agent"].values()) == 3
    assert sum(summary["received_count_by_agent"].values()) == 3


def test_lifecycle_messages_never_carry_decision_context_even_if_context_was_set():
    log = telemetry.start_run("test-run")
    telemetry.set_context(episode_id=5, environment_step=12, global_environment_step=99)
    build_message("gatekeeper@localhost", _metadata(message_type=MessageType.STOP_EXPERIMENT))
    event = log.events[0]
    assert event.episode_id is None
    assert event.environment_step is None
    assert event.global_environment_step is None


def test_advisory_messages_carry_decision_context_when_set():
    log = telemetry.start_run("test-run")
    telemetry.set_context(episode_id=3, environment_step=7, global_environment_step=None)
    build_message("gatekeeper@localhost", _metadata(message_type=MessageType.ADVISORY_REQUEST, request_id="req-9"))
    event = log.events[0]
    assert event.episode_id == 3
    assert event.environment_step == 7


def test_messages_before_any_set_context_call_have_none_context():
    log = telemetry.start_run("test-run")
    build_message("gatekeeper@localhost", _metadata(message_type=MessageType.ADVISORY_REQUEST))
    assert log.events[0].episode_id is None


def test_summary_message_type_and_consultation_counts():
    log = telemetry.start_run("test-run")
    build_message("gatekeeper@localhost", _metadata(message_type=MessageType.ADVISORY_REQUEST))
    build_message("rl-orchestrator@localhost", _metadata(sender="gatekeeper", receiver="rl_orchestrator", message_type=MessageType.ADVISORY_RESPONSE))
    build_message("plan-maker@localhost", _metadata(sender="gatekeeper", receiver="plan_maker_1", message_type=MessageType.CORRECTION_REQUEST))
    build_message("gatekeeper@localhost", _metadata(message_type=MessageType.READY_CHECK))

    summary = log.summarize()
    assert summary["consultation_request_count"] == 1
    assert summary["consultation_response_count"] == 1
    assert summary["retry_count"] == 1
    assert summary["lifecycle_message_count"] == 1
    assert summary["message_count_by_type"]["ADVISORY_REQUEST"] == 1


def test_summarize_empty_log_returns_empty_dict():
    log = telemetry.start_run("test-run")
    assert log.summarize() == {}


def test_stop_run_returns_the_log_and_clears_active():
    telemetry.start_run("test-run")
    build_message("gatekeeper@localhost", _metadata())
    log = telemetry.stop_run()
    assert log is not None and len(log.events) == 1
    assert telemetry.get_active() is None
    # further build_message() calls after stop are no-ops for telemetry.
    build_message("gatekeeper@localhost", _metadata())
    assert len(log.events) == 1


def test_write_message_artifacts_writes_csv_and_summary(tmp_path):
    from marla.metrics.writer import write_message_artifacts

    log = telemetry.start_run("test-run")
    build_message("gatekeeper@localhost", _metadata(message_type=MessageType.ADVISORY_REQUEST))
    write_message_artifacts(tmp_path, log)

    assert (tmp_path / "messages.csv").is_file()
    assert (tmp_path / "message_summary.json").is_file()
    import csv

    with (tmp_path / "messages.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["message_type"] == "ADVISORY_REQUEST"


def test_write_message_artifacts_is_a_noop_for_none_or_empty_log(tmp_path):
    from marla.metrics.writer import write_message_artifacts

    write_message_artifacts(tmp_path, None)
    assert not (tmp_path / "messages.csv").exists()

    empty_log = telemetry.start_run("test-run")
    write_message_artifacts(tmp_path, empty_log)
    assert not (tmp_path / "messages.csv").exists()


def test_two_sequential_runs_in_the_same_process_do_not_leak_events():
    telemetry.start_run("run-a")
    build_message("gatekeeper@localhost", _metadata())
    log_a = telemetry.stop_run()

    telemetry.start_run("run-b")
    log_b = telemetry.get_active()
    assert log_b.events == []
    build_message("gatekeeper@localhost", _metadata())
    build_message("gatekeeper@localhost", _metadata())
    assert len(log_a.events) == 1  # unaffected by the second run
    assert len(log_b.events) == 2
