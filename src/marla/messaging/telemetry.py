"""Raw inter-agent message-event telemetry (``messages.csv``).

**Canonical counting rule**: exactly ONE event is recorded per logical
MARLA message, at build time (:func:`marla.messaging.builders.build_message`
calls :func:`record_if_active` itself, at its one call site), tagged with
both ``sender_alias`` and ``receiver_alias`` already present on the
envelope constructed there. "messages sent by agent X" and "messages
received by agent Y" are BOTH derived by grouping this SAME event stream
(by ``sender_alias`` / ``receiver_alias`` respectively) -- never two
independently instrumented counters, which is exactly the design that
could double-count a message or let sent/received drift apart. This is a
deliberate simplification, documented rather than silently assumed:
"received" here means "addressed to and dispatched toward" this agent, not
"confirmed processed by its own behaviour" -- SPADE delivery within one
process's local-mode embedded XMPP server is not separately confirmed.

Only LOGICAL MARLA envelope messages are counted -- raw XMPP transport
packets (presence stanzas, IQ pings, SPADE/pyjabber internals) never pass
through ``build_message()`` at all, so they can never appear here.

Usage (mirrors ``monitoring.resources.ResourceMonitor``'s start/stop/
set_context shape, but as a lightweight module-level active-log rather
than an object every call site would otherwise need threading through --
``build_message()`` has no other access to run state):

    telemetry.start_run(run_id)
    ...
    telemetry.set_context(episode_id=1, environment_step=5, global_environment_step=5)
    ...  # any build_message() call during this window is tagged with the above
    log = telemetry.stop_run()
    write_message_artifacts(run_dir, log)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


@dataclass
class MessageEvent:
    timestamp: float  # time.time(), seconds since epoch
    elapsed_seconds: float  # seconds since telemetry.start_run() for this run
    run_id: str
    sender_alias: str
    receiver_alias: str
    performative: str
    message_type: str
    conversation_id: str
    request_id: str | None
    # Decision-time context, populated for messages built from within the
    # rollout decision loop (advisory request/response/correction) via
    # set_context() -- None for lifecycle messages (READY_CHECK/
    # START_EXPERIMENT/STOP_EXPERIMENT), which are not tied to any one
    # decision and are typed distinctly via message_type instead.
    episode_id: int | None
    environment_step: int | None
    global_environment_step: int | None


_LIFECYCLE_MESSAGE_TYPES = frozenset({"READY_CHECK", "READY", "START_EXPERIMENT", "STOP_EXPERIMENT", "EXPERIMENT_FAILED"})
_CONSULTATION_REQUEST_TYPES = frozenset({"ADVISORY_REQUEST"})
_CONSULTATION_RESPONSE_TYPES = frozenset({"ADVISORY_RESPONSE"})
_CORRECTION_TYPES = frozenset({"CORRECTION_REQUEST"})


class MessageEventLog:
    """Per-run collector. One instance per ``marla run`` process/
    :func:`start_run` call -- never shared across runs, so two sequential
    ``collect()``-style test invocations in the same process never see
    each other's events."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._start = time.monotonic()
        self.events: list[MessageEvent] = []
        self._episode_id: int | None = None
        self._environment_step: int | None = None
        self._global_environment_step: int | None = None

    def set_context(
        self, episode_id: int | None = None, environment_step: int | None = None, global_environment_step: int | None = None
    ) -> None:
        self._episode_id = episode_id
        self._environment_step = environment_step
        self._global_environment_step = global_environment_step

    def clear_context(self) -> None:
        """Explicitly drop decision context -- called around lifecycle
        messages so they never inherit stale context from whatever
        decision last set it."""
        self._episode_id = self._environment_step = self._global_environment_step = None

    def record(
        self, sender_alias: str, receiver_alias: str, performative: str, message_type: str,
        conversation_id: str, request_id: str | None,
    ) -> None:
        # Lifecycle messages are never decision-scoped, regardless of
        # whatever context a prior (or, in principle, a still-pending)
        # decision last set -- enforced here, at the one recording site,
        # rather than trusting every lifecycle call site to remember to
        # clear_context() first.
        is_lifecycle = message_type in _LIFECYCLE_MESSAGE_TYPES
        self.events.append(
            MessageEvent(
                timestamp=time.time(), elapsed_seconds=time.monotonic() - self._start, run_id=self.run_id,
                sender_alias=sender_alias, receiver_alias=receiver_alias, performative=performative,
                message_type=message_type, conversation_id=conversation_id, request_id=request_id,
                episode_id=None if is_lifecycle else self._episode_id,
                environment_step=None if is_lifecycle else self._environment_step,
                global_environment_step=None if is_lifecycle else self._global_environment_step,
            )
        )

    def to_rows(self) -> list[dict]:
        return [asdict(e) for e in self.events]

    def summarize(self) -> dict:
        """Run-level derived counts -- raw events remain the authoritative
        data; this is a convenience aggregate, always reproducible from
        ``to_rows()`` alone.

        ``lifecycle_message_count`` reflects only whatever lifecycle
        messages happened to be sent while THIS log was active -- see
        ``learning/trainer.py``'s ``telemetry.start_run()`` call site for
        a documented scope limitation: the pre-training READY_CHECK/
        START_EXPERIMENT and post-training STOP_EXPERIMENT messages,
        sent from ``OrchestratorLifecycleBehaviour`` immediately outside
        ``run_training_loop``'s own duration, are not currently captured.
        """
        if not self.events:
            return {}
        sent: dict[str, int] = {}
        received: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for e in self.events:
            sent[e.sender_alias] = sent.get(e.sender_alias, 0) + 1
            received[e.receiver_alias] = received.get(e.receiver_alias, 0) + 1
            by_type[e.message_type] = by_type.get(e.message_type, 0) + 1
        # NOTE: one logical consultation is TWO ADVISORY_REQUEST messages
        # (RL Orchestrator -> Gatekeeper, then Gatekeeper -> Plan Maker) --
        # this counts MESSAGES, not distinct consultations. Compare against
        # episodes.csv/decisions.csv's own consultation_count (or
        # decisions.csv's queried column) for the distinct-consultation
        # figure; do not assume this equals that number directly.
        consultation_request_count = sum(1 for e in self.events if e.message_type in _CONSULTATION_REQUEST_TYPES)
        consultation_response_count = sum(1 for e in self.events if e.message_type in _CONSULTATION_RESPONSE_TYPES)
        retry_count = sum(1 for e in self.events if e.message_type in _CORRECTION_TYPES)
        lifecycle_count = sum(1 for e in self.events if e.message_type in _LIFECYCLE_MESSAGE_TYPES)
        return {
            "total_message_count": len(self.events),
            "sent_count_by_agent": sent,
            "received_count_by_agent": received,
            "message_count_by_type": by_type,
            "consultation_request_count": consultation_request_count,
            "consultation_response_count": consultation_response_count,
            "retry_count": retry_count,
            "lifecycle_message_count": lifecycle_count,
        }


_active: MessageEventLog | None = None


def start_run(run_id: str) -> MessageEventLog:
    global _active
    _active = MessageEventLog(run_id)
    return _active


def stop_run() -> MessageEventLog | None:
    global _active
    log = _active
    _active = None
    return log


def get_active() -> MessageEventLog | None:
    return _active


def set_context(
    episode_id: int | None = None, environment_step: int | None = None, global_environment_step: int | None = None
) -> None:
    if _active is not None:
        _active.set_context(episode_id, environment_step, global_environment_step)


def clear_context() -> None:
    if _active is not None:
        _active.clear_context()


def record_if_active(
    sender_alias: str, receiver_alias: str, performative: str, message_type: str,
    conversation_id: str, request_id: str | None,
) -> None:
    """Called from exactly one place (``builders.build_message``) -- never
    call this from anywhere else, or the "one event per logical message"
    invariant this module exists to guarantee would no longer hold.
    """
    if _active is not None:
        _active.record(sender_alias, receiver_alias, performative, message_type, conversation_id, request_id)
