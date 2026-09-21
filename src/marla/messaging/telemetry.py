"""Raw inter-agent message-event telemetry (``messages.csv``).

**Canonical counting rule**: every logical MARLA message can produce AT
MOST two events in this log:

- a ``"sent"`` event, recorded once, at build time
  (:func:`marla.messaging.builders.build_message` calls
  :func:`record_sent_if_active` at its one call site) -- this proves the
  message was built and addressed to a receiver, nothing more.
- a ``"handled"`` event, recorded once, at the point the INTENDED MARLA
  behaviour actually accepts the message for processing (after
  ``self.receive()`` returns it and its envelope/correlation checks pass --
  see each agent behaviour's ``run()`` method for its own call site) -- this
  proves a real SPADE behaviour actually matched, parsed, and acted on it.

These are deliberately NOT the same event, and NOT assumed to always both
exist: MARLA has observed real "No behaviour matched for message ..."
situations, where a message was built and addressed but nothing ever
processed it. Counting one event at build time only -- the previous design
of this module -- could not distinguish "addressed and processed" from
"addressed and silently dropped." A message that is sent but never handled
is a real, visible discrepancy (``sent_not_handled_count`` in
:meth:`MessageEventLog.summarize`), not an error swallowed by this module.

**Correlating a "sent" event with its "handled" event**: none of the
envelope fields callers already had (``run_id``, ``conversation_id``,
``request_id``, ``message_type``, ``sender_alias``, ``receiver_alias``) are
sufficient by themselves -- a retried ``CORRECTION_REQUEST`` reuses the same
``request_id``/``conversation_id``/``message_type`` across every retry
(spec: retries correlate to the same request, on purpose), so those fields
alone cannot tell two retries apart. :func:`marla.messaging.builders.build_message`
therefore mints a fresh ``message_id`` (a UUID, distinct from
``request_id``) for every call -- i.e. every logical message, retries
included -- and stamps it into the SPADE message's own metadata. The
receiving behaviour reads it back off the parsed envelope
(``MessageMetadata.message_id``) and passes the SAME value to
:func:`record_handled_if_active`, which is what lets "sent" and "handled"
be joined on ``message_id`` after the fact, including across retries.

Only LOGICAL MARLA envelope messages are counted -- raw XMPP transport
packets (presence stanzas, IQ pings, SPADE/pyjabber internals) never pass
through ``build_message()`` at all, so they can never appear here.

**Scope limitation, unchanged from the previous design**: this module is a
single process-local, module-level active log (see ``start_run``/
``stop_run`` below). In local mode (the tested, production `marla run`
mode), every agent runs in one process/event loop, so one log genuinely
captures every sent AND handled event for the whole run. In distributed
mode, a Gatekeeper or Plan Maker running in its own process has its own,
separate ``_active`` (or none at all, if it never calls ``start_run``) --
this was already true for "sent" events before "handled" events existed,
and is not a new limitation introduced here.

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
from dataclasses import asdict, dataclass
from typing import Literal

EventType = Literal["sent", "handled"]


@dataclass
class MessageEvent:
    timestamp: float  # time.time(), seconds since epoch
    elapsed_seconds: float  # seconds since telemetry.start_run() for this run
    run_id: str
    event_type: str  # "sent" | "handled" -- see module docstring
    message_id: str  # correlates a message's "sent" event with its "handled" event
    sender_alias: str
    receiver_alias: str
    performative: str
    message_type: str
    conversation_id: str
    request_id: str | None
    # Decision-time context, populated for messages built/handled from
    # within the rollout decision loop (advisory request/response/
    # correction) via set_context() -- None for lifecycle messages
    # (READY_CHECK/START_EXPERIMENT/STOP_EXPERIMENT/...), which are not
    # tied to any one decision and are typed distinctly via message_type
    # instead. This rule applies identically to "sent" and "handled"
    # events for the same message.
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
        self, event_type: EventType, message_id: str, sender_alias: str, receiver_alias: str, performative: str,
        message_type: str, conversation_id: str, request_id: str | None,
    ) -> None:
        # Lifecycle messages are never decision-scoped, regardless of
        # whatever context a prior (or, in principle, a still-pending)
        # decision last set -- enforced here, at the one recording site,
        # rather than trusting every lifecycle call site to remember to
        # clear_context() first. Applies identically to sent and handled
        # events.
        is_lifecycle = message_type in _LIFECYCLE_MESSAGE_TYPES
        self.events.append(
            MessageEvent(
                timestamp=time.time(), elapsed_seconds=time.monotonic() - self._start, run_id=self.run_id,
                event_type=event_type, message_id=message_id,
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

        ``sent_*``/``handled_*`` figures are two SEPARATE groupings of two
        SEPARATE event kinds (see the module docstring) -- never assume
        ``handled_count_by_agent`` sums to the same total as
        ``sent_count_by_agent``: a message with no matching behaviour is
        sent but never handled, and this is exactly the gap
        ``sent_not_handled_count`` makes visible rather than silently
        absorbing.

        ``addressed_count_by_agent`` (derived from "sent" events, grouped
        by receiver) is the honest replacement for this module's previous
        ``received_count_by_agent``: it proves a message was built and
        addressed to that agent, never that agent's own behaviour actually
        processed it -- use ``handled_count_by_agent`` for that.

        ``lifecycle_message_count`` reflects only whatever lifecycle
        messages happened to be sent while THIS log was active.
        """
        if not self.events:
            return {}
        sent_events = [e for e in self.events if e.event_type == "sent"]
        handled_events = [e for e in self.events if e.event_type == "handled"]

        sent_by_agent: dict[str, int] = {}
        addressed_by_agent: dict[str, int] = {}
        sent_by_type: dict[str, int] = {}
        for e in sent_events:
            sent_by_agent[e.sender_alias] = sent_by_agent.get(e.sender_alias, 0) + 1
            addressed_by_agent[e.receiver_alias] = addressed_by_agent.get(e.receiver_alias, 0) + 1
            sent_by_type[e.message_type] = sent_by_type.get(e.message_type, 0) + 1

        handled_by_agent: dict[str, int] = {}
        handled_by_type: dict[str, int] = {}
        for e in handled_events:
            # "handled by" means the RECEIVING agent's own behaviour did the
            # handling -- grouped by receiver_alias, not sender_alias.
            handled_by_agent[e.receiver_alias] = handled_by_agent.get(e.receiver_alias, 0) + 1
            handled_by_type[e.message_type] = handled_by_type.get(e.message_type, 0) + 1

        sent_message_ids = {e.message_id for e in sent_events}
        handled_message_ids = {e.message_id for e in handled_events}
        sent_not_handled_count = len(sent_message_ids - handled_message_ids)

        # NOTE: one logical consultation is TWO ADVISORY_REQUEST messages
        # (RL Orchestrator -> Gatekeeper, then Gatekeeper -> Plan Maker) --
        # this counts MESSAGES, not distinct consultations. Compare against
        # episodes.csv/decisions.csv's own consultation_count (or
        # decisions.csv's queried column) for the distinct-consultation
        # figure; do not assume this equals that number directly. Counted
        # over "sent" events only -- each logical message has exactly one
        # sent event by construction, so this is never double-counted by
        # also including its "handled" event.
        consultation_request_count = sum(1 for e in sent_events if e.message_type in _CONSULTATION_REQUEST_TYPES)
        consultation_response_count = sum(1 for e in sent_events if e.message_type in _CONSULTATION_RESPONSE_TYPES)
        retry_count = sum(1 for e in sent_events if e.message_type in _CORRECTION_TYPES)
        lifecycle_count = sum(1 for e in sent_events if e.message_type in _LIFECYCLE_MESSAGE_TYPES)
        return {
            "total_message_count": len(sent_events),
            "total_event_count": len(self.events),
            "sent_count_by_agent": sent_by_agent,
            "handled_count_by_agent": handled_by_agent,
            "addressed_count_by_agent": addressed_by_agent,
            "sent_count_by_type": sent_by_type,
            "handled_count_by_type": handled_by_type,
            "sent_not_handled_count": sent_not_handled_count,
            "message_count_by_type": sent_by_type,
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


def record_sent_if_active(
    message_id: str, sender_alias: str, receiver_alias: str, performative: str, message_type: str,
    conversation_id: str, request_id: str | None,
) -> None:
    """Called from exactly one place (``builders.build_message``) -- never
    call this from anywhere else, or the "one sent-event per logical
    message" invariant this module exists to guarantee would no longer
    hold.
    """
    if _active is not None:
        _active.record("sent", message_id, sender_alias, receiver_alias, performative, message_type, conversation_id, request_id)


def record_handled_if_active(
    message_id: str, sender_alias: str, receiver_alias: str, performative: str, message_type: str,
    conversation_id: str, request_id: str | None,
) -> None:
    """Called by the intended MARLA behaviour once it has actually accepted
    a received message for processing (parsed its envelope and confirmed
    it belongs to this run/request) -- never at ``self.receive()`` time
    alone, since a message that fails envelope parsing or correlates to no
    known request was never really "handled," just delivered.
    """
    if _active is not None:
        _active.record("handled", message_id, sender_alias, receiver_alias, performative, message_type, conversation_id, request_id)
