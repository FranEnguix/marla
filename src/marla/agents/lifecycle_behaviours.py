"""SPADE behaviour wiring for the lifecycle events in :mod:`marla.runtime.lifecycle`.

Reusable by any agent role: the RL Orchestrator uses
:class:`ReadyListenerBehaviour`/:class:`FailureListenerBehaviour` to collect
events, and non-orchestrator participants (Gatekeeper, Plan Maker --
Milestone 6) use :class:`ReadyCheckResponderBehaviour`/
:class:`StopExperimentBehaviour` to answer the handshake and shut down
cleanly.
"""

from __future__ import annotations

import asyncio
import logging

from spade.behaviour import CyclicBehaviour
from spade.template import Template

from marla.messaging.builders import build_message, new_id
from marla.messaging.parsers import MessageValidationError, parse_metadata
from marla.messaging.schemas import MESSAGE_SCHEMA_VERSION, MessageMetadata, MessageType
from marla.runtime.lifecycle import FailureSignal, ReadySignal

logger = logging.getLogger(__name__)


def message_type_template(message_type: MessageType) -> Template:
    template = Template()
    template.set_metadata("message_type", message_type.value)
    return template


# SPADE's XMPPClient enables XEP-0199 keepalive pings every 55s and
# *reconnects on a missed one* (slixmpp's xep_0199 plugin docstring: "will
# send a ping at `interval` and reconnect if the ping times out"). All
# agents in local mode share one event loop, and every agent's Plan Maker
# consultation blocks that whole loop for real, possibly multi-second-to-
# multi-minute stretches (spec section 4.1's asyncio.to_thread would hang
# indefinitely here instead -- see local_backend.py's module docstring) --
# long enough to miss a ping round trip. The resulting reconnect attempts
# to re-register (SPADE's default auto_register=True), and if the loop is
# blocked *again* at that moment, registration itself times out, raising
# an unhandled, fatal RegistrationException -- observed in practice as a
# run crashing partway through with no relation to anything actually going
# wrong. Raising the interval only narrows the window rather than closing
# it, since a long enough correction-loop stretch can still exceed any
# fixed interval; disabling it outright removes ping-based reconnection as
# a failure mode entirely. A dead connection is still discovered the moment
# anything is actually sent over it, and a genuinely dropped *peer* is
# already handled by the presence-based disconnect detection above
# (``make_disconnect_detector``), which doesn't depend on ping timing.
def disable_reconnect_on_missed_ping(agent) -> None:
    """Disable XEP-0199's ping-triggered reconnect; see the module-level comment.

    SPADE's ``XMPPClient.__init__`` already calls ``enable_keepalive()``
    once before ``setup()`` ever runs, scheduling a recurring ping.
    ``disable_keepalive()`` cancels that scheduled event.
    """
    agent.client["xep_0199"].disable_keepalive()


def make_disconnect_detector(agent, watched_jid: str):
    """Build ``(on_available, on_unavailable, expect_disconnect)`` presence callbacks
    that stop ``agent`` only on a genuine, *unexpected* disconnect of ``watched_jid``.

    Used by support agents (Gatekeeper, Plan Maker) to satisfy "explicit
    disconnect... must fail the experiment" (spec section 5) in distributed
    mode, where losing the process on the other end of a JID is a real,
    detectable event rather than a same-process assumption. Local mode's
    embedded XMPP server is not used here on purpose -- this only matters
    once processes are genuinely separate.

    An "unavailable" presence stanza for ``watched_jid`` can arrive before
    the two agents' startup ordering has settled -- e.g. a presence probe
    answered for a peer that has simply not come online yet -- and looks
    identical on the wire to a peer that really did drop. Only a peer
    previously observed *available* and then reported unavailable counts as
    a disconnect; both callbacks must be wired (``on_available`` as well as
    ``on_unavailable``) for this distinction to work.

    A peer's disconnect is also unremarkable, not a failure, once the run has
    reached its own STOP_EXPERIMENT -- calling ``expect_disconnect()`` (from
    :class:`StopExperimentBehaviour`, right before this agent stops itself)
    suppresses the callback for the rest of this agent's lifetime, so a
    normal end-of-run teardown race doesn't get reported as a crash.
    """
    watched_bare = str(watched_jid).split("/")[0]
    seen_available = False
    expecting_disconnect = False

    def _on_available(peer_jid, presence_info=None, last_presence=None) -> None:
        nonlocal seen_available
        if str(peer_jid).split("/")[0] == watched_bare:
            seen_available = True

    def _on_unavailable(peer_jid, presence_info=None, last_presence=None) -> None:
        if str(peer_jid).split("/")[0] != watched_bare or not seen_available or expecting_disconnect:
            return
        logger.error("%s: watched peer %s disconnected; stopping", agent.jid, peer_jid)
        agent.failure = ConnectionError(f"{watched_jid} disconnected")
        agent.submit(agent.stop())

    def _expect_disconnect() -> None:
        nonlocal expecting_disconnect
        expecting_disconnect = True

    return _on_available, _on_unavailable, _expect_disconnect


class ReadyListenerBehaviour(CyclicBehaviour):
    """RL-Orchestrator side: turns incoming READY messages into ReadySignal events."""

    def __init__(self, events: "asyncio.Queue", run_id: str):
        super().__init__()
        self._events = events
        self._run_id = run_id

    async def run(self) -> None:
        message = await self.receive(timeout=None)
        if message is None:
            return
        try:
            metadata = parse_metadata(message)
        except MessageValidationError as exc:
            logger.warning("Dropping malformed READY message: %s", exc)
            return
        if metadata.run_id != self._run_id:
            logger.warning(
                "Dropping READY message with mismatched run_id %r (expected %r)",
                metadata.run_id, self._run_id,
            )
            return

        signal = ReadySignal(
            alias=metadata.sender_alias,
            jid=str(message.sender),
            schema_version=metadata.schema_version,
            model_version=message.get_metadata("model_version") or "",
            resolved_device=message.get_metadata("resolved_device") or "",
        )
        await self._events.put(signal)


class FailureListenerBehaviour(CyclicBehaviour):
    """RL-Orchestrator side: turns incoming EXPERIMENT_FAILED messages into FailureSignal events."""

    def __init__(self, events: "asyncio.Queue", run_id: str):
        super().__init__()
        self._events = events
        self._run_id = run_id

    async def run(self) -> None:
        message = await self.receive(timeout=None)
        if message is None:
            return
        try:
            metadata = parse_metadata(message)
        except MessageValidationError as exc:
            logger.warning("Dropping malformed EXPERIMENT_FAILED message: %s", exc)
            return
        if metadata.run_id != self._run_id:
            return
        reason = message.body or "unspecified"
        await self._events.put(FailureSignal(alias=metadata.sender_alias, reason=reason))


class ReadyCheckResponderBehaviour(CyclicBehaviour):
    """Participant side: replies READY to READY_CHECK once local setup has succeeded.

    Local model loading/device resolution must happen *before* this
    behaviour is added (spec section 18: an agent must not send READY after
    failed initialization) -- if setup failed, the agent process should
    raise instead of ever starting this behaviour.
    """

    def __init__(self, run_id: str, own_alias: str, model_version: str, resolved_device: str):
        super().__init__()
        self._run_id = run_id
        self._own_alias = own_alias
        self._model_version = model_version
        self._resolved_device = resolved_device

    async def run(self) -> None:
        message = await self.receive(timeout=None)
        if message is None:
            return
        try:
            metadata = parse_metadata(message)
        except MessageValidationError as exc:
            logger.warning("Dropping malformed READY_CHECK message: %s", exc)
            return
        if metadata.run_id != self._run_id:
            return

        reply_metadata = MessageMetadata(
            performative="inform",
            message_type=MessageType.READY,
            schema_version=MESSAGE_SCHEMA_VERSION,
            run_id=self._run_id,
            conversation_id=metadata.conversation_id,
            request_id=new_id("request"),
            sender_alias=self._own_alias,
            receiver_alias=metadata.sender_alias,
        )
        reply = build_message(
            str(message.sender),
            reply_metadata,
            extra_metadata={"model_version": self._model_version, "resolved_device": self._resolved_device},
        )
        await self.send(reply)


class StopExperimentBehaviour(CyclicBehaviour):
    """Any agent: stops itself upon receiving STOP_EXPERIMENT for this run."""

    def __init__(self, run_id: str):
        super().__init__()
        self._run_id = run_id

    async def run(self) -> None:
        message = await self.receive(timeout=None)
        if message is None:
            return
        try:
            metadata = parse_metadata(message)
        except MessageValidationError as exc:
            logger.warning("Dropping malformed STOP_EXPERIMENT message: %s", exc)
            return
        if metadata.run_id != self._run_id:
            return
        logger.info("Received STOP_EXPERIMENT for run %s; stopping agent %s", self._run_id, self.agent.jid)
        expect_peer_disconnect = getattr(self.agent, "expect_peer_disconnect", None)
        if expect_peer_disconnect is not None:
            expect_peer_disconnect()
        await self.agent.stop()
