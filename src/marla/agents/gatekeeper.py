"""The Gatekeeper: authoritative schema validator for advisory traffic (spec section 3.2).

All Plan Maker advisory requests/responses route through here:
``RL Orchestrator -> Gatekeeper -> Plan Maker`` and back. The Gatekeeper
validates communication and advisory payloads, never final NASimEmu
actions. It holds only the transient pending-request registry described in
spec section 6 -- no Blackboard, no cross-run memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import ValidationError
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour

from marla.agents.lifecycle_behaviours import (
    ReadyCheckResponderBehaviour,
    StopExperimentBehaviour,
    disable_reconnect_on_missed_ping,
    make_disconnect_detector,
    message_type_template,
)
from marla.messaging.advisory_validation import AdvisoryValidationError, validate_advisory_response_body
from marla.messaging.builders import build_message
from marla.messaging.parsers import MessageValidationError, parse_json_body, parse_metadata
from marla.messaging.schemas import (
    MESSAGE_SCHEMA_VERSION,
    AdvisoryRequestPayload,
    MessageMetadata,
    MessageType,
)

logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    """Transient correlation state for one in-flight advisory request (spec section 6)."""

    run_id: str
    request_id: str
    source_observation_id: str
    episode_id: int
    environment_step: int
    expected_agent_alias: str
    legal_action_ids: tuple[str, ...]
    schema_revision_count: int = 0


class AdvisoryRequestBehaviour(CyclicBehaviour):
    """RL Orchestrator -> Gatekeeper: validate the request, then forward it verbatim to the Plan Maker."""

    async def run(self) -> None:
        message = await self.receive(timeout=None)
        if message is None:
            return
        agent: GatekeeperAgent = self.agent

        try:
            metadata = parse_metadata(message)
            body = parse_json_body(message)
            request = AdvisoryRequestPayload.model_validate(body)
        except (MessageValidationError, ValidationError) as exc:
            logger.error("Rejecting malformed advisory request from RL Orchestrator: %s", exc)
            return

        if request.run_id != agent.run_id:
            logger.warning("Dropping advisory request with mismatched run_id: %s", request.run_id)
            return

        agent.pending[request.request_id] = PendingRequest(
            run_id=request.run_id,
            request_id=request.request_id,
            source_observation_id=request.source_observation_id,
            episode_id=request.episode_id,
            environment_step=request.step,
            expected_agent_alias=agent.plan_maker_alias,
            legal_action_ids=tuple(a.action_id for a in request.legal_actions),
        )

        forward_metadata = MessageMetadata(
            performative="request",
            message_type=MessageType.ADVISORY_REQUEST,
            schema_version=MESSAGE_SCHEMA_VERSION,
            run_id=agent.run_id,
            conversation_id=metadata.conversation_id,
            request_id=request.request_id,
            sender_alias=agent.alias,
            receiver_alias=agent.plan_maker_alias,
        )
        await self.send(build_message(agent.plan_maker_jid, forward_metadata, payload=body))


class AdvisoryResponseBehaviour(CyclicBehaviour):
    """Plan Maker -> Gatekeeper: validate, request correction, or forward/reject to the RL Orchestrator."""

    async def run(self) -> None:
        message = await self.receive(timeout=None)
        if message is None:
            return
        agent: GatekeeperAgent = self.agent

        try:
            metadata = parse_metadata(message)
            body = parse_json_body(message)
        except MessageValidationError as exc:
            logger.error("Dropping malformed advisory response (cannot correlate): %s", exc)
            return

        pending = agent.pending.get(metadata.request_id)
        if pending is None:
            logger.warning("No pending request for advisory response request_id=%s", metadata.request_id)
            return

        if metadata.sender_alias != pending.expected_agent_alias:
            error = AdvisoryValidationError(
                "unexpected_sender", f"{metadata.sender_alias} != {pending.expected_agent_alias}"
            )
        else:
            try:
                response = validate_advisory_response_body(
                    body, agent.run_id, pending.request_id, set(pending.legal_action_ids)
                )
                error = None
            except AdvisoryValidationError as exc:
                error = exc

        if error is None:
            logger.info("Gatekeeper: accepted advisory response for request_id=%s", metadata.request_id)
            await self._accept(pending, body)
            del agent.pending[metadata.request_id]
            return

        pending.schema_revision_count += 1
        if pending.schema_revision_count > agent.max_schema_revisions:
            logger.info(
                "Gatekeeper: rejecting request_id=%s after %d failed revision(s): %s",
                metadata.request_id, pending.schema_revision_count - 1, error.reason,
            )
            await self._reject(pending)
            del agent.pending[metadata.request_id]
            return

        logger.info(
            "Gatekeeper: requesting correction %d/%d for request_id=%s: %s",
            pending.schema_revision_count, agent.max_schema_revisions, metadata.request_id, error.reason,
        )
        await self._request_correction(pending, error)

    async def _accept(self, pending: PendingRequest, raw_body: dict) -> None:
        agent: GatekeeperAgent = self.agent
        artifact = {"response_status": "accepted", "validation": {"status": "accepted"}, "payload": raw_body}
        metadata = MessageMetadata(
            performative="inform",
            message_type=MessageType.ADVISORY_RESPONSE,
            schema_version=MESSAGE_SCHEMA_VERSION,
            run_id=agent.run_id,
            conversation_id=pending.request_id,
            request_id=pending.request_id,
            sender_alias=agent.alias,
            receiver_alias=agent.orchestrator_alias,
        )
        await self.send(build_message(agent.orchestrator_jid, metadata, payload=artifact))

    async def _reject(self, pending: PendingRequest) -> None:
        agent: GatekeeperAgent = self.agent
        artifact = {
            "response_status": "schema_rejected",
            "validation": {"status": "rejected", "reason": "maximum_schema_revisions_exceeded"},
            "payload": None,
        }
        metadata = MessageMetadata(
            performative="inform",
            message_type=MessageType.ADVISORY_RESPONSE,
            schema_version=MESSAGE_SCHEMA_VERSION,
            run_id=agent.run_id,
            conversation_id=pending.request_id,
            request_id=pending.request_id,
            sender_alias=agent.alias,
            receiver_alias=agent.orchestrator_alias,
        )
        await self.send(build_message(agent.orchestrator_jid, metadata, payload=artifact))

    async def _request_correction(self, pending: PendingRequest, error: AdvisoryValidationError) -> None:
        agent: GatekeeperAgent = self.agent
        metadata = MessageMetadata(
            performative="request",
            message_type=MessageType.CORRECTION_REQUEST,
            schema_version=MESSAGE_SCHEMA_VERSION,
            run_id=agent.run_id,
            conversation_id=pending.request_id,
            request_id=pending.request_id,
            sender_alias=agent.alias,
            receiver_alias=pending.expected_agent_alias,
        )
        payload = {
            "reason": error.reason,
            "detail": error.detail,
            "legal_action_ids": list(pending.legal_action_ids),
        }
        await self.send(build_message(agent.plan_maker_jid, metadata, payload=payload))


class GatekeeperAgent(Agent):
    """Validates advisory communication; never validates final NASimEmu actions."""

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        alias: str,
        run_id: str,
        orchestrator_alias: str,
        orchestrator_jid: str,
        plan_maker_alias: str,
        plan_maker_jid: str,
        max_schema_revisions: int,
        enable_disconnect_detection: bool = False,
    ):
        super().__init__(jid, password)
        self.alias = alias
        self.run_id = run_id
        self.orchestrator_alias = orchestrator_alias
        self.orchestrator_jid = orchestrator_jid
        self.plan_maker_alias = plan_maker_alias
        self.plan_maker_jid = plan_maker_jid
        self.max_schema_revisions = max_schema_revisions
        self.enable_disconnect_detection = enable_disconnect_detection

        self.pending: dict[str, PendingRequest] = {}
        self.failure: Exception | None = None

    async def setup(self) -> None:
        disable_reconnect_on_missed_ping(self)
        self.presence.on_subscribe = lambda peer_jid: self.presence.approve_subscription(peer_jid)
        self.presence.set_available()
        if self.enable_disconnect_detection:
            # Distributed mode only (spec: "explicit disconnect... must fail
            # the experiment"): local mode skips this to avoid adding more
            # presence-stanza traffic against the embedded XMPP server than
            # necessary (see README's known pyjabber flakiness note).
            (
                self.presence.on_available,
                self.presence.on_unavailable,
                self.expect_peer_disconnect,
            ) = make_disconnect_detector(self, self.orchestrator_jid)
            self.presence.subscribe(self.orchestrator_jid)

        self.add_behaviour(
            ReadyCheckResponderBehaviour(
                self.run_id, self.alias, model_version="gatekeeper-v1", resolved_device="n/a"
            ),
            message_type_template(MessageType.READY_CHECK),
        )
        self.add_behaviour(StopExperimentBehaviour(self.run_id), message_type_template(MessageType.STOP_EXPERIMENT))
        self.add_behaviour(AdvisoryRequestBehaviour(), message_type_template(MessageType.ADVISORY_REQUEST))
        self.add_behaviour(AdvisoryResponseBehaviour(), message_type_template(MessageType.ADVISORY_RESPONSE))
