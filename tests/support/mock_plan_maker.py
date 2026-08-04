"""Deterministic mock Plan Maker agent for Gatekeeper integration tests.

Not part of the shipped ``marla`` package -- the real Plan Maker (frozen
RAG + LM backend) lands in Milestones 8-9. This double only needs to be
deterministic and support three strategies exercising the Gatekeeper's full
accept / correct / exhaust-and-reject paths.
"""

from __future__ import annotations

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour

from marla.agents.lifecycle_behaviours import (
    ReadyCheckResponderBehaviour,
    StopExperimentBehaviour,
    message_type_template,
)
from marla.messaging.builders import build_message
from marla.messaging.parsers import MessageValidationError, parse_json_body, parse_metadata
from marla.messaging.schemas import MESSAGE_SCHEMA_VERSION, MessageMetadata, MessageType


def _build_response_body(strategy: str, run_id: str, request_id: str, legal_action_ids: list[str], attempt: int) -> dict:
    def valid_scores() -> dict[str, float]:
        return {action_id: 0.5 for action_id in legal_action_ids}

    def invalid_scores() -> dict[str, float]:
        # Drop the last action id: triggers action_id_coverage_mismatch.
        return {action_id: 0.5 for action_id in legal_action_ids[:-1]}

    if strategy == "always_valid":
        scores = valid_scores()
    elif strategy == "always_invalid":
        scores = invalid_scores()
    elif strategy == "invalid_then_valid":
        scores = invalid_scores() if attempt == 1 else valid_scores()
    else:
        raise ValueError(f"Unknown mock Plan Maker strategy: {strategy!r}")

    return {
        "schema_version": MESSAGE_SCHEMA_VERSION,
        "run_id": run_id,
        "request_id": request_id,
        "scores": scores,
        "model_version": "mock-plan-maker-v1",
        "prompt_version": "mock-prompt-v1",
        "knowledge_version": "mock-knowledge-v1",
        "inference_latency_ms": 1.0,
        "retrieved_rule_ids": [],
    }


class _AdvisoryRequestHandler(CyclicBehaviour):
    async def run(self) -> None:
        message = await self.receive(timeout=None)
        if message is None:
            return
        agent: MockPlanMakerAgent = self.agent
        try:
            metadata = parse_metadata(message)
            body = parse_json_body(message)
        except MessageValidationError:
            return

        legal_action_ids = [a["action_id"] for a in body["legal_actions"]]
        agent.attempt_count[metadata.request_id] = 1
        await self._reply(agent, metadata.request_id, legal_action_ids, attempt=1)

    async def _reply(self, agent: "MockPlanMakerAgent", request_id: str, legal_action_ids: list[str], attempt: int) -> None:
        response_body = _build_response_body(agent.strategy, agent.run_id, request_id, legal_action_ids, attempt)
        metadata = MessageMetadata(
            performative="inform",
            message_type=MessageType.ADVISORY_RESPONSE,
            schema_version=MESSAGE_SCHEMA_VERSION,
            run_id=agent.run_id,
            conversation_id=request_id,
            request_id=request_id,
            sender_alias=agent.alias,
            receiver_alias=agent.gatekeeper_alias,
        )
        await self.send(build_message(agent.gatekeeper_jid, metadata, payload=response_body))


class _CorrectionRequestHandler(CyclicBehaviour):
    async def run(self) -> None:
        message = await self.receive(timeout=None)
        if message is None:
            return
        agent: MockPlanMakerAgent = self.agent
        try:
            metadata = parse_metadata(message)
            body = parse_json_body(message)
        except MessageValidationError:
            return

        legal_action_ids = body["legal_action_ids"]
        attempt = agent.attempt_count.get(metadata.request_id, 1) + 1
        agent.attempt_count[metadata.request_id] = attempt

        response_body = _build_response_body(agent.strategy, agent.run_id, metadata.request_id, legal_action_ids, attempt)
        reply_metadata = MessageMetadata(
            performative="inform",
            message_type=MessageType.ADVISORY_RESPONSE,
            schema_version=MESSAGE_SCHEMA_VERSION,
            run_id=agent.run_id,
            conversation_id=metadata.request_id,
            request_id=metadata.request_id,
            sender_alias=agent.alias,
            receiver_alias=agent.gatekeeper_alias,
        )
        await self.send(build_message(agent.gatekeeper_jid, reply_metadata, payload=response_body))


class MockPlanMakerAgent(Agent):
    def __init__(
        self,
        jid: str,
        password: str,
        *,
        alias: str,
        run_id: str,
        gatekeeper_alias: str,
        gatekeeper_jid: str,
        strategy: str = "always_valid",
    ):
        super().__init__(jid, password)
        self.alias = alias
        self.run_id = run_id
        self.gatekeeper_alias = gatekeeper_alias
        self.gatekeeper_jid = gatekeeper_jid
        self.strategy = strategy
        self.attempt_count: dict[str, int] = {}

    async def setup(self) -> None:
        self.presence.on_subscribe = lambda peer_jid: self.presence.approve_subscription(peer_jid)
        self.presence.set_available()
        self.presence.subscribe(self.gatekeeper_jid)

        self.add_behaviour(
            ReadyCheckResponderBehaviour(
                self.run_id, self.alias, model_version="mock-plan-maker-v1", resolved_device="cpu"
            ),
            message_type_template(MessageType.READY_CHECK),
        )
        self.add_behaviour(StopExperimentBehaviour(self.run_id), message_type_template(MessageType.STOP_EXPERIMENT))
        self.add_behaviour(_AdvisoryRequestHandler(), message_type_template(MessageType.ADVISORY_REQUEST))
        self.add_behaviour(_CorrectionRequestHandler(), message_type_template(MessageType.CORRECTION_REQUEST))
