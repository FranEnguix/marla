"""The Plan Maker SPADE agent (spec sections 3.3, 8, 9).

Frozen and advisory: no PPO gradients, no cross-run memory, no hidden
simulator access. It only ever sees what the Gatekeeper forwards -- the
current visible observation, current legal action descriptions, the
configured objective, and the static versioned knowledge base.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

from pydantic import ValidationError
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour

from marla.agents.lifecycle_behaviours import (
    ReadyCheckResponderBehaviour,
    StopExperimentBehaviour,
    disable_reconnect_on_missed_ping,
    message_type_template,
    make_disconnect_detector,
)
from marla.knowledge.retriever import KnowledgeBase, retrieve_rules
from marla.messaging.builders import build_message
from marla.messaging.parsers import MessageValidationError, parse_json_body, parse_metadata
from marla.messaging.schemas import MESSAGE_SCHEMA_VERSION, AdvisoryRequestPayload, MessageMetadata, MessageType
from marla.models.plan_maker_backend import BackendResponse, PlanMakerBackend
from marla.models.prompt import PROMPT_VERSION, build_correction_prompt, build_prompt
from marla.models.response_parser import coerce_scores, extract_json_object

logger = logging.getLogger(__name__)


class AdvisoryRequestHandler(CyclicBehaviour):
    """Gatekeeper -> Plan Maker: retrieve rules, build a prompt, and respond."""

    async def run(self) -> None:
        message = await self.receive(timeout=None)
        if message is None:
            return
        agent: PlanMakerAgent = self.agent

        try:
            metadata = parse_metadata(message)
            body = parse_json_body(message)
            request = AdvisoryRequestPayload.model_validate(body)
        except (MessageValidationError, ValidationError) as exc:
            logger.error("Rejecting malformed advisory request from Gatekeeper: %s", exc)
            return
        if request.run_id != agent.run_id:
            logger.warning("Dropping advisory request with mismatched run_id: %s", request.run_id)
            return

        legal_action_types = {action.type for action in request.legal_actions}
        retrieved = retrieve_rules(agent.knowledge_base, request.observation, legal_action_types)
        legal_action_ids = [action.action_id for action in request.legal_actions]
        prompt = build_prompt(retrieved, request.objective, request.observation, request.legal_actions)

        agent.pending_prompts[request.request_id] = (prompt, legal_action_ids)
        await agent.generate_and_respond(
            self, request.run_id, request.request_id, prompt, [rule.id for rule in retrieved], legal_action_ids
        )


class CorrectionRequestHandler(CyclicBehaviour):
    """Gatekeeper -> Plan Maker: retry with a corrective instruction, same request ID."""

    async def run(self) -> None:
        message = await self.receive(timeout=None)
        if message is None:
            return
        agent: PlanMakerAgent = self.agent

        try:
            metadata = parse_metadata(message)
            body = parse_json_body(message)
        except MessageValidationError as exc:
            logger.error("Dropping malformed correction request: %s", exc)
            return
        if metadata.run_id != agent.run_id:
            return

        stored = agent.pending_prompts.get(metadata.request_id)
        if stored is None:
            logger.warning("No stored prompt for correction request_id=%s", metadata.request_id)
            return
        original_prompt, legal_action_ids = stored
        legal_action_ids = body.get("legal_action_ids", legal_action_ids)

        corrected_prompt = build_correction_prompt(
            original_prompt,
            reason=body.get("reason", ""),
            detail=body.get("detail", ""),
            legal_action_ids=legal_action_ids,
        )
        await agent.generate_and_respond(
            self, agent.run_id, metadata.request_id, corrected_prompt, [], legal_action_ids
        )


class PlanMakerAgent(Agent):
    def __init__(
        self,
        jid: str,
        password: str,
        *,
        alias: str,
        run_id: str,
        gatekeeper_alias: str,
        gatekeeper_jid: str,
        backend: PlanMakerBackend,
        model_version: str,
        knowledge_base: KnowledgeBase,
        resolved_device: str,
        enable_disconnect_detection: bool = False,
        debug_dir: Path | None = None,
    ):
        super().__init__(jid, password)
        self.alias = alias
        self.run_id = run_id
        self.gatekeeper_alias = gatekeeper_alias
        self.gatekeeper_jid = gatekeeper_jid
        self.backend = backend
        self.model_version = model_version
        self.knowledge_base = knowledge_base
        self.resolved_device = resolved_device
        self.enable_disconnect_detection = enable_disconnect_detection
        self.debug_dir = debug_dir
        self._debug_sequence = itertools.count(1)

        self.pending_prompts: dict[str, tuple[str, list[str]]] = {}
        self.failure: Exception | None = None

    async def setup(self) -> None:
        disable_reconnect_on_missed_ping(self)
        self.presence.on_subscribe = lambda peer_jid: self.presence.approve_subscription(peer_jid)
        self.presence.set_available()
        if self.enable_disconnect_detection:
            # Distributed mode only; see GatekeeperAgent for rationale.
            (
                self.presence.on_available,
                self.presence.on_unavailable,
                self.expect_peer_disconnect,
            ) = make_disconnect_detector(self, self.gatekeeper_jid)
            self.presence.subscribe(self.gatekeeper_jid)

        self.add_behaviour(
            ReadyCheckResponderBehaviour(
                self.run_id, self.alias, model_version=self.model_version, resolved_device=self.resolved_device
            ),
            message_type_template(MessageType.READY_CHECK),
        )
        self.add_behaviour(StopExperimentBehaviour(self.run_id), message_type_template(MessageType.STOP_EXPERIMENT))
        self.add_behaviour(AdvisoryRequestHandler(), message_type_template(MessageType.ADVISORY_REQUEST))
        self.add_behaviour(CorrectionRequestHandler(), message_type_template(MessageType.CORRECTION_REQUEST))

    async def generate_and_respond(
        self,
        behaviour: CyclicBehaviour,
        run_id: str,
        request_id: str,
        prompt: str,
        retrieved_rule_ids: list[str],
        legal_action_ids: list[str],
    ) -> None:
        logger.info("Plan Maker: generating response for request_id=%s (prompt: %d chars)...", request_id, len(prompt))
        try:
            response = await self.backend.generate(prompt, legal_action_ids)
            logger.info(
                "Plan Maker: generated response for request_id=%s in %.1fs", request_id, response.latency_ms / 1000
            )
        except Exception:
            # A backend failure (e.g. a prompt exceeding a small model's
            # context window) must not leave the RL Orchestrator waiting
            # forever (spec: no elapsed timeout). Responding with an empty,
            # unparseable result routes this through the same
            # correction/rejection path already used for a malformed LM
            # output, rather than inventing a second failure mode.
            logger.exception("Plan Maker backend failed to generate a response for request_id=%s", request_id)
            response = BackendResponse(raw_text="", latency_ms=0.0)

        self._write_debug_files(request_id, prompt, response.raw_text)

        raw_scores = extract_json_object(response.raw_text) or {}
        scores = coerce_scores(raw_scores)

        payload = {
            "schema_version": MESSAGE_SCHEMA_VERSION,
            "run_id": run_id,
            "request_id": request_id,
            "scores": scores,
            "model_version": self.model_version,
            "prompt_version": PROMPT_VERSION,
            "knowledge_version": self.knowledge_base.version,
            "inference_latency_ms": response.latency_ms,
            "retrieved_rule_ids": retrieved_rule_ids,
        }
        metadata = MessageMetadata(
            performative="inform",
            message_type=MessageType.ADVISORY_RESPONSE,
            schema_version=MESSAGE_SCHEMA_VERSION,
            run_id=run_id,
            conversation_id=request_id,
            request_id=request_id,
            sender_alias=self.alias,
            receiver_alias=self.gatekeeper_alias,
        )
        await behaviour.send(build_message(self.gatekeeper_jid, metadata, payload=payload))

    def _write_debug_files(self, request_id: str, prompt: str, raw_response: str) -> None:
        """``--debug``: dump this attempt's exact query and generated output to disk.

        A schema-rejected response can only really be diagnosed by looking
        at what the model was actually asked and what it actually said --
        the sequence number distinguishes an initial request from its
        correction retries, which share the same request_id.
        """
        if self.debug_dir is None:
            return
        sequence = next(self._debug_sequence)
        prefix = f"{sequence:04d}_{request_id}"
        (self.debug_dir / f"{prefix}_query.txt").write_text(prompt, encoding="utf-8")
        (self.debug_dir / f"{prefix}_response.txt").write_text(raw_response, encoding="utf-8")
