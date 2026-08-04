"""RL-Orchestrator-side plumbing for a single Gatekeeper consultation round-trip.

The RL Orchestrator sends an ``ADVISORY_REQUEST`` and waits synchronously
(no elapsed timeout, spec section 10/14) for the correlated
``ADVISORY_RESPONSE`` -- either an accepted payload or a
``schema_rejected`` artifact after the Gatekeeper exhausts
``max_schema_revisions``. Milestone 7 wires the learned query gate around
this; this module only implements the request/await mechanics, reusable by
tests today and by the query gate later.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Literal

from spade.behaviour import CyclicBehaviour

from marla.messaging.builders import build_message, new_id
from marla.messaging.parsers import MessageValidationError, parse_json_body, parse_metadata
from marla.messaging.schemas import (
    MESSAGE_SCHEMA_VERSION,
    AdvisoryActionDescriptor,
    AdvisoryObjective,
    AdvisoryRequestPayload,
    AdvisoryResponsePayload,
    MessageMetadata,
    MessageType,
)

AdvisoryStatus = Literal["accepted", "schema_rejected"]


@dataclass(frozen=True)
class AdvisoryOutcome:
    status: AdvisoryStatus
    payload: AdvisoryResponsePayload | None
    latency_seconds: float
    request_id: str


class AdvisoryResponseListenerBehaviour(CyclicBehaviour):
    """Resolves the pending future matching an incoming ADVISORY_RESPONSE's request_id.

    The RL Orchestrator ignores direct Plan Maker advisory messages (spec
    section 10) -- this behaviour is only ever registered with a template
    matching messages *from the Gatekeeper*.
    """

    def __init__(self, pending: dict[str, "asyncio.Future"]):
        super().__init__()
        self._pending = pending

    async def run(self) -> None:
        message = await self.receive(timeout=None)
        if message is None:
            return
        try:
            metadata = parse_metadata(message)
            body = parse_json_body(message)
        except MessageValidationError:
            return

        future = self._pending.pop(metadata.request_id, None)
        if future is None or future.done():
            return
        future.set_result(body)


async def send_advisory_request(
    behaviour: CyclicBehaviour,
    *,
    gatekeeper_jid: str,
    gatekeeper_alias: str,
    run_id: str,
    sender_alias: str,
    episode_id: int,
    step: int,
    source_observation_id: str,
    objective: AdvisoryObjective,
    observation: dict,
    legal_actions: list[AdvisoryActionDescriptor],
    pending: dict[str, "asyncio.Future"],
) -> AdvisoryOutcome:
    """Send one advisory request and await its correlated response, indefinitely."""
    request_id = new_id("request")
    request_payload = AdvisoryRequestPayload(
        schema_version=MESSAGE_SCHEMA_VERSION,
        run_id=run_id,
        request_id=request_id,
        episode_id=episode_id,
        step=step,
        source_observation_id=source_observation_id,
        objective=objective,
        observation=observation,
        legal_actions=legal_actions,
    )

    future: asyncio.Future = asyncio.get_event_loop().create_future()
    pending[request_id] = future

    metadata = MessageMetadata(
        performative="request",
        message_type=MessageType.ADVISORY_REQUEST,
        schema_version=MESSAGE_SCHEMA_VERSION,
        run_id=run_id,
        conversation_id=request_id,
        request_id=request_id,
        sender_alias=sender_alias,
        receiver_alias=gatekeeper_alias,
    )
    message = build_message(gatekeeper_jid, metadata, payload=request_payload.model_dump())

    start = time.monotonic()
    await behaviour.send(message)
    body = await future  # no elapsed timeout: spec sections 5, 10, 14
    latency = time.monotonic() - start

    if body.get("response_status") == "schema_rejected":
        return AdvisoryOutcome(status="schema_rejected", payload=None, latency_seconds=latency, request_id=request_id)

    payload = AdvisoryResponsePayload.model_validate(body["payload"])
    return AdvisoryOutcome(status="accepted", payload=payload, latency_seconds=latency, request_id=request_id)
