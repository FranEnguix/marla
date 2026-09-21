"""Message envelope schema shared by every SPADE message MARLA sends.

Per spec section 10, every message carries this metadata regardless of
payload: ``performative``, ``message_type``, ``schema_version``, ``run_id``,
``conversation_id``, ``request_id``, ``sender_alias``, ``receiver_alias``.
Advisory request/response payload schemas (section 9) are added in
Milestone 6/7; this module covers the envelope plus the five lifecycle
message types that may travel directly between the RL Orchestrator and any
participant (section 4/5): ``READY_CHECK``, ``READY``, ``START_EXPERIMENT``,
``STOP_EXPERIMENT``, ``EXPERIMENT_FAILED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

MESSAGE_SCHEMA_VERSION = "1.1"  # AdvisoryRequestPayload: legal_actions -> candidate_actions, + consulted_subnet/global_candidate_action_count (subnet-scoped consultation)


class MessageType(str, Enum):
    READY_CHECK = "READY_CHECK"
    READY = "READY"
    START_EXPERIMENT = "START_EXPERIMENT"
    STOP_EXPERIMENT = "STOP_EXPERIMENT"
    EXPERIMENT_FAILED = "EXPERIMENT_FAILED"
    ADVISORY_REQUEST = "ADVISORY_REQUEST"
    ADVISORY_RESPONSE = "ADVISORY_RESPONSE"
    CORRECTION_REQUEST = "CORRECTION_REQUEST"


# Lifecycle messages travel directly RL Orchestrator <-> participant;
# advisory messages must always be routed through the Gatekeeper (Milestone 6).
LIFECYCLE_MESSAGE_TYPES = frozenset(
    {
        MessageType.READY_CHECK,
        MessageType.READY,
        MessageType.START_EXPERIMENT,
        MessageType.STOP_EXPERIMENT,
        MessageType.EXPERIMENT_FAILED,
    }
)


@dataclass(frozen=True)
class MessageMetadata:
    """The envelope fields required on every MARLA SPADE message.

    ``message_id`` is deliberately NOT something callers set when
    constructing this object to pass into ``build_message()`` -- it always
    defaults to ``""`` there, since ``build_message()`` mints a fresh one
    itself for every call (see ``marla.messaging.builders`` and
    ``marla.messaging.telemetry``'s module docstring for why: it is the
    correlation key between a message's "sent" and "handled" telemetry
    events, including across retries that reuse the same ``request_id``).
    It is only ever meaningful on the ``MessageMetadata`` returned by
    ``parse_metadata()`` for an actually-received message, where it carries
    the real value the sender generated.
    """

    performative: str
    message_type: MessageType
    schema_version: str
    run_id: str
    conversation_id: str
    request_id: str
    sender_alias: str
    receiver_alias: str
    message_id: str = ""


class MessageValidationError(Exception):
    """Raised when a received message's envelope is missing or malformed."""


# --- Advisory request/response payloads (spec section 9) -------------------


class AdvisoryObjective(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    description: str


class AdvisoryActionDescriptor(BaseModel):
    """The Plan Maker's view of a legal action: public fields only."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    type: str
    target: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class AdvisoryRequestPayload(BaseModel):
    """Subnet-scoped consultation (schema 1.1+): ``candidate_actions`` is the
    CONSULTED subset only (one subnet's actions plus FINISH), never the
    full global candidate set -- renamed from ``legal_actions`` because that
    name implied completeness this payload no longer has. ``consulted_subnet``/
    ``global_candidate_action_count`` are accounting-only fields for the
    Plan Maker's prompt and MARLA's own metrics; they are never used to
    reconstruct or validate the global action set, which this payload does
    not carry at all (sending it would defeat the whole point of scoping).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    run_id: str
    request_id: str
    episode_id: int
    step: int
    source_observation_id: str
    objective: AdvisoryObjective
    observation: dict[str, Any]
    candidate_actions: list[AdvisoryActionDescriptor]
    consulted_subnet: int | None = None
    global_candidate_action_count: int


class AdvisoryResponsePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    run_id: str
    request_id: str
    scores: dict[str, float]
    model_version: str
    prompt_version: str
    knowledge_version: str
    inference_latency_ms: float
    retrieved_rule_ids: list[str] = Field(default_factory=list)
    # Token accounting (research/aamas2027) -- None for a backend that
    # can't report them (e.g. the not-yet-implemented remote backend).
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    @field_validator("scores")
    @classmethod
    def _scores_in_unit_interval(cls, value: dict[str, float]) -> dict[str, float]:
        for action_id, score in value.items():
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"score for {action_id!r} must be numeric, got {type(score).__name__}")
            if not (0.0 <= float(score) <= 1.0):
                raise ValueError(f"score for {action_id!r} out of [0,1]: {score}")
        return value
