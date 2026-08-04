"""Parse the MARLA envelope and JSON payload out of a received SPADE message."""

from __future__ import annotations

import json
from typing import Any

from spade.message import Message

from marla.messaging.schemas import MessageMetadata, MessageType, MessageValidationError

_REQUIRED_METADATA_KEYS = (
    "performative",
    "message_type",
    "schema_version",
    "run_id",
    "conversation_id",
    "request_id",
    "sender_alias",
    "receiver_alias",
)


def parse_metadata(message: Message) -> MessageMetadata:
    values: dict[str, str] = {}
    for key in _REQUIRED_METADATA_KEYS:
        value = message.get_metadata(key)
        if value is None:
            raise MessageValidationError(f"Message is missing required metadata field: {key!r}")
        values[key] = value

    try:
        message_type = MessageType(values["message_type"])
    except ValueError as exc:
        raise MessageValidationError(f"Unknown message_type: {values['message_type']!r}") from exc

    return MessageMetadata(
        performative=values["performative"],
        message_type=message_type,
        schema_version=values["schema_version"],
        run_id=values["run_id"],
        conversation_id=values["conversation_id"],
        request_id=values["request_id"],
        sender_alias=values["sender_alias"],
        receiver_alias=values["receiver_alias"],
    )


def parse_json_body(message: Message) -> dict[str, Any]:
    if not message.body:
        raise MessageValidationError("Message body is empty; expected a JSON payload")
    try:
        payload = json.loads(message.body)
    except json.JSONDecodeError as exc:
        raise MessageValidationError(f"Message body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MessageValidationError("Message body must decode to a JSON object")
    return payload
