"""Build SPADE messages carrying the MARLA envelope (spec section 10)."""

from __future__ import annotations

import json
import uuid
from typing import Any

from spade.message import Message

from marla.messaging.schemas import MessageMetadata


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def build_message(
    to_jid: str,
    metadata: MessageMetadata,
    extra_metadata: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> Message:
    """Build a SPADE ``Message`` with the full MARLA envelope as metadata.

    ``payload``, if given, is JSON-encoded into the message body; envelope
    fields always live in metadata, never in the body, so the Gatekeeper can
    validate correlation/routing without parsing JSON.

    The body is never left empty, even when there is no ``payload``: slixmpp
    only fires its generic ``"message"`` event (the one SPADE's dispatcher
    listens on) for stanzas matching ``message/body`` -- a message that is
    all metadata and no body is silently never delivered to any behaviour,
    on any server, in-process or across a real connection. ``"{}"`` is a
    harmless placeholder (valid, empty JSON) for messages that carry no
    payload, such as the READY_CHECK/READY/STOP_EXPERIMENT handshake.
    """
    message = Message(to=to_jid)
    message.set_metadata("performative", metadata.performative)
    message.set_metadata("message_type", metadata.message_type.value)
    message.set_metadata("schema_version", metadata.schema_version)
    message.set_metadata("run_id", metadata.run_id)
    message.set_metadata("conversation_id", metadata.conversation_id)
    message.set_metadata("request_id", metadata.request_id)
    message.set_metadata("sender_alias", metadata.sender_alias)
    message.set_metadata("receiver_alias", metadata.receiver_alias)

    if extra_metadata:
        for key, value in extra_metadata.items():
            message.set_metadata(key, value)

    message.thread = metadata.conversation_id
    message.body = json.dumps(payload) if payload is not None else "{}"

    return message
