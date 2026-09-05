"""Dependency-injection interface for Plan Maker inference backends.

The Plan Maker agent depends on this interface, not on any concrete model
library -- ``local_backend.py`` (HF transformers) and ``remote_backend.py``
(API-based, not implemented in v0.1) both satisfy it, and tests can inject
a trivial fake without touching either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BackendResponse:
    raw_text: str
    latency_ms: float
    # Token counts for cost/efficiency accounting (research/aamas2027).
    # None for a backend that can't report them (e.g. remote_backend.py,
    # not implemented in this release) rather than a fabricated 0.
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class PlanMakerBackend(Protocol):
    async def generate(self, prompt: str, legal_action_ids: list[str]) -> BackendResponse:
        """Generate a response scoring every one of ``legal_action_ids``.

        Backends that size their own generation budget (see
        ``local_backend.py``) need the actual action count -- and the
        actual ID strings, not just how many there are, since a small
        model's response is one JSON entry per action ID and longer IDs
        (e.g. ``exploit:host-2-0:e_elasticsearch`` vs ``finish``) cost more
        tokens -- to avoid truncating the response as an episode's legal
        action space grows mid-run.
        """
        ...
