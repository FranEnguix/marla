"""Content-addressed cache for Plan Maker advisory responses (evaluation only).

Key = SHA-256 of a canonical JSON encoding of exactly the inputs that
determine the Plan Maker's prompt (see ``models/prompt.build_prompt``):
the observation summary, the sorted set of legal action IDs, the model
identity, the prompt version, and the knowledge version. Deliberately
excludes ``request_id``/``observation_id`` -- both are confirmed
non-deterministic-or-positional (a fresh UUID and an episode/step counter,
respectively), never a function of the environment state, so including
either would make every key unique and defeat the cache entirely without
adding any safety.

Because the key *is* a content hash of the state, two different ablations
that reach the same observation with the same legal-action set are safe to
share a cache entry (nothing about *how* that state was reached leaks into
the key), and two ablations whose trajectories have already diverged
naturally produce different keys and therefore fresh, independent Plan
Maker calls. No cross-state leakage is possible by construction.

Persisted as append-only JSON Lines so a partially-completed evaluation run
can be resumed without re-querying everything already answered.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def compute_cache_key(
    observation: dict[str, Any],
    legal_action_ids: list[str],
    model_name: str,
    model_revision: str,
    prompt_version: str,
    knowledge_version: str,
) -> str:
    canonical = json.dumps(
        {
            "observation": observation,
            "legal_action_ids": sorted(legal_action_ids),
            "model_name": model_name,
            "model_revision": model_revision,
            "prompt_version": prompt_version,
            "knowledge_version": knowledge_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedAdvisoryResponse:
    status: str  # "accepted" | "schema_rejected"
    scores: dict[str, float] | None
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


class AdvisoryCache:
    """In-memory dict backed by an append-only JSON Lines file."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._entries: dict[str, CachedAdvisoryResponse] = {}
        if self._path.is_file():
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._entries[record["key"]] = CachedAdvisoryResponse(**record["response"])

    def get(self, key: str) -> CachedAdvisoryResponse | None:
        return self._entries.get(key)

    def put(self, key: str, response: CachedAdvisoryResponse) -> None:
        if key in self._entries:
            return
        self._entries[key] = response
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "response": asdict(response)}) + "\n")

    def __len__(self) -> int:
        return len(self._entries)
