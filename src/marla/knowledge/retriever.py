"""Deterministic lightweight RAG over the static NASimEmu knowledge base (spec section 8).

No vector database, no embeddings, no learned retrieval: rules are matched
by evaluating fixed boolean predicates over the request's visible
``observation`` summary and the current legal action types, and returned in
the knowledge file's own stable order. Nothing here changes across runs
given the same inputs, and nothing here is learned or updated from
previous MARLA runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class KnowledgeRule:
    id: str
    observation_flags: tuple[str, ...]
    legal_action_types: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class KnowledgeBase:
    version: str
    rules: tuple[KnowledgeRule, ...]


def resolve_knowledge_path(path: str) -> Path:
    """Resolve a ``package://marla/...`` or plain filesystem knowledge path reference."""
    if path.startswith("package://"):
        relative = path[len("package://") :]
        parts = relative.split("/", 1)
        if len(parts) != 2 or parts[0] != "marla":
            raise ValueError(f"Unsupported package:// reference: {path!r}")
        with resources.as_file(resources.files("marla") / parts[1]) as resolved:
            return Path(resolved)
    return Path(path)


def load_knowledge_base(path: str | Path) -> KnowledgeBase:
    resolved = resolve_knowledge_path(str(path)) if isinstance(path, str) else Path(path)
    with resolved.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    rules = tuple(
        KnowledgeRule(
            id=rule["id"],
            observation_flags=tuple(rule.get("triggers", {}).get("observation_flags", [])),
            legal_action_types=tuple(rule.get("triggers", {}).get("legal_action_types", [])),
            text=" ".join(rule["text"].split()),
        )
        for rule in raw["rules"]
    )
    return KnowledgeBase(version=raw["version"], rules=rules)


def compute_observation_flags(observation: dict[str, Any]) -> set[str]:
    """Derive boolean observation flags from the request's visible observation summary.

    ``observation`` is the JSON summary built by
    ``environment.observation_summary.build_observation_summary``: a dict
    with a ``hosts`` list, each host carrying ``access``, ``reachable``, and
    ``known_services``/``known_processes`` counts -- all visible-only.
    """
    hosts = observation.get("hosts", [])
    flags: set[str] = set()

    if any(host.get("reachable") and host.get("known_services", 0) == 0 for host in hosts):
        flags.add("services_unknown")
    if any(host.get("access") in ("user", "root") for host in hosts):
        flags.add("user_access_present")
    if not any(host.get("access") == "root" for host in hosts):
        flags.add("root_access_missing")

    return flags


def retrieve_rules(
    knowledge_base: KnowledgeBase,
    observation: dict[str, Any],
    legal_action_types: set[str],
) -> list[KnowledgeRule]:
    """Deterministic predicate evaluation; returns a stable ordered subset.

    A rule matches when ALL of its ``observation_flags`` are currently
    active (vacuously true if empty) AND at least one of its
    ``legal_action_types`` is present among the current legal actions
    (vacuously true if empty, i.e. the rule applies regardless of type).
    """
    active_flags = compute_observation_flags(observation)

    def matches(rule: KnowledgeRule) -> bool:
        flags_ok = all(flag in active_flags for flag in rule.observation_flags)
        types_ok = not rule.legal_action_types or bool(set(rule.legal_action_types) & legal_action_types)
        return flags_ok and types_ok

    return [rule for rule in knowledge_base.rules if matches(rule)]
