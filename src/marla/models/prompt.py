"""Plan Maker prompt construction (spec section 9).

The model is only asked to produce the ``{action_id: score}`` mapping --
the surrounding advisory-response envelope (schema/run/request IDs, model
and knowledge versions, latency) is assembled deterministically by the
Plan Maker agent itself, never hallucinated by the LM.
"""

from __future__ import annotations

import json

from marla.knowledge.retriever import KnowledgeRule
from marla.messaging.schemas import AdvisoryActionDescriptor, AdvisoryObjective

PROMPT_VERSION = "nasimemu-plan-maker-v1"

_TEMPLATE = """You are the MARLA Plan Maker. You provide advisory confidence scores for
currently legal NASimEmu actions. You do not execute actions.

NASIMEMU KNOWLEDGE
{retrieved_rules}

EXPERIMENT OBJECTIVE
{objective}

CURRENT VISIBLE OBSERVATION
{observation}

LEGAL ACTIONS
{legal_actions}

Assign an independent confidence in the inclusive range [0,1] to every
supplied action ID. Scores are not required to sum to one. Do not add or
omit action IDs. Return strict JSON mapping each action ID to its
confidence score, and no explanatory text."""


def build_prompt(
    retrieved_rules: list[KnowledgeRule],
    objective: AdvisoryObjective,
    observation: dict,
    legal_actions: list[AdvisoryActionDescriptor],
) -> str:
    rules_text = "\n".join(f"- {rule.text}" for rule in retrieved_rules) or "(no specific rules retrieved)"
    objective_text = json.dumps({"type": objective.type, "description": objective.description})
    observation_text = json.dumps(observation)
    legal_actions_text = json.dumps([action.model_dump() for action in legal_actions])

    return _TEMPLATE.format(
        retrieved_rules=rules_text,
        objective=objective_text,
        observation=observation_text,
        legal_actions=legal_actions_text,
    )


def build_correction_prompt(original_prompt: str, reason: str, detail: str, legal_action_ids: list[str]) -> str:
    # Ending on a bare JSON array of IDs invites a small model to echo that
    # array back verbatim instead of transforming it into a scored mapping
    # (observed in practice with Qwen2.5-0.5B-Instruct) -- an explicit
    # negative instruction plus a worked example, with the actual directive
    # last (not the array), corrects this far more reliably.
    example_id = legal_action_ids[0] if legal_action_ids else "example_action_id"
    example_mapping = json.dumps({example_id: 0.5})
    return (
        f"{original_prompt}\n\n"
        f"Your previous response was rejected: {reason} ({detail}).\n"
        f"These are the action IDs you must score, exactly, no more and no fewer: "
        f"{json.dumps(legal_action_ids)}.\n"
        f"Respond with ONLY a JSON object mapping each of those action IDs to a confidence "
        f"score in [0,1], for example: {example_mapping}\n"
        "Do not return a JSON array of action IDs -- return the scored mapping."
    )
