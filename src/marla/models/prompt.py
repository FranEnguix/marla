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

# Bumped from "nasimemu-plan-maker-v1": subnet-scoped consultation changes
# prompt semantics (a subset of candidates, not the complete legal action
# space) -- see the template's own CONSULTED SUBNET/GLOBAL PROGRESS/LOCAL
# VISIBLE OBSERVATION/CONSULTED CANDIDATE ACTIONS sections below.
PROMPT_VERSION = "nasimemu-plan-maker-v2-subnet-scoped"

_TEMPLATE = """You are the MARLA Plan Maker. You provide advisory confidence scores for
a SUBSET of the currently legal NASimEmu actions. You do not execute actions.

The RL policy that ultimately chooses an action has a LARGER GLOBAL ACTION
SPACE than what is shown to you here -- {global_candidate_action_count} total
candidate action(s) exist right now across the whole visible network. This
consultation covers exactly ONE subnet, CONSULTED SUBNET {consulted_subnet},
plus the global "finish" action: every non-"finish" action below targets a
host in that one subnet. Actions targeting other subnets exist for the
policy but are deliberately NOT shown to you -- never invent or refer to an
action ID that is not listed in CONSULTED CANDIDATE ACTIONS below.

GOAL (always true for this task): gain root access on every host in the
network with "is_sensitive_target": true. GLOBAL PROGRESS below is visible
across the WHOLE network (not just the consulted subnet) and tells you
directly how much of that is already done. The special "finish" action ends
the episode immediately with no further reward -- it is only a good choice
once visible_sensitive_targets_remaining == 0; choosing it earlier ends the
episode without completing the objective.

NASIMEMU KNOWLEDGE
{retrieved_rules}

EXPERIMENT OBJECTIVE
{objective}

GLOBAL PROGRESS (whole visible network)
{global_progress}

LOCAL VISIBLE OBSERVATION (consulted subnet {consulted_subnet} only)
Per-host state below lists only *confirmed* facts: a service, process, or
OS name that is absent from a host's list has simply not been confirmed
yet, never confirmed absent. Hosts outside the consulted subnet are not
shown here.
{local_hosts}

CONSULTED CANDIDATE ACTIONS
This is NOT the complete global action space -- only the consulted
subnet's actions plus "finish". Each exploit/privilege-escalation action's
"parameters" give the specific service/os/process it requires on its
target -- cross-reference these against the target host's own
known_services/known_os/known_processes above before scoring it.
{candidate_actions}

Score every action by how much it helps reach the goal from the current
observation: prefer scanning a reachable host whose services, processes,
or OS are not yet confirmed; prefer an exploit or privilege-escalation
action whose required service/os/process is already confirmed present on
its target over one that isn't; and score "finish" according to the rule
above. Assign an independent confidence in the inclusive range [0,1] to
every supplied action ID, exactly once. Scores are not required to sum to
one. Do not add or omit action IDs, and do not invent or reference any
action ID not listed in CONSULTED CANDIDATE ACTIONS. Return strict JSON
mapping each action ID to its confidence score, and no explanatory text."""


def build_prompt(
    retrieved_rules: list[KnowledgeRule],
    objective: AdvisoryObjective,
    observation: dict,
    candidate_actions: list[AdvisoryActionDescriptor],
    consulted_subnet: int | None,
    global_candidate_action_count: int,
) -> str:
    """``observation`` is already the SCOPED ``{global_progress, local_hosts}``
    dict (see ``marla.environment.consultation_scope.build_scoped_observation``)
    -- this function only renders it, it never filters/scopes anything
    itself, so the SPADE path and DirectConsultant can never disagree
    about what the Plan Maker actually sees.
    """
    rules_text = "\n".join(f"- {rule.text}" for rule in retrieved_rules) or "(no specific rules retrieved)"
    objective_text = json.dumps({"type": objective.type, "description": objective.description})
    global_progress_text = json.dumps(observation["global_progress"])
    local_hosts_text = json.dumps(observation["local_hosts"])
    candidate_actions_text = json.dumps([action.model_dump() for action in candidate_actions])

    return _TEMPLATE.format(
        retrieved_rules=rules_text,
        objective=objective_text,
        global_progress=global_progress_text,
        local_hosts=local_hosts_text,
        candidate_actions=candidate_actions_text,
        consulted_subnet=consulted_subnet,
        global_candidate_action_count=global_candidate_action_count,
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
