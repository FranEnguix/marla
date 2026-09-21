"""Subnet-scoped Plan Maker consultation -- the ONE authoritative
implementation of "which subnet gets consulted, which actions/observation
fields it sees" (spec: subnet-scoped consultation).

Both the real SPADE consultation path (``RolloutCollector`` -> ``consult_fn``
-> ``advisory_client``/``Gatekeeper``/``Plan Maker``) and the in-process
:class:`~marla.evaluation.direct_consult.DirectConsultant` evaluation path
are scope-agnostic downstream: neither filters actions/observations itself
(see their own module docstrings). ``RolloutCollector._decide`` is the ONE
call site that builds a :class:`ConsultationScope` from the GLOBAL candidate
set and passes the already-scoped subset to ``consult_fn`` -- so both
consumers automatically stay in lockstep with no duplicated routing logic.

Design (spec sections 2-7):

    A_t            = global candidate-action set (unchanged, every visible
                      host x scenario action list, plus FINISH)
    s_t            = subnet(argmax_i base_logit_i, i != FINISH)  -- DETERMINISTIC,
                      no learned subnet-selection head, no sampling.
    A_t^PM         = {a in A_t : subnet(a) == s_t} U {FINISH}

The Plan Maker never selects s_t and is never shown it as something to
choose -- it is purely a local action expert inside the already-selected
subnet. PPO retains base logits (and therefore the ability to act) over
the FULL global A_t regardless of what was consulted (spec section 16:
sampling still happens over the whole global vector after advice is
scattered back in -- see marla.learning.advice/recurrent_policy.apply_advice).

Edge case: if no non-FINISH candidate exists (e.g. an episode's very last
step, or a degenerate legal-action set), there is no action to route by,
so ``consulted_subnet=None`` and the consulted set collapses to
``[FINISH]`` alone -- never a crash, never an invented subnet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from marla.environment.actions import ActionDescriptor, parse_host_target_key

# Bumped whenever subnet-scoped consultation's DECISION semantics change in
# a way that makes a previously-trained MARLA_FULL policy's learned weights
# (in particular TrustHead, which learned to interpret advice-summary/
# agreement statistics computed over whatever scope was used at training
# time) no longer a faithful match for what the runtime now feeds it --
# even when every tensor SHAPE stays identical (see
# marla.learning.checkpoint's own docstring: this is a decision-semantics
# version, not a weight-shape version, and is checked independently of
# POLICY_REPRESENTATION_VERSION). 1 = the initial subnet-scoped design
# (deterministic single-subnet routing, sparse local advice residual). A
# checkpoint saved before this field existed (None) predates subnet-scoped
# consultation entirely (the old dense/global design) and is never
# compatible with it.
CONSULTATION_SCOPE_VERSION = 1


def extract_action_subnet(action: ActionDescriptor) -> int | None:
    """The subnet number encoded in ``action.target_key`` (``host-X-Y`` -> X).

    FINISH (and any other non-host-targeted action, none exist today) has
    no subnet -- ``None``, never an invented value. Works uniformly for
    every host-targeted action type (service/OS/process scan, subnet scan,
    exploit, privilege escalation) because all of them share the exact
    same ``target_key = host_target_key(subnet, host)`` convention (see
    ``environment/actions.py``'s ``build_legal_actions``) -- there is
    deliberately no per-action-type special case here.
    """
    if action.is_finish or action.target_key is None:
        return None
    subnet, _host = parse_host_target_key(action.target_key)
    return subnet


def select_consulted_subnet(legal_actions: list[ActionDescriptor], base_logits: Tensor) -> int | None:
    """s_t = subnet of the highest-base-logit NON-FINISH candidate.

    Stable tie semantics: ``torch.argmax`` returns the FIRST maximal index
    (PyTorch's documented tie-breaking rule), applied here over only the
    non-FINISH candidates in their existing global order -- the same
    ordering convention every other stable-ID lookup in this codebase
    relies on. FINISH itself is excluded from the argmax candidates by
    construction, so it can never determine s_t (spec: "FINISH must not
    determine s_t").

    Returns ``None`` when no non-FINISH candidate exists at all -- the one
    documented edge case (spec section 3), not an error.
    """
    non_finish_indices = [i for i, a in enumerate(legal_actions) if not a.is_finish]
    if not non_finish_indices:
        return None
    non_finish_logits = base_logits[non_finish_indices]
    local_argmax = int(torch.argmax(non_finish_logits).item())
    i_star = non_finish_indices[local_argmax]
    return extract_action_subnet(legal_actions[i_star])


def compute_routing_margin(legal_actions: list[ActionDescriptor], base_logits: Tensor) -> float | None:
    """How close ``select_consulted_subnet`` is to picking a DIFFERENT
    subnet, given ``base_logits``: the winning non-FINISH action's own
    logit, minus the highest non-FINISH logit belonging to any OTHER
    subnet. A small/negative margin means routing is close to (or already
    past) a tie with a competing subnet; the PPO route-switch diagnostic
    (``learning.ppo``) uses this purely for reporting, never to alter
    training.

    ``None`` when there is no non-FINISH candidate at all, or when every
    non-FINISH candidate belongs to the SAME subnet (nothing to switch
    to) -- both documented "not applicable" cases, never a fabricated 0.
    """
    non_finish = [(i, a) for i, a in enumerate(legal_actions) if not a.is_finish]
    if not non_finish:
        return None
    non_finish_indices = [i for i, _ in non_finish]
    non_finish_logits = base_logits[non_finish_indices]
    local_argmax = int(torch.argmax(non_finish_logits).item())
    winner_index, winner_action = non_finish[local_argmax]
    winner_subnet = extract_action_subnet(winner_action)

    other_subnet_logits = [
        float(base_logits[i].item()) for i, a in non_finish if extract_action_subnet(a) != winner_subnet
    ]
    if not other_subnet_logits:
        return None
    return float(base_logits[winner_index].item()) - max(other_subnet_logits)


@dataclass(frozen=True)
class ConsultationScope:
    """The exact subset of the global candidate set the Plan Maker is asked about.

    ``consulted_indices`` are positions into the GLOBAL ``legal_actions``
    list this scope was built from -- global order is preserved verbatim
    (never reordered), and these indices are the authoritative link back
    to the global action space for sparse-residual scattering (spec
    section 10) and PPO replay (spec section 13).
    """

    consulted_subnet: int | None
    consulted_indices: list[int]
    consulted_actions: list[ActionDescriptor]
    global_candidate_action_count: int


def select_consulted_actions(legal_actions: list[ActionDescriptor], consulted_subnet: int | None) -> ConsultationScope:
    """Every global action whose subnet matches ``consulted_subnet``, plus FINISH exactly once.

    When ``consulted_subnet`` is ``None`` (the no-non-FINISH-candidate edge
    case), only FINISH matches -- the documented ``consulted_subnet=None,
    consulted set=[FINISH]`` fallback, never an empty set (FINISH is
    always present in ``legal_actions`` by construction) and never every
    action leaking through untargeted.
    """
    indices = [
        i for i, a in enumerate(legal_actions)
        if a.is_finish or extract_action_subnet(a) == consulted_subnet
    ]
    return ConsultationScope(
        consulted_subnet=consulted_subnet,
        consulted_indices=indices,
        consulted_actions=[legal_actions[i] for i in indices],
        global_candidate_action_count=len(legal_actions),
    )


def build_scoped_observation(observation: dict[str, Any], consulted_subnet: int | None) -> dict[str, Any]:
    """Two components only (spec section 6):

    - ``global_progress``: a compact, fixed-width summary built ONLY from
      fields already present in ``observation`` (itself already visible-
      only, see ``marla.environment.observation_summary``'s own
      docstring) -- no new fact source, no hidden information.
    - ``local_hosts``: the SAME per-host dicts ``observation["hosts"]``
      already carries (target/access/compromised/reachable/
      is_sensitive_target/known_os/known_services/known_processes),
      filtered down to hosts in ``consulted_subnet`` only. "Not
      confirmed" vs. "confirmed absent" semantics are unchanged -- this
      function only filters which hosts appear, never alters a host's own
      fields.

    When ``consulted_subnet`` is ``None``, ``local_hosts`` is empty (there
    is nothing to reason about locally -- the consulted action set is
    FINISH alone) rather than falling back to showing every host, which
    would silently defeat the whole scoping guarantee.
    """
    sensitive_total = observation["sensitive_hosts_total"]
    sensitive_with_root = observation["sensitive_hosts_with_root_access"]
    exploration = observation["network_exploration"]

    local_hosts = [
        host for host in observation["hosts"]
        if consulted_subnet is not None and parse_host_target_key(host["target"])[0] == consulted_subnet
    ]

    return {
        "global_progress": {
            "visible_sensitive_targets_total": sensitive_total,
            "visible_sensitive_targets_with_root": sensitive_with_root,
            "visible_sensitive_targets_remaining": sensitive_total - sensitive_with_root,
            "known_subnets_count": len(exploration["known_subnets"]),
            "successfully_scanned_subnets_count": len(exploration["successfully_scanned_subnets"]),
            "known_unscanned_subnets_count": len(exploration["known_unscanned_subnets"]),
            "selected_subnet": consulted_subnet,
        },
        "local_hosts": local_hosts,
    }
