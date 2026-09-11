r"""Fixed-width, scenario-vocabulary-independent relational compatibility
features between a candidate action and the visible facts of its target.

**Why this exists.** ``ActionEncoder``'s old parameter features
(:data:`marla.learning.action_encoder.PARAMETER_FEATURE_DIM`) only encode
*whether* an action has a service/process parameter and *whether* it has an
OS parameter -- never which one, and never whether it matches anything
observed. Combined with GraphSAGE's deliberately scenario-vocabulary-free
node features (:mod:`marla.environment.graph`), this meant every exploit
targeting the same host with the same broad shape (e.g. "requires a
service, requires an OS") received an *identical* action embedding --
``e_elasticsearch`` was indistinguishable from ``e_wp_ninja`` on the same
Windows host, even after both facts were fully observed. This module fixes
that by encoding *relationships* ("the service this action requires has
been observed on this target") instead of *identities* ("this action
requires Elasticsearch") -- so the feature width never depends on how many
services/OSes/processes/exploits/privescs a scenario happens to define.

**Partial observability, protected.** Every input here comes from
:class:`marla.environment.visible_facts.VisibleHostFacts` -- the same
visible-only facts the Plan Maker's observation summary uses (see that
module's docstring) -- nothing hidden is read. Critically, ``unknown`` and
``mismatch`` are never collapsed: NASimEmu's partial-observation wrapper
only ever merges in positive discoveries and never clears one back to
unknown, so "this service has not been observed" must never be treated as
"this service is confirmed absent". For OS specifically, the state machine
is: no OS observed yet -> ``UNKNOWN``; the required OS is among the
observed OS facts -> ``MATCH``; some OS *has* been positively identified
and it is not the required one -> ``MISMATCH`` (real NASimEmu hosts carry
exactly one OS, so a positively-identified OS that differs from the
requirement is a genuine, positively-known contradiction, not a guess).

**Status is a separate, coarser summary of the same facts** (spec section
15), driving decision-metrics/probability-mass diagnostics as well as the
feature vector -- computed by the same function so the three systems
cannot drift apart. Only ``exploit``/``privilege_escalation`` actions get a
real status: they are the only action types with a service/process/OS
*requirement* to confirm or contradict. Scans and FINISH are always
``NOT_APPLICABLE`` -- there is no "compatible service" concept for a scan.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

import torch
from nasimemu.nasim.envs.utils import AccessLevel
from torch import Tensor

from marla.environment.actions import ActionDescriptor
from marla.environment.visible_facts import VisibleHostFacts

# Index : meaning (spec section 5's suggested minimum, plus one addition,
# target_compromised, justified below).
#
#  0  has_service_or_process_requirement -- static, from the action's own parameters
#  1  has_os_requirement                 -- static, from the action's own parameters
#  2  required_service_known_present     -- MATCH only (never inferred from silence)
#  3  required_process_known_present     -- MATCH only
#  4  required_os_known_present          -- MATCH only
#  5  required_os_known_incompatible     -- MISMATCH only (a *different* OS was positively observed)
#  6  target_reachable                   -- visible target precondition
#  7  target_has_user_access             -- visible target context
#  8  target_has_root_access             -- visible target context
#  9  target_compromised                 -- visible target precondition: NASimEmu's own
#     Network.perform_action() requires host_compromised for privilege_escalation
#     (and process_scan) specifically -- a real, positively-observable
#     precondition distinct from access level (a compromised host can still
#     be access=NONE-adjacent before its first successful exploit sets
#     access; compromised and access are tracked as separate HostVector
#     fields in real NASimEmu).
COMPATIBILITY_FEATURE_DIM = 10


class CompatibilityStatus(str, Enum):
    CONFIRMED_COMPATIBLE = "confirmed_compatible"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


_REQUIREMENT_ACTION_TYPES = frozenset({"exploit", "privilege_escalation"})


def _access_features(facts: VisibleHostFacts) -> tuple[float, float, float]:
    return (
        float(facts.reachable),
        float(facts.access == AccessLevel.USER),
        float(facts.access == AccessLevel.ROOT),
    )


def _status_from_parts(
    action_type: str,
    has_requirement: bool,
    has_os: bool,
    required_service_known_present: bool,
    required_process_known_present: bool,
    required_os_known_present: bool,
    required_os_known_incompatible: bool,
    reachable: bool,
    compromised: bool,
    has_service: bool,
    has_process: bool,
) -> CompatibilityStatus:
    """Status derivation (spec section 15), shared verbatim between
    :func:`compute_action_compatibility` (rollout/encoder time, computed
    straight from :class:`~marla.environment.visible_facts.VisibleHostFacts`)
    and :func:`compatibility_status_from_vector` (decisions.csv/probability-
    mass time, reconstructed from the already-stored feature vector) -- the
    one place this logic lives, so the two call sites cannot drift apart.
    Contradiction (a positively observed fact proving incompatibility)
    always wins over unknown.
    """
    if action_type not in _REQUIREMENT_ACTION_TYPES:
        return CompatibilityStatus.NOT_APPLICABLE
    if action_type == "privilege_escalation" and not compromised:
        return CompatibilityStatus.CONTRADICTED
    if not reachable:
        return CompatibilityStatus.CONTRADICTED
    if required_os_known_incompatible:
        return CompatibilityStatus.CONTRADICTED
    if (
        (has_service and not required_service_known_present)
        or (has_process and not required_process_known_present)
        or (has_os and not required_os_known_present)
    ):
        return CompatibilityStatus.UNKNOWN
    return CompatibilityStatus.CONFIRMED_COMPATIBLE


def compute_action_compatibility(
    descriptor: ActionDescriptor, facts_by_target: dict[str, VisibleHostFacts]
) -> tuple[list[float], CompatibilityStatus]:
    """The single authoritative computation -- used identically for the
    action encoder's input features, ``decisions.csv``'s per-decision
    status/match fields, and the compatible-action probability-mass
    diagnostic. Returns ``(feature_vector, status)``; the feature vector
    always has length :data:`COMPATIBILITY_FEATURE_DIM`.
    """
    facts = facts_by_target.get(descriptor.target_key) if descriptor.target_key else None

    service = descriptor.parameters.get("service")
    process = descriptor.parameters.get("process")
    os_req = descriptor.parameters.get("os")
    has_requirement = float(bool(service) or bool(process))
    has_os = float(bool(os_req))

    if descriptor.action_type not in _REQUIREMENT_ACTION_TYPES:
        # FINISH and every scan type: no service/process/OS requirement
        # concept exists for these, so status is always NOT_APPLICABLE.
        # The feature vector still carries whatever visible target context
        # exists (harmless, and lets the encoder see e.g. "this scan
        # target is already compromised" without inventing a second
        # pathway for that information).
        vec = [has_requirement, has_os, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if facts is not None:
            reachable, user, root = _access_features(facts)
            vec[6], vec[7], vec[8] = reachable, user, root
            vec[9] = float(facts.compromised)
        return vec, CompatibilityStatus.NOT_APPLICABLE

    if facts is None:
        # Defensive: a legal action should only ever be built for a
        # currently-visible target (see actions.build_legal_actions), so
        # this shouldn't occur in practice -- if it does, treat it as
        # genuinely unknown rather than guessing either way.
        return [has_requirement, has_os, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], CompatibilityStatus.UNKNOWN

    required_service_known_present = float(bool(service) and service in facts.known_services)
    required_process_known_present = float(bool(process) and process in facts.known_processes)
    required_os_known_present = float(bool(os_req) and os_req in facts.known_os)
    # MISMATCH requires a POSITIVELY-observed, DIFFERENT os -- never
    # inferred from `not required_os_known_present` alone (that would
    # conflate "no OS observed yet" with "the wrong OS was observed").
    required_os_known_incompatible = float(bool(os_req) and bool(facts.known_os) and os_req not in facts.known_os)
    reachable, user_access, root_access = _access_features(facts)
    compromised = float(facts.compromised)

    vec = [
        has_requirement,
        has_os,
        required_service_known_present,
        required_process_known_present,
        required_os_known_present,
        required_os_known_incompatible,
        reachable,
        user_access,
        root_access,
        compromised,
    ]

    status = _status_from_parts(
        descriptor.action_type,
        bool(has_requirement),
        bool(has_os),
        bool(required_service_known_present),
        bool(required_process_known_present),
        bool(required_os_known_present),
        bool(required_os_known_incompatible),
        bool(reachable),
        bool(compromised),
        bool(service),
        bool(process),
    )

    return vec, status


def compute_compatibility_matrix(
    descriptors: list[ActionDescriptor], facts_by_target: dict[str, VisibleHostFacts]
) -> Tensor:
    """``(N, COMPATIBILITY_FEATURE_DIM)`` tensor, one row per descriptor, in
    the same order -- this is exactly what gets stored on ``StepRecord``
    and fed to :class:`marla.learning.action_encoder.ActionEncoder`, so
    rollout collection and PPO replay use the identical tensor (spec
    section 11: PPO replay must never recompute this from live state).
    """
    if not descriptors:
        return torch.zeros((0, COMPATIBILITY_FEATURE_DIM), dtype=torch.float32)
    rows = [compute_action_compatibility(d, facts_by_target)[0] for d in descriptors]
    return torch.tensor(rows, dtype=torch.float32)


def compute_compatibility_statuses(
    descriptors: list[ActionDescriptor], facts_by_target: dict[str, VisibleHostFacts]
) -> list[CompatibilityStatus]:
    """Status only (no tensor) -- for decisions.csv/probability-mass
    diagnostics computed from a policy's probability vector rather than
    the encoder's input.
    """
    return [compute_action_compatibility(d, facts_by_target)[1] for d in descriptors]


def compatibility_status_from_vector(action_type: str, vector: Sequence[float]) -> CompatibilityStatus:
    """Reconstructs :class:`CompatibilityStatus` from an already-computed
    :data:`COMPATIBILITY_FEATURE_DIM`-wide feature vector plus the action's
    type -- for call sites (``decisions.csv`` row-building, the compatible-
    action probability-mass diagnostic) that only have a ``StepRecord``'s
    stored ``compatibility_features`` tensor and ``legal_action_descriptors``
    on hand, not the original ``VisibleHostFacts`` a rollout step saw.
    Drives the *same* :func:`_status_from_parts` branch logic
    :func:`compute_action_compatibility` uses, so decision metrics and
    probability-mass diagnostics can never drift from the encoder's own
    notion of compatibility (spec section 15).
    """
    if len(vector) != COMPATIBILITY_FEATURE_DIM:
        raise ValueError(f"expected a {COMPATIBILITY_FEATURE_DIM}-wide compatibility vector, got {len(vector)}")
    (
        has_requirement,
        has_os,
        required_service_known_present,
        required_process_known_present,
        required_os_known_present,
        required_os_known_incompatible,
        reachable,
        _user_access,
        _root_access,
        compromised,
    ) = (float(v) for v in vector)
    # has_service/has_process aren't separately recoverable from
    # has_requirement (index 0) alone -- but by construction
    # (actions.build_legal_actions) an "exploit" descriptor's parameters
    # only ever carry "service" and a "privilege_escalation" descriptor's
    # only ever carry "process", never both, so the action_type already
    # tells us which one has_requirement refers to.
    has_service = action_type == "exploit" and bool(has_requirement)
    has_process = action_type == "privilege_escalation" and bool(has_requirement)
    return _status_from_parts(
        action_type,
        bool(has_requirement),
        bool(has_os),
        bool(required_service_known_present),
        bool(required_process_known_present),
        bool(required_os_known_present),
        bool(required_os_known_incompatible),
        bool(reachable),
        bool(compromised),
        has_service,
        has_process,
    )
