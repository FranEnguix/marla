r"""Evidence-aware action features and episode-local host-action memory.

Two optional, independently switchable per-action inputs to the ordinary-
action scorer (``policy.action_evidence`` / ``policy.action_history``, see
:class:`marla.learning.base_scorer.BaseActionScorer`):

``c_t(h, a)`` -- **evidence features** (:data:`EVIDENCE_FEATURE_DIM`).
:mod:`marla.environment.action_compatibility` already tells the policy
whether a candidate's required service/process/OS has been *positively*
observed on its target. What it cannot express is whether the absence of a
required fact means "not looked at yet" or "looked at, and not there", nor
how much a scan would still reveal. These features add exactly that:

- whether the target's service / OS / process information is *available*
  (a successful scan of that kind -- or a successful exploit, which NASimEmu
  answers with the host's services and OS -- has been observed, or the
  observation already lists such facts);
- whether a required service/process was **not found by a completed scan**.
  NASimEmu's ``ServiceScan``/``ProcessScan`` return the host's complete
  service/process vector, so once that information is available a required
  name missing from it is a genuine negative observation -- never inferred
  from silence (no scan) and never read from the simulator's hidden host
  configuration;
- the fraction of an exploit/privesc's prerequisites that are confirmed,
  still unknown, and whether any is contradicted;
- for scans: whether the information this scan returns is already
  available, and the fraction of the target's exploits/privescs whose
  prerequisite of that kind is still unknown (a visible proxy for the scan's
  information value).

``m_t(h, a)`` -- **history features** (:data:`HISTORY_FEATURE_DIM`): an
episode-local record of this exact action against this exact host (keyed by
the stable ``action_id``, which names both), and the observable outcome of
its previous attempts. Failing ``e_drupal`` on host A never changes host B's
``e_drupal`` row; there is no global per-exploit counter (the per-name
failure set kept here feeds a metric only, never the policy).

Nothing here masks, penalizes or forbids anything: both blocks are plain
inputs whose weights PPO learns. A matching service with a previous
stochastic failure stays a normal candidate.

**Observability.** Inputs are restricted to (a) the visible host rows
(:class:`marla.environment.visible_facts.VisibleHostFacts`), (b) MARLA's own
visible ``successfully_scanned_subnets`` record, and (c) the observable
result of the agent's own previous actions: NASimEmu's observation carries
an auxiliary row ``[success, connection_error, permission_error,
undefined_error]`` for the action just taken (``nasim.envs.observation``),
which :func:`observed_action_outcome` reads verbatim. ``undefined_error`` is
NASimEmu's random-failure signal (the action's success draw failed); a
failure with no flag set means the host did not meet the action's
preconditions -- both are what the observation itself reports, not hidden
causes. No simulator-only state (true services, success probabilities,
sensitivity of unexploited hosts) is read.

**Replay.** Both matrices are computed once per decision from the
pre-action state and stored on the ``StepRecord``; PPO replay reuses those
exact tensors (see :mod:`marla.learning.ppo`). The memory is owned by one
``RolloutCollector`` (one environment stream), reset at every episode start,
and only updated after an action's transition, so it never contains future
information.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from marla.environment.actions import ActionDescriptor, parse_host_target_key
from marla.environment.nasimemu_adapter import EnvironmentState, TransitionResult
from marla.environment.visible_facts import VisibleHostFacts

# Index : meaning (exploit/privesc = "requirement actions"; every row of a
# targetless action, i.e. FINISH, is all zeros).
#
#  0  service_information_available  -- target's services have been revealed
#  1  os_information_available       -- target's OS has been revealed
#  2  process_information_available  -- target's processes have been revealed
#  3  required_service_not_found     -- exploit only: services revealed, required one absent
#  4  required_process_not_found     -- privesc only: processes revealed, required one absent
#  5  prerequisite_confirmed_fraction  -- requirement actions: confirmed / applicable prerequisites
#  6  prerequisite_unknown_fraction    -- requirement actions: unknown / applicable prerequisites
#  7  prerequisite_contradicted        -- requirement actions: any prerequisite observed absent/different
#  8  scan_information_available     -- scans only: what this scan returns is already available
#  9  scan_unresolved_fraction       -- scans only: share of the target's requirement actions whose
#                                       prerequisite of this scan's kind is still unknown
EVIDENCE_FEATURE_DIM = 10

#  0  attempted_before
#  1  attempt_count                  -- min(n, 4) / 4
#  2  unproductive_attempt_count     -- attempts that changed nothing visible, min(k, 4) / 4
#  3  last_attempt_succeeded         -- auxiliary-row success flag of the latest attempt
#  4  last_attempt_productive        -- the latest attempt changed the visible observation
#  5  recency                        -- 1 / (steps since the latest attempt)
#  6  target_changed_since_last_attempt      -- target's visible facts differ from right after it
#  7  new_target_information_since_last_attempt -- target's known services/OS/processes grew since
#  8  last_failure_connection_error  -- the latest attempt failed with this observable category ...
#  9  last_failure_permission_error
# 10  last_failure_stochastic        -- ... NASimEmu's undefined_error (random failure)
# 11  last_failure_unspecified       -- ... failed with no error flag
HISTORY_FEATURE_DIM = 12
_COUNT_CAP = 4

_REQUIREMENT_ACTION_TYPES = frozenset({"exploit", "privilege_escalation"})
_SCAN_INFORMATION = {"service_scan": "services", "os_scan": "os", "process_scan": "processes"}


class PrerequisiteState(str, Enum):
    CONFIRMED = "confirmed"
    UNKNOWN = "unknown"
    CONTRADICTED = "contradicted"


class FailureCategory(str, Enum):
    """Observable result category of one action (auxiliary observation row)."""

    NONE = "none"  # the action succeeded
    CONNECTION_ERROR = "connection_error"
    PERMISSION_ERROR = "permission_error"
    STOCHASTIC = "stochastic"  # undefined_error: the action's random success draw failed
    UNSPECIFIED = "unspecified"  # failed with no error flag set


_FAILURE_FEATURE_INDEX = {
    FailureCategory.CONNECTION_ERROR: 8,
    FailureCategory.PERMISSION_ERROR: 9,
    FailureCategory.STOCHASTIC: 10,
    FailureCategory.UNSPECIFIED: 11,
}


@dataclass(frozen=True)
class ObservedActionOutcome:
    """What the agent observed about the action it just took."""

    succeeded: bool
    # The visible observation changed: a host row (by address), the set of
    # visible hosts, or the successfully-scanned-subnet record.
    productive: bool
    failure_category: FailureCategory


@dataclass(frozen=True)
class SelectedActionContext:
    """Raw (unnormalized) evidence/history summary of the selected action,
    recorded per decision for decisions.csv metrics -- never a policy input."""

    evidence_status: str  # PrerequisiteState value, or "not_applicable"
    required_service_status: str | None  # confirmed/unknown/contradicted; None without a service requirement
    service_information_available: bool | None  # None for a targetless action
    scan_information_available: bool | None  # scans only
    prior_attempts: int
    prior_unproductive_attempts: int
    last_attempt_productive: bool | None  # None if never attempted
    target_changed_since_last_attempt: bool | None  # None if never attempted
    # Exploit/privesc only: the same-named action failed unproductively on a
    # different host earlier this episode (different_host_retry_rate).
    failed_on_other_host: bool | None

    @property
    def unproductive_repeat(self) -> bool:
        """Repeat of an attempt that changed nothing, with nothing changed on its target since."""
        return (
            self.prior_attempts > 0
            and self.last_attempt_productive is False
            and self.target_changed_since_last_attempt is False
        )


def observed_action_outcome(
    before: EnvironmentState,
    transition: TransitionResult,
    facts_before: dict[str, VisibleHostFacts],
    facts_after: dict[str, VisibleHostFacts],
) -> ObservedActionOutcome:
    """Reads the action result from the auxiliary observation row of the
    post-action observation. An action the adapter absorbed as invalid
    (``info["invalid_action"]``) leaves the previous observation -- and its
    stale auxiliary row -- in place, so it is reported as an unspecified
    failure instead. A state without a raw observation (a synthetic adapter)
    falls back to the same flags in NASimEmu's ``ActionResult.info()``,
    which carries no random-failure flag."""
    after = transition.state
    assert after is not None, "FINISH has no observable action outcome"
    productive = (
        facts_after != facts_before
        or after.successfully_scanned_subnets != before.successfully_scanned_subnets
    )
    if transition.info.get("invalid_action"):
        return ObservedActionOutcome(False, productive, FailureCategory.UNSPECIFIED)
    if after.raw_observation is not None:
        aux = after.raw_observation[-1]
    else:
        info = transition.info
        aux = (
            info.get("success", False), info.get("connection_error", False),
            info.get("permission_error", False), False,
        )
    if bool(aux[0]):
        return ObservedActionOutcome(True, productive, FailureCategory.NONE)
    if bool(aux[1]):
        category = FailureCategory.CONNECTION_ERROR
    elif bool(aux[2]):
        category = FailureCategory.PERMISSION_ERROR
    elif bool(aux[3]):
        category = FailureCategory.STOCHASTIC
    else:
        category = FailureCategory.UNSPECIFIED
    return ObservedActionOutcome(False, productive, category)


def _action_name(descriptor: ActionDescriptor) -> str:
    parts = descriptor.action_id.split(":")
    return parts[2] if len(parts) >= 3 else descriptor.action_type


def _known_information(facts: VisibleHostFacts) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    return facts.known_services, facts.known_os, facts.known_processes


@dataclass
class _HostInformation:
    services: bool = False
    os: bool = False
    processes: bool = False


@dataclass
class _ActionHistory:
    attempts: int = 0
    unproductive_attempts: int = 0
    last_step: int = 0
    last_succeeded: bool = False
    last_productive: bool = False
    last_failure: FailureCategory = FailureCategory.NONE
    target_facts_after_last_attempt: VisibleHostFacts | None = None


class EpisodeActionMemory:
    """Episode-local evidence and host-action history for one environment stream."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._information: dict[str, _HostInformation] = {}
        self._history: dict[str, _ActionHistory] = {}
        # (action_type, name) -> target keys where it failed without changing anything (metric only).
        self._unproductive_failures: dict[tuple[str, str], set[str]] = {}

    # --- evidence ----------------------------------------------------------------------------------------

    def information_available(self, target_key: str, facts: VisibleHostFacts) -> tuple[bool, bool, bool]:
        """``(services, os, processes)`` revealed for this host. NASimEmu never
        clears a confirmed fact, so a non-empty known set also counts (e.g.
        the services and OS a successful exploit returns)."""
        info = self._information.get(target_key, _HostInformation())
        return (
            info.services or bool(facts.known_services),
            info.os or bool(facts.known_os),
            info.processes or bool(facts.known_processes),
        )

    def prerequisite_states(
        self, descriptor: ActionDescriptor, facts: VisibleHostFacts
    ) -> dict[str, PrerequisiteState]:
        """State of each prerequisite (``service``/``process``/``os``) an
        exploit/privesc names; empty for every other action type."""
        if descriptor.action_type not in _REQUIREMENT_ACTION_TYPES:
            return {}
        services_info, os_info, processes_info = self.information_available(facts.target_key, facts)
        states: dict[str, PrerequisiteState] = {}
        for kind, known, available in (
            ("service", facts.known_services, services_info),
            ("process", facts.known_processes, processes_info),
            ("os", facts.known_os, os_info),
        ):
            required = descriptor.parameters.get(kind)
            if not required:
                continue
            if required in known:
                states[kind] = PrerequisiteState.CONFIRMED
            elif available:
                states[kind] = PrerequisiteState.CONTRADICTED
            else:
                states[kind] = PrerequisiteState.UNKNOWN
        return states

    def evidence_matrix(
        self,
        descriptors: list[ActionDescriptor],
        facts_by_target: dict[str, VisibleHostFacts],
        state: EnvironmentState,
    ) -> Tensor:
        """``(len(descriptors), EVIDENCE_FEATURE_DIM)``, in descriptor order."""
        prerequisites = [
            self.prerequisite_states(d, facts_by_target[d.target_key]) if d.target_key in facts_by_target else {}
            for d in descriptors
        ]
        # Per target and prerequisite kind: (#unknown, #requiring) over its requirement actions.
        unresolved: dict[tuple[str, str], list[int]] = {}
        for d, states in zip(descriptors, prerequisites):
            for kind, prerequisite in states.items():
                counts = unresolved.setdefault((d.target_key, kind), [0, 0])
                counts[0] += prerequisite == PrerequisiteState.UNKNOWN
                counts[1] += 1

        rows = []
        for d, states in zip(descriptors, prerequisites):
            row = [0.0] * EVIDENCE_FEATURE_DIM
            facts = facts_by_target.get(d.target_key) if d.target_key else None
            if facts is not None:
                services_info, os_info, processes_info = self.information_available(d.target_key, facts)
                row[0], row[1], row[2] = float(services_info), float(os_info), float(processes_info)
                if states:
                    row[3] = float(states.get("service") == PrerequisiteState.CONTRADICTED)
                    row[4] = float(states.get("process") == PrerequisiteState.CONTRADICTED)
                    values = list(states.values())
                    row[5] = values.count(PrerequisiteState.CONFIRMED) / len(values)
                    row[6] = values.count(PrerequisiteState.UNKNOWN) / len(values)
                    row[7] = float(PrerequisiteState.CONTRADICTED in values)
                elif d.action_type == "subnet_scan":
                    subnet, _ = parse_host_target_key(d.target_key)
                    row[8] = float(subnet in state.successfully_scanned_subnets)
                elif d.action_type in _SCAN_INFORMATION:
                    row[8] = float({"service_scan": services_info, "os_scan": os_info,
                                    "process_scan": processes_info}[d.action_type])
                    kind = {"service_scan": "service", "os_scan": "os", "process_scan": "process"}[d.action_type]
                    unknown, requiring = unresolved.get((d.target_key, kind), (0, 0))
                    row[9] = unknown / requiring if requiring else 0.0
            rows.append(row)
        if not rows:
            return torch.zeros((0, EVIDENCE_FEATURE_DIM), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    # --- history -----------------------------------------------------------------------------------------

    def history_matrix(
        self, descriptors: list[ActionDescriptor], facts_by_target: dict[str, VisibleHostFacts], step: int
    ) -> Tensor:
        """``(len(descriptors), HISTORY_FEATURE_DIM)``; ``step`` is the
        current decision's episode step index."""
        rows = []
        for d in descriptors:
            row = [0.0] * HISTORY_FEATURE_DIM
            history = self._history.get(d.action_id)
            if history is not None and history.attempts > 0:
                changed, new_information = self._changes_since_last_attempt(history, facts_by_target.get(d.target_key))
                row[0] = 1.0
                row[1] = min(history.attempts, _COUNT_CAP) / _COUNT_CAP
                row[2] = min(history.unproductive_attempts, _COUNT_CAP) / _COUNT_CAP
                row[3] = float(history.last_succeeded)
                row[4] = float(history.last_productive)
                row[5] = 1.0 / max(step - history.last_step, 1)
                row[6] = float(changed)
                row[7] = float(new_information)
                if history.last_failure in _FAILURE_FEATURE_INDEX:
                    row[_FAILURE_FEATURE_INDEX[history.last_failure]] = 1.0
            rows.append(row)
        if not rows:
            return torch.zeros((0, HISTORY_FEATURE_DIM), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    @staticmethod
    def _changes_since_last_attempt(
        history: _ActionHistory, facts_now: VisibleHostFacts | None
    ) -> tuple[bool, bool]:
        before = history.target_facts_after_last_attempt
        if before is None or facts_now is None:
            return facts_now != before, False
        changed = facts_now != before
        new_information = any(
            now > then for now, then in zip(_known_information(facts_now), _known_information(before))
        )
        return changed, new_information

    def selected_action_context(
        self,
        descriptor: ActionDescriptor,
        facts_by_target: dict[str, VisibleHostFacts],
        evidence_row: Tensor,
    ) -> SelectedActionContext:
        """Metric summary of the selected action, from the same pre-action
        memory and evidence row the policy was given."""
        facts = facts_by_target.get(descriptor.target_key) if descriptor.target_key else None
        states = self.prerequisite_states(descriptor, facts) if facts is not None else {}
        if not states:
            evidence_status = "not_applicable"
        elif PrerequisiteState.CONTRADICTED in states.values():
            evidence_status = PrerequisiteState.CONTRADICTED.value
        elif all(s == PrerequisiteState.CONFIRMED for s in states.values()):
            evidence_status = PrerequisiteState.CONFIRMED.value
        else:
            evidence_status = PrerequisiteState.UNKNOWN.value

        history = self._history.get(descriptor.action_id)
        attempted = history is not None and history.attempts > 0
        changed = self._changes_since_last_attempt(history, facts)[0] if attempted else None
        failed_elsewhere = None
        if descriptor.action_type in _REQUIREMENT_ACTION_TYPES:
            targets = self._unproductive_failures.get((descriptor.action_type, _action_name(descriptor)), set())
            failed_elsewhere = bool(targets - {descriptor.target_key})
        is_scan = descriptor.action_type in _SCAN_INFORMATION or descriptor.action_type == "subnet_scan"
        return SelectedActionContext(
            evidence_status=evidence_status,
            required_service_status=states["service"].value if "service" in states else None,
            service_information_available=bool(evidence_row[0]) if facts is not None else None,
            scan_information_available=bool(evidence_row[8]) if is_scan and facts is not None else None,
            prior_attempts=history.attempts if history is not None else 0,
            prior_unproductive_attempts=history.unproductive_attempts if history is not None else 0,
            last_attempt_productive=history.last_productive if attempted else None,
            target_changed_since_last_attempt=changed,
            failed_on_other_host=failed_elsewhere,
        )

    # --- update ------------------------------------------------------------------------------------------

    def record(
        self,
        descriptor: ActionDescriptor,
        step: int,
        facts_after: dict[str, VisibleHostFacts],
        outcome: ObservedActionOutcome,
    ) -> None:
        """Fold one completed non-FINISH action into the memory (after its
        transition -- the memory only ever holds past outcomes)."""
        if descriptor.is_finish or descriptor.target_key is None:
            return
        target = descriptor.target_key
        if outcome.succeeded and descriptor.action_type in _SCAN_INFORMATION:
            info = self._information.setdefault(target, _HostInformation())
            setattr(info, _SCAN_INFORMATION[descriptor.action_type], True)

        history = self._history.setdefault(descriptor.action_id, _ActionHistory())
        history.attempts += 1
        history.unproductive_attempts += not outcome.productive
        history.last_step = step
        history.last_succeeded = outcome.succeeded
        history.last_productive = outcome.productive
        history.last_failure = outcome.failure_category
        history.target_facts_after_last_attempt = facts_after.get(target)

        if descriptor.action_type in _REQUIREMENT_ACTION_TYPES and not outcome.succeeded and not outcome.productive:
            key = (descriptor.action_type, _action_name(descriptor))
            self._unproductive_failures.setdefault(key, set()).add(target)
