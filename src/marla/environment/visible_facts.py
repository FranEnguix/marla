"""One authoritative extraction of visible-only per-host facts.

Both the Plan Maker's observation summary (:mod:`marla.environment.observation_summary`)
and PPO's action-compatibility layer (:mod:`marla.environment.action_compatibility`)
need the *same* facts -- which services/OS/processes have actually been
discovered, current access level, reachability, compromised state -- read
from exactly the same ``HostVector`` fields, so there is exactly one place
that decides what "visible" means (spec: "Do not create a second subtly-
different interpretation of the same observation"). This is also what
makes PPO_ONLY's compatibility features and the Plan Maker's advisory
observation provably see the same facts (see
``tests/test_visible_facts_fairness.py``) -- nothing here is exposed to one
consumer and withheld from the other.

Only ever exposes *positively confirmed* facts: NASimEmu's partial-
observability wrapper merges in nonzero (positive) discoveries and never
clears a bit back to unknown (``nasimemu.env.PartiallyObservableWrapper.__update_obs``),
so an absent name in ``known_services``/``known_processes``/``known_os``
means "not yet confirmed", never "confirmed absent". No hidden simulator
state (true service list, true OS, undiscovered facts) is read here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel
from torch import Tensor

from marla.environment.actions import host_target_key
from marla.environment.nasimemu_adapter import EnvironmentState


@dataclass(frozen=True)
class VisibleHostFacts:
    """Everything about one host that is currently visible to the agent --
    never more. ``known_*`` sets contain only positively-confirmed names;
    absence means "not yet observed", not "confirmed absent" (see module
    docstring).
    """

    target_key: str
    reachable: bool
    compromised: bool
    access: AccessLevel
    known_os: frozenset[str]
    known_services: frozenset[str]
    known_processes: frozenset[str]
    # The host's own visible `value` field (0 for a non-sensitive host,
    # positive for a sensitive/objective-relevant one) -- not hidden
    # scenario metadata, the same field observation_summary.py already
    # reads to compute is_sensitive_target.
    value: float = 0.0

    @property
    def is_sensitive_target(self) -> bool:
        return self.value > 0


def extract_visible_host_facts(state: EnvironmentState) -> dict[str, VisibleHostFacts]:
    """One :class:`VisibleHostFacts` per currently-visible host, keyed by
    the same stable ``target_key`` used throughout :mod:`marla.environment.actions`.
    """
    facts: dict[str, VisibleHostFacts] = {}
    for row, address in zip(state.host_rows, state.host_addresses):
        host = HostVector(row)
        target_key = host_target_key(*address)
        facts[target_key] = VisibleHostFacts(
            target_key=target_key,
            reachable=bool(host.reachable),
            compromised=bool(host.compromised),
            access=AccessLevel(int(host.access)),
            known_os=frozenset(name for name, present in host.os.items() if present),
            known_services=frozenset(name for name, present in host.services.items() if present),
            known_processes=frozenset(name for name, present in host.processes.items() if present),
            value=float(host.value),
        )
    return facts


# FINISH-learnability investigation (Case B: the actor did not represent
# "the objective is already complete" strongly enough for FINISH to be
# preferred once it was). A compact, fixed-width, VISIBLE-ONLY summary of
# global objective progress, fed into the recurrent core alongside the
# graph embedding -- see recurrent_core.RecurrentCore's updated x_t.
#
# "Visible sensitive target" here means a host whose sensitivity has
# actually been confirmed (VisibleHostFacts.is_sensitive_target, i.e. its
# real .value has been revealed) -- NASimEmu's own observation model
# reveals a host's true value upon a successful exploit against it (see
# nasim.envs.state.Observation.from_action_result's
# obs_kwargs["value"] = True on action.is_exploit()), not merely from
# scanning it, so this genuinely tracks "targets the agent has learned
# matter", never a hidden simulator count. An UNDISCOVERED sensitive host
# cannot appear here at all (it isn't in facts_by_target), so
# "every visible sensitive target has ROOT" is explicitly NOT conflated
# with "the true global objective is complete" -- see has_visible_sensitive_target
# below, which is False (not True) whenever nothing sensitive has been
# confirmed yet, precisely to avoid that conflation being read into the
# fraction/without-root fields when there is nothing to summarize.
VISIBLE_TARGET_PROGRESS_DIM = 3
# Backward-compatible alias -- the v3 FINISH-diagnostics phase introduced
# this under the unqualified name before the exploration-progress vector
# (below) existed as a second, independently-ablatable component.
VISIBLE_PROGRESS_DIM = VISIBLE_TARGET_PROGRESS_DIM


def compute_target_progress(facts_by_target: dict[str, VisibleHostFacts]) -> Tensor:
    """``(VISIBLE_TARGET_PROGRESS_DIM,)`` -- ``[has_visible_sensitive_target,
    fraction_visible_sensitive_targets_with_root, any_visible_sensitive_target_without_root]``.

    Deliberately does NOT reveal how many sensitive targets exist in
    total (that would let the policy learn "FINISH exactly when a hidden
    counter hits some threshold") -- only a fraction/boolean over
    whatever has actually been confirmed sensitive so far.
    """
    sensitive = [f for f in facts_by_target.values() if f.is_sensitive_target]
    total = len(sensitive)
    with_root = sum(1 for f in sensitive if f.access == AccessLevel.ROOT)
    has_visible_sensitive_target = float(total > 0)
    fraction_with_root = float(with_root) / total if total > 0 else 0.0
    any_without_root = float(with_root < total)
    return torch.tensor(
        [has_visible_sensitive_target, fraction_with_root, any_without_root], dtype=torch.float32
    )


# Backward-compatible alias for the same reason as VISIBLE_PROGRESS_DIM above.
compute_visible_progress = compute_target_progress


def all_visible_sensitive_targets_rooted(facts_by_target: dict[str, VisibleHostFacts]) -> bool | None:
    """``True``/``False`` when at least one sensitive target is visible;
    ``None`` when none is -- "no visible sensitive targets" must NOT be
    silently treated as strong "all complete" evidence (spec: a caller
    that conflates ``None`` with ``True`` is overclaiming what the
    observation actually proves). Callers that need a plain boolean for a
    diagnostic split should pair this with
    :func:`compute_target_progress`'s ``has_visible_sensitive_target`` or
    check for ``None`` explicitly.
    """
    sensitive = [f for f in facts_by_target.values() if f.is_sensitive_target]
    if not sensitive:
        return None
    return all(f.access == AccessLevel.ROOT for f in sensitive)


# --- Observable subnet-exploration progress (second visible-only signal) ---
#
# Distinguishes "all currently known sensitive targets are rooted" from
# "...AND there is no known subnet exploration frontier left" -- without
# ever exposing unknown subnets or hidden targets (see
# VisibleNetworkExploration's own docstring for the exact semantics and
# NasimEmuAdapter.step()'s tracking of successfully_scanned_subnets for
# how "successfully scanned" is actually detected).
VISIBLE_EXPLORATION_PROGRESS_DIM = 2


@dataclass(frozen=True)
class VisibleNetworkExploration:
    """Everything about subnet-exploration progress that is currently
    visible to the agent -- never more.

    ``known_subnets``: subnet IDs with at least one currently-visible host
    (the same set :mod:`marla.environment.graph` already uses to build
    subnet graph nodes -- NASimEmu's own ``_get_subnets()`` is defined
    identically, over currently-visible host rows only). A subnet the
    simulator has but that has never had a host discovered in it is not a
    member of this set and has zero influence on any property below.

    ``successfully_scanned_subnets``: subnets from which a ``SubnetScan``
    action has *succeeded* at least once (never merely attempted -- see
    ``NasimEmuAdapter.step()``'s tracking). A successful scan that reveals
    zero new subnets still counts; a failed/absorbed-invalid scan never
    does. Always a subset of ``known_subnets`` (scanning targets a host,
    which must already be visible to be a legal target).
    """

    known_subnets: frozenset[int]
    successfully_scanned_subnets: frozenset[int]

    @property
    def known_unscanned_subnets(self) -> frozenset[int]:
        return self.known_subnets - self.successfully_scanned_subnets

    @property
    def fraction_known_subnets_scanned(self) -> float:
        if not self.known_subnets:
            return 0.0
        scanned = self.successfully_scanned_subnets & self.known_subnets
        return len(scanned) / len(self.known_subnets)

    @property
    def known_exploration_frontier_remaining(self) -> bool:
        """``True`` iff at least one currently-*known* subnet has never
        been successfully scanned. Deliberately NOT named
        "network_fully_explored" when ``False`` -- that would overclaim:
        undiscovered subnets may still exist even when every known one has
        been scanned (the POMDP's fundamental ambiguity is preserved, not
        hidden).
        """
        return bool(self.known_unscanned_subnets)


def extract_visible_network_exploration(state: EnvironmentState) -> "VisibleNetworkExploration":
    """The one authoritative visible-exploration extraction, mirroring
    :func:`extract_visible_host_facts` -- reused identically by the
    per-subnet-node GraphSAGE feature, the recurrent progress vector,
    decision/rollout metrics, and (optionally) the Plan Maker's
    observation summary, so none of them can drift into a subtly
    different notion of "known"/"scanned".
    """
    known_subnets = frozenset(addr[0] for addr in state.host_addresses)
    scanned = frozenset(state.successfully_scanned_subnets) & known_subnets
    return VisibleNetworkExploration(known_subnets=known_subnets, successfully_scanned_subnets=scanned)


def compute_exploration_progress(exploration: VisibleNetworkExploration) -> Tensor:
    """``(VISIBLE_EXPLORATION_PROGRESS_DIM,)`` -- ``[fraction_known_subnets_scanned,
    any_known_subnet_unscanned]``. Never reveals the true/total subnet
    count -- only a fraction/boolean over currently-known subnets.
    """
    return torch.tensor(
        [
            float(exploration.fraction_known_subnets_scanned),
            float(exploration.known_exploration_frontier_remaining),
        ],
        dtype=torch.float32,
    )


def visible_progress_dim(target_enabled: bool, exploration_enabled: bool) -> int:
    """Total width of the assembled visible-progress vector for a given
    ablation configuration (spec: v2 / v3-target / v3-full) -- the single
    place that decides this, so ``RecurrentCore`` construction and
    :func:`assemble_visible_progress` can never disagree about the width.
    """
    return (VISIBLE_TARGET_PROGRESS_DIM if target_enabled else 0) + (
        VISIBLE_EXPLORATION_PROGRESS_DIM if exploration_enabled else 0
    )


def assemble_visible_progress(
    facts_by_target: dict[str, VisibleHostFacts],
    exploration: VisibleNetworkExploration,
    target_enabled: bool,
    exploration_enabled: bool,
) -> Tensor:
    """Builds exactly the tensor a policy constructed with this ablation
    configuration expects -- target-progress features first, then
    exploration-progress features (spec section 9's suggested layout),
    each entirely omitted (not zeroed) when its flag is off, so a
    "v3-target" policy's ``RecurrentCore`` never even has weights
    connected to the exploration features it wasn't given -- disabling a
    component removes it from the input, it does not just zero it out.
    """
    parts: list[Tensor] = []
    if target_enabled:
        parts.append(compute_target_progress(facts_by_target))
    if exploration_enabled:
        parts.append(compute_exploration_progress(exploration))
    if not parts:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(parts, dim=0)
