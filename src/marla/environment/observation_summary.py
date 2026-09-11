"""A JSON-serializable, visible-only summary of the current observation.

Sent to the Plan Maker as the advisory request's ``observation`` field
(spec section 9) and used by the deterministic RAG retriever to derive
observation flags (spec section 8). Built from
:func:`marla.environment.visible_facts.extract_visible_host_facts` -- the
one authoritative visible-fact extraction layer, also used by
:mod:`marla.environment.action_compatibility` for PPO's action-compatibility
features, so the Plan Maker and PPO_ONLY are guaranteed to see the same
observable facts (see ``tests/test_visible_facts_fairness.py``) even though
they consume them through different representations.

Lists specific discovered facts (which services/processes/OS are
*confirmed present*), not just counts -- a count alone ("2 services
known") gives the Plan Maker nothing to cross-reference against a
specific exploit/privesc action's own required service/OS/process (see
``environment/actions.py``'s ``parameters``), so its confidence scores
can't actually reflect whether a given action's prerequisites are met.
NASimEmu's partial-observability wrapper only ever merges in *positive*
(nonzero) facts and never clears one back to unknown (see
``nasimemu.env.PartiallyObservableWrapper.__update_obs``), so an absent
name here means "not yet confirmed", never "confirmed absent" -- nothing
below claims a service/process/OS is confirmed missing.
"""

from __future__ import annotations

from typing import Any

from nasimemu.nasim.envs.utils import AccessLevel

from marla.environment.nasimemu_adapter import EnvironmentState
from marla.environment.visible_facts import extract_visible_host_facts, extract_visible_network_exploration

_ACCESS_NAMES = {AccessLevel.NONE: "none", AccessLevel.USER: "user", AccessLevel.ROOT: "root"}


def build_observation_summary(state: EnvironmentState) -> dict[str, Any]:
    """Per-host visible state, plus the scenario's fixed capture-target progress.

    NASimEmu's objective is always "gain root access on every sensitive
    (``is_sensitive_target``) host" (spec section 1's ``capture_target``
    objective -- the only supported type); ``sensitive_hosts_total`` and
    ``sensitive_hosts_with_root_access`` make that progress explicit
    rather than requiring the Plan Maker to infer it by scanning every
    host's ``value``/``access`` fields itself.
    """
    facts_by_target = extract_visible_host_facts(state)

    hosts = []
    sensitive_hosts_total = 0
    sensitive_hosts_with_root_access = 0
    for facts in facts_by_target.values():
        has_root = facts.access == AccessLevel.ROOT
        if facts.is_sensitive_target:
            sensitive_hosts_total += 1
            if has_root:
                sensitive_hosts_with_root_access += 1

        hosts.append(
            {
                "target": facts.target_key,
                "access": _ACCESS_NAMES[facts.access],
                "compromised": facts.compromised,
                "reachable": facts.reachable,
                "is_sensitive_target": facts.is_sensitive_target,
                "known_os": sorted(facts.known_os),
                "known_services": sorted(facts.known_services),
                "known_processes": sorted(facts.known_processes),
            }
        )
    # Observable subnet-exploration facts (spec section 16): the same
    # VisibleNetworkExploration source PPO's graph/recurrent-progress
    # features use, so the two consumers can never disagree about which
    # subnets are known or already successfully scanned (see
    # tests/test_visible_facts_fairness.py's PPO/Plan Maker cross-check).
    # Only known subnets appear here -- an undiscovered one is never
    # listed, by construction (it isn't a member of VisibleNetworkExploration
    # .known_subnets at all).
    exploration = extract_visible_network_exploration(state)
    network_exploration = {
        "known_subnets": sorted(exploration.known_subnets),
        "successfully_scanned_subnets": sorted(exploration.successfully_scanned_subnets),
        "known_unscanned_subnets": sorted(exploration.known_unscanned_subnets),
    }

    return {
        "hosts": hosts,
        "sensitive_hosts_total": sensitive_hosts_total,
        "sensitive_hosts_with_root_access": sensitive_hosts_with_root_access,
        "network_exploration": network_exploration,
    }
