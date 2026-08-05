"""A JSON-serializable, visible-only summary of the current observation.

Sent to the Plan Maker as the advisory request's ``observation`` field
(spec section 9) and used by the deterministic RAG retriever to derive
observation flags (spec section 8). Built from exactly the same visible
``HostVector`` fields used for graph node features (spec section 12) --
no hidden simulator state.

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

from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel

from marla.environment.actions import host_target_key
from marla.environment.nasimemu_adapter import EnvironmentState

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
    hosts = []
    sensitive_hosts_total = 0
    sensitive_hosts_with_root_access = 0
    for row, address in zip(state.host_rows, state.host_addresses):
        host = HostVector(row)
        is_sensitive = bool(host.value > 0)
        has_root = int(host.access) == int(AccessLevel.ROOT)
        if is_sensitive:
            sensitive_hosts_total += 1
            if has_root:
                sensitive_hosts_with_root_access += 1

        hosts.append(
            {
                "target": host_target_key(*address),
                "access": _ACCESS_NAMES[AccessLevel(int(host.access))],
                "compromised": bool(host.compromised),
                "reachable": bool(host.reachable),
                "is_sensitive_target": is_sensitive,
                "known_os": [name for name, present in host.os.items() if present],
                "known_services": [name for name, present in host.services.items() if present],
                "known_processes": [name for name, present in host.processes.items() if present],
            }
        )
    return {
        "hosts": hosts,
        "sensitive_hosts_total": sensitive_hosts_total,
        "sensitive_hosts_with_root_access": sensitive_hosts_with_root_access,
    }
