"""A JSON-serializable, visible-only summary of the current observation.

Sent to the Plan Maker as the advisory request's ``observation`` field
(spec section 9) and used by the deterministic RAG retriever to derive
observation flags (spec section 8). Built from exactly the same visible
``HostVector`` fields used for graph node features (spec section 12) --
no hidden simulator state.
"""

from __future__ import annotations

from typing import Any

from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel

from marla.environment.actions import host_target_key
from marla.environment.nasimemu_adapter import EnvironmentState

_ACCESS_NAMES = {AccessLevel.NONE: "none", AccessLevel.USER: "user", AccessLevel.ROOT: "root"}


def build_observation_summary(state: EnvironmentState) -> dict[str, Any]:
    hosts = []
    for row, address in zip(state.host_rows, state.host_addresses):
        host = HostVector(row)
        hosts.append(
            {
                "target": host_target_key(*address),
                "access": _ACCESS_NAMES[AccessLevel(int(host.access))],
                "reachable": bool(host.reachable),
                "known_services": int(sum(host.services.values())),
                "known_processes": int(sum(host.processes.values())),
                "is_objective_target": bool(host.value > 0),
            }
        )
    return {"hosts": hosts}
