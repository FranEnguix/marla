"""Cheap, scalar-only observable state-change deltas between two steps.

Computed directly from the two :class:`EnvironmentState` snapshots already
produced by a normal ``NasimEmuAdapter.step()`` call -- nothing here stores
either snapshot; the caller is expected to discard both immediately after
calling :func:`compute_state_delta` and keep only the returned
:class:`StateDelta` (five small numbers) as part of a decision's metrics.

Relies on NASimEmu's partial-observability invariant (see
``environment/observation_summary``'s module docstring): a host's
``discovered``/service/process facts are only ever added within an episode,
never cleared back to unknown. That invariant does NOT extend to the row
*count*, though: for a scenario whose subnets have a randomized host count
(most bundled ones -- a "ranges of hosts" scenario file), revealing a new
subnet (e.g. via ``subnet_scan``) appends wholly new host rows to
``EnvironmentState.host_rows``/``host_addresses`` mid-episode, already
marked discovered on arrival, rather than flipping a ``discovered`` bit on
a row that was present all along (verified empirically: a 3-host
``before`` snapshot followed by a 6-host ``after`` snapshot after a
``subnet_scan``, the 3 new rows appearing with ``discovered=True``
immediately). A comparison keyed by array *position* silently misses every
one of these -- it only ever iterates positions common to both snapshots.
Every comparison here is therefore keyed by host **address**, matching
snapshots up by identity rather than position, and a host address present
in ``after`` but absent from ``before`` is itself counted as a discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel

from marla.environment.nasimemu_adapter import EnvironmentState


class AccessGain(IntEnum):
    """The best access-level improvement achieved by one compound decision.

    An int enum (not a bool) because a single exploit action can jump
    straight to ROOT (some exploits grant root directly) -- collapsing that
    to a single "access gained" boolean would lose exactly the USER-vs-ROOT
    distinction the spec asks decisions.csv to preserve.
    """

    NONE = 0
    USER = 1
    ROOT = 2


@dataclass(frozen=True)
class StateDelta:
    new_hosts_discovered: int
    new_subnets_discovered: int
    new_services_confirmed: int
    new_processes_confirmed: int
    access_gain: AccessGain


_NO_CHANGE = StateDelta(0, 0, 0, 0, AccessGain.NONE)


def _access_gain(before: int, after: int) -> AccessGain:
    if after <= before:
        return AccessGain.NONE
    return AccessGain.ROOT if after == int(AccessLevel.ROOT) else AccessGain.USER


def compute_state_delta(before: EnvironmentState, after: EnvironmentState | None) -> StateDelta:
    """``after`` is ``None`` for a FINISH decision (no environment step ran)."""
    if after is None:
        return _NO_CHANGE

    before_by_address = dict(zip(before.host_addresses, before.host_rows))
    after_by_address = dict(zip(after.host_addresses, after.host_rows))

    new_hosts = 0
    new_services = 0
    new_processes = 0
    subnets_before: set[int] = set()
    subnets_after: set[int] = set()
    best_gain = AccessGain.NONE

    for address, before_row in before_by_address.items():
        subnet = address[0]
        host_before = HostVector(before_row)
        was_discovered = bool(host_before.discovered)
        if was_discovered:
            subnets_before.add(subnet)

        after_row = after_by_address.get(address)
        if after_row is None:
            # A host address present before but not after would mean NASimEmu
            # forgot a host mid-episode -- never observed, but if it ever
            # happens there is nothing to diff this host against.
            continue
        host_after = HostVector(after_row)
        is_discovered = bool(host_after.discovered)
        if is_discovered:
            subnets_after.add(subnet)
        if is_discovered and not was_discovered:
            new_hosts += 1

        services_before = host_before.services
        for name, value in host_after.services.items():
            if value and not services_before.get(name):
                new_services += 1

        processes_before = host_before.processes
        for name, value in host_after.processes.items():
            if value and not processes_before.get(name):
                new_processes += 1

        gain = _access_gain(int(host_before.access), int(host_after.access))
        if gain > best_gain:
            best_gain = gain

    # Host addresses that only exist in `after` -- a newly-revealed subnet
    # (e.g. via subnet_scan) appends wholly new rows rather than flipping a
    # bit on a row that was present all along (see module docstring). There
    # is no "before" row to diff these against: their mere appearance,
    # already-discovered, is itself the discovery event.
    for address, after_row in after_by_address.items():
        if address in before_by_address:
            continue
        host_after = HostVector(after_row)
        if bool(host_after.discovered):
            new_hosts += 1
            subnets_after.add(address[0])
        gain = _access_gain(int(AccessLevel.NONE), int(host_after.access))
        if gain > best_gain:
            best_gain = gain
        for name, value in host_after.services.items():
            if value:
                new_services += 1
        for name, value in host_after.processes.items():
            if value:
                new_processes += 1

    return StateDelta(
        new_hosts_discovered=new_hosts,
        new_subnets_discovered=len(subnets_after - subnets_before),
        new_services_confirmed=new_services,
        new_processes_confirmed=new_processes,
        access_gain=best_gain,
    )
