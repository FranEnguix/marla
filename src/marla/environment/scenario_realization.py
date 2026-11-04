"""Offline reconstruction of the exact network NASimEmu generates for a seed.

A randomized NASimEmu scenario (``host_configurations: _random``, subnet
size ranges, per-subnet sensitivity probabilities) is a *distribution*; what
an episode actually plays on is one realization of it. This module rebuilds
that realization by running the real generator -- :meth:`NasimEmuAdapter.reset`
with the requested seed, including its deterministic zero-target
regeneration -- and reading the resulting simulator state. It never samples
anything itself, so a realization here is the one MARLA trains/evaluates on
for that seed, by construction.

**Subnet permutation.** ``nasimemu.nasim.scenarios.Scenario`` randomly
permutes the non-internet subnet IDs of every generated scenario (subnet 0,
the internet, stays fixed): the YAML's subnet 1 ("DMZ", say) may be realized
as subnet 2 in one episode and subnet 1 in the next, and every host address
moves with it. To let figures place the *same* YAML subnet in the same spot
for every seed, :func:`_recording_subnet_permutation` wraps
``Scenario._permute_subnets`` for the duration of one reset and recomputes
the permutation from a *copy* of numpy's global RNG state taken right before
the original method draws it -- the global stream itself is never advanced
or altered by the wrapper (the original method still draws it). The
recovered permutation is then checked against the realized subnet sizes and
topology before anything uses it.

**Ground truth, never an observation.** Everything here is hidden simulator
state (OS, services, processes, sensitivity of every host). It is for
offline documentation/analysis only and must never be fed to a policy; the
agent observes host attributes only through its own scans and exploits.

The caller's ``random``/``numpy`` global RNG states are saved and restored
around every reconstruction, so using this module does not perturb any
other generation in the same process.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from marla.environment.actions import host_target_key
from marla.environment.nasimemu_adapter import NasimEmuAdapter

# numpy's legacy global RNG (which NasimEmuAdapter.reset seeds) accepts 0 <= seed < 2**32.
MAX_SCENARIO_SEED = 2**32 - 1

_ACCESS_NAMES = {1: "user", 2: "root"}


class ScenarioRealizationError(RuntimeError):
    """The realization could not be reconstructed or failed a consistency check."""


@dataclass(frozen=True)
class ApplicableAction:
    """An exploit/privilege escalation whose host-side preconditions (service/
    process and OS, exactly as NASimEmu's ``HostVector.perform_action`` checks
    them) hold on a host. Says nothing about reachability or firewalls."""

    name: str
    access: str
    prob: float


@dataclass(frozen=True)
class RealizedHost:
    address: tuple[int, int]
    os: str | None
    services: tuple[str, ...]
    # Without the ``~`` (None) placeholder that scenarios defining no
    # processes put in every host's process list.
    processes: tuple[str, ...]
    sensitive: bool
    value: float
    discovery_value: float
    # Host firewall: source address -> services this host refuses from it.
    firewall_denied: tuple[tuple[tuple[int, int], tuple[str, ...]], ...]
    applicable_exploits: tuple[ApplicableAction, ...]
    applicable_privescs: tuple[ApplicableAction, ...]

    @property
    def key(self) -> str:
        return host_target_key(*self.address)

    @property
    def subnet_id(self) -> int:
        return self.address[0]


@dataclass(frozen=True)
class RealizedSubnet:
    """A realized non-internet subnet. ``subnet_id`` is the ID host addresses
    use in this realization; ``source_index`` is the subnet's index in the
    scenario YAML (stable across seeds)."""

    subnet_id: int
    source_index: int
    label: str | None
    size: int
    size_bounds: tuple[int, int]
    internet_facing: bool


@dataclass(frozen=True)
class SubnetConnection:
    """One topology edge between two subnets (realized IDs, ``a < b``; ``0`` is
    the internet). ``a_to_b``/``b_to_a`` are the topology matrix entries;
    ``services_*`` are the services the subnet firewall permits in each
    direction."""

    a: int
    b: int
    a_to_b: bool
    b_to_a: bool
    services_a_to_b: tuple[str, ...]
    services_b_to_a: tuple[str, ...]
    restricted: bool  # some connected direction does not permit every scenario service

    @property
    def bidirectional(self) -> bool:
        return self.a_to_b and self.b_to_a


@dataclass(frozen=True)
class ScenarioRealization:
    scenario_path: str
    scenario_name: str
    scenario_sha256: str
    scenario_format: str  # "v1" | "v2"
    requested_seed: int
    realization_seed: int
    regeneration_attempt: int
    # subnet_permutation[source_index] = realized subnet ID (index 0 = internet).
    subnet_permutation: tuple[int, ...]
    subnets: tuple[RealizedSubnet, ...]  # non-internet subnets, ordered by source_index
    connections: tuple[SubnetConnection, ...]
    hosts: tuple[RealizedHost, ...]  # ordered by address
    os_names: tuple[str, ...]
    services: tuple[str, ...]
    processes: tuple[str, ...]  # scenario process vocabulary without the ``~`` placeholder
    address_space_bounds: tuple[int, int] | None
    warnings: tuple[str, ...] = field(default=())

    @property
    def sensitive_hosts(self) -> tuple[RealizedHost, ...]:
        return tuple(h for h in self.hosts if h.sensitive)

    @property
    def defines_processes(self) -> bool:
        return bool(self.processes)

    def subnet(self, subnet_id: int) -> RealizedSubnet:
        for s in self.subnets:
            if s.subnet_id == subnet_id:
                return s
        raise KeyError(subnet_id)

    def hosts_in(self, subnet_id: int) -> tuple[RealizedHost, ...]:
        return tuple(h for h in self.hosts if h.subnet_id == subnet_id)

    def structure_dict(self) -> dict:
        """What the default (topology-only) view shows: subnets, connectivity
        and which hosts are sensitive targets."""
        return {
            "subnets": [
                {"subnet_id": s.subnet_id, "source_index": s.source_index, "label": s.label, "size": s.size,
                 "internet_facing": s.internet_facing}
                for s in self.subnets
            ],
            "connections": [
                {"a": c.a, "b": c.b, "a_to_b": c.a_to_b, "b_to_a": c.b_to_a,
                 "services_a_to_b": list(c.services_a_to_b), "services_b_to_a": list(c.services_b_to_a)}
                for c in self.connections
            ],
            "targets": [list(h.address) for h in self.sensitive_hosts],
        }

    def hosts_dict(self) -> list[dict]:
        return [
            {
                "address": list(h.address), "key": h.key, "os": h.os, "services": list(h.services),
                "processes": list(h.processes), "sensitive": h.sensitive, "value": h.value,
                "discovery_value": h.discovery_value,
                "firewall_denied": [[list(src), list(svcs)] for src, svcs in h.firewall_denied],
                "applicable_exploits": [[a.name, a.access, a.prob] for a in h.applicable_exploits],
                "applicable_privescs": [[a.name, a.access, a.prob] for a in h.applicable_privescs],
            }
            for h in self.hosts
        ]

    @property
    def topology_fingerprint(self) -> str:
        return _fingerprint(self.structure_dict())

    @property
    def realization_fingerprint(self) -> str:
        """Identical for two seeds iff they generate the same network (subnet
        IDs, connectivity, firewall, and every host's sampled attributes)."""
        return _fingerprint({"structure": self.structure_dict(), "hosts": self.hosts_dict()})

    def to_dict(self) -> dict:
        return {
            "scenario_path": self.scenario_path,
            "scenario_name": self.scenario_name,
            "scenario_sha256": self.scenario_sha256,
            "scenario_format": self.scenario_format,
            "requested_seed": self.requested_seed,
            "realization_seed": self.realization_seed,
            "regeneration_attempt": self.regeneration_attempt,
            "subnet_permutation": list(self.subnet_permutation),
            **self.structure_dict(),
            "hosts": self.hosts_dict(),
            "topology_fingerprint": self.topology_fingerprint,
            "realization_fingerprint": self.realization_fingerprint,
            "warnings": list(self.warnings),
        }


def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def scenario_display_name(path: str | Path) -> str:
    """``evidence_c_multi_target.v2.yaml`` -> ``evidence_c_multi_target``."""
    name = Path(path).name
    for suffix in (".yaml", ".yml"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.endswith(".v2"):
        name = name[:-3]
    return name


def validate_scenario_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"scenario seed must be an integer, got {seed!r}")
    if not 0 <= int(seed) <= MAX_SCENARIO_SEED:
        raise ValueError(f"scenario seed must be in [0, {MAX_SCENARIO_SEED}] (numpy's seed range), got {seed}")
    return int(seed)


@dataclass
class _PrePermutation:
    permutation: list[int]
    sizes: list[int]
    topology: list[list[int]]


@contextmanager
def _recording_subnet_permutation(records: list[_PrePermutation]) -> Iterator[None]:
    """See the module docstring. Appends one record per generated scenario."""
    import nasimemu.nasim.scenarios.utils as nasim_utils
    from nasimemu.nasim.scenarios.scenario import Scenario

    original = Scenario._permute_subnets

    def recording(self) -> None:
        replay = np.random.RandomState()
        replay.set_state(np.random.get_state())
        subnets = [int(s) for s in self.scenario_dict[nasim_utils.SUBNETS]]
        tail = replay.permutation(np.arange(1, len(subnets)))
        records.append(_PrePermutation(
            permutation=[0, *(int(x) for x in tail)],
            sizes=subnets,
            topology=[[int(c) for c in row] for row in np.asarray(self.scenario_dict[nasim_utils.TOPOLOGY])],
        ))
        original(self)

    Scenario._permute_subnets = recording
    try:
        yield
    finally:
        Scenario._permute_subnets = original


@contextmanager
def _preserved_global_rng() -> Iterator[None]:
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _yaml_subnet_metadata(raw: dict, n_subnets: int, realized_sizes: list[int]) -> tuple[dict[int, str], dict[int, tuple[int, int]]]:
    labels: dict[int, str] = {}
    for key, value in (raw.get("subnet_labels") or {}).items():
        try:
            labels[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    bounds: dict[int, tuple[int, int]] = {}
    raw_subnets = raw.get("subnets") or []
    for idx in range(1, n_subnets):
        entry = raw_subnets[idx - 1] if idx - 1 < len(raw_subnets) else None
        if isinstance(entry, str) and re.fullmatch(r"\s*\d+\s*-\s*\d+\s*", entry):
            lo, hi = (int(x) for x in entry.split("-"))
            bounds[idx] = (lo, hi)
        elif isinstance(entry, int):
            bounds[idx] = (entry, entry)
        else:
            bounds[idx] = (realized_sizes[idx], realized_sizes[idx])
    return labels, bounds


def _true_keys(flags: dict) -> list:
    return [name for name, on in flags.items() if on]


def _access_name(value) -> str:
    try:
        return _ACCESS_NAMES.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


class ScenarioRealizer:
    """Reconstructs realizations of one scenario file; reusable across seeds."""

    def __init__(self, scenario_path: str | Path) -> None:
        path = Path(scenario_path)
        if not path.is_file():
            raise FileNotFoundError(f"scenario file not found: {path}")
        self.path = path.resolve()
        content = self.path.read_bytes()
        self.sha256 = hashlib.sha256(content).hexdigest()
        self.name = scenario_display_name(self.path)
        self.format = "v2" if ".v2" in self.path.suffixes else "v1"
        self._raw = yaml.safe_load(content.decode("utf-8")) or {}
        # max_episode_steps is irrelevant here: the adapter is only ever reset,
        # never stepped. NASimEmuEnv's constructor reseeds the global RNGs from
        # OS entropy, so it is built inside the guard too.
        with _preserved_global_rng():
            self._adapter = NasimEmuAdapter(str(self.path), max_episode_steps=1)

    def realize(self, seed: int) -> ScenarioRealization:
        seed = validate_scenario_seed(seed)
        records: list[_PrePermutation] = []
        with _preserved_global_rng(), _recording_subnet_permutation(records):
            self._adapter.reset(seed=seed)
            env = self._adapter._env.env  # offline ground truth (see module docstring)
            network = env.network
            scenario = env.scenario
            initial_state = env.current_state
            attempt = int(self._adapter.regeneration_attempt)
            realization_seed = int(self._adapter.realization_seed)
            if len(records) != attempt + 1:
                raise ScenarioRealizationError(
                    f"expected {attempt + 1} generated scenario(s) for seed {seed}, observed {len(records)}"
                )
            return self._extract(seed, realization_seed, attempt, records[-1], network, scenario, initial_state)

    def _extract(self, seed, realization_seed, attempt, pre, network, scenario, initial_state) -> ScenarioRealization:
        perm = pre.permutation
        sizes = [int(s) for s in network.subnets]
        topology = [[int(c) for c in row] for row in np.asarray(network.topology)]
        n = len(sizes)
        if len(perm) != n or sorted(perm) != list(range(n)) or perm[0] != 0:
            raise ScenarioRealizationError(f"recovered subnet permutation {perm} is not a permutation of {n} subnets")
        for i in range(n):
            if sizes[perm[i]] != pre.sizes[i]:
                raise ScenarioRealizationError(f"subnet permutation check failed (sizes) for seed {seed}")
            for j in range(n):
                if topology[perm[i]][perm[j]] != pre.topology[i][j]:
                    raise ScenarioRealizationError(f"subnet permutation check failed (topology) for seed {seed}")

        all_services = tuple(scenario.services)
        labels, bounds = _yaml_subnet_metadata(self._raw, n, pre.sizes)
        subnets = tuple(
            RealizedSubnet(
                subnet_id=perm[source],
                source_index=source,
                label=labels.get(source),
                size=sizes[perm[source]],
                size_bounds=bounds[source],
                internet_facing=bool(network.subnet_public(perm[source])),
            )
            for source in range(1, n)
        )

        connections = []
        for a in range(n):
            for b in range(a + 1, n):
                ab, ba = topology[a][b] == 1, topology[b][a] == 1
                if not (ab or ba):
                    continue
                svc_ab = tuple(sorted(network.firewall.get((a, b), ()))) if ab else ()
                svc_ba = tuple(sorted(network.firewall.get((b, a), ()))) if ba else ()
                full = set(all_services)
                restricted = (ab and set(svc_ab) != full) or (ba and set(svc_ba) != full)
                connections.append(SubnetConnection(a, b, ab, ba, svc_ab, svc_ba, bool(restricted)))

        exploits = sorted(scenario.exploits.items())
        privescs = sorted(scenario.privescs.items())
        sensitive = {tuple(addr) for addr in network.sensitive_addresses}
        hosts = []
        for address in sorted(network.address_space):
            host = network.hosts[address]
            os_list = _true_keys(host.os)
            services = sorted(_true_keys(host.services))
            processes = sorted(p for p in _true_keys(host.processes) if p is not None)
            # The simulator's initial state must carry the same configuration.
            state_host = initial_state.get_host(address)
            if (_true_keys(state_host.os) != os_list or sorted(_true_keys(state_host.services)) != services
                    or sorted(p for p in _true_keys(state_host.processes) if p is not None) != processes):
                raise ScenarioRealizationError(f"host {address} differs between scenario and initial state (seed {seed})")
            os_name = os_list[0] if len(os_list) == 1 else None
            applicable_exploits = tuple(
                ApplicableAction(name, _access_name(e["access"]), float(e["prob"]))
                for name, e in exploits
                if host.services.get(e["service"], False) and (e["os"] is None or host.os.get(e["os"], False))
            )
            applicable_privescs = tuple(
                ApplicableAction(name, _access_name(p["access"]), float(p["prob"]))
                for name, p in privescs
                if (p["process"] is None or host.processes.get(p["process"], False))
                and (p["os"] is None or host.os.get(p["os"], False))
            )
            firewall_denied = tuple(
                (tuple(int(x) for x in src), tuple(sorted(svcs)))
                for src, svcs in sorted(host.firewall.items())
                if svcs
            )
            hosts.append(RealizedHost(
                address=(int(address[0]), int(address[1])),
                os=os_name,
                services=tuple(services),
                processes=tuple(processes),
                sensitive=tuple(address) in sensitive,
                value=float(host.value),
                discovery_value=float(host.discovery_value),
                firewall_denied=firewall_denied,
                applicable_exploits=applicable_exploits,
                applicable_privescs=applicable_privescs,
            ))

        warnings = []
        if any(len(_true_keys(network.hosts[a].os)) != 1 for a in network.address_space):
            warnings.append("a host does not run exactly one OS; its OS is shown as unknown")
        bounds_raw = getattr(network, "address_space_bounds", None)
        return ScenarioRealization(
            scenario_path=str(self.path),
            scenario_name=self.name,
            scenario_sha256=self.sha256,
            scenario_format=self.format,
            requested_seed=seed,
            realization_seed=realization_seed,
            regeneration_attempt=attempt,
            subnet_permutation=tuple(perm),
            subnets=tuple(sorted(subnets, key=lambda s: s.source_index)),
            connections=tuple(connections),
            hosts=tuple(hosts),
            os_names=tuple(scenario.os),
            services=all_services,
            processes=tuple(p for p in scenario.processes if p is not None),
            address_space_bounds=tuple(int(x) for x in bounds_raw) if bounds_raw is not None else None,
            warnings=tuple(warnings),
        )


def realize_scenario(scenario_path: str | Path, seed: int) -> ScenarioRealization:
    """One-shot convenience wrapper around :class:`ScenarioRealizer`."""
    return ScenarioRealizer(scenario_path).realize(seed)
