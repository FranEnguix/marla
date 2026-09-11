"""MARLA-owned reading of NASimEmu scenario *templates*.

For a V1 (static) scenario, NASimEmu's own loader (``nasim.load_scenario``)
is fully deterministic (grep-verified: no ``random.*``/``np.random.*`` call
anywhere in ``nasimemu.nasim.scenarios.loader``) -- so a single real load
already gives the one and only realization, and this module just wraps that
real, already-loaded :class:`~nasimemu.nasim.scenarios.scenario.Scenario`.

For a V2 scenario, ``ScenarioLoaderV2.load()`` *resolves* every random
dimension (variable subnet sizes, per-host sensitivity, and -- when
``host_configurations: _random`` -- every host's OS/services/processes) the
moment it runs, via bare ``random.random()``/``random.choice()`` calls. The
returned :class:`Scenario` is one concrete sample, not the generator
template; the generative parameters themselves (subnet size ranges,
per-subnet sensitivity probabilities, the ``_random`` marker,
``sensitive_services``) are never preserved anywhere NASimEmu exposes after
loading. To reason about *every* realization a V2 file can produce, this
module reads the same YAML keys ``ScenarioLoaderV2`` reads, but keeps the
*unresolved* ranges/probabilities/markers instead of sampling them --
mirroring its parsing exactly (same key names, same ``is_for_os`` rule)
without duplicating its randomness.

This is the one documented case (spec section 21) where MARLA reads a
scenario's raw YAML directly rather than going through NASimEmu's loader,
because the semantics needed (the generator template) are only present in
the file, not in anything the loader constructs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

AccessLevel = Literal["user", "root"]
ACCESS_NAME_TO_INT = {"user": 1, "root": 2}


def is_for_os(name: str | None, os: str) -> bool:
    """Exact port of ``ScenarioLoaderV2._parse_host_configs.is_for_os``:
    a service/process name encodes its OS as the second ``_``-separated
    token (e.g. ``21_linux_proftpd``, ``3306_any_mysql``); ``None``
    (MARLA's ``~``-parsed "no process required" sentinel) matches every OS.
    """
    if name is None:
        return True
    parts = name.split("_")
    x_os = parts[1] if len(parts) > 1 else None
    return x_os == os or x_os == "any"


@dataclass(frozen=True)
class ExploitDef:
    name: str
    service: str
    os: str | None
    prob: float
    cost: float
    access: int


@dataclass(frozen=True)
class PrivescDef:
    name: str
    process: str | None
    os: str | None
    prob: float
    cost: float
    access: int


@dataclass(frozen=True)
class SubnetSpec:
    """One subnet in the topology, 0 = internet."""

    id: int
    size_min: int
    size_max: int
    sensitive_prob: float
    label: str | None = None

    @property
    def is_sensitive_capable(self) -> bool:
        return self.sensitive_prob > 0.0 and self.size_max > 0


@dataclass(frozen=True)
class StaticHostSpec:
    """A concrete (non-randomized) host -- from a V1 scenario, or a V2
    scenario whose ``host_configurations`` is an explicit dict rather than
    ``_random``."""

    address: tuple[int, int]
    os: str
    services: frozenset[str]
    processes: frozenset[str]
    firewall: dict[tuple[int, int], frozenset[str]]
    is_sensitive: bool


FirewallRule = dict[tuple[int, int], frozenset[str] | None]  # None = "_all"


@dataclass
class ScenarioSpec:
    """Everything the solvability checker needs, either read directly (V1,
    via the real loader) or reconstructed from the raw V2 YAML template
    without resolving its randomness.
    """

    path: Path
    format: Literal["v1", "v2"]
    randomized: bool

    os_list: tuple[str, ...]
    services: tuple[str, ...]
    processes: tuple[str, ...]
    sensitive_services: tuple[str, ...]

    exploits: tuple[ExploitDef, ...]
    privescs: tuple[PrivescDef, ...]

    subnets: tuple[SubnetSpec, ...]  # index 0 = internet
    topology: tuple[tuple[int, ...], ...]
    firewall: FirewallRule
    firewall_mode: Literal["_subnets", "explicit"]

    # Populated only when not randomized (V1, or V2 with an explicit
    # host_configurations dict) -- a fully concrete, enumerable host set.
    static_hosts: dict[tuple[int, int], StaticHostSpec] | None = field(default=None)

    raw_content: str = ""

    @property
    def exploit_map(self) -> dict[str, dict[str | None, ExploitDef]]:
        m: dict[str, dict[str | None, ExploitDef]] = {}
        for e in self.exploits:
            m.setdefault(e.service, {})[e.os] = e
        return m

    @property
    def privesc_map(self) -> dict[str | None, dict[str | None, PrivescDef]]:
        m: dict[str | None, dict[str | None, PrivescDef]] = {}
        for p in self.privescs:
            m.setdefault(p.process, {})[p.os] = p
        return m


def _access_to_int(value) -> int:
    if isinstance(value, str):
        return ACCESS_NAME_TO_INT[value.lower()]
    return int(value)


def _none_if_tilde(value):
    if value is None or (isinstance(value, str) and value.lower() == "none"):
        return None
    return value


def _parse_exploits(raw: dict) -> tuple[ExploitDef, ...]:
    out = []
    for name, e in raw.items():
        out.append(
            ExploitDef(
                name=name,
                service=e["service"],
                os=_none_if_tilde(e.get("os")),
                prob=float(e["prob"]),
                cost=float(e["cost"]),
                access=_access_to_int(e["access"]),
            )
        )
    return tuple(out)


def _parse_privescs(raw: dict) -> tuple[PrivescDef, ...]:
    out = []
    for name, p in raw.items():
        out.append(
            PrivescDef(
                name=name,
                process=_none_if_tilde(p.get("process")),
                os=_none_if_tilde(p.get("os")),
                prob=float(p["prob"]),
                cost=float(p["cost"]),
                access=_access_to_int(p["access"]),
            )
        )
    return tuple(out)


def _parse_v2_subnets(raw_subnets: list) -> tuple[SubnetSpec, ...]:
    subnets: list[SubnetSpec] = [SubnetSpec(id=0, size_min=1, size_max=1, sensitive_prob=0.0, label="internet")]
    for idx, entry in enumerate(raw_subnets, start=1):
        if isinstance(entry, str):
            lo, hi = entry.split("-")
            size_min, size_max = int(lo), int(hi)
        else:
            size_min = size_max = int(entry)
        subnets.append(SubnetSpec(id=idx, size_min=size_min, size_max=size_max, sensitive_prob=0.0))
    return tuple(subnets)


def _apply_sensitive_probs(subnets: tuple[SubnetSpec, ...], raw_sensitive: dict) -> tuple[SubnetSpec, ...]:
    out = []
    for s in subnets:
        prob = float(raw_sensitive.get(s.id, 0.0)) if s.id != 0 else 0.0
        out.append(SubnetSpec(id=s.id, size_min=s.size_min, size_max=s.size_max, sensitive_prob=prob, label=s.label))
    return tuple(out)


def _resolve_firewall(raw_firewall, topology: tuple[tuple[int, ...], ...]) -> tuple[FirewallRule, str]:
    if raw_firewall == "_subnets":
        fw: FirewallRule = {}
        for src, row in enumerate(topology):
            for dest, col in enumerate(row):
                if src != dest and col == 1:
                    fw[(src, dest)] = None  # None == "_all": every service permitted on this edge
        return fw, "_subnets"

    fw = {}
    for key, value in raw_firewall.items():
        edge = eval(key) if isinstance(key, str) else tuple(key)
        if "_all" in value:
            fw[tuple(edge)] = None
        else:
            fw[tuple(edge)] = frozenset(value)
    return fw, "explicit"


def _parse_v2_spec(path: Path, raw: dict, raw_content: str) -> ScenarioSpec:
    subnets = _parse_v2_subnets(raw["subnets"])
    subnets = _apply_sensitive_probs(subnets, raw.get("sensitive_hosts", {}))
    topology = tuple(tuple(int(c) for c in row) for row in raw["topology"])

    os_list = tuple(raw["os"])
    services = tuple(raw["services"])
    processes = tuple(_none_if_tilde(p) for p in raw.get("processes", []))
    sensitive_services = tuple(raw.get("sensitive_services", []))

    exploits = _parse_exploits(raw["exploits"])
    privescs = _parse_privescs(raw["privilege_escalation"])

    firewall, firewall_mode = _resolve_firewall(raw["firewall"], topology)

    host_cfgs = raw["host_configurations"]
    randomized = host_cfgs == "_random"

    static_hosts = None
    if not randomized:
        # Sensitivity for an explicit-host V2 scenario is subnet-probability
        # -based (`sensitive_hosts`, the same key `_random` uses), not
        # address-based -- there is no separate per-host sensitivity field
        # in host_configurations itself. A probability of exactly 1.0 or
        # 0.0 is fully deterministic regardless of host_configurations
        # being explicit (every/no host in that subnet is sensitive, in
        # every realization); anything strictly between the two means
        # sensitive-host *placement* is still randomized even though the
        # host identities (OS/services/processes) are fixed -- a shape
        # _check_static's fixed-host-set reasoning cannot represent (it
        # would either over- or under-count sensitive hosts silently), so
        # that combination is rejected loudly here instead.
        subnets_by_id = {s.id: s for s in subnets}
        for s in subnets:
            if s.id != 0 and 0.0 < s.sensitive_prob < 1.0:
                raise ValueError(
                    f"Scenario {path}: explicit host_configurations combined with a fractional "
                    f"sensitive_hosts probability ({s.sensitive_prob}) for subnet {s.id} is not "
                    "supported -- sensitive-host placement would still be randomized per-episode "
                    "even though every host's identity is fixed, which the static-host solvability "
                    "path cannot represent. Use host_configurations: _random, or a sensitive_hosts "
                    "probability of exactly 0.0 or 1.0 for every subnet."
                )

        static_hosts = {}
        for addr_key, cfg in host_cfgs.items():
            addr = eval(addr_key) if isinstance(addr_key, str) else tuple(addr_key)
            fw = {}
            for other_key, denied in cfg.get("firewall", {}).items():
                other = eval(other_key) if isinstance(other_key, str) else tuple(other_key)
                fw[tuple(other)] = frozenset(denied)
            static_hosts[tuple(addr)] = StaticHostSpec(
                address=tuple(addr),
                os=cfg["os"],
                services=frozenset(cfg.get("services", [])),
                processes=frozenset(_none_if_tilde(p) for p in cfg.get("processes", [])),
                firewall=fw,
                is_sensitive=subnets_by_id[tuple(addr)[0]].sensitive_prob >= 1.0,
            )

    return ScenarioSpec(
        path=path,
        format="v2",
        randomized=randomized,
        os_list=os_list,
        services=services,
        processes=processes,
        sensitive_services=sensitive_services,
        exploits=exploits,
        privescs=privescs,
        subnets=subnets,
        topology=topology,
        firewall=firewall,
        firewall_mode=firewall_mode,
        static_hosts=static_hosts,
        raw_content=raw_content,
    )


def _parse_v1_spec(path: Path, raw_content: str) -> ScenarioSpec:
    # V1 loading is fully deterministic (see module docstring) -- a single
    # real load already gives the complete, concrete scenario.
    import nasimemu.nasim as nasim

    scenario = nasim.load_scenario(str(path))

    subnets = tuple(
        SubnetSpec(id=idx, size_min=int(size), size_max=int(size), sensitive_prob=0.0)
        for idx, size in enumerate(scenario.subnets)
    )
    sensitive_addrs = set(scenario.sensitive_addresses)
    subnets = tuple(
        SubnetSpec(
            id=s.id,
            size_min=s.size_min,
            size_max=s.size_max,
            sensitive_prob=1.0 if any(addr[0] == s.id for addr in sensitive_addrs) else 0.0,
        )
        for s in subnets
    )
    topology = tuple(tuple(int(c) for c in row) for row in scenario.topology)

    exploits = tuple(
        ExploitDef(
            name=name,
            service=e["service"],
            os=e["os"],
            prob=float(e["prob"]),
            cost=float(e["cost"]),
            access=int(e["access"]),
        )
        for name, e in scenario.exploits.items()
    )
    privescs = tuple(
        PrivescDef(
            name=name,
            process=p["process"],
            os=p["os"],
            prob=float(p["prob"]),
            cost=float(p["cost"]),
            access=int(p["access"]),
        )
        for name, p in scenario.privescs.items()
    )

    firewall: FirewallRule = {}
    for edge, services in scenario.firewall.items():
        firewall[tuple(edge)] = frozenset(services)

    static_hosts = {}
    for addr, host in scenario.hosts.items():
        fw = {}
        for other, denied in host.firewall.items():
            fw[tuple(other)] = frozenset(denied)
        static_hosts[tuple(addr)] = StaticHostSpec(
            address=tuple(addr),
            os=next(name for name, present in host.os.items() if present),
            services=frozenset(name for name, present in host.services.items() if present),
            processes=frozenset(name for name, present in host.processes.items() if present),
            firewall=fw,
            is_sensitive=tuple(addr) in sensitive_addrs,
        )

    return ScenarioSpec(
        path=path,
        format="v1",
        randomized=False,
        os_list=tuple(scenario.os),
        services=tuple(scenario.services),
        processes=tuple(scenario.processes),
        sensitive_services=(),
        exploits=exploits,
        privescs=privescs,
        subnets=subnets,
        topology=topology,
        firewall=firewall,
        firewall_mode="explicit",
        static_hosts=static_hosts,
        raw_content=raw_content,
    )


def load_scenario_spec(path: str | Path) -> ScenarioSpec:
    """Load a :class:`ScenarioSpec` from a scenario YAML file.

    Dispatches on the same ``.v2`` suffix rule NASimEmu itself uses
    (``'.v2' in Path(path).suffixes``, see
    ``nasimemu.nasim.scenarios.load_scenario``) so MARLA never
    second-guesses which loader NASimEmu would pick for this file.
    """
    path = Path(path)
    raw_content = path.read_text(encoding="utf-8")
    if ".v2" in path.suffixes:
        raw = yaml.safe_load(raw_content)
        return _parse_v2_spec(path, raw, raw_content)
    return _parse_v1_spec(path, raw_content)
