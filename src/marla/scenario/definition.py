"""What a scenario YAML *defines*: its topology, host-count ranges and every
generation probability, read without generating any network.

Nothing here samples, seeds or calls NASimEmu: the YAML is parsed directly
and the probabilities are derived in closed form from the exact rules of
NASimEmu's loaders (``nasimemu.nasim.scenarios.loader_v2.ScenarioLoaderV2``
for ``*.v2.yaml``, ``loader.ScenarioLoader`` otherwise), so a definition
figure describes every network a scenario can produce, not one of them.

**V2 ``host_configurations: _random`` -- the rules being described.** For
each non-internet subnet ``s`` (YAML order):

* size: a fixed ``n``, or ``"a-b"`` -> ``random.randint(a, b)``, i.e. uniform
  over ``a..b``;
* each host is sensitive independently with probability
  ``sensitive_hosts[s]`` (value 100; non-sensitive hosts have value 0);
* each host's OS is ``random.choice(os)``: uniform over the OS list;
* given the OS, the host's *non-sensitive* candidate services are the
  scenario services for that OS (``<port>_<os>_<name>`` or ``..._any_...``)
  minus ``sensitive_services``; it gets ``k`` of them, a uniformly random
  subset of size ``k`` with ``P(k) proportional to 0.5**k`` (``k = 1..n``)
  -- so each candidate is present with probability ``E[k] / n``, and the
  presence events are *not* independent (exactly ``k`` are present);
* a sensitive host additionally gets one of the sensitive services
  available for its OS (uniformly); a non-sensitive host gets one with
  probability 0.1;
* processes follow the same ``P(k)`` rule over the processes for that OS
  (``~``, MARLA's "no process", counts for every OS).

NASimEmu then permutes the non-internet subnet IDs uniformly at random, so
a realization's subnet 1 is not necessarily the YAML's subnet 1 (figures of
definitions use YAML indices). MARLA additionally discards every
realization without a sensitive host and regenerates it
(:meth:`marla.environment.nasimemu_adapter.NasimEmuAdapter.reset`): the
networks used in experiments are conditioned on having at least one
target. The probabilities here are the generator's, before that
conditioning; :attr:`ScenarioDefinition.p_zero_targets` is how often it
applies.

V1 scenarios (and V2 with an explicit ``host_configurations`` mapping) are
static: every host is listed; only V2's per-subnet sensitivity stays random.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from marla.scenario.spec import is_for_os

# NASimEmu's loader_v2 constants.
SKEW_BASE = 0.5  # skew_dist(n, p=0.5)
SENSITIVE_SERVICE_ON_NON_TARGET = 0.1
SENSITIVE_HOST_VALUE = 100.0
DEFAULT_HOST_VALUE = 0.0

_ACCESS_NAMES = {1: "user", 2: "root", "user": "user", "root": "root"}


class ScenarioDefinitionError(ValueError):
    """The YAML cannot be read as a NASimEmu scenario."""


@dataclass(frozen=True)
class SubnetLink:
    """Same fields as :class:`marla.environment.scenario_realization.SubnetConnection`
    (subnet IDs here are YAML indices), so diagrams treat both alike."""

    a: int
    b: int
    a_to_b: bool
    b_to_a: bool
    services_a_to_b: tuple[str, ...]
    services_b_to_a: tuple[str, ...]
    restricted: bool

    @property
    def bidirectional(self) -> bool:
        return self.a_to_b and self.b_to_a


@dataclass(frozen=True)
class StaticHost:
    """A host listed explicitly in the YAML."""

    address: tuple[int, int]
    os: str | None
    services: tuple[str, ...]
    processes: tuple[str, ...]
    firewall_denied: tuple[tuple[tuple[int, int], tuple[str, ...]], ...] = ()
    value: float | None = None


@dataclass(frozen=True)
class SubnetDefinition:
    source_index: int
    label: str | None
    size_min: int | None  # None: not determinable from the YAML (see size_note)
    size_max: int | None
    size_distribution: tuple[tuple[int, float], ...]  # (hosts, probability)
    # V2: probability that each host is sensitive; None when the YAML gives none.
    target_probability: float | None
    targets_min: int | None
    targets_max: int | None
    internet_facing: bool
    # Exact marginal probability that a host of this subnet runs each sensitive
    # service (V2 _random; over OS and target status).
    sensitive_service_presence: tuple[tuple[str, float], ...] = ()
    static_hosts: tuple[StaticHost, ...] = ()
    # V1: the listed sensitive hosts and their values.
    explicit_targets: tuple[tuple[tuple[int, int], float], ...] = ()
    size_note: str | None = None

    @property
    def subnet_id(self) -> int:
        """Definitions are drawn with YAML indices."""
        return self.source_index

    @property
    def size_bounds(self) -> tuple[int, int]:
        lo = self.size_min if self.size_min is not None else 1
        hi = self.size_max if self.size_max is not None else lo
        return lo, hi


@dataclass(frozen=True)
class OsHostModel:
    """Attribute distribution of a randomly generated host, given its OS."""

    os: str
    probability: float
    service_candidates: tuple[str, ...]  # non-sensitive services for this OS
    service_count_distribution: tuple[tuple[int, float], ...]
    service_presence: float  # P(a given candidate is present | OS)
    sensitive_services: tuple[str, ...]  # available to this OS
    process_candidates: tuple[str, ...]  # without the "~" placeholder
    process_count_distribution: tuple[tuple[int, float], ...]
    process_presence: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionDefinition:
    kind: str  # "exploit" | "privilege_escalation"
    name: str
    requires: str | None  # service (exploit) or process (privilege escalation); None: none
    os: str | None  # None: any OS
    prob: float
    access: str
    cost: float


@dataclass(frozen=True)
class ScenarioDefinition:
    path: str
    name: str
    sha256: str
    format: str  # "v1" | "v2"
    randomized_hosts: bool
    subnets: tuple[SubnetDefinition, ...]  # non-internet, YAML order
    connections: tuple[SubnetLink, ...]
    os_names: tuple[str, ...]
    services: tuple[str, ...]
    processes: tuple[str, ...]  # without "~"
    sensitive_services: tuple[str, ...]
    host_model: tuple[OsHostModel, ...]  # empty unless randomized_hosts
    actions: tuple[ActionDefinition, ...]
    scan_costs: tuple[tuple[str, float], ...]
    targets_min: int | None
    targets_max: int | None
    p_zero_targets: float | None  # before MARLA's zero-target regeneration
    warnings: tuple[str, ...] = field(default=())

    @property
    def subnet_permutation(self) -> tuple[int, ...]:
        """Identity: a definition is drawn in YAML subnet order."""
        return tuple(range(len(self.subnets) + 1))

    @property
    def host_count_range(self) -> tuple[int | None, int | None]:
        if any(s.size_min is None or s.size_max is None for s in self.subnets):
            return None, None
        return sum(s.size_min for s in self.subnets), sum(s.size_max for s in self.subnets)

    @property
    def defines_processes(self) -> bool:
        return bool(self.processes)

    def to_dict(self) -> dict:
        return {
            "scenario_path": self.path, "scenario_name": self.name, "scenario_sha256": self.sha256,
            "scenario_format": self.format, "randomized_hosts": self.randomized_hosts,
            "subnets": [
                {"subnet": s.source_index, "label": s.label, "hosts_min": s.size_min, "hosts_max": s.size_max,
                 "target_probability": s.target_probability, "targets_min": s.targets_min,
                 "targets_max": s.targets_max, "internet_facing": s.internet_facing}
                for s in self.subnets
            ],
            "hosts_min": self.host_count_range[0], "hosts_max": self.host_count_range[1],
            "targets_min": self.targets_min, "targets_max": self.targets_max,
            "p_zero_targets": self.p_zero_targets,
            "warnings": list(self.warnings),
        }


def skew_count_distribution(n: int, base: float = SKEW_BASE) -> tuple[tuple[int, float], ...]:
    """``loader_v2.skew_dist(n)``: ``P(k) proportional to base**k`` for ``k = 1..n``."""
    if n <= 0:
        return ()
    weights = [base**k for k in range(1, n + 1)]
    total = sum(weights)
    return tuple((k, w / total) for k, w in zip(range(1, n + 1), weights))


def _expected(distribution) -> float:
    return sum(k * p for k, p in distribution)


def _literal(key):
    if isinstance(key, str):
        try:
            return ast.literal_eval(key)
        except (ValueError, SyntaxError) as exc:
            raise ScenarioDefinitionError(f"cannot read address {key!r}") from exc
    return key


def _none_if_tilde(value):
    if value is None or (isinstance(value, str) and value.strip().lower() in ("none", "~", "")):
        return None
    return value


def _parse_actions(raw: dict) -> tuple[ActionDefinition, ...]:
    actions = []
    for kind, section, key in (("exploit", "exploits", "service"), ("privilege_escalation", "privilege_escalation", "process")):
        for name, spec in sorted((raw.get(section) or {}).items()):
            access = spec.get("access")
            actions.append(ActionDefinition(
                kind=kind, name=str(name), requires=_none_if_tilde(spec.get(key)), os=_none_if_tilde(spec.get("os")),
                prob=float(spec.get("prob")), access=_ACCESS_NAMES.get(access, str(access)), cost=float(spec.get("cost")),
            ))
    return tuple(actions)


def _parse_firewall(raw_firewall, topology: list[list[int]], services: tuple[str, ...]) -> dict[tuple[int, int], tuple[str, ...]]:
    if raw_firewall == "_subnets":
        return {(i, j): services for i, row in enumerate(topology) for j, c in enumerate(row) if i != j and c == 1}
    rules = {}
    for key, value in (raw_firewall or {}).items():
        edge = tuple(_literal(key))
        rules[edge] = services if "_all" in (value or []) else tuple(value or ())
    return rules


def _parse_size(entry) -> tuple[int | None, int | None, str | None]:
    if isinstance(entry, bool):
        return None, None, f"unrecognized subnet size {entry!r}"
    if isinstance(entry, int):
        return entry, entry, None
    if isinstance(entry, str):
        match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", entry)
        if match and int(match.group(1)) <= int(match.group(2)):
            return int(match.group(1)), int(match.group(2)), None
    return None, None, f"unrecognized subnet size {entry!r}"


def load_scenario_definition(path: str | Path) -> ScenarioDefinition:
    """Reads ``path`` (YAML only; no generation, no randomness)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"scenario file not found: {path}")
    content = path.read_bytes()
    try:
        raw = yaml.safe_load(content.decode("utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ScenarioDefinitionError(f"{path} is not valid YAML: {exc}") from exc
    for key in ("subnets", "topology", "os", "services", "exploits"):
        if key not in raw:
            raise ScenarioDefinitionError(f"{path} is not a NASimEmu scenario (no {key!r})")
    fmt = "v2" if ".v2" in path.suffixes else "v1"
    name = path.name
    for suffix in (".yaml", ".yml"):
        name = name[: -len(suffix)] if name.endswith(suffix) else name
    name = name[:-3] if name.endswith(".v2") else name

    os_names = tuple(raw["os"])
    services = tuple(raw["services"])
    raw_processes = list(raw.get("processes") or [])
    processes = tuple(p for p in (_none_if_tilde(x) for x in raw_processes) if p is not None)
    sensitive_services = tuple(raw.get("sensitive_services") or ())
    topology = [[int(c) for c in row] for row in raw["topology"]]
    n = len(raw["subnets"]) + 1
    warnings: list[str] = []
    if len(topology) != n or any(len(row) != n for row in topology):
        raise ScenarioDefinitionError(f"{path}: topology is not {n}x{n} (internet + {n - 1} subnets)")
    firewall = _parse_firewall(raw.get("firewall"), topology, services)
    labels = {}
    for key, value in (raw.get("subnet_labels") or {}).items():
        try:
            labels[int(key)] = str(value)
        except (TypeError, ValueError):
            continue

    host_configs = raw.get("host_configurations")
    randomized_hosts = fmt == "v2" and host_configs == "_random"
    static_by_subnet: dict[int, list[StaticHost]] = {}
    if not randomized_hosts:
        for key, cfg in (host_configs or {}).items():
            address = tuple(_literal(key))
            fw = tuple((tuple(_literal(src)), tuple(sorted(svcs or ()))) for src, svcs in sorted((cfg.get("firewall") or {}).items()) if svcs)
            static_by_subnet.setdefault(address[0], []).append(StaticHost(
                address=address, os=_none_if_tilde(cfg.get("os")), services=tuple(sorted(cfg.get("services") or ())),
                processes=tuple(sorted(p for p in (cfg.get("processes") or []) if _none_if_tilde(p) is not None)),
                firewall_denied=fw, value=float(cfg["value"]) if "value" in cfg else None,
            ))

    host_model = _host_model(os_names, services, raw_processes, sensitive_services) if randomized_hosts else ()

    raw_sensitive = raw.get("sensitive_hosts") or {}
    explicit_targets: dict[int, list[tuple[tuple[int, int], float]]] = {}
    target_prob: dict[int, float] = {}
    if fmt == "v1":
        for key, value in raw_sensitive.items():
            address = tuple(_literal(key))
            explicit_targets.setdefault(address[0], []).append((address, float(value)))
    else:
        for key, value in raw_sensitive.items():
            try:
                target_prob[int(key)] = float(value)
            except (TypeError, ValueError):
                warnings.append(f"sensitive_hosts entry {key!r} is not a subnet probability")

    subnets = []
    for idx in range(1, n):
        lo, hi, note = _parse_size(raw["subnets"][idx - 1])
        if not randomized_hosts and idx in static_by_subnet:
            count = len(static_by_subnet[idx])
            if lo is not None and not lo <= count <= hi:
                note = f"{count} hosts listed but subnets[{idx - 1}] = {raw['subnets'][idx - 1]!r}"
        size_dist = tuple((k, 1 / (hi - lo + 1)) for k in range(lo, hi + 1)) if lo is not None else ()
        p = target_prob.get(idx) if fmt == "v2" else None
        if fmt == "v2" and p is None:
            warnings.append(f"subnet {idx} has no sensitive_hosts probability (NASimEmu's v2 loader would fail)")
        if fmt == "v1":
            listed = sorted(explicit_targets.get(idx, []))
            t_min = t_max = len(listed)
        elif p is None or lo is None:
            t_min = t_max = None
        else:
            t_min = lo if p >= 1.0 else 0
            t_max = hi if p > 0.0 else 0
        presence = ()
        if randomized_hosts and p is not None:
            presence = tuple(
                (svc, sum(m.probability * (p + (1 - p) * SENSITIVE_SERVICE_ON_NON_TARGET) / len(m.sensitive_services)
                          for m in host_model if svc in m.sensitive_services))
                for svc in sensitive_services
            )
        subnets.append(SubnetDefinition(
            source_index=idx, label=labels.get(idx), size_min=lo, size_max=hi, size_distribution=size_dist,
            target_probability=p, targets_min=t_min, targets_max=t_max,
            internet_facing=topology[idx][0] == 1 or topology[0][idx] == 1,
            sensitive_service_presence=presence,
            static_hosts=tuple(sorted(static_by_subnet.get(idx, []), key=lambda h: h.address)),
            explicit_targets=tuple(sorted(explicit_targets.get(idx, []))), size_note=note,
        ))

    connections = []
    for a in range(n):
        for b in range(a + 1, n):
            ab, ba = topology[a][b] == 1, topology[b][a] == 1
            if not (ab or ba):
                continue
            svc_ab = tuple(sorted(firewall.get((a, b), ()))) if ab else ()
            svc_ba = tuple(sorted(firewall.get((b, a), ()))) if ba else ()
            full = set(services)
            restricted = (ab and set(svc_ab) != full) or (ba and set(svc_ba) != full)
            connections.append(SubnetLink(a, b, ab, ba, svc_ab, svc_ba, bool(restricted)))

    known = [s for s in subnets if s.targets_min is not None]
    targets_min = sum(s.targets_min for s in known) if len(known) == len(subnets) else None
    targets_max = sum(s.targets_max for s in known) if len(known) == len(subnets) else None
    p_zero = None
    if fmt == "v2" and all(s.target_probability is not None and s.size_distribution for s in subnets):
        p_zero = math.prod(sum(q * (1 - s.target_probability) ** k for k, q in s.size_distribution) for s in subnets)
    elif fmt == "v1":
        p_zero = 0.0 if targets_max else 1.0

    scan_costs = tuple((label, float(raw[key])) for label, key in (
        ("service scan", "service_scan_cost"), ("OS scan", "os_scan_cost"),
        ("subnet scan", "subnet_scan_cost"), ("process scan", "process_scan_cost")) if key in raw)
    return ScenarioDefinition(
        path=str(path.resolve()), name=name, sha256=hashlib.sha256(content).hexdigest(), format=fmt,
        randomized_hosts=randomized_hosts, subnets=tuple(subnets), connections=tuple(connections),
        os_names=os_names, services=services, processes=processes, sensitive_services=sensitive_services,
        host_model=host_model, actions=_parse_actions(raw), scan_costs=scan_costs,
        targets_min=targets_min, targets_max=targets_max, p_zero_targets=p_zero,
        warnings=tuple(warnings) + tuple(w for m in host_model for w in m.warnings),
    )


def _host_model(os_names, services, raw_processes, sensitive_services) -> tuple[OsHostModel, ...]:
    models = []
    for os in os_names:
        own = [s for s in services if is_for_os(s, os)]
        sensitive = tuple(s for s in sensitive_services if s in own)
        candidates = tuple(s for s in own if s not in sensitive)
        procs = [_none_if_tilde(p) for p in raw_processes if is_for_os(_none_if_tilde(p), os)]
        notes = []
        if not candidates:
            notes.append(f"{os}: no non-sensitive service available (NASimEmu cannot generate such a host)")
        if not sensitive:
            notes.append(f"{os}: no sensitive service available (a sensitive {os} host cannot be generated)")
        service_dist = skew_count_distribution(len(candidates))
        process_dist = skew_count_distribution(len(procs))
        real_procs = tuple(p for p in procs if p is not None)
        models.append(OsHostModel(
            os=os, probability=1 / len(os_names), service_candidates=candidates,
            service_count_distribution=service_dist,
            service_presence=_expected(service_dist) / len(candidates) if candidates else 0.0,
            sensitive_services=sensitive, process_candidates=real_procs,
            process_count_distribution=process_dist if real_procs else (),
            process_presence=_expected(process_dist) / len(procs) if real_procs else 0.0,
            warnings=tuple(notes),
        ))
    return tuple(models)
