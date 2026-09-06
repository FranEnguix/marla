"""Typed result models for the scenario solvability subsystem.

Kept independent of both NASimEmu's runtime classes and any CLI formatting
(spec: "core checker depend[s] [on] formatted CLI strings" is explicitly
disallowed) -- :mod:`marla.scenario.solvability` returns
:class:`ScenarioSolvabilityResult` instances, and ``marla scenario check``
only formats them for humans/JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SolvabilityStatus(str, Enum):
    """Outcome of a universal-solvability analysis.

    ``UNKNOWN`` is a real, distinct outcome -- e.g. a scenario shape the
    checker cannot yet reason about exhaustively -- and must never be
    silently treated as solvable by any caller (spec section 4).
    """

    PROVEN_SOLVABLE = "proven_solvable"
    PROVEN_UNSOLVABLE = "proven_unsolvable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HostConfigClass:
    """One symbolic equivalence class of a host configuration considered
    during rootability analysis -- an OS plus a (possibly minimal, possibly
    worst-case) service/process set, not one concrete host address. V2
    random generation produces many concrete hosts that collapse onto a
    small number of these classes (spec section 26).
    """

    os: str
    services: frozenset[str]
    processes: frozenset[str]
    is_sensitive: bool = False


@dataclass(frozen=True)
class HostRootabilityFailure:
    """A structural reason some legally-generatable host configuration
    class cannot ever reach ROOT, independent of network reachability.
    """

    os: str
    services: frozenset[str]
    processes: frozenset[str]
    available_attack_paths: tuple[str, ...]
    missing_capability: str
    affected_subnet_ids: tuple[int, ...] = ()
    is_sensitive_class: bool = True

    def describe(self) -> str:
        services = ", ".join(sorted(self.services)) or "(none)"
        paths = ", ".join(self.available_attack_paths) or "(none)"
        subnets = ", ".join(str(s) for s in self.affected_subnet_ids) or "any"
        return (
            f"OS={self.os} services=[{services}] -> available: {paths}; "
            f"missing: {self.missing_capability} (affected sensitive subnets: {subnets})"
        )


@dataclass(frozen=True)
class NetworkReachabilityFailure:
    """A structural reason some subnet that may hold a sensitive host is
    not guaranteed reachable from the attacker's starting foothold.
    """

    target_subnet_id: int
    required_path: tuple[int, ...]
    blocking_edge: tuple[int, int] | None
    reason: str

    def describe(self) -> str:
        path = " -> ".join("internet" if s == 0 else str(s) for s in self.required_path)
        blocking = f"{self.blocking_edge[0]} -> {self.blocking_edge[1]}" if self.blocking_edge else "(none)"
        return (
            f"Sensitive-capable subnet {self.target_subnet_id} is not guaranteed reachable.\n"
            f"  Required path: {path}\n"
            f"  Blocking edge: {blocking}\n"
            f"  Reason: {self.reason}"
        )


@dataclass(frozen=True)
class StepBoundEstimate:
    """A conservative, honestly-labeled bound on action count -- never
    claimed exact unless genuinely proven (spec section 19).
    """

    lower_bound: int | None
    lower_bound_proven: bool
    upper_bound: int | None
    upper_bound_proven: bool


@dataclass
class ScenarioSolvabilityResult:
    """Everything :mod:`marla.scenario.solvability` determines about one
    scenario, for one objective semantics."""

    status: SolvabilityStatus
    universally_solvable: bool
    scenario_path: str
    scenario_format: str  # "v1" | "v2"
    objective: str
    randomized: bool

    host_rootability_failures: list[HostRootabilityFailure] = field(default_factory=list)
    network_reachability_failures: list[NetworkReachabilityFailure] = field(default_factory=list)

    sensitive_subnet_ids: list[int] = field(default_factory=list)
    sensitive_target_count_range: tuple[int, int] | None = None

    structurally_solvable: bool | None = None
    step_bound: StepBoundEstimate | None = None

    notes: list[str] = field(default_factory=list)

    scenario_hash: str = ""
    validator_version: str = ""

    def to_dict(self) -> dict:
        """JSON-serializable structure (spec section 11) -- independent of
        any CLI text formatting."""
        return {
            "status": self.status.value,
            "scenario": self.scenario_path,
            "scenario_format": self.scenario_format,
            "objective": self.objective,
            "universally_solvable": self.universally_solvable,
            "randomized": self.randomized,
            "structurally_solvable": self.structurally_solvable,
            "sensitive_subnet_ids": self.sensitive_subnet_ids,
            "sensitive_target_count_range": (
                list(self.sensitive_target_count_range) if self.sensitive_target_count_range else None
            ),
            "step_bound": (
                None
                if self.step_bound is None
                else {
                    "lower_bound": self.step_bound.lower_bound,
                    "lower_bound_proven": self.step_bound.lower_bound_proven,
                    "upper_bound": self.step_bound.upper_bound,
                    "upper_bound_proven": self.step_bound.upper_bound_proven,
                }
            ),
            "failure_classes": [
                {
                    "os": f.os,
                    "services": sorted(f.services),
                    "processes": sorted(f.processes),
                    "available_attack_paths": list(f.available_attack_paths),
                    "missing_capability": f.missing_capability,
                    "affected_subnet_ids": list(f.affected_subnet_ids),
                }
                for f in self.host_rootability_failures
            ],
            "network_issues": [
                {
                    "target_subnet_id": nf.target_subnet_id,
                    "required_path": list(nf.required_path),
                    "blocking_edge": list(nf.blocking_edge) if nf.blocking_edge else None,
                    "reason": nf.reason,
                }
                for nf in self.network_reachability_failures
            ],
            "host_rootability_issues": [f.describe() for f in self.host_rootability_failures],
            "notes": self.notes,
            "scenario_hash": self.scenario_hash,
            "validator_version": self.validator_version,
        }
