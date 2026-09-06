"""Minimal, deterministic scenario repair (spec sections 12-18).

Operates on the scenario's real YAML structure (parsed with
``yaml.safe_load``, per-key line ranges located in the original text), never
via blind string substitution, and never overwrites the source file. Only
Priority 1 ("add the missing vulnerability/attack capability", spec section
13) is implemented -- the only class this repository's own scenarios
actually need (see :mod:`marla.scenario.solvability`'s worked analysis of
``sm_entry_user_three_subnets.v2.yaml``). Firewall/topology repairs
(Priorities 3-4) are intentionally NOT implemented: every failure class this
checker can currently produce reduces to a missing exploit/privilege-
escalation capability (see the module docstring of
:mod:`marla.scenario.solvability` -- a "network reachability" failure here
is always caused by some (service, OS) pair having no exploit at all, the
exact same root cause a rootability failure has, so the same Priority-1 fix
resolves both). If a future failure class genuinely required a firewall or
topology change, :func:`plan_repair` raises :class:`RepairNotSupported`
rather than silently attempting something this module was never designed to
verify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from marla.scenario.models import ScenarioSolvabilityResult
from marla.scenario.solvability import ROOT, USER, check_solvability
from marla.scenario.spec import ScenarioSpec, load_scenario_spec

REPAIR_PREFIX = "marla_repair"


class RepairNotSupported(Exception):
    """Raised when a failure class has no implemented minimal repair."""


class RepairVerificationError(Exception):
    """Raised when a generated repair candidate fails to load in NASimEmu
    or fails re-validation -- the caller must not treat this as success
    (spec section 18)."""


@dataclass(frozen=True)
class RepairChange:
    section: str  # "privilege_escalation" | "exploits"
    name: str
    definition: dict
    reason: str

    def describe(self) -> str:
        fields = "\n".join(
            f"      {k}: {'null' if v is None else v}" for k, v in self.definition.items()
        )
        return f"  + {self.section}.{self.name}\n{fields}\n  Reason:\n    {self.reason}"


@dataclass
class RepairPlan:
    changes: list[RepairChange] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.changes


def _privesc_name(os: str) -> str:
    return f"{REPAIR_PREFIX}_{os}_root_privesc"


def _exploit_name(os: str, service: str) -> str:
    return f"{REPAIR_PREFIX}_{os}_{service}_root_exploit"


def plan_repair(spec: ScenarioSpec, result: ScenarioSolvabilityResult) -> RepairPlan:
    """Derive the minimal set of structural changes needed, deterministically.

    Preference order (spec section 15), applied per distinct (os, missing
    capability) class, sorted by OS name for determinism:

    1. If USER access is already achievable for this OS but no compatible
       privilege escalation exists, add exactly one artificial privilege
       escalation for that OS (``process: null`` -- unconstrained, so it
       cannot fail to match a generated host -- see spec section 14). This
       is the exact repair worked through in spec section 16.
    2. If NO exploit at all exists for some (service, OS) pair required by
       a failure class, add exactly one artificial ROOT exploit reusing
       that existing service name (never inventing a new service).

    Each distinct (os) needing a privesc gets exactly one privesc, even if
    multiple failure classes for that OS exist -- adding it once already
    fixes every USER-but-no-privesc failure for that OS.
    """
    if result.universally_solvable:
        return RepairPlan(changes=[])

    plan = RepairPlan()
    seen_privesc_os: set[str] = set()
    seen_exploit_targets: set[tuple[str, str]] = set()

    for failure in sorted(result.host_rootability_failures, key=lambda f: (f.os, sorted(f.services))):
        os = failure.os
        if "no compatible privilege escalation" in (failure.missing_capability or ""):
            if os in seen_privesc_os:
                continue
            seen_privesc_os.add(os)
            plan.changes.append(
                RepairChange(
                    section="privilege_escalation",
                    name=_privesc_name(os),
                    definition={"process": None, "os": os, "prob": 1.0, "cost": 1, "access": "root"},
                    reason=(
                        f"Sensitive {os} hosts can be generated with only USER-level exploitable "
                        f"services (e.g. services={sorted(failure.services)}, attack paths="
                        f"{list(failure.available_attack_paths)}). No {os} privilege escalation to "
                        "ROOT existed. This is an artificial, scenario-level capability added only "
                        "to guarantee benchmark solvability -- it is not a real vulnerability."
                    ),
                )
            )
        elif "no exploit reaches" in (failure.missing_capability or ""):
            # Reuse the alphabetically-first already-installed service in
            # the failing class that has no exploit at all for this OS, so
            # the added exploit can only ever match a legally-generatable
            # host (spec section 14: never add an exploit that can't match).
            existing_services_without_exploit = sorted(
                s for s in failure.services if not any(e.service == s for e in spec.exploits)
            )
            if not existing_services_without_exploit:
                # every installed service already has *some* exploit for
                # some OS, just not this one -- reuse the first service.
                existing_services_without_exploit = sorted(failure.services)
            service = existing_services_without_exploit[0]
            key = (os, service)
            if key in seen_exploit_targets:
                continue
            seen_exploit_targets.add(key)
            plan.changes.append(
                RepairChange(
                    section="exploits",
                    name=_exploit_name(os, service),
                    definition={"service": service, "os": os, "prob": 1.0, "cost": 1, "access": "root"},
                    reason=(
                        f"No exploit exists for service={service!r} on OS={os}, so a host installing "
                        "only this service cannot be compromised at all. Added a direct-ROOT exploit "
                        "reusing the existing service name (spec section 14: never a new service). "
                        "This is an artificial, scenario-level capability added only to guarantee "
                        "benchmark solvability -- it is not a real vulnerability."
                    ),
                )
            )
        else:
            raise RepairNotSupported(
                f"No minimal repair implemented for failure class: {failure.describe()}"
            )

    if result.network_reachability_failures and plan.is_empty:
        # This checker only ever produces a network failure as a side
        # effect of a missing-exploit gap (see module docstring) -- if one
        # occurs with no corresponding host failure captured above, this is
        # a genuinely new failure shape this module was not designed for.
        raise RepairNotSupported(
            "Network reachability failure with no corresponding host-rootability failure -- "
            "firewall/topology repair (Priority 3/4) is not implemented. Manual scenario "
            "changes are required: "
            + "; ".join(nf.describe() for nf in result.network_reachability_failures)
        )

    return plan


def _find_top_level_key_range(lines: list[str], key: str) -> tuple[int, int]:
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"{key}:"):
            start = i
            break
    if start is None:
        raise RepairNotSupported(f"Could not find top-level key {key!r} in scenario file to repair")
    end = len(lines)
    for j in range(start + 1, len(lines)):
        stripped = lines[j]
        if stripped.strip() and not stripped[0].isspace() and not stripped.startswith("#"):
            end = j
            break
    return start, end


def _format_block(name: str, definition: dict) -> str:
    out = [f"  {name}:\n"]
    for k, v in definition.items():
        rendered = "~" if v is None else v
        out.append(f"    {k}: {rendered}\n")
    return "".join(out)


def _apply_changes_to_text(raw_content: str, changes: list[RepairChange]) -> str:
    """Insert each change's block at the end of its section, leaving every
    other line of the file (including all comments/formatting) untouched."""
    lines = raw_content.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    by_section: dict[str, list[RepairChange]] = {}
    for change in changes:
        by_section.setdefault(change.section, []).append(change)

    # Insert from the bottom-most section upward so earlier insertions
    # don't shift the line indices of sections not yet processed.
    insertions = []
    for section, section_changes in by_section.items():
        start, end = _find_top_level_key_range(lines, section)
        block = "".join(_format_block(c.name, c.definition) for c in section_changes)
        insertions.append((end, block))

    for end_idx, block in sorted(insertions, key=lambda x: -x[0]):
        lines.insert(end_idx, block)

    return "".join(lines)


def _default_repaired_path(source: Path) -> Path:
    suffixes = "".join(source.suffixes)  # e.g. ".v2.yaml"
    stem = source.name[: -len(suffixes)] if suffixes else source.stem
    if ".v2" in source.suffixes:
        return source.with_name(f"{stem}.solvable.v2.yaml")
    return source.with_name(f"{stem}.solvable.yaml")


@dataclass
class RepairOutcome:
    repaired: bool
    output_path: Path | None
    plan: RepairPlan
    before: ScenarioSolvabilityResult
    after: ScenarioSolvabilityResult | None
    message: str


def repair_scenario(
    scenario_path: str | Path,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> RepairOutcome:
    """Analyze ``scenario_path``; if unsolvable, write a minimally-repaired
    copy and re-verify it through the real NASimEmu loader plus this same
    checker (spec section 18) before ever reporting success.
    """
    scenario_path = Path(scenario_path)
    spec = load_scenario_spec(scenario_path)
    before = check_solvability(spec)

    if before.universally_solvable:
        return RepairOutcome(
            repaired=False,
            output_path=None,
            plan=RepairPlan(changes=[]),
            before=before,
            after=before,
            message=f"{scenario_path} is already universally solvable -- no repair needed.",
        )

    plan = plan_repair(spec, before)
    if plan.is_empty:
        raise RepairNotSupported("Scenario is unsolvable but no repair plan could be derived.")

    out_path = Path(output_path) if output_path is not None else _default_repaired_path(scenario_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Pass an explicit --output path or --overwrite to replace it."
        )

    repaired_content = _apply_changes_to_text(spec.raw_content, plan.changes)

    # Structural round-trip sanity check before ever writing to disk.
    yaml.safe_load(repaired_content)

    out_path.write_text(repaired_content, encoding="utf-8")

    try:
        repaired_spec = load_scenario_spec(out_path)
        # Spec section 18 step 1-2: load through the REAL NASimEmu loader
        # too, so a structurally-valid-but-NASimEmu-rejected file is caught.
        import nasimemu.nasim as nasim

        nasim.load_scenario(str(out_path))
        after = check_solvability(repaired_spec)
    except Exception as exc:
        out_path.unlink(missing_ok=True)
        raise RepairVerificationError(
            f"Repaired candidate at {out_path} failed verification and was removed: {exc}"
        ) from exc

    if not after.universally_solvable:
        out_path.unlink(missing_ok=True)
        raise RepairVerificationError(
            f"Repaired candidate at {out_path} is still not universally solvable "
            f"({after.status.value}) after applying the planned changes; removed. "
            f"Remaining issues: {[f.describe() for f in after.host_rootability_failures]}"
        )

    return RepairOutcome(
        repaired=True,
        output_path=out_path,
        plan=plan,
        before=before,
        after=after,
        message=f"Wrote {out_path}, PROVEN_SOLVABLE after repair.",
    )


def format_repair_report(scenario_path: Path, outcome: RepairOutcome) -> str:
    lines = [
        "Original:",
        f"  {scenario_path}",
        "",
    ]
    if not outcome.repaired:
        lines.append(outcome.message)
        return "\n".join(lines)

    lines += [
        "Generated:",
        f"  {outcome.output_path}",
        "",
        "Changes:",
    ]
    for change in outcome.plan.changes:
        lines.append(change.describe())
        lines.append("")
    lines.append(f"Universal solvability after repair: {outcome.after.status.value.upper()}")
    return "\n".join(lines)
