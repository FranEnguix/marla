"""``marla scenario`` command group: solvability checking and repair.

All CLI formatting lives here, deliberately separate from the typed
analysis results in :mod:`marla.scenario.models` -- the checker itself
never depends on any of these strings (spec section 11).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from marla.scenario.models import ScenarioSolvabilityResult, SolvabilityStatus
from marla.scenario.repair import (
    RepairNotSupported,
    RepairVerificationError,
    format_repair_report,
    repair_scenario,
)
from marla.scenario.solvability import check_solvability
from marla.scenario.spec import load_scenario_spec
from marla.scenarios.uri import ScenarioReferenceError, resolve_scenario_reference

scenario_app = typer.Typer(help="Validate and repair NASimEmu scenario solvability.")


def format_check_report(result: ScenarioSolvabilityResult, verbose: bool = False) -> str:
    lines = [
        f"Scenario: {result.scenario_path}",
        f"Scenario format: {result.scenario_format}"
        + (" (randomized)" if result.randomized else " (static)"),
        f"Objective: {result.objective}",
        f"Universal solvability: {result.status.value.upper()}",
    ]
    if result.sensitive_subnet_ids:
        lines.append(f"Possible sensitive-host locations: subnets {result.sensitive_subnet_ids}")
    else:
        lines.append("Possible sensitive-host locations: (none -- vacuous objective)")
    if result.sensitive_target_count_range:
        lo, hi = result.sensitive_target_count_range
        lines.append(f"Possible sensitive target count: {lo}-{hi}")
    lines.append(
        "Network reachability: "
        + ("OK" if not result.network_reachability_failures else f"{len(result.network_reachability_failures)} issue(s)")
    )
    lines.append(
        "Host rootability: "
        + ("OK" if not result.host_rootability_failures else "FAILED")
    )
    if result.step_bound is not None:
        sb = result.step_bound
        lower = f"{sb.lower_bound}{'' if sb.lower_bound_proven else ' (unproven)'}"
        upper = f"{sb.upper_bound}{'' if sb.upper_bound_proven else ' (unproven)'}"
        lines.append(f"Step bound: lower>={lower}, upper<={upper}")

    if result.status == SolvabilityStatus.PROVEN_SOLVABLE:
        lines += [
            "",
            "Scenario solvability check PASSED",
            "",
            "All realizations permitted by this scenario have at least one valid path to:",
            "  ROOT on every sensitive host.",
        ]
        if verbose and result.notes:
            lines.append("")
            lines.append("Notes:")
            for note in result.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)

    lines += ["", "Scenario solvability check FAILED", ""]
    if result.status == SolvabilityStatus.UNKNOWN:
        lines.append("Result: could not be proven solvable or unsolvable.")
        for note in result.notes:
            lines.append(f"  - {note}")
        return "\n".join(lines)

    lines.append("Result:")
    lines.append("  Not every NASimEmu-generated realization can reach the capture_target objective.")
    lines.append("")

    for i, failure in enumerate(result.host_rootability_failures, start=1):
        lines.append(f"Failure class {i}")
        lines.append(f"  OS: {failure.os}")
        lines.append(f"  services: {sorted(failure.services)}")
        result_str = "ROOT unattainable" if not failure.available_attack_paths else "USER attainable, ROOT unattainable"
        lines.append(f"  result: {result_str}")
        lines.append(f"  available attack paths: {list(failure.available_attack_paths) or '(none)'}")
        lines.append(f"  missing: {failure.missing_capability}")
        lines.append(f"  affected sensitive subnets: {list(failure.affected_subnet_ids)}")
        lines.append("")

    for i, nf in enumerate(result.network_reachability_failures, start=1):
        lines.append(f"Network issue {i}")
        for line in nf.describe().splitlines():
            lines.append(f"  {line}")
        lines.append("")

    lines.append("To generate the smallest repaired scenario, run:")
    lines.append("")
    lines.append(f"  marla scenario repair {result.scenario_path}")

    if verbose and result.notes:
        lines.append("")
        lines.append("Notes:")
        for note in result.notes:
            lines.append(f"  - {note}")

    return "\n".join(lines)


def format_preflight_failure(result: ScenarioSolvabilityResult) -> str:
    """Concise version of :func:`format_check_report`, for ``marla run``'s
    preflight failure (spec section 8) -- same content, shorter, always
    ends with the exact repair command and the "no agents started" note.
    """
    lines = [
        "Scenario solvability check FAILED",
        "",
        "Scenario:",
        f"  {result.scenario_path}",
        "",
    ]
    if result.status == SolvabilityStatus.UNKNOWN:
        lines.append("Result:")
        lines.append("  Solvability could not be proven (UNKNOWN) -- treated as unsafe to run.")
        for note in result.notes:
            lines.append(f"  {note}")
    else:
        lines.append("Result:")
        lines.append("  Not every NASimEmu-generated realization can reach the "
                      f"{result.objective} objective.")
        lines.append("")
        for i, failure in enumerate(result.host_rootability_failures, start=1):
            lines.append(f"Problem {i}:")
            lines.append(f"  Sensitive {failure.os} hosts may be generated without a ROOT-capable attack path.")
            lines.append("")
            lines.append("  Example unsolvable host configuration:")
            lines.append(f"    OS: {failure.os}")
            lines.append("    services:")
            for s in sorted(failure.services):
                lines.append(f"      - {s}")
            lines.append("")
            lines.append(f"  Available attack chain: {', '.join(failure.available_attack_paths) or '(none)'}")
            lines.append("")
            lines.append(f"  Missing: {failure.missing_capability}")
            lines.append("")
        for i, nf in enumerate(result.network_reachability_failures, start=1):
            lines.append(f"Network problem {i}:")
            for line in nf.describe().splitlines():
                lines.append(f"  {line}")
            lines.append("")
        lines.append(
            "Because this host configuration may occur on a sensitive host, the scenario is "
            "not universally solvable."
        )
    lines.append("")
    lines.append("No MARLA agents were started.")
    lines.append("")
    lines.append("To generate the smallest repaired scenario, run:")
    lines.append("")
    lines.append(f"  marla scenario repair {result.scenario_path}")
    return "\n".join(lines)


def _resolve_cli_scenario_argument(reference: str) -> Path:
    """Resolves a scenario CLI argument -- a ``marla://...`` reference or a
    plain filesystem path (relative to the current working directory,
    matching Typer's previous bare-``Path`` behavior) -- to a real file,
    or exits with an actionable error (spec: ``marla scenario check``/
    ``repair`` must resolve ``marla://`` identically to every other call
    site, via the one authoritative resolver).
    """
    try:
        resolved = resolve_scenario_reference(reference, config_dir=Path.cwd())
    except ScenarioReferenceError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    return Path(resolved)


@scenario_app.command("check")
def check(
    scenario: str = typer.Argument(
        ..., help="Path to a NASimEmu scenario YAML file, or a marla://<name> reference."
    ),
    objective: str = typer.Option(
        "capture_target", "--objective", help="Objective semantics to check against (defaults to MARLA's current capture_target)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON instead of a text report."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Include extra diagnostic notes."),
) -> None:
    """Analyze whether every NASimEmu realization of SCENARIO can reach the
    configured objective, without starting any MARLA agents."""
    scenario = _resolve_cli_scenario_argument(scenario)
    if not scenario.is_file():
        typer.secho(f"Not a file: {scenario}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    spec = load_scenario_spec(scenario)
    result = check_solvability(spec, objective=objective)

    if json_output:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        color = typer.colors.GREEN if result.status == SolvabilityStatus.PROVEN_SOLVABLE else typer.colors.RED
        typer.secho(format_check_report(result, verbose=verbose), fg=color)

    if result.status != SolvabilityStatus.PROVEN_SOLVABLE:
        raise typer.Exit(code=1)


@scenario_app.command("repair")
def repair(
    scenario: str = typer.Argument(
        ..., help="Path to a NASimEmu scenario YAML file, or a marla://<name> reference."
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Path for the repaired scenario (default: <name>.solvable.v2.yaml next to the source)."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite the output path if it already exists (never overwrites the source scenario)."
    ),
) -> None:
    """Produce the smallest repaired copy of SCENARIO that is PROVEN_SOLVABLE.

    Never modifies SCENARIO itself; writes a new file (see --output).
    """
    scenario = _resolve_cli_scenario_argument(scenario)
    if not scenario.is_file():
        typer.secho(f"Not a file: {scenario}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        outcome = repair_scenario(scenario, output_path=output, overwrite=overwrite)
    except FileExistsError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except (RepairNotSupported, RepairVerificationError) as exc:
        typer.secho(f"Repair failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    color = typer.colors.GREEN if (not outcome.repaired or outcome.after.universally_solvable) else typer.colors.RED
    typer.secho(format_repair_report(scenario, outcome), fg=color)

    if outcome.repaired and not outcome.after.universally_solvable:
        # Should be unreachable -- repair_scenario() itself raises
        # RepairVerificationError in this case -- but never claim success
        # if it somehow weren't.
        raise typer.Exit(code=1)
