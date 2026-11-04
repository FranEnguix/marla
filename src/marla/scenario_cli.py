"""``marla scenario`` command group: solvability checking and repair, and
seeded scenario figures (``plot``/``plot-batch``, see
:mod:`marla.visualization.scenario_figures`).

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

scenario_app = typer.Typer(help="Validate, repair and plot NASimEmu scenarios.")


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


def _parse_view(view: str | None, verbose: bool, default: str) -> tuple[str, ...]:
    from marla.visualization.scenario_layout import VIEWS

    chosen = view or ("verbose" if verbose else default)
    if chosen == "both":
        return VIEWS
    if chosen not in VIEWS:
        raise ValueError(f"--view must be default, verbose or both, not {chosen!r}")
    if view is not None and verbose and chosen == "default":
        raise ValueError("--verbose conflicts with --view default")
    return (chosen,)


def _run_scenario(run_dir: Path) -> Path:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"no metadata.json in {run_dir} -- is this a run directory written by `marla run`?")
    scenario = json.loads(metadata_path.read_text(encoding="utf-8")).get("scenario")
    if not scenario:
        raise ValueError(f"{metadata_path} does not record a scenario")
    return Path(scenario)


def _report_batch(result, output_dir: Path) -> None:
    color = typer.colors.GREEN if not result.unreconstructed else typer.colors.YELLOW
    typer.secho(
        f"Rendered {len(result.rendered)} realization(s) into {output_dir}/"
        + (f"; {len(result.skipped_existing)} already present (skipped)" if result.skipped_existing else "")
        + (f"; {result.aliases} seed(s) generate an already-listed identical network (recorded, not re-rendered)"
           if result.aliases else ""),
        fg=color,
    )
    if result.files_written:
        typer.echo("  files: " + ", ".join(f"{n} {fmt.upper()}" for fmt, n in sorted(result.files_written.items())))
    typer.echo(f"  manifest: {result.manifest_path}")
    if result.layout_issue_figures:
        typer.secho(f"  layout warnings in {len(result.layout_issue_figures)} figure(s); see layout_issues in the manifest",
                    fg=typer.colors.YELLOW)
    if result.unreconstructed:
        typer.secho(f"  {len(result.unreconstructed)} seed(s) could NOT be reconstructed reliably; no figure written:",
                    fg=typer.colors.RED, err=True)
        for u in result.unreconstructed[:20]:
            typer.secho(f"    {u['scenario']} seed {u['requested_seed']}: {u['reason']}", fg=typer.colors.RED, err=True)


_FORMAT_HELP = "Comma-separated output formats (pdf, svg, png)."
_SIZE_HELP = ("Figure width: natural (fonts at their nominal print size), single (3.33 in column), "
              "double (7.0 in text width) or a width in inches; content is scaled down, never up.")


@scenario_app.command("plot")
def plot(
    scenario: str = typer.Argument(
        None, help="Scenario YAML path or marla://<name> reference (default: the scenario of --run-dir)."
    ),
    seed: list[int] = typer.Option(None, "--seed", help="Scenario seed to reconstruct (repeatable)."),
    seeds: str = typer.Option(None, "--seeds", help="Comma-separated seeds and inclusive ranges, e.g. 11,12,20-29."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Add each host's sampled attributes (simulator ground truth)."),
    view: str = typer.Option(None, "--view", help="default, verbose or both (overrides --verbose)."),
    run_dir: Path = typer.Option(None, "--run-dir", help="A `marla run` directory: its scenario is plotted and the "
                                 "default output is <run-dir>/plots/scenarios/."),
    output_dir: Path = typer.Option(None, "--output-dir", "-o", help="Output directory (default: plots/scenarios/)."),
    formats: str = typer.Option("pdf,svg,png", "--format", help=_FORMAT_HELP),
    dpi: int = typer.Option(300, "--dpi", help="PNG resolution."),
    size: str = typer.Option("natural", "--size", help=_SIZE_HELP),
    caption: bool = typer.Option(False, "--caption/--no-caption", help="Print scenario, seed and counts under the figure."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace figure files that already exist."),
) -> None:
    """Draw the network NASimEmu generates for SCENARIO at the given seed(s).

    The realization is rebuilt with the real scenario generator (including
    MARLA's zero-target regeneration), so the figure shows the exact network
    an episode with that seed plays on. Default view: topology only; --verbose
    adds every host's sampled OS, services, processes and target status.
    """
    from marla.environment.scenario_realization import ScenarioRealizationError, ScenarioRealizer, validate_scenario_seed
    from marla.visualization.scenario_figures import (
        ScenarioFigureError, ScenarioRequest, parse_formats, parse_seed_list, plot_requests,
    )

    try:
        views = _parse_view(view, verbose, "default")
        fmt = parse_formats(formats)
        if scenario is None and run_dir is None:
            raise ValueError("give a SCENARIO or --run-dir")
        path = _resolve_cli_scenario_argument(scenario) if scenario is not None else _run_scenario(run_dir)
        requested = [validate_scenario_seed(s) for s in (seed or [])] + (parse_seed_list(seeds) if seeds else [])
        if not requested:
            raise ValueError("give at least one --seed or --seeds")
        realizer = ScenarioRealizer(path)
        request = ScenarioRequest(realizer.path)
        for s in requested:
            request.record(s).sources.add("cli")
        out = output_dir or ((run_dir / "plots" / "scenarios") if run_dir is not None else Path("plots") / "scenarios")
        result = plot_requests({realizer.sha256: request}, out, views=views, formats=fmt, dpi=dpi, size=size,
                               caption=caption, deduplicate=False, overwrite=overwrite)
    except (ScenarioFigureError, ScenarioRealizationError, ValueError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    _report_batch(result, out)
    if result.unreconstructed:
        raise typer.Exit(code=1)


@scenario_app.command("plot-batch")
def plot_batch(
    scenario: str = typer.Argument(
        None, help="Scenario YAML path or marla://<name>. Required with --from-census/--seeds; with "
                   "--from-episodes it keeps only that scenario's episodes."
    ),
    from_census: Path = typer.Option(None, "--from-census", help="Scenario census CSV (seed,host,os,services,sensitive): "
                                     "plot its seeds and verify every host against it."),
    from_episodes: list[Path] = typer.Option(None, "--from-episodes", help="A MARLA episodes.csv, or a directory searched "
                                             "recursively for them (runs, evaluations); repeatable."),
    seeds: str = typer.Option(None, "--seeds", help="Additional seeds/ranges for SCENARIO, e.g. 3000000000-3000000199."),
    split: str = typer.Option(None, "--split", help="Only census/episode rows of this split (e.g. test)."),
    eval_only: bool = typer.Option(False, "--eval-only", help="Only episodes flagged is_eval."),
    view: str = typer.Option("both", "--view", help="default, verbose or both."),
    output_dir: Path = typer.Option(Path("plots") / "scenarios", "--output-dir", "-o", help="Output directory."),
    formats: str = typer.Option("pdf,svg,png", "--format", help=_FORMAT_HELP),
    dpi: int = typer.Option(300, "--dpi", help="PNG resolution."),
    size: str = typer.Option("natural", "--size", help=_SIZE_HELP),
    caption: bool = typer.Option(False, "--caption/--no-caption", help="Print scenario, seed and counts under each figure."),
    dedupe: bool = typer.Option(True, "--dedupe/--no-dedupe", help="Render identical realizations once (default)."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace figure files that already exist."),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Keep realizations whose files all exist."),
    jobs: int = typer.Option(1, "--jobs", "-j", help="Parallel rendering processes."),
) -> None:
    """Draw every distinct realization used by an experiment, deduplicated.

    Seeds come from a scenario census, from recorded MARLA episodes (training
    runs or evaluations -- the same seed evaluated by several conditions or
    training seeds is plotted once), and/or --seeds. Each reconstruction is
    checked against what was recorded (census host attributes, episode
    target counts, the run's scenario hash); seeds that do not match are
    reported and get no figure.
    """
    from marla.environment.scenario_realization import ScenarioRealizationError, ScenarioRealizer
    from marla.visualization.scenario_figures import (
        ScenarioFigureError, ScenarioRequest, collect_episode_seeds, parse_formats, parse_seed_list, plot_requests,
        read_census,
    )

    try:
        views = _parse_view(view, False, "both")
        fmt = parse_formats(formats)
        if not (from_census or from_episodes or seeds):
            raise ValueError("give at least one of --from-census, --from-episodes, --seeds")
        realizer = ScenarioRealizer(_resolve_cli_scenario_argument(scenario)) if scenario is not None else None
        if (from_census or seeds) and realizer is None:
            raise ValueError("--from-census and --seeds need a SCENARIO argument")
        requests: dict = {}
        if realizer is not None:
            requests[realizer.sha256] = ScenarioRequest(realizer.path)
        if from_episodes:
            stats = collect_episode_seeds(from_episodes, requests, eval_only=eval_only, split=split, only_scenario=realizer)
            typer.echo(f"Read {stats.get('episodes', 0)} episode row(s) from {stats.get('files', 0)} file(s)"
                       + (f"; ignored {stats['skipped_other_scenario']} row(s) of other scenarios"
                          if stats.get("skipped_other_scenario") else ""))
        if from_census:
            census = read_census(from_census, split=split)
            request = requests[realizer.sha256]
            for s, hosts in census.items():
                rec = request.record(s)
                rec.sources.add("census")
                rec.census_hosts = hosts
        if seeds:
            for s in parse_seed_list(seeds):
                requests[realizer.sha256].record(s).sources.add("cli")
        requests = {sha: r for sha, r in requests.items() if r.seeds}
        if not requests:
            raise ValueError("the inputs select no seeds")
        result = plot_requests(requests, output_dir, views=views, formats=fmt, dpi=dpi, size=size, caption=caption,
                               deduplicate=dedupe, overwrite=overwrite, skip_existing=skip_existing, jobs=jobs,
                               progress=typer.echo)
    except (ScenarioFigureError, ScenarioRealizationError, ValueError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    _report_batch(result, output_dir)
    if result.unreconstructed:
        raise typer.Exit(code=1)
