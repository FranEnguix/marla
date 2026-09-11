"""MARLA command-line interface, built with Typer.

Commands: ``run``, ``validate``, ``summarize``, ``version``, and the
``scenario`` command group (``check``/``repair``). Also invocable as
``python -m marla``.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

from marla import __version__
from marla.config.loader import ConfigError, load_config
from marla.config.models import Config, effective_batch_size
from marla.config.templates import render_templates
from marla.runtime.device import DeviceResolutionError, resolve_device
from marla.scenario.models import SolvabilityStatus
from marla.scenario.preflight import preflight_check
from marla.optuna_cli import register_optimize_command, study_app
from marla.scenario_cli import format_preflight_failure, scenario_app
from marla.scenarios.uri import MARLA_SCENARIO_URI_PREFIX
from marla.utils.formatting import format_quantity
from marla.utils.python_version import UnsupportedPythonVersionError, check_python_version

app = typer.Typer(
    name="marla",
    help="MARLA: Multi-Agent Reinforcement Learning Architecture for Offensive AI.",
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(scenario_app, name="scenario")
app.add_typer(study_app, name="study")
register_optimize_command(app)


@app.callback()
def _main() -> None:
    """Enforce the supported Python version before any subcommand runs."""
    try:
        check_python_version()
    except UnsupportedPythonVersionError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


# Third-party libraries this process pulls in (SPADE/slixmpp, pyjabber,
# transformers, sqlalchemy) all log at INFO/DEBUG too; without this they'd
# drown out MARLA's own progress messages the moment logging is turned on.
_NOISY_THIRD_PARTY_LOGGERS = (
    "slixmpp", "spade", "pyjabber", "sqlalchemy", "alembic", "transformers", "urllib3", "httpx", "asyncio",
)


def _configure_progress_logging() -> None:
    """Make ``marla``'s own INFO-level progress logging visible on stdout.

    A run with no elapsed timeout (spec section 5) can legitimately run for
    a long time between anything printed by ``run()`` itself -- a rollout
    is one ``await``, and an assisted rollout's per-step Plan Maker
    consultation is a real, possibly slow, network+inference round trip in
    the middle of it. Without this, there was nothing to distinguish "still
    working" from "hung" for the entire duration of a rollout.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _print_config_error(exc: ConfigError) -> None:
    typer.secho("Configuration is invalid:", fg=typer.colors.RED, err=True)
    for problem in exc.problems:
        typer.secho(f"  - {problem}", fg=typer.colors.RED, err=True)


def _resolve_agent_selectors(config: Config, selectors: list[str]) -> list[str]:
    """Resolve ``--agent`` values (alias or JID) to canonical aliases.

    Raises ``typer.BadParameter`` on unknown or duplicate selectors.
    """
    known: dict[str, str] = {config.rl_orchestrator.alias: config.rl_orchestrator.alias}
    known[config.rl_orchestrator.jid] = config.rl_orchestrator.alias
    if config.gatekeeper is not None:
        known[config.gatekeeper.alias] = config.gatekeeper.alias
        known[config.gatekeeper.jid] = config.gatekeeper.alias
    for agent in config.agents:
        known[agent.alias] = agent.alias
        known[agent.jid] = agent.alias

    resolved: list[str] = []
    seen: set[str] = set()
    for selector in selectors:
        if selector not in known:
            raise typer.BadParameter(
                f"--agent '{selector}' does not match any configured alias or JID"
            )
        alias = known[selector]
        if alias in seen:
            raise typer.BadParameter(f"--agent '{selector}' selected more than once")
        seen.add(alias)
        resolved.append(alias)
    return resolved


@app.command()
def init(
    directory: Path = typer.Argument(
        Path("experiment_templates"), help="Directory to create with starter baseline/assisted configs."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite template files that already exist."),
) -> None:
    """Generate starter baseline/assisted config templates for a quick first run."""
    conflicts = [directory / name for name in ("baseline.yaml", "assisted.yaml") if (directory / name).exists()]
    if conflicts and not force:
        typer.secho("Refusing to overwrite existing files (use --force to overwrite):", fg=typer.colors.RED, err=True)
        for path in conflicts:
            typer.secho(f"  - {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Always the MARLA-owned, pre-validated scenario (spec: "do not
    # generate configs that immediately fail MARLA's own preflight
    # validation"), named through the portable marla:// scheme (see
    # marla.scenarios.uri) rather than a materialized filesystem path --
    # the generated config keeps working unchanged no matter where it's
    # later copied to or which MARLA install (checkout, editable install,
    # wheel) runs it, unlike the old NASimEmu/scenarios/ filesystem search
    # this replaced (which, even when it found something, found the
    # original *unsolvable* sm_entry_user_three_subnets.v2.yaml).
    scenario_value = f"{MARLA_SCENARIO_URI_PREFIX}sm_entry_user_three_subnets.solvable.v2.yaml"

    directory.mkdir(parents=True, exist_ok=True)
    for name, content in render_templates(scenario_value).items():
        (directory / name).write_text(content, encoding="utf-8")

    typer.secho(f"Created {directory}/baseline.yaml and {directory}/assisted.yaml", fg=typer.colors.GREEN)
    typer.echo("")
    typer.echo("Try the baseline variant (no external setup needed):")
    typer.echo(f"  export MARLA_RL_ORCHESTRATOR_PASSWORD=changeme")
    typer.echo(f"  marla run {directory}/baseline.yaml")
    typer.echo("")
    typer.echo("Try the assisted variant (adds a Gatekeeper + a small local Plan Maker model):")
    typer.echo('  pip install -e ".[local-lm]"   # needed once, for the local Plan Maker backend')
    typer.echo(f"  export MARLA_RL_ORCHESTRATOR_PASSWORD=changeme")
    typer.echo(f"  export MARLA_GATEKEEPER_PASSWORD=changeme")
    typer.echo(f"  export MARLA_PLAN_MAKER_1_PASSWORD=changeme")
    typer.echo(f"  marla run {directory}/assisted.yaml")


@app.command()
def run(
    config_path: Path = typer.Argument(..., exists=False, help="Path to the experiment YAML file."),
    agent: list[str] = typer.Option(
        [], "--agent", help="Alias or JID of an agent to start (distributed mode only, repeatable)."
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Write every Plan Maker query and generated response to debug/<run_id>/.",
    ),
    resume: Path = typer.Option(
        None,
        "--resume",
        help=(
            "Path to a checkpoint.pt to resume from (local mode only): loads policy/optimizer/RNG "
            "state and continues training toward this config's ppo.total_environment_steps, rather "
            "than starting over. The config's own hyperparameters are used for the resumed portion; "
            "only ppo.total_environment_steps is expected to differ from what originally produced "
            "the checkpoint (research/aamas2027's staged-training design)."
        ),
    ),
) -> None:
    """Run an experiment (baseline or assisted, local or distributed)."""
    _configure_progress_logging()
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        _print_config_error(exc)
        raise typer.Exit(code=1) from exc

    if config.execution.mode == "local" and agent:
        typer.secho(
            "local mode does not accept --agent (all configured agents are started together)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    if config.execution.mode == "distributed" and not agent:
        typer.secho(
            "distributed mode requires at least one --agent",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    selected_aliases = _resolve_agent_selectors(config, agent) if agent else None

    try:
        resolved_device = resolve_device(config.device)
    except DeviceResolutionError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Config OK: experiment={config.experiment.name!r} "
        f"mode={config.execution.mode} consultation={config.consultation.mode} "
        f"device(requested={resolved_device.requested}, resolved={resolved_device.resolved})"
    )
    if selected_aliases is not None:
        typer.echo(f"Selected agents: {selected_aliases}")

    # Preflight scenario-solvability check (spec: "MARLA must never start a
    # training/evaluation run on a scenario for which one or more valid
    # NASimEmu-generated realizations cannot reach the configured success
    # objective"). Runs before any agent -- RL Orchestrator, Gatekeeper,
    # Plan Maker, XMPP -- is started, for both local and distributed mode.
    # Deliberately quiet on success (a few lines); a failure gets the full
    # diagnostic plus the exact repair command, and no agents are started.
    from marla.runtime.local import resolve_scenario_path

    scenario_path = resolve_scenario_path(config, config_path.parent)
    typer.echo("Validating scenario solvability...")
    try:
        solvability_result = preflight_check(config, config_path.parent, scenario_path)
    except Exception as exc:
        typer.secho(f"Scenario solvability check could not be completed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if solvability_result.status != SolvabilityStatus.PROVEN_SOLVABLE:
        typer.secho(format_preflight_failure(solvability_result), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho("Scenario solvability: PASSED", fg=typer.colors.GREEN)
    # Compact only (spec section 31) -- proof a paper run used a validated
    # scenario, not a copy of the (potentially large) diagnostic result.
    scenario_validation_metadata = {
        "status": solvability_result.status.value,
        "validator_version": solvability_result.validator_version,
        "scenario_hash": solvability_result.scenario_hash,
    }

    if resume is not None and config.execution.mode != "local":
        typer.secho("--resume is only supported in local mode", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if config.execution.mode == "local":
        import math

        from marla.learning.checkpoint import peek_checkpoint_metadata
        from marla.metrics.writer import finalize_run_directory, initialize_run_directory
        from marla.runtime.local import LocalRunError, resolve_run_dir, run_local

        run_dir = resolve_run_dir(config)
        ppo = config.policy.ppo
        already_done = 0
        if resume is not None:
            if not resume.is_file():
                typer.secho(f"--resume checkpoint not found: {resume}", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1)
            already_done = peek_checkpoint_metadata(resume).environment_steps
            if already_done >= ppo.total_environment_steps:
                typer.secho(
                    f"--resume checkpoint already has {already_done} environment steps, "
                    f">= this config's ppo.total_environment_steps ({ppo.total_environment_steps}); nothing to do.",
                    fg=typer.colors.RED, err=True,
                )
                raise typer.Exit(code=1)
        # total_environment_steps is a GLOBAL budget across every
        # independent environment stream, never per-stream -- for
        # num_envs=4 this must count 4*steps_per_env transitions per
        # collection cycle, not just steps_per_env (which would silently
        # run 4x the intended total). effective_batch_size(ppo) ==
        # ppo.steps_per_env exactly when num_envs=1.
        batch_size = effective_batch_size(ppo)
        num_rollouts = math.ceil((ppo.total_environment_steps - already_done) / batch_size)
        if resume is not None:
            typer.echo(
                f"Resuming from {resume} ({already_done} environment step(s) already done): "
                f"{num_rollouts} more rollout(s) of {batch_size} steps each "
                f"(target: {ppo.total_environment_steps})."
            )
        else:
            typer.echo(f"Starting local {config.consultation.mode} run: {num_rollouts} rollout(s) of {batch_size} steps each.")
        start_time = datetime.now(timezone.utc)
        # Spec section 2: resolved config, provisional metadata.json, and
        # every metrics CSV's header row exist before a single environment
        # step runs -- the training loop (run_training_loop, via the RL
        # Orchestrator) then flushes each rollout's results here as it goes,
        # so an interrupted run's results up to the last completed rollout
        # are already on disk however the process ends.
        initialize_run_directory(run_dir, config, resolved_device, start_time, scenario_validation_metadata)
        try:
            orchestrator = run_local(
                config, config_path.parent, num_rollouts=num_rollouts, debug=debug,
                resume_from=resume, run_dir=run_dir,
            )
        except LocalRunError as exc:
            finalize_run_directory(
                run_dir, config, None, resolved_device, start_time, datetime.now(timezone.utc), "failed",
                scenario_validation_metadata,
            )
            typer.secho(f"Run failed: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

        result = orchestrator.training_result
        status = "stopped_by_user" if result.stopped_by_user else "completed"
        finalize_run_directory(
            run_dir, config, result, resolved_device, start_time, datetime.now(timezone.utc), status,
            scenario_validation_metadata,
        )
        typer.echo(f"Metrics written to {run_dir}/")

        verb = "Stopped by user" if result.stopped_by_user else "Run complete"
        typer.secho(
            f"{verb}: {result.environment_steps} environment steps, "
            f"{len(result.episode_summaries)} episodes.",
            fg=typer.colors.GREEN,
        )
        return

    import math

    from marla.metrics.writer import finalize_run_directory, initialize_run_directory
    from marla.runtime.distributed import DistributedRunError, run_distributed
    from marla.runtime.local import resolve_run_dir

    run_dir = resolve_run_dir(config)
    ppo = config.policy.ppo
    num_rollouts = math.ceil(ppo.total_environment_steps / effective_batch_size(ppo))
    assert selected_aliases is not None  # enforced above: distributed mode requires --agent
    owns_orchestrator = config.rl_orchestrator.alias in selected_aliases
    typer.echo(f"Starting distributed process for agents: {selected_aliases}")
    start_time = datetime.now(timezone.utc)
    if owns_orchestrator:
        # Only the process hosting the RL Orchestrator writes central
        # metrics (spec section 21) -- see the incremental-persistence note
        # on the local-mode branch above for what this enables.
        initialize_run_directory(run_dir, config, resolved_device, start_time, scenario_validation_metadata)
    try:
        orchestrator = run_distributed(
            config, config_path.parent, set(selected_aliases), num_rollouts=num_rollouts, debug=debug,
            run_dir=run_dir if owns_orchestrator else None,
        )
    except DistributedRunError as exc:
        if owns_orchestrator:
            finalize_run_directory(
                run_dir, config, None, resolved_device, start_time, datetime.now(timezone.utc), "failed",
                scenario_validation_metadata,
            )
        typer.secho(f"Run failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if orchestrator is not None:
        # Spec section 21: the RL Orchestrator is the only central metrics
        # writer -- a Gatekeeper- or Plan-Maker-only process never does this.
        result = orchestrator.training_result
        status = "stopped_by_user" if result.stopped_by_user else "completed"
        finalize_run_directory(
            run_dir, config, result, resolved_device, start_time, datetime.now(timezone.utc), status,
            scenario_validation_metadata,
        )
        typer.echo(f"Metrics written to {run_dir}/")

        verb = "Stopped by user" if result.stopped_by_user else "Run complete"
        typer.secho(
            f"{verb}: {result.environment_steps} environment steps, "
            f"{len(result.episode_summaries)} episodes.",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho("Process complete (no RL Orchestrator in this process).", fg=typer.colors.GREEN)


@app.command()
def validate(
    config_path: Path = typer.Argument(..., exists=False, help="Path to the experiment YAML file."),
) -> None:
    """Validate an experiment YAML file without running anything."""
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        _print_config_error(exc)
        raise typer.Exit(code=1) from exc

    typer.secho(f"OK: {config_path} is a valid MARLA configuration.", fg=typer.colors.GREEN)
    typer.echo(f"  experiment: {config.experiment.name} (run_id={config.experiment.run_id})")
    typer.echo(f"  execution.mode: {config.execution.mode}")
    typer.echo(f"  consultation.mode: {config.consultation.mode}")
    typer.echo(f"  device: {config.device}")


@app.command()
def summarize(
    run_or_directory: Path = typer.Argument(
        ..., exists=False, help="Path to a run directory (runs/<experiment>/<run-id>)."
    ),
) -> None:
    """Summarize metrics for a completed run and plot them (episodes.csv / summary.json)."""
    import json

    from marla.metrics.plots import generate_plots

    if not run_or_directory.is_dir():
        typer.secho(f"Not a directory: {run_or_directory}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    summary_path = run_or_directory / "summary.json"
    metadata_path = run_or_directory / "metadata.json"
    if not summary_path.is_file():
        typer.secho(
            f"No summary.json in {run_or_directory} -- is this a run directory written by `marla run`?",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        typer.secho(
            f"Run {metadata['run_id']!r} ({metadata['variant']}, {metadata['status']})", fg=typer.colors.GREEN
        )
        typer.echo(f"  scenario: {metadata['scenario']}")
        typer.echo(f"  device: requested={metadata['device_requested']} resolved={metadata['device_resolved']}")

    # _fmt is a bare-number alias of format_quantity (unit="") -- every
    # call below that appends a unit suffix uses format_quantity directly
    # so the suffix is only ever glued onto an actual number, never onto
    # the literal string "n/a" (see marla.utils.formatting's docstring).
    def _fmt(value, digits: int = 3) -> str:
        return format_quantity(value, digits=digits)

    typer.echo(f"  episodes: {summary['episode_count']}")
    # Reported separately, never collapsed into one number (spec section
    # 25): objective_reached doesn't require FINISH; successful_finish
    # does. See EpisodeSummary's docstring for the exact distinction.
    obj_ci_low, obj_ci_high = summary.get("objective_reached_rate_ci_low"), summary.get("objective_reached_rate_ci_high")
    obj_ci_suffix = f" (95% CI {_fmt(obj_ci_low)}-{_fmt(obj_ci_high)})" if obj_ci_low is not None else ""
    typer.echo(f"  objective reached rate: {_fmt(summary.get('objective_reached_rate'))}{obj_ci_suffix}")
    ci_low, ci_high = summary.get("goal_success_rate_ci_low"), summary.get("goal_success_rate_ci_high")
    ci_suffix = f" (95% CI {_fmt(ci_low)}-{_fmt(ci_high)})" if ci_low is not None else ""
    typer.echo(f"  successful finish rate: {_fmt(summary.get('successful_finish_rate', summary['goal_success_rate']))}{ci_suffix}")
    if summary.get("premature_finish_rate") is not None:
        typer.echo(
            f"  premature finish rate: {_fmt(summary['premature_finish_rate'])}  "
            f"timeout rate: {_fmt(summary.get('timeout_rate'))}"
        )
    typer.echo(f"  mean benchmark return: {_fmt(summary['mean_benchmark_return'])}")
    typer.echo(f"  mean episode duration: {format_quantity(summary['mean_episode_duration_seconds'], 's')}")
    typer.echo(f"  total consultations: {summary['total_consultations']}")
    if summary.get("queries_per_successful_episode") is not None:
        typer.echo(f"  queries per successful episode: {_fmt(summary['queries_per_successful_episode'])}")
    typer.echo(f"  mean Plan Maker latency: {format_quantity(summary['mean_plan_maker_latency_ms'], 'ms')}")
    if summary.get("p95_plan_maker_latency_ms") is not None:
        typer.echo(f"  p95 Plan Maker latency: {format_quantity(summary['p95_plan_maker_latency_ms'], 'ms')}")
    typer.echo(f"  schema rejection rate: {_fmt(summary['schema_rejection_rate'])}")
    if summary.get("advice_acceptance_rate") is not None:
        typer.echo(f"  advice acceptance rate: {_fmt(summary['advice_acceptance_rate'])}")
    typer.echo(f"  advice changed top action rate: {_fmt(summary['advice_changed_top_action_rate'])}")
    if summary.get("eval_episode_count"):  # absent in summary.json written before eval episodes existed
        typer.echo(f"  eval episodes: {summary['eval_episode_count']}")
        typer.echo(f"  eval objective reached rate: {_fmt(summary.get('eval_objective_reached_rate'))}")
        typer.echo(
            f"  eval successful finish rate: "
            f"{_fmt(summary.get('eval_successful_finish_rate', summary.get('eval_goal_success_rate')))}"
        )
        typer.echo(f"  mean eval return: {_fmt(summary.get('mean_eval_return'))}")
    typer.echo(
        f"  total training: {summary['total_training_environment_steps']} steps, "
        f"{format_quantity(summary['total_training_seconds'], 's', digits=1)}"
    )

    carbon_summary_path = run_or_directory / "carbon" / "carbon_summary.json"
    if carbon_summary_path.is_file():
        carbon = json.loads(carbon_summary_path.read_text(encoding="utf-8"))
        if carbon.get("enabled"):
            typer.echo(
                f"  energy consumed: {format_quantity(carbon.get('energy_consumed_kwh'), ' kWh', digits=6)}  "
                f"estimated CO2eq: {format_quantity(carbon.get('emissions_kg_co2eq'), ' kg', digits=6)}"
                "  (CodeCarbon best-effort estimate, not an exact physical measurement)"
            )

    plots_dir = run_or_directory / "plots"
    written = generate_plots(run_or_directory, plots_dir)
    if written:
        typer.secho(f"\nWrote {len(written)} plot(s) to {plots_dir}/:", fg=typer.colors.GREEN)
        for path in written:
            typer.echo(f"  {path.name}")
    else:
        typer.secho("\nNo plots generated (no episode/update/decision data found).", fg=typer.colors.YELLOW)


@app.command()
def version() -> None:
    """Print MARLA and key dependency versions."""
    from marla.utils.versions import collect_dependency_versions, python_version

    typer.echo(f"marla {__version__}")
    typer.echo(f"python {python_version()}")
    for module_name, module_version in collect_dependency_versions().items():
        typer.echo(f"{module_name} {module_version}")
