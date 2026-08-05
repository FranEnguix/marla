"""MARLA command-line interface, built with Typer.

Commands: ``run``, ``validate``, ``summarize``, ``version``. Also invocable
as ``python -m marla``.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

from marla import __version__
from marla.config.loader import ConfigError, load_config
from marla.config.models import Config
from marla.config.templates import find_nasimemu_scenario, render_templates
from marla.runtime.device import DeviceResolutionError, resolve_device
from marla.utils.python_version import UnsupportedPythonVersionError, check_python_version

app = typer.Typer(
    name="marla",
    help="MARLA: Multi-Agent Reinforcement Learning Architecture for Offensive AI.",
    add_completion=False,
    no_args_is_help=True,
)


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

    scenario_path = find_nasimemu_scenario(Path.cwd())
    if scenario_path is None:
        typer.secho(
            "Warning: could not find NASimEmu/scenarios/ near the current directory; "
            "the generated 'environment.scenario' path will need to be fixed by hand "
            "before these configs will run.",
            fg=typer.colors.YELLOW,
        )
        scenario_value = str(Path("NASimEmu") / "scenarios" / "sm_entry_dmz_one_subnet.v2.yaml")
    else:
        scenario_value = str(scenario_path)

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

    if config.execution.mode == "local":
        import math

        from marla.metrics.writer import write_run_artifacts
        from marla.runtime.local import LocalRunError, resolve_run_dir, run_local

        run_dir = resolve_run_dir(config)
        ppo = config.policy.ppo
        num_rollouts = math.ceil(ppo.total_environment_steps / ppo.rollout_steps)
        typer.echo(f"Starting local {config.consultation.mode} run: {num_rollouts} rollout(s) of {ppo.rollout_steps} steps each.")
        start_time = datetime.now(timezone.utc)
        try:
            orchestrator = run_local(config, config_path.parent, num_rollouts=num_rollouts, debug=debug)
        except LocalRunError as exc:
            write_run_artifacts(run_dir, config, None, resolved_device, start_time, datetime.now(timezone.utc), "failed")
            typer.secho(f"Run failed: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

        result = orchestrator.training_result
        status = "stopped_by_user" if result.stopped_by_user else "completed"
        write_run_artifacts(run_dir, config, result, resolved_device, start_time, datetime.now(timezone.utc), status)
        typer.echo(f"Metrics written to {run_dir}/")

        verb = "Stopped by user" if result.stopped_by_user else "Run complete"
        typer.secho(
            f"{verb}: {result.environment_steps} environment steps, "
            f"{len(result.episode_summaries)} episodes.",
            fg=typer.colors.GREEN,
        )
        return

    import math

    from marla.metrics.writer import write_run_artifacts
    from marla.runtime.distributed import DistributedRunError, run_distributed
    from marla.runtime.local import resolve_run_dir

    run_dir = resolve_run_dir(config)
    ppo = config.policy.ppo
    num_rollouts = math.ceil(ppo.total_environment_steps / ppo.rollout_steps)
    assert selected_aliases is not None  # enforced above: distributed mode requires --agent
    owns_orchestrator = config.rl_orchestrator.alias in selected_aliases
    typer.echo(f"Starting distributed process for agents: {selected_aliases}")
    start_time = datetime.now(timezone.utc)
    try:
        orchestrator = run_distributed(
            config, config_path.parent, set(selected_aliases), num_rollouts=num_rollouts, debug=debug
        )
    except DistributedRunError as exc:
        if owns_orchestrator:
            write_run_artifacts(run_dir, config, None, resolved_device, start_time, datetime.now(timezone.utc), "failed")
        typer.secho(f"Run failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if orchestrator is not None:
        # Spec section 21: the RL Orchestrator is the only central metrics
        # writer -- a Gatekeeper- or Plan-Maker-only process never does this.
        result = orchestrator.training_result
        status = "stopped_by_user" if result.stopped_by_user else "completed"
        write_run_artifacts(run_dir, config, result, resolved_device, start_time, datetime.now(timezone.utc), status)
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

    def _fmt(value, digits: int = 3) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)

    typer.echo(f"  episodes: {summary['episode_count']}")
    ci_low, ci_high = summary.get("goal_success_rate_ci_low"), summary.get("goal_success_rate_ci_high")
    ci_suffix = f" (95% CI {_fmt(ci_low)}-{_fmt(ci_high)})" if ci_low is not None else ""
    typer.echo(f"  goal success rate: {_fmt(summary['goal_success_rate'])}{ci_suffix}")
    if summary.get("premature_finish_rate") is not None:
        typer.echo(
            f"  premature finish rate: {_fmt(summary['premature_finish_rate'])}  "
            f"timeout rate: {_fmt(summary.get('timeout_rate'))}"
        )
    typer.echo(f"  mean benchmark return: {_fmt(summary['mean_benchmark_return'])}")
    typer.echo(f"  mean episode duration: {_fmt(summary['mean_episode_duration_seconds'])}s")
    typer.echo(f"  total consultations: {summary['total_consultations']}")
    if summary.get("queries_per_successful_episode") is not None:
        typer.echo(f"  queries per successful episode: {_fmt(summary['queries_per_successful_episode'])}")
    typer.echo(f"  mean Plan Maker latency: {_fmt(summary['mean_plan_maker_latency_ms'])}ms")
    if summary.get("p95_plan_maker_latency_ms") is not None:
        typer.echo(f"  p95 Plan Maker latency: {_fmt(summary['p95_plan_maker_latency_ms'])}ms")
    typer.echo(f"  schema rejection rate: {_fmt(summary['schema_rejection_rate'])}")
    if summary.get("advice_acceptance_rate") is not None:
        typer.echo(f"  advice acceptance rate: {_fmt(summary['advice_acceptance_rate'])}")
    typer.echo(f"  advice changed top action rate: {_fmt(summary['advice_changed_top_action_rate'])}")
    if summary.get("eval_episode_count"):  # absent in summary.json written before eval episodes existed
        typer.echo(f"  eval episodes: {summary['eval_episode_count']}")
        typer.echo(f"  eval goal success rate: {_fmt(summary.get('eval_goal_success_rate'))}")
        typer.echo(f"  mean eval return: {_fmt(summary.get('mean_eval_return'))}")
    typer.echo(
        f"  total training: {summary['total_training_environment_steps']} steps, "
        f"{_fmt(summary['total_training_seconds'], 1)}s"
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
