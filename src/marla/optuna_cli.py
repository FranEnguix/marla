"""``marla optimize`` and the ``marla study`` command group -- runs and
inspects a persistent Optuna PPO hyperparameter study.

All CLI formatting lives here, separate from the study driver logic in
:mod:`marla.optuna_study` (same separation-of-concerns pattern as
``scenario_cli.py``/``marla.scenario``).
"""

from __future__ import annotations

import asyncio

import typer

study_app = typer.Typer(help="Inspect a persistent Optuna PPO hyperparameter study.")


@study_app.command("status")
def study_status(study_config_path: str) -> None:
    """Prints how many trials are COMPLETE/FAIL/RUNNING/waiting, without
    running anything.
    """
    from optuna.trial import TrialState

    from marla.optuna_study.config import load_study_config
    from marla.optuna_study.study import create_or_load_study

    config, base_dir = load_study_config(study_config_path)
    study = create_or_load_study(config, base_dir)
    trials = study.get_trials(deepcopy=False)
    counts: dict[str, int] = {}
    for t in trials:
        counts[t.state.name] = counts.get(t.state.name, 0) + 1

    typer.echo(f"Study: {config.study_name}")
    typer.echo(f"Storage: {config.storage}")
    typer.echo(f"Total trials recorded: {len(trials)}")
    for state_name, count in sorted(counts.items()):
        typer.echo(f"  {state_name}: {count}")
    completed = counts.get("COMPLETE", 0)
    typer.echo(f"Progress: {completed}/{config.n_completed_trials_target} completed trials")
    if completed >= config.n_completed_trials_target:
        typer.secho("Trial budget reached.", fg=typer.colors.GREEN)


@study_app.command("summarize")
def study_summarize(study_config_path: str) -> None:
    """Prints a compact trial comparison table (spec section 66) --
    stable/unstable, worst/mean ROOT AUC, key hyperparameters, carbon.
    """
    from optuna.trial import TrialState

    from marla.optuna_study.config import load_study_config
    from marla.optuna_study.study import create_or_load_study

    config, base_dir = load_study_config(study_config_path)
    study = create_or_load_study(config, base_dir)
    trials = study.get_trials(deepcopy=False)

    header = (
        f"{'trial':>5} {'state':>10} {'stable':>7} {'worst_auc':>10} {'mean_auc':>9} "
        f"{'sched':>9} {'lr':>10} {'gamma':>7} {'lambda':>7} {'clip':>6} {'epochs':>7} "
        f"{'mb_seq':>7} {'v_coef':>7} {'ent_coef':>9} {'co2_kg':>10} {'kwh':>10} {'wall_s':>8}"
    )
    typer.echo(header)
    def _f(value) -> float:
        return value if value is not None else float("nan")

    for t in sorted(trials, key=lambda x: x.number):
        if t.state == TrialState.COMPLETE:
            attrs = t.user_attrs
            typer.echo(
                f"{t.number:>5} {t.state.name:>10} {str(attrs.get('trial_stable')):>7} "
                f"{_f(attrs.get('worst_seed_root_auc')):>10.4f} {_f(attrs.get('mean_root_auc')):>9.4f} "
                f"{t.params.get('scheduler_type', ''):>9} {_f(t.params.get('learning_rate')):>10.2e} "
                f"{_f(t.params.get('gamma')):>7.4f} {_f(t.params.get('gae_lambda')):>7.4f} "
                f"{_f(t.params.get('clip_epsilon')):>6.3f} {t.params.get('epochs', ''):>7} "
                f"{t.params.get('minibatch_sequences', ''):>7} {_f(t.params.get('value_coefficient')):>7.3f} "
                f"{_f(t.params.get('action_entropy_coefficient')):>9.2e} "
                f"{_f(attrs.get('total_co2eq_kg')):>10.6f} {_f(attrs.get('total_energy_kwh')):>10.6f} "
                f"{_f(attrs.get('total_wall_clock_seconds')):>8.1f}"
            )
        else:
            typer.echo(f"{t.number:>5} {t.state.name:>10} {'':>7} {'':>10} {'':>9}  (failed/incomplete -- see trial user_attrs)")


def register_optimize_command(app: typer.Typer) -> None:
    """Adds the top-level ``marla optimize`` command onto the main Typer
    ``app`` (kept as a separate registration function, called from
    ``cli.py``, so this module's own import stays cheap and optional --
    optuna is only imported once ``optimize``/``study ...`` actually run).
    """

    @app.command("optimize")
    def optimize(study_config_path: str) -> None:
        """Runs (or resumes) a persistent Optuna PPO hyperparameter study
        until its declared COMPLETED-trial budget is met (spec: PHASE
        B/C). Safe to interrupt and rerun -- resumes from the same
        SQLite-backed study, never re-running an already-COMPLETE trial.
        """
        from marla.optuna_study.config import load_study_config
        from marla.optuna_study.study import run_study

        config, base_dir = load_study_config(study_config_path)
        typer.echo(f"Study '{config.study_name}': target {config.n_completed_trials_target} completed trials, storage={config.storage}")
        asyncio.run(run_study(config, base_dir))
        typer.secho(f"Study '{config.study_name}' reached its {config.n_completed_trials_target}-completed-trial target.", fg=typer.colors.GREEN)
