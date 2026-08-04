"""Plot generation for ``marla summarize``, from a written run directory's CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: marla summarize is a CLI command, never a GUI
import matplotlib.pyplot as plt
import pandas as pd


def generate_plots(run_dir: Path, plots_dir: Path) -> list[Path]:
    """Render whichever plots the available CSVs support; returns the files written.

    Every plot is independently best-effort: a run with no episodes yet (or
    ``metrics.record_decisions: false``) still gets whatever plots its data
    supports, rather than failing the whole command over one missing file.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    episodes_path = run_dir / "episodes.csv"
    if episodes_path.exists() and episodes_path.stat().st_size > 0:
        episodes = pd.read_csv(episodes_path)
        if not episodes.empty:
            written += _plot_episode_returns(episodes, plots_dir)
            written += _plot_consultation_activity(episodes, plots_dir)
            written += _plot_episode_efficiency(episodes, plots_dir)

    updates_path = run_dir / "updates.csv"
    if updates_path.exists() and updates_path.stat().st_size > 0:
        updates = pd.read_csv(updates_path)
        if not updates.empty:
            written += _plot_losses(updates, plots_dir)
            written += _plot_query_behavior(updates, plots_dir)
            written += _plot_training_dynamics(updates, plots_dir)
            written += _plot_optimization_diagnostics(updates, plots_dir)

    decisions_path = run_dir / "decisions.csv"
    if decisions_path.exists() and decisions_path.stat().st_size > 0:
        decisions = pd.read_csv(decisions_path)
        if not decisions.empty:
            written += _plot_decision_diagnostics(decisions, plots_dir)
            written += _plot_advice_influence(decisions, plots_dir)

    return written


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_episode_returns(episodes: pd.DataFrame, plots_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episodes["episode_id"], episodes["benchmark_return"], marker=".", label="benchmark_return")
    ax.plot(episodes["episode_id"], episodes["training_return"], marker=".", label="training_return", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Episode return over training")
    ax.legend()
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "episode_returns.png")]


def _plot_consultation_activity(episodes: pd.DataFrame, plots_dir: Path) -> list[Path]:
    if episodes["consultation_count"].sum() == 0:
        return []  # baseline variant (or a run with no queries): nothing to show
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(episodes["episode_id"], episodes["consultation_count"])
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Consultation count")
    ax1.set_title("Plan Maker consultations per episode")
    ax1.grid(alpha=0.3)

    ax2.bar(episodes["episode_id"], episodes["schema_rejection_count"], color="tab:red")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Schema rejections")
    ax2.set_title("Gatekeeper rejections per episode")
    ax2.grid(alpha=0.3)
    return [_save(fig, plots_dir / "consultation_activity.png")]


def _plot_losses(updates: pd.DataFrame, plots_dir: Path) -> list[Path]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    ax1.plot(updates["update"], updates["policy_loss"], label="policy_loss")
    ax1.plot(updates["update"], updates["value_loss"], label="value_loss")
    ax1.set_xlabel("PPO update")
    ax1.set_ylabel("Loss")
    ax1.set_title("PPO losses")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(updates["update"], updates["approximate_kl"], label="approximate_kl", color="tab:purple")
    ax2.plot(updates["update"], updates["explained_variance"], label="explained_variance", color="tab:green")
    ax2.set_xlabel("PPO update")
    ax2.set_title("PPO diagnostics")
    ax2.legend()
    ax2.grid(alpha=0.3)
    return [_save(fig, plots_dir / "ppo_losses.png")]


def _plot_query_behavior(updates: pd.DataFrame, plots_dir: Path) -> list[Path]:
    if updates["mean_query_probability"].isna().all():
        return []  # baseline variant: no query gate at all
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(updates["update"], updates["mean_query_probability"], label="mean query probability")
    ax.plot(updates["update"], updates["actual_query_rate"], label="actual query rate", alpha=0.7)
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Query gate behavior over training")
    ax.legend()
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "query_behavior.png")]


def _plot_decision_diagnostics(decisions: pd.DataFrame, plots_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(decisions.index, decisions["base_policy_entropy"], label="base policy entropy", alpha=0.8)
    ax.plot(decisions.index, decisions["base_top_two_margin"], label="base top-two margin", alpha=0.8)
    ax.set_xlabel("Decision (in collection order)")
    ax.set_title("Base policy confidence over training")
    ax.legend()
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "policy_confidence.png")]


def _plot_episode_efficiency(episodes: pd.DataFrame, plots_dir: Path) -> list[Path]:
    # NASimEmu-agents' reference implementation tracks goal-success rate and
    # episode length as its primary training-progress signals (alongside
    # return, already covered by episode_returns.png); this is the MARLA
    # equivalent, using a rolling window so the trend is readable even
    # though goal_success is a per-episode 0/1.
    window = max(1, min(10, len(episodes)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    success_rate = episodes["goal_success"].astype(float).rolling(window, min_periods=1).mean()
    ax1.plot(episodes["episode_id"], success_rate)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(f"Goal success rate (rolling mean, window={window})")
    ax1.set_ylim(0, 1)
    ax1.set_title("Goal success rate over training")
    ax1.grid(alpha=0.3)

    ax2.plot(episodes["episode_id"], episodes["environment_steps"], label="environment steps", alpha=0.8)
    if episodes["steps_to_goal"].notna().any():
        ax2.plot(episodes["episode_id"], episodes["steps_to_goal"], label="steps to goal", alpha=0.8)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title("Episode length over training")
    ax2.legend()
    ax2.grid(alpha=0.3)
    return [_save(fig, plots_dir / "episode_efficiency.png")]


def _plot_training_dynamics(updates: pd.DataFrame, plots_dir: Path) -> list[Path]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    ax1.plot(updates["update"], updates["action_entropy"], label="action_entropy")
    ax1.plot(updates["update"], updates["query_entropy"], label="query_entropy")
    ax1.set_xlabel("PPO update")
    ax1.set_ylabel("Entropy")
    ax1.set_title("Policy entropy over training")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(updates["update"], updates["learning_rate"], color="tab:orange")
    ax2.set_xlabel("PPO update")
    ax2.set_ylabel("Learning rate")
    ax2.set_title("Learning rate over training")
    ax2.grid(alpha=0.3)
    return [_save(fig, plots_dir / "training_dynamics.png")]


def _plot_optimization_diagnostics(updates: pd.DataFrame, plots_dir: Path) -> list[Path]:
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(updates["update"], updates["gradient_norm"], color="tab:blue", label="gradient_norm")
    ax1.set_xlabel("PPO update")
    ax1.set_ylabel("Gradient norm", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(updates["update"], updates["clip_fraction"], color="tab:red", label="clip_fraction")
    ax2.set_ylabel("Clip fraction", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title("Gradient norm and PPO clip fraction over training")
    return [_save(fig, plots_dir / "gradient_and_clipping.png")]


def _plot_advice_influence(decisions: pd.DataFrame, plots_dir: Path) -> list[Path]:
    queried = decisions[decisions["queried"] == True]  # noqa: E712 (pandas bool comparison)
    if queried.empty or queried["beta"].isna().all():
        return []  # baseline variant (or a run with no accepted advice): nothing to show

    window = max(1, min(20, len(queried)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(range(len(queried)), queried["beta"].rolling(window, min_periods=1).mean())
    ax1.set_xlabel("Queried decision (in collection order)")
    ax1.set_ylabel(f"Mean beta (rolling, window={window})")
    ax1.set_title("Advice trust weight over training")
    ax1.grid(alpha=0.3)

    changed = queried["advice_changed_top_action"].dropna().astype(float)
    if not changed.empty:
        ax2.plot(range(len(changed)), changed.rolling(window, min_periods=1).mean(), color="tab:green")
    ax2.set_xlabel("Accepted-advice decision (in collection order)")
    ax2.set_ylabel(f"Top action changed (rolling, window={window})")
    ax2.set_ylim(0, 1)
    ax2.set_title("How often advice changes the top action")
    ax2.grid(alpha=0.3)
    return [_save(fig, plots_dir / "advice_influence.png")]
