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

    updates_path = run_dir / "updates.csv"
    if updates_path.exists() and updates_path.stat().st_size > 0:
        updates = pd.read_csv(updates_path)
        if not updates.empty:
            written += _plot_losses(updates, plots_dir)
            written += _plot_query_behavior(updates, plots_dir)

    decisions_path = run_dir / "decisions.csv"
    if decisions_path.exists() and decisions_path.stat().st_size > 0:
        decisions = pd.read_csv(decisions_path)
        if not decisions.empty:
            written += _plot_decision_diagnostics(decisions, plots_dir)

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
