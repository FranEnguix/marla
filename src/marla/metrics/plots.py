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
            written += _plot_learning_rate(updates, plots_dir)

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


def _split_train_eval(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Splits into (train, eval, x_column), tolerating older episodes.csv
    files written before ``rollout``/``is_eval`` existed (see writer.py)."""
    if "is_eval" not in episodes.columns or "rollout" not in episodes.columns:
        return episodes, episodes.iloc[0:0], "episode_id"
    train = episodes[episodes["is_eval"] == False]  # noqa: E712 (pandas bool comparison)
    eval_ = episodes[episodes["is_eval"] == True]  # noqa: E712
    x_col = "rollout" if train["rollout"].notna().any() or eval_["rollout"].notna().any() else "episode_id"
    return train, eval_, x_col


def _grouped_mean_and_band(df: pd.DataFrame, x_col: str, y_col: str, window: int = 5):
    """Mean +/- an uncertainty band of ``y_col`` per ``x_col`` value.

    Real cross-sample std where more than one row shares an x value (e.g. a
    batch of eval episodes at one checkpoint); otherwise a rolling std
    across nearby points as a local-variability estimate, plus a light
    rolling mean for readability -- the standard smoothed-learning-curve-
    with-uncertainty-band presentation for single-run RL training curves.
    """
    grouped = df.groupby(x_col)[y_col].agg(["mean", "std"]).sort_index()
    window = max(1, min(window, len(grouped)))
    mean = grouped["mean"].rolling(window, min_periods=1).mean()
    rolling_std = grouped["mean"].rolling(window, min_periods=1).std()
    band = grouped["std"].fillna(rolling_std).fillna(0.0)
    return grouped.index.to_numpy(), mean.to_numpy(), band.to_numpy()


def _plot_episode_returns(episodes: pd.DataFrame, plots_dir: Path) -> list[Path]:
    train, eval_, x_col = _split_train_eval(episodes)
    x_label = "Rollout" if x_col == "rollout" else "Episode"

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if not train.empty:
        x, mean, band = _grouped_mean_and_band(train, x_col, "benchmark_return")
        ax.plot(x, mean, label="train (stochastic policy)", color="tab:blue")
        ax.fill_between(x, mean - band, mean + band, color="tab:blue", alpha=0.2)

    if not eval_.empty:
        x, mean, band = _grouped_mean_and_band(eval_, x_col, "benchmark_return")
        ax.plot(x, mean, label="eval (greedy policy)", color="tab:orange", linestyle="--", marker=".")
        ax.fill_between(x, mean - band, mean + band, color="tab:orange", alpha=0.2)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Average reward (benchmark return)")
    ax.set_title("Average reward over training")
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
    # episode length (train vs. eval) as its primary training-progress
    # signals alongside return (episode_returns.png); same train/eval split
    # and smoothed-mean-with-band presentation as there (see
    # _grouped_mean_and_band), tolerating older runs with no eval data.
    train, eval_, x_col = _split_train_eval(episodes)
    train = train.assign(goal_success=train["goal_success"].astype(float))
    if not eval_.empty:
        eval_ = eval_.assign(goal_success=eval_["goal_success"].astype(float))
    x_label = "Rollout" if x_col == "rollout" else "Episode"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x, mean, band = _grouped_mean_and_band(train, x_col, "goal_success")
    ax1.plot(x, mean, label="train (stochastic policy)", color="tab:blue")
    ax1.fill_between(x, (mean - band).clip(0, 1), (mean + band).clip(0, 1), color="tab:blue", alpha=0.2)
    if not eval_.empty:
        x, mean, band = _grouped_mean_and_band(eval_, x_col, "goal_success")
        ax1.plot(x, mean, label="eval (greedy policy)", color="tab:orange", linestyle="--", marker=".")
        ax1.fill_between(x, (mean - band).clip(0, 1), (mean + band).clip(0, 1), color="tab:orange", alpha=0.2)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Goal success rate")
    ax1.set_ylim(0, 1)
    ax1.set_title("Goal success rate over training")
    ax1.legend()
    ax1.grid(alpha=0.3)

    x, mean, band = _grouped_mean_and_band(train, x_col, "environment_steps")
    ax2.plot(x, mean, label="train (stochastic policy)", color="tab:blue")
    ax2.fill_between(x, mean - band, mean + band, color="tab:blue", alpha=0.2)
    if not eval_.empty:
        x, mean, band = _grouped_mean_and_band(eval_, x_col, "environment_steps")
        ax2.plot(x, mean, label="eval (greedy policy)", color="tab:orange", linestyle="--", marker=".")
        ax2.fill_between(x, mean - band, mean + band, color="tab:orange", alpha=0.2)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Episode length (environment steps)")
    ax2.set_title("Episode length over training")
    ax2.legend()
    ax2.grid(alpha=0.3)
    return [_save(fig, plots_dir / "episode_efficiency.png")]


def _plot_training_dynamics(updates: pd.DataFrame, plots_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(updates["update"], updates["action_entropy"], label="action_entropy")
    ax.plot(updates["update"], updates["query_entropy"], label="query_entropy")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy entropy over training")
    ax.legend()
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "training_dynamics.png")]


def _plot_learning_rate(updates: pd.DataFrame, plots_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(updates["update"], updates["learning_rate"], color="tab:orange")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Learning rate")
    is_constant = updates["learning_rate"].nunique() <= 1
    title = "Learning rate over training"
    if is_constant:
        title += " (constant -- no scheduler configured)"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "learning_rate.png")]


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
