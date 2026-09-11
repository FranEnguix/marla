"""Plot generation for ``marla summarize``, from a written run directory's CSVs."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: marla summarize is a CLI command, never a GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marla.metrics.csv_schema import read_decisions_csv

# A plain [0, 1] y-limit puts a perfectly flat "always 0" or "always 1"
# line exactly on the axis border, where it's easy to mistake for no data
# at all (observed in practice: ~135 consecutive accepted-advice decisions
# that all changed the top action rendered as an empty-looking plot). This
# small margin keeps such a line visibly distinct from the frame.
_RATE_YLIM = (-0.05, 1.05)

# _grouped_mean_and_band's rolling window (spec: a plot must never smooth
# data without saying so) -- every title built from its output includes
# this suffix, so the disclosure lives in the image itself, not only in
# _grouped_mean_and_band's own docstring/comments. Deliberately says
# "point" rather than "rollout": the x-axis grouping is rollout for a
# normal run but falls back to episode_id for an older episodes.csv
# written before the rollout column existed (see _split_train_eval).
_SMOOTHING_NOTE = " (5-point rolling mean)"

# Minimum resource-telemetry samples needed before a time-series resource
# plot is considered meaningful. Below this, a line/area plot either draws
# nothing visible (a single point has no line to connect, see
# _plot_resource_by_phase's own history) or is too short a window to say
# anything about a trend -- an explicit "not enough data" panel is always
# clearer than a chart that LOOKS complete but silently shows ~nothing.
_MIN_RESOURCE_SAMPLES_FOR_TIMESERIES = 3


def _set_title_with_note(ax, title: str, note: str | None = None, *, fontsize: int = 12, note_fontsize: int = 8) -> None:
    """A short, bold main title with an optional smaller gray note
    directly beneath it, inside the axes' own top margin -- keeps
    secondary conditions/caveats (filtering rules, "diagnostic only",
    smoothing disclosures, ...) out of an oversized, wrapping main title
    that clips or overlaps a neighboring subplot. ``note`` may contain
    explicit ``\\n`` line breaks for multi-line notes; it is never
    auto-wrapped (matplotlib's own text auto-wrap is approximate and can
    still overflow a narrow subplot), so callers keep each line short by
    construction.

    The note sits directly above the axes' own top edge (an ``annotate``
    anchored to axes-fraction (0.5, 1.0) with a small points-based
    offset); the title's own ``pad`` is sized in points from the note's
    line count so the two never overlap regardless of figure size/DPI --
    a fixed pad only worked for a single-line note (see git history: an
    earlier fixed-pad version visibly overlapped a real two-line note).
    """
    n_note_lines = note.count("\n") + 1 if note else 0
    pad = 6 + n_note_lines * (note_fontsize * 1.5) if note else 6
    ax.set_title(title, fontsize=fontsize, pad=pad)
    if note:
        ax.annotate(
            note, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=note_fontsize, color="dimgray",
        )


def _insufficient_samples_panel(ax, message: str, *, wrap_width: int = 46) -> None:
    """A deliberate, explicit "not enough data" panel -- used instead of
    silently rendering a blank-looking axes (spec: never a misleading
    empty chart). Keeps the axes frame/labels visible (so the reader can
    still see what metric this would have been) but replaces the data
    area with a plain, honest sentence.

    Hard-wraps at ``wrap_width`` characters via ``textwrap`` rather than
    matplotlib's own ``wrap=True`` -- the latter wraps against the whole
    FIGURE's width, not this axes' own (narrower, in a multi-panel
    figure) width, and visibly overflowed a real two-panel resource plot
    when tried. Existing ``\\n`` breaks in ``message`` are preserved;
    each resulting line is wrapped independently.
    """
    wrapped = "\n".join(textwrap.fill(line, width=wrap_width) for line in message.split("\n"))
    ax.text(
        0.5, 0.5, wrapped, ha="center", va="center", transform=ax.transAxes,
        fontsize=9, color="dimgray",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _accepted_advice_rows(decisions: pd.DataFrame) -> pd.DataFrame:
    """Rows where consultation was queried *and accepted* -- the only rows
    where advice was actually applied.

    Deliberately not ``decisions["queried"] == True``: a schema-rejected
    query also has a real, non-null ``beta`` -- forced to exactly 0 (see
    learning/decision.py's ``compute_final_decision``), which in turn makes
    ``final_logits`` identical to ``base_logits`` and
    ``advice_changed_top_action`` deterministically ``False``. A plot
    meant to show "how much does accepted advice actually change the
    decision" must not silently mix in those zero/False rejected-query
    rows, which would pull any such statistic toward 0 in proportion to
    the schema rejection rate rather than reflecting the trust head's
    actual behavior on advice it was actually given.
    """
    if "response_status" not in decisions.columns:
        return decisions.iloc[0:0]
    return decisions[decisions["response_status"] == "accepted"]


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
            written += _plot_episode_outcomes(episodes, plots_dir)
            written += _plot_finish_efficiency(episodes, plots_dir)

    updates_path = run_dir / "updates.csv"
    if updates_path.exists() and updates_path.stat().st_size > 0:
        updates = pd.read_csv(updates_path)
        if not updates.empty:
            written += _plot_losses(updates, plots_dir)
            written += _plot_training_dynamics(updates, plots_dir)
            written += _plot_optimization_diagnostics(updates, plots_dir)
            written += _plot_learning_rate(updates, plots_dir)

    # rollouts.csv (spec section 6): rollout-level aggregates that used to
    # be repeated on every updates.csv row -- absent entirely for a run
    # written before rollouts.csv existed, in which case the two plots
    # sourced from it are simply skipped (see module docstring's note on
    # tolerating older run directories).
    rollouts_path = run_dir / "rollouts.csv"
    if rollouts_path.exists() and rollouts_path.stat().st_size > 0:
        rollouts = pd.read_csv(rollouts_path)
        if not rollouts.empty:
            written += _plot_query_behavior(rollouts, plots_dir)
            written += _plot_reward(rollouts, plots_dir)
            written += _plot_rollout_timing(rollouts, plots_dir)
            written += _plot_compatible_action_probability(rollouts, plots_dir)
            written += _plot_finish_probability_by_objective_state(rollouts, plots_dir)
            written += _plot_finish_probability_by_visible_completion_and_frontier(rollouts, plots_dir)
            written += _plot_known_subnet_exploration_progress(rollouts, plots_dir)

    decisions_path = run_dir / "decisions.csv"
    if decisions_path.exists() and decisions_path.stat().st_size > 0:
        decisions = read_decisions_csv(decisions_path)
        if not decisions.empty:
            written += _plot_decision_diagnostics(decisions, plots_dir)
            written += _plot_advice_influence(decisions, plots_dir)
            written += _plot_gatekeeper_reliability(decisions, plots_dir)
            written += _plot_plan_maker_latency(decisions, plots_dir)
            written += _plot_query_decision_analysis(decisions, plots_dir)
            written += _plot_critic_quality(decisions, plots_dir)
            written += _plot_reward_vs_credit_assignment(decisions, plots_dir)
            written += _plot_action_type_distribution(decisions, plots_dir)
            written += _plot_assisted_policy_influence(decisions, plots_dir)
            written += _plot_finish_probability_by_remaining_targets(decisions, plots_dir)

    resources_path = run_dir / "resources.csv"
    if resources_path.exists() and resources_path.stat().st_size > 0:
        resources = pd.read_csv(resources_path)
        if not resources.empty:
            written += _plot_resource_cpu(resources, plots_dir)
            written += _plot_resource_ram(resources, plots_dir)
            written += _plot_resource_gpu(resources, plots_dir)
            written += _plot_resource_gpu_memory(resources, plots_dir)

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
    """Average episodic return over training -- the single most important
    training-progress plot (episode-level environment return, the correct
    metric for comparing baseline/assisted/other agents, vs. the
    cost-adjusted training return PPO actually optimizes; a large gap
    between them means the agent is paying substantial consultation cost).
    """
    train, eval_, x_col = _split_train_eval(episodes)
    x_label = "Rollout" if x_col == "rollout" else "Episode"
    has_cost_gap = not episodes["benchmark_return"].equals(episodes["training_return"])

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if not train.empty:
        x, mean, band = _grouped_mean_and_band(train, x_col, "benchmark_return")
        # marker="." matters more than it looks: a line with a single point
        # (a run stopped after just one rollout) is otherwise invisible --
        # nothing to connect, and no marker to draw the point itself.
        ax.plot(x, mean, label="train env. return (benchmark)", color="tab:blue", marker=".")
        ax.fill_between(x, mean - band, mean + band, color="tab:blue", alpha=0.2)
        if has_cost_gap:
            x, mean, _band = _grouped_mean_and_band(train, x_col, "training_return")
            ax.plot(x, mean, label="train cost-adjusted return", color="tab:blue", linestyle=":", marker=".", alpha=0.8)

    if not eval_.empty:
        x, mean, band = _grouped_mean_and_band(eval_, x_col, "benchmark_return")
        ax.plot(x, mean, label="eval env. return (benchmark)", color="tab:orange", linestyle="--", marker=".")
        ax.fill_between(x, mean - band, mean + band, color="tab:orange", alpha=0.2)
        if has_cost_gap:
            x, mean, _band = _grouped_mean_and_band(eval_, x_col, "training_return")
            ax.plot(x, mean, label="eval cost-adjusted return", color="tab:orange", linestyle=":", alpha=0.8)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Average episodic return")
    ax.set_title("Average episodic return over training" + _SMOOTHING_NOTE)
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

    # Separate axes: explained_variance is unbounded below (an occasional
    # badly-fit update can send it to -1000s) and would otherwise dwarf
    # approximate_kl's much smaller-scale variation into an invisible flat
    # line on a shared axis.
    ax2.plot(updates["update"], updates["approximate_kl"], color="tab:purple", label="approximate_kl")
    ax2.set_xlabel("PPO update")
    ax2.set_ylabel("Approximate KL", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")
    ax2.grid(alpha=0.3)

    ax3 = ax2.twinx()
    ax3.plot(updates["update"], updates["explained_variance"], color="tab:green", label="explained_variance")
    ax3.set_ylabel("Explained variance", color="tab:green")
    ax3.tick_params(axis="y", labelcolor="tab:green")

    ax2.set_title("PPO diagnostics")
    return [_save(fig, plots_dir / "ppo_losses.png")]


def _plot_query_behavior(rollouts: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Sourced from rollouts.csv (spec section 6), not updates.csv -- these
    are rollout-level aggregates, computed once per rollout, not per PPO
    minibatch update.
    """
    if rollouts["mean_query_probability"].isna().all():
        return []  # baseline variant: no query gate at all
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rollouts["rollout"], rollouts["mean_query_probability"], label="mean query probability", marker=".")
    ax.plot(rollouts["rollout"], rollouts["actual_query_rate"], label="actual query rate", alpha=0.7, marker=".")
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Rate")
    ax.set_ylim(*_RATE_YLIM)
    ax.set_title("Query gate behavior over training")
    ax.legend()
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "query_behavior.png")]


def _plot_decision_diagnostics(decisions: pd.DataFrame, plots_dir: Path) -> list[Path]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(decisions.index, decisions["base_policy_entropy"], label="base policy entropy", alpha=0.8)
    ax1.plot(decisions.index, decisions["base_top_two_margin"], label="base top-two margin", alpha=0.8)
    ax1.set_xlabel("Decision (in collection order)")
    ax1.set_title("Base policy confidence over training")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Raw entropy's range depends on legal_action_count (more legal actions
    # -> higher max possible entropy), which varies as NASimEmu discovers
    # more hosts within an episode -- normalizing by log(N) makes
    # uncertainty comparable across states with different action counts.
    # log(1) == 0 (a single legal action has no uncertainty to normalize).
    log_n = np.log(decisions["legal_action_count"].clip(lower=1))
    normalized_entropy = (decisions["base_policy_entropy"] / log_n.replace(0, np.nan)).clip(0, 1)
    ax2.plot(decisions.index, normalized_entropy, color="tab:purple", alpha=0.8)
    ax2.set_xlabel("Decision (in collection order)")
    ax2.set_ylabel(r"Normalized entropy $H / \log N$")
    ax2.set_ylim(*_RATE_YLIM)
    ax2.set_title("Action-count-normalized policy entropy")
    ax2.grid(alpha=0.3)
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
    # objective_reached (independent of FINISH) is a newer column -- tolerate
    # an older episodes.csv written before this field existed rather than
    # KeyError'ing (same tolerance pattern used elsewhere in this module).
    has_objective_reached = "objective_reached" in episodes.columns
    if has_objective_reached:
        train = train.assign(objective_reached=train["objective_reached"].astype(float))
        if not eval_.empty:
            eval_ = eval_.assign(objective_reached=eval_["objective_reached"].astype(float))
    x_label = "Rollout" if x_col == "rollout" else "Episode"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.3))

    # Reported as two distinct lines, never collapsed into one (spec
    # section 25): objective_reached doesn't require FINISH;
    # successful_finish (== goal_success) does -- see EpisodeSummary's
    # docstring for the exact distinction.
    if has_objective_reached:
        x, mean, band = _grouped_mean_and_band(train, x_col, "objective_reached")
        ax1.plot(x, mean, label="objective reached (train)", color="tab:green", marker="^", alpha=0.7)
        if not eval_.empty:
            x, mean, band = _grouped_mean_and_band(eval_, x_col, "objective_reached")
            ax1.plot(x, mean, label="objective reached (eval)", color="tab:green", linestyle="--", marker="^", alpha=0.7)

    x, mean, band = _grouped_mean_and_band(train, x_col, "goal_success")
    ax1.plot(x, mean, label="successful FINISH (train)", color="tab:blue", marker=".")
    ax1.fill_between(x, (mean - band).clip(0, 1), (mean + band).clip(0, 1), color="tab:blue", alpha=0.2)
    if not eval_.empty:
        x, mean, band = _grouped_mean_and_band(eval_, x_col, "goal_success")
        ax1.plot(x, mean, label="successful FINISH (eval)", color="tab:orange", linestyle="--", marker=".")
        ax1.fill_between(x, (mean - band).clip(0, 1), (mean + band).clip(0, 1), color="tab:orange", alpha=0.2)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Rate")
    ax1.set_ylim(*_RATE_YLIM)
    _set_title_with_note(ax1, "Objective reached vs. successful FINISH", "Sensitive hosts captured" + _SMOOTHING_NOTE)
    ax1.legend(fontsize="x-small")
    ax1.grid(alpha=0.3)

    # Episode length can't be negative; clip the band's lower edge so a
    # locally large std (small scenarios can regularly finish in 1-2 steps,
    # so std can exceed the mean) doesn't draw a misleadingly negative
    # shaded region for an inherently non-negative quantity.
    x, mean, band = _grouped_mean_and_band(train, x_col, "environment_steps")
    ax2.plot(x, mean, label="train (stochastic policy)", color="tab:blue", marker=".")
    ax2.fill_between(x, (mean - band).clip(min=0), mean + band, color="tab:blue", alpha=0.2)
    if not eval_.empty:
        x, mean, band = _grouped_mean_and_band(eval_, x_col, "environment_steps")
        ax2.plot(x, mean, label="eval (greedy policy)", color="tab:orange", linestyle="--", marker=".")
        ax2.fill_between(x, (mean - band).clip(min=0), mean + band, color="tab:orange", alpha=0.2)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Episode length (environment steps)")
    _set_title_with_note(ax2, "Episode length", "All episodes (success + failure)" + _SMOOTHING_NOTE)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # steps_to_goal is set the instant the simulator's true objective
    # becomes satisfied (rollout.py's `objective_became_satisfied`),
    # regardless of whether FINISH is ever selected afterward -- NOT only
    # on a successful FINISH (a stale claim this comment used to make;
    # corrected here since a code comment being wrong is itself a
    # correctness bug, even though nothing user-facing read it). It is
    # NaN for an episode that never reached the objective at all.
    # groupby's mean/std already skip NaN, so this is the mean *only over
    # objective-reached episodes* at each point, with no separate
    # filtering needed -- a rollout with zero such episodes simply leaves
    # a gap rather than a misleading zero.
    any_success = False
    if train["steps_to_goal"].notna().any():
        any_success = True
        x, mean, band = _grouped_mean_and_band(train, x_col, "steps_to_goal")
        ax3.plot(x, mean, label="train (stochastic policy)", color="tab:blue", marker=".")
        ax3.fill_between(x, (mean - band).clip(min=0), mean + band, color="tab:blue", alpha=0.2)
    if not eval_.empty and eval_["steps_to_goal"].notna().any():
        any_success = True
        x, mean, band = _grouped_mean_and_band(eval_, x_col, "steps_to_goal")
        ax3.plot(x, mean, label="eval (greedy policy)", color="tab:orange", linestyle="--", marker=".")
        ax3.fill_between(x, (mean - band).clip(min=0), mean + band, color="tab:orange", alpha=0.2)
    ax3.set_xlabel(x_label)
    ax3.set_ylabel("Steps to objective")
    _set_title_with_note(
        ax3, "Steps to objective",
        "Only episodes that reached the objective are included\nFINISH is not required for inclusion" + _SMOOTHING_NOTE,
    )
    if any_success:
        ax3.legend()
    else:
        _insufficient_samples_panel(ax3, "Objective never reached yet in this run")
    ax3.grid(alpha=0.3)
    return [_save(fig, plots_dir / "episode_efficiency.png")]


_OUTCOME_COLORS = {
    "success": "tab:green",
    "premature_finish": "tab:red",
    "timeout": "tab:gray",
}


def _episode_outcome(row: pd.Series) -> str:
    if row["goal_success"]:
        return "success"
    if row["finish_reason"] == "finish":
        return "premature_finish"
    return "timeout"


def _plot_episode_outcomes(episodes: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Stacked distribution of *why* each episode ended, over training.

    A low success rate has very different causes -- the agent never makes
    enough progress (timeout), or it incorrectly chooses FINISH too early
    (premature_finish) -- and episode_efficiency.png's success-rate curve
    alone can't distinguish them. There is no fourth "environment/protocol
    failure" category here: NASimEmu never internally terminates a
    non-FINISH action, and an absorbed action-space precondition failure
    (see NasimEmuAdapter._absorb_invalid_action) is a failed *attempt*,
    not an episode-ending error -- every episode is exactly one of the
    three outcomes below.
    """
    train, eval_, x_col = _split_train_eval(episodes)
    x_label = "Rollout" if x_col == "rollout" else "Episode"
    if train.empty:
        return []

    train = train.assign(outcome=train.apply(_episode_outcome, axis=1))
    fractions = (
        train.groupby(x_col)["outcome"]
        .value_counts(normalize=True)
        .unstack("outcome")
        .reindex(columns=list(_OUTCOME_COLORS), fill_value=0.0)
        .sort_index()
    )

    # A stacked *bar* chart per the design brief (not a stacked-area plot):
    # a run with only one rollout so far -- exactly the case a short/just-
    # stopped run hits -- has only one x-value, and a filled area needs at
    # least two points to draw anything visible at all (confirmed in
    # practice: a single-rollout run rendered a completely blank plot).
    # Bars have no such degenerate case.
    width = fractions.index.to_series().diff().min() if len(fractions) > 1 else 1
    width = 0.8 * (width if pd.notna(width) else 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(fractions))
    for outcome, color in _OUTCOME_COLORS.items():
        ax.bar(fractions.index, fractions[outcome], width=width, bottom=bottom, label=outcome, color=color, alpha=0.85)
        bottom += fractions[outcome].to_numpy()
    ax.set_xlabel(x_label)
    ax.set_ylabel("Fraction of episodes")
    ax.set_ylim(0, _RATE_YLIM[1])
    ax.set_title("Episode outcome distribution over training")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "episode_outcomes.png")]


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


_PHASE_COLOR = {"rollout_collection": "tab:blue", "ppo_update": "tab:orange", "evaluation": "tab:green"}


def _resource_wall_clock_seconds(resources: pd.DataFrame) -> pd.Series:
    """Wall-clock seconds since this run's own first resource sample --
    resource samples are on a fixed ~1s cadence (config.metrics.
    resource_monitoring.sampling_interval_seconds), not aligned to
    rollout/environment-step boundaries, so wall time (not environment
    steps) is the meaningful x-axis here (spec: "global environment steps
    or wall time as appropriate").
    """
    return resources["timestamp"] - resources["timestamp"].iloc[0]


def _resource_insufficient_message(n_samples: int) -> str:
    return (
        f"Insufficient resource telemetry samples (n={n_samples}, "
        f"need >= {_MIN_RESOURCE_SAMPLES_FOR_TIMESERIES}).\n"
        "This run was likely too short for the background sampler to record more."
    )


def _plot_resource_by_phase(resources: pd.DataFrame, y_col: str, ylabel: str, title: str, filename: str, plots_dir: Path) -> list[Path]:
    if y_col not in resources.columns or resources[y_col].dropna().empty:
        return []
    n = len(resources[y_col].dropna())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if n < _MIN_RESOURCE_SAMPLES_FOR_TIMESERIES:
        _set_title_with_note(ax, title)
        _insufficient_samples_panel(ax, _resource_insufficient_message(n))
        return [_save(fig, plots_dir / filename)]
    x = _resource_wall_clock_seconds(resources)
    if "phase" in resources.columns:
        for phase, color in _PHASE_COLOR.items():
            mask = resources["phase"] == phase
            if mask.any():
                ax.scatter(x[mask], resources.loc[mask, y_col], s=8, color=color, label=phase, alpha=0.7)
        ax.legend(fontsize=8)
    else:
        ax.plot(x, resources[y_col], color="tab:blue")
    ax.set_xlabel("Wall-clock seconds since run start")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / filename)]


def _plot_resource_cpu(resources: pd.DataFrame, plots_dir: Path) -> list[Path]:
    # TWO PANELS, not one overlay: process CPU % (psutil's "summed across
    # logical cores" convention -- can exceed 100%) and system CPU %
    # (psutil's "averaged across logical cores" convention -- capped at
    # 100%) use DIFFERENT scales; overlaying them on one axis would
    # visually imply they're directly comparable when they are not -- see
    # monitoring/resources.py's module docstring.
    if "cpu_process_pct" not in resources.columns:
        return []
    n = len(resources["cpu_process_pct"].dropna())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    if n < _MIN_RESOURCE_SAMPLES_FOR_TIMESERIES:
        for ax, title in ((ax1, "Process CPU utilization"), (ax2, "System-wide CPU utilization")):
            _set_title_with_note(ax, title)
            _insufficient_samples_panel(ax, _resource_insufficient_message(n), wrap_width=28)
        return [_save(fig, plots_dir / "resource_cpu_over_time.png")]
    x = _resource_wall_clock_seconds(resources)
    ax1.plot(x, resources["cpu_process_pct"], color="tab:blue")
    ax1.set_xlabel("Wall-clock seconds since run start")
    ax1.set_ylabel("Process CPU % (sum across cores, can exceed 100%)")
    ax1.set_title("Process CPU utilization")
    ax1.grid(alpha=0.3)
    ax2.plot(x, resources["cpu_system_pct"], color="tab:gray")
    ax2.set_xlabel("Wall-clock seconds since run start")
    ax2.set_ylabel("System CPU % (average across cores, 0-100)")
    ax2.set_ylim(0, 105)
    ax2.set_title("System-wide CPU utilization")
    ax2.grid(alpha=0.3)
    return [_save(fig, plots_dir / "resource_cpu_over_time.png")]


def _plot_resource_ram(resources: pd.DataFrame, plots_dir: Path) -> list[Path]:
    if "ram_rss_mb" not in resources.columns:
        return []
    n = len(resources["ram_rss_mb"].dropna())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if n < _MIN_RESOURCE_SAMPLES_FOR_TIMESERIES:
        _set_title_with_note(ax, "RAM usage over training")
        _insufficient_samples_panel(ax, _resource_insufficient_message(n))
        return [_save(fig, plots_dir / "resource_ram_over_time.png")]
    x = _resource_wall_clock_seconds(resources)
    ax.plot(x, resources["ram_rss_mb"], label="process RSS (MB)", color="tab:purple")
    if "ram_system_used_mb" in resources.columns:
        ax.plot(x, resources["ram_system_used_mb"], label="system RAM used (MB)", color="tab:gray", alpha=0.7)
    ax.set_xlabel("Wall-clock seconds since run start")
    ax.set_ylabel("RAM (MB)")
    ax.set_title("RAM usage over training")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "resource_ram_over_time.png")]


def _plot_resource_gpu(resources: pd.DataFrame, plots_dir: Path) -> list[Path]:
    # Absent (not zero) whenever NVML wasn't available -- see
    # monitoring/resources.py's module docstring on GPU-field semantics.
    return _plot_resource_by_phase(
        resources, "gpu_util_pct", "GPU utilization %", "GPU utilization over training (NVML)",
        "resource_gpu_over_time.png", plots_dir,
    )


def _plot_resource_gpu_memory(resources: pd.DataFrame, plots_dir: Path) -> list[Path]:
    if "torch_cuda_allocated_mb" not in resources.columns or resources["torch_cuda_allocated_mb"].dropna().empty:
        return []
    n = len(resources["torch_cuda_allocated_mb"].dropna())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if n < _MIN_RESOURCE_SAMPLES_FOR_TIMESERIES:
        _set_title_with_note(ax, "GPU memory over training")
        _insufficient_samples_panel(ax, _resource_insufficient_message(n))
        return [_save(fig, plots_dir / "resource_gpu_memory_over_time.png")]
    x = _resource_wall_clock_seconds(resources)
    ax.plot(x, resources["torch_cuda_allocated_mb"], label="torch allocated (MB)", color="tab:red")
    ax.plot(x, resources["torch_cuda_reserved_mb"], label="torch reserved (MB)", color="tab:orange", alpha=0.7)
    if "gpu_memory_used_mb" in resources.columns and resources["gpu_memory_used_mb"].notna().any():
        ax.plot(x, resources["gpu_memory_used_mb"], label="NVML GPU memory used (MB)", color="tab:gray", alpha=0.6)
    ax.set_xlabel("Wall-clock seconds since run start")
    ax.set_ylabel("GPU memory (MB)")
    ax.set_title("GPU memory over training")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "resource_gpu_memory_over_time.png")]


def _plot_reward(rollouts: pd.DataFrame, plots_dir: Path) -> list[Path]:
    # Absent in rollouts.csv files written before this field existed, and
    # in any run directory written before rollouts.csv existed at all --
    # fields are additive, never backfilled into old runs (see the module
    # docstring's note on permanently-empty columns for the same pattern).
    if "mean_nasimemu_reward" not in rollouts.columns:
        return []

    # The raw per-step reward signal actually driving PPO, over training --
    # distinct from episode_returns.png's per-episode *return* (the sum of
    # this over a whole episode). The two lines only diverge for the
    # assisted variant, where mean_training_reward additionally reflects
    # consultation-cost deductions that mean_nasimemu_reward does not.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rollouts["rollout"], rollouts["mean_nasimemu_reward"], label="mean nasimemu reward", marker=".")
    if not rollouts["mean_training_reward"].equals(rollouts["mean_nasimemu_reward"]):
        ax.plot(rollouts["rollout"], rollouts["mean_training_reward"], label="mean training reward", alpha=0.7, marker=".")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Mean reward per step")
    ax.set_title("Reward over training")
    ax.legend()
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "reward_over_training.png")]


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
    """Trust weight and top-action-change rate, over *accepted* advice only
    (see :func:`_accepted_advice_rows`)."""
    accepted = _accepted_advice_rows(decisions)
    if accepted.empty or accepted["beta"].isna().all():
        return []  # baseline variant (or a run with no accepted advice): nothing to show

    window = max(1, min(20, len(accepted)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(range(len(accepted)), accepted["beta"].rolling(window, min_periods=1).mean())
    ax1.set_xlabel("Accepted-advice decision (in collection order)")
    ax1.set_ylabel(rf"Mean $\beta$ (rolling, window={window})")
    ax1.set_title(r"Advice trust weight $\beta$ over training")
    ax1.grid(alpha=0.3)

    changed = accepted["advice_changed_top_action"].dropna().astype(float)
    if not changed.empty:
        ax2.plot(range(len(changed)), changed.rolling(window, min_periods=1).mean(), color="tab:green")
    ax2.set_xlabel("Accepted-advice decision (in collection order)")
    ax2.set_ylabel(f"Top action changed (rolling, window={window})")
    ax2.set_ylim(*_RATE_YLIM)
    ax2.set_title("How often advice changes the top action")
    ax2.grid(alpha=0.3)
    return [_save(fig, plots_dir / "advice_influence.png")]


def _plot_gatekeeper_reliability(decisions: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Accepted vs. rejected consultation outcomes over training.

    A high query rate is not useful if many responses get rejected -- the
    agent pays the consultation cost either way. ``response_status`` only
    ever takes two values in this release (``accepted`` /
    ``schema_rejected``): there is no elapsed-timeout status (spec
    sections 5/10/14 -- consultation waits indefinitely) and no separate
    "Plan Maker unavailable" status (a dead peer fails the whole run
    instead, see agents/orchestrator.py's disconnect handling).
    """
    queried = decisions[decisions["queried"] == True]  # noqa: E712
    if queried.empty or queried["response_status"].isna().all():
        return []

    window = max(1, min(20, len(queried)))
    accepted = (queried["response_status"] == "accepted").astype(float)
    rejected = (queried["response_status"] == "schema_rejected").astype(float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        range(len(accepted)), accepted.rolling(window, min_periods=1).mean(),
        label="accepted", color="tab:green",
    )
    ax.plot(
        range(len(rejected)), rejected.rolling(window, min_periods=1).mean(),
        label="rejected", color="tab:red",
    )
    ax.set_xlabel("Queried decision (in collection order)")
    ax.set_ylabel(f"Rate (rolling, window={window})")
    ax.set_ylim(*_RATE_YLIM)
    ax.set_title("Gatekeeper consultation outcomes over training")
    ax.legend()
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "gatekeeper_reliability.png")]


def _plot_plan_maker_latency(decisions: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Response latency distribution and trend -- the real wall-clock cost
    of consultation, separate from its reward cost (see
    reward_over_training.png). Median/p95 are reported alongside the mean
    since latency distributions are typically long-tailed.
    """
    latencies = decisions["response_latency_ms"].dropna()
    if latencies.empty:
        return []

    mean, median, p95, p99 = (
        latencies.mean(), latencies.median(), latencies.quantile(0.95), latencies.quantile(0.99)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(latencies, bins=min(30, max(5, latencies.nunique())), color="tab:blue", alpha=0.8)
    for value, label, color in (
        (mean, "mean", "black"), (median, "median", "tab:orange"), (p95, "p95", "tab:red"),
    ):
        ax1.axvline(value, color=color, linestyle="--", linewidth=1, label=f"{label}={value:.0f}ms")
    ax1.set_xlabel("Response latency (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title("Plan Maker latency distribution")
    ax1.legend(fontsize="small")
    ax1.grid(alpha=0.3)

    window = max(1, min(20, len(latencies)))
    ax2.plot(range(len(latencies)), latencies.rolling(window, min_periods=1).mean(), color="tab:blue")
    ax2.set_xlabel("Queried decision (in collection order)")
    ax2.set_ylabel(f"Latency (ms, rolling mean, window={window})")
    ax2.set_title("Plan Maker latency over training")
    ax2.grid(alpha=0.3)

    fig.suptitle(f"mean={mean:.0f}ms  median={median:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms", fontsize=9)
    return [_save(fig, plots_dir / "plan_maker_latency.png")]


def _plot_query_decision_analysis(decisions: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Does the query gate actually ask under uncertainty?

    Splits base-policy entropy by whether that decision was queried, and
    scatters entropy against the query gate's own predicted probability --
    if the query gate uses uncertainty as intended, query probability
    should generally trend upward with entropy, though the relationship
    need not be monotonic since the GRU state and consultation cost also
    matter. Skipped for the baseline variant (no query gate at all).
    """
    if "query_probability" not in decisions.columns or decisions["query_probability"].isna().all():
        return []

    queried_mask = decisions["queried"] == True  # noqa: E712
    entropy_queried = decisions["base_policy_entropy"].where(queried_mask)
    entropy_non_queried = decisions["base_policy_entropy"].where(~queried_mask)
    window = max(1, min(20, len(decisions)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(
        decisions.index, entropy_queried.rolling(window, min_periods=1).mean(),
        label="queried", color="tab:purple",
    )
    ax1.plot(
        decisions.index, entropy_non_queried.rolling(window, min_periods=1).mean(),
        label="not queried", color="tab:gray", alpha=0.8,
    )
    ax1.set_xlabel("Decision (in collection order)")
    ax1.set_ylabel(f"Base policy entropy (rolling, window={window})")
    ax1.set_title("Entropy: queried vs. non-queried decisions")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.scatter(decisions["base_policy_entropy"], decisions["query_probability"], s=8, alpha=0.2, color="tab:blue")
    ax2.set_xlabel("Base policy entropy")
    ax2.set_ylabel("Query probability")
    ax2.set_title("Query probability vs. base policy entropy")
    ax2.grid(alpha=0.3)
    return [_save(fig, plots_dir / "query_decision_analysis.png")]


def _plot_rollout_timing(rollouts: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Where rollout wall-clock time actually goes -- environment/rollout
    collection vs. PPO optimization vs. periodic evaluation (spec section
    17). Plan Maker latency is never a separate term here: it is already
    included in collection_seconds (a consultation happens *during*
    collection, see learning/rollout.py's ``_decide``), never in
    optimization_seconds (PPO replay never calls the Plan Maker at all).
    """
    if "collection_seconds" not in rollouts.columns:
        return []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rollouts["rollout"], rollouts["collection_seconds"], label="collection (env + Plan Maker)", marker=".")
    ax.plot(rollouts["rollout"], rollouts["optimization_seconds"], label="PPO optimization", marker=".")
    if rollouts["evaluation_seconds"].notna().any():
        ax.plot(rollouts["rollout"], rollouts["evaluation_seconds"], label="periodic evaluation", marker=".")
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Seconds")
    ax.set_title("Rollout wall-clock time breakdown")
    ax.legend()
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "rollout_timing.png")]


def _plot_compatible_action_probability(rollouts: pd.DataFrame, plots_dir: Path) -> list[Path]:
    r"""$P(\text{action compatible with observed facts})$ vs. environment
    steps -- the direct evidence that the compatibility-feature fix (see
    :mod:`marla.environment.action_compatibility`) is actually usable by
    the trained policy, not just representable in principle: the policy's
    probability mass on ``CONFIRMED_COMPATIBLE`` actions should rise over
    training as the actor learns to prefer them over ones it cannot yet
    confirm are safe. Absent for a run written before these columns
    existed (spec section 16's tolerance pattern for older rollouts.csv
    files) -- not a crash, just fewer plots. Both base and final series are
    always plotted; for the baseline/PPO_ONLY variant they coincide
    exactly (no advice ever perturbs the logits), which is itself the
    expected, verifiable picture rather than a reason to hide one series.
    """
    if "mean_base_compatible_action_probability_mass" not in rollouts.columns:
        return []
    if rollouts["mean_base_compatible_action_probability_mass"].isna().all():
        return []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = rollouts["environment_steps_total"]
    ax.plot(x, rollouts["mean_base_compatible_action_probability_mass"], label="base policy", color="tab:blue", marker=".")
    if not rollouts["mean_final_compatible_action_probability_mass"].equals(
        rollouts["mean_base_compatible_action_probability_mass"]
    ):
        ax.plot(
            x, rollouts["mean_final_compatible_action_probability_mass"],
            label="final policy (post-advice)", color="tab:purple", linestyle="--", marker=".",
        )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(r"$P(\mathrm{action\ compatible\ with\ observed\ facts})$")
    ax.set_ylim(*_RATE_YLIM)
    ax.set_title("Compatible-action probability mass over training")
    ax.legend()
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "compatible_action_probability.png")]


def _plot_finish_probability_by_objective_state(rollouts: pd.DataFrame, plots_dir: Path) -> list[Path]:
    r"""$P(\mathrm{FINISH})$ conditioned on whether the true objective was
    already satisfied *at decision time* (``objective_satisfied_before_action``,
    never the post-action field -- see ``StepRecord``'s own docstring for
    why), over training environment steps. The central FINISH-learnability
    question this plot answers directly: does the policy learn
    $P(\mathrm{FINISH}\mid \text{objective reached}) \to 1$ while keeping
    $P(\mathrm{FINISH}\mid \text{objective not reached}) \to 0$? Absent for
    a run written before these ``rollouts.csv`` columns existed. Base and
    final-policy series both plotted when they differ (assisted variant);
    for PPO_ONLY they coincide exactly.
    """
    if "mean_base_finish_probability_objective_reached" not in rollouts.columns:
        return []
    if rollouts["mean_base_finish_probability_objective_reached"].isna().all() and rollouts[
        "mean_base_finish_probability_objective_not_reached"
    ].isna().all():
        return []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = rollouts["environment_steps_total"]
    ax.plot(
        x, rollouts["mean_base_finish_probability_objective_reached"],
        label=r"$P(\mathrm{FINISH}\mid$objective reached$)$ -- base", color="tab:green", marker=".",
    )
    ax.plot(
        x, rollouts["mean_base_finish_probability_objective_not_reached"],
        label=r"$P(\mathrm{FINISH}\mid$objective not reached$)$ -- base", color="tab:red", marker=".",
    )
    has_final_series = (
        "mean_final_finish_probability_objective_reached" in rollouts.columns
        and not rollouts["mean_final_finish_probability_objective_reached"].equals(
            rollouts["mean_base_finish_probability_objective_reached"]
        )
    )
    if has_final_series:
        ax.plot(
            x, rollouts["mean_final_finish_probability_objective_reached"],
            label=r"$P(\mathrm{FINISH}\mid$objective reached$)$ -- final", color="tab:green", linestyle="--", marker=".",
        )
        ax.plot(
            x, rollouts["mean_final_finish_probability_objective_not_reached"],
            label=r"$P(\mathrm{FINISH}\mid$objective not reached$)$ -- final", color="tab:red", linestyle="--", marker=".",
        )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(r"$P(\mathrm{FINISH})$")
    ax.set_ylim(*_RATE_YLIM)
    ax.set_title("FINISH probability, conditioned on decision-time objective state")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "finish_probability_by_objective_state.png")]


_MIN_DECISIONS_PER_REMAINING_TARGET_COUNT = 5


def _plot_finish_probability_by_remaining_targets(decisions: pd.DataFrame, plots_dir: Path) -> list[Path]:
    r"""$P(\mathrm{FINISH})$ (here, the base-policy FINISH probability) as a
    function of the number of true sensitive targets still missing ROOT
    *at decision time* (``sensitive_targets_remaining_before_action`` --
    diagnostic simulator truth, never a policy input). Expected shape:
    remaining=0 -> high FINISH probability, remaining>0 -> low. A purely
    diagnostic plot, not a paper result: a remaining-target count with
    fewer than :data:`_MIN_DECISIONS_PER_REMAINING_TARGET_COUNT` decisions
    this run is dropped rather than shown as a fabricated stable estimate.
    Absent entirely for a run written before these columns existed.

    Line+markers (not bars): the x values (remaining-target counts) are
    ordered discrete counts where the *trend* across them is the point of
    the plot (does FINISH probability fall as more targets remain?) --
    bars would visually suggest unordered categories, and a connecting
    line makes that trend legible at a glance in a way a bare scatter of
    points would not.
    """
    required = {"sensitive_targets_remaining_before_action", "base_finish_probability"}
    if not required.issubset(decisions.columns):
        return []
    data = decisions.dropna(subset=list(required))
    if data.empty:
        return []

    grouped = data.groupby("sensitive_targets_remaining_before_action")["base_finish_probability"]
    counts = grouped.count()
    means = grouped.mean()
    stds = grouped.std().fillna(0.0)
    kept = counts[counts >= _MIN_DECISIONS_PER_REMAINING_TARGET_COUNT].index
    if len(kept) == 0:
        return []
    means, stds, counts = means.loc[kept].sort_index(), stds.loc[kept], counts.loc[kept]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = means.index.to_numpy()
    ax.errorbar(x, means.to_numpy(), yerr=stds.reindex(means.index).to_numpy(), fmt="o-", color="tab:blue", capsize=3)
    for xi, n in zip(x, counts.reindex(means.index).to_numpy()):
        ax.annotate(f"n={n}", (xi, means.loc[xi]), textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")
    ax.set_xlabel("Sensitive targets remaining without ROOT")
    ax.set_ylabel(r"Mean $P(\mathrm{FINISH})$ (base policy)")
    ax.set_ylim(*_RATE_YLIM)
    ax.set_xticks(x)
    if len(x) == 1:
        # A single x-category still needs to look intentional, not like a
        # sliver of a wider plot that failed to render -- give the one
        # point some visible breathing room either side.
        ax.set_xlim(x[0] - 1, x[0] + 1)
    _set_title_with_note(
        ax, "FINISH probability vs. remaining targets",
        "Diagnostic only -- uses true simulator state, not the policy's own observation\n"
        f"Only remaining-target counts with >= {_MIN_DECISIONS_PER_REMAINING_TARGET_COUNT} decisions are shown",
    )
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "finish_probability_by_remaining_targets.png")]


def _plot_finish_probability_by_visible_completion_and_frontier(rollouts: pd.DataFrame, plots_dir: Path) -> list[Path]:
    r"""The main new FINISH/exploration diagnostic (spec sections 23/25/38):
    $P(\mathrm{FINISH})$ (base policy), restricted to decisions where every
    currently-*visible* sensitive target is already rooted, split by
    whether a known-subnet exploration frontier remains -- built entirely
    from VISIBLE evidence (``all_visible_sensitive_targets_rooted_before_action``/
    ``known_exploration_frontier_remaining_before_action``), never the
    true objective. Desirable shape: the "no frontier" curve rises above
    the "frontier remains" curve -- $\Delta_{\text{frontier}}$
    (``mean_finish_probability_visible_complete_no_frontier`` minus
    ``..._frontier_remaining``) growing over training means the policy is
    learning to condition FINISH on observable exploration completion,
    not just on visible target completion alone. Absent for a run written
    before these columns existed.
    """
    required = {
        "mean_finish_probability_visible_complete_frontier_remaining",
        "mean_finish_probability_visible_complete_no_frontier",
    }
    if not required.issubset(rollouts.columns):
        return []
    if rollouts[list(required)].isna().all().all():
        return []

    fig, ax = plt.subplots(figsize=(8, 4.7))
    x = rollouts["environment_steps_total"]
    ax.plot(
        x, rollouts["mean_finish_probability_visible_complete_no_frontier"],
        label="targets complete, no unscanned subnets known", color="tab:green", marker=".",
    )
    ax.plot(
        x, rollouts["mean_finish_probability_visible_complete_frontier_remaining"],
        label="targets complete, an unscanned subnet remains", color="tab:orange", marker=".",
    )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(r"$P(\mathrm{FINISH})$ (base policy)")
    ax.set_ylim(*_RATE_YLIM)
    _set_title_with_note(
        ax, "FINISH probability by exploration status",
        "Visible-target-complete decisions only, split by whether an unscanned known subnet remains",
    )
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "finish_probability_by_visible_completion_and_frontier.png")]


def _plot_known_subnet_exploration_progress(rollouts: pd.DataFrame, plots_dir: Path) -> list[Path]:
    r"""Does the policy actually explore the known topology? Fraction of
    currently-*known* subnets successfully scanned (mean over each
    rollout's decisions), and the fraction of decisions where a known
    exploration frontier still remains, over training environment steps.
    Never the true subnet count -- see
    :mod:`marla.environment.visible_facts`. Absent for a run written
    before these ``rollouts.csv`` columns existed.
    """
    if "mean_fraction_known_subnets_scanned" not in rollouts.columns:
        return []
    if rollouts["mean_fraction_known_subnets_scanned"].isna().all():
        return []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = rollouts["environment_steps_total"]
    ax.plot(
        x, rollouts["mean_fraction_known_subnets_scanned"],
        label="mean fraction of known subnets scanned", color="tab:blue", marker=".",
    )
    if "known_frontier_remaining_rate" in rollouts.columns and not rollouts["known_frontier_remaining_rate"].isna().all():
        ax.plot(
            x, rollouts["known_frontier_remaining_rate"],
            label="decisions with an unscanned known subnet remaining", color="tab:purple", linestyle="--", marker=".",
        )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Rate")
    ax.set_ylim(*_RATE_YLIM)
    ax.set_title("Known subnet exploration progress over training")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "known_subnet_exploration_progress.png")]


def _plot_critic_quality(decisions: pd.DataFrame, plots_dir: Path) -> list[Path]:
    r"""Is the critic well-calibrated? ``critic_value`` ($V_t$, predicted at
    collection time) vs. ``return_target`` ($R_t$, the GAE-derived target it
    was trained toward) -- points near the y=x line mean $V_t \approx R_t$.
    """
    if "critic_value" not in decisions.columns or "return_target" not in decisions.columns:
        return []
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(decisions["return_target"], decisions["critic_value"], s=6, alpha=0.15, color="tab:blue")
    lo = min(decisions["return_target"].min(), decisions["critic_value"].min())
    hi = max(decisions["return_target"].max(), decisions["critic_value"].max())
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, label=r"$V_t = R_t$")
    ax.set_xlabel(r"Return target $R_t$")
    ax.set_ylabel(r"Critic value $V_t$")
    ax.set_title(r"Critic calibration: predicted value $V_t$ vs. GAE return target $R_t$")
    ax.legend()
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "critic_quality.png")]


def _plot_reward_vs_credit_assignment(decisions: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """A temporal-credit-assignment diagnostic: one point per training
    decision, immediate reward vs. the credit PPO actually assigned that
    step (``gae_advantage``) -- the highlighted quadrant is negative
    reward with positive advantage: an immediately costly step (e.g. a
    paid consultation, or a failed exploit attempt) that GAE still
    credits because it led to later success. Everything else is plotted
    too, unhighlighted, so the highlighted quadrant's actual density is
    visible in context rather than shown in isolation.
    """
    if "gae_advantage" not in decisions.columns:
        return []
    fig, ax = plt.subplots(figsize=(7, 5.8))
    negative_reward_positive_advantage = (decisions["training_reward"] < 0) & (decisions["gae_advantage"] > 0)
    # Exactly one condition is highlighted; its complement is genuinely
    # "every other decision" (not a fabricated 3- or 4-way breakdown this
    # plot doesn't actually compute) -- "All other decisions" says that
    # honestly, where a bare "other" would not explain itself.
    ax.scatter(
        decisions.loc[~negative_reward_positive_advantage, "training_reward"],
        decisions.loc[~negative_reward_positive_advantage, "gae_advantage"],
        s=6, alpha=0.15, color="tab:gray", label="All other decisions",
    )
    ax.scatter(
        decisions.loc[negative_reward_positive_advantage, "training_reward"],
        decisions.loc[negative_reward_positive_advantage, "gae_advantage"],
        s=10, alpha=0.4, color="tab:red", label="Negative reward, positive advantage",
    )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Training reward (this step)")
    ax.set_ylabel("GAE advantage")
    _set_title_with_note(
        ax, "Reward vs. GAE advantage",
        "Each point is one training decision\nHighlighted: an immediately costly step PPO still credits for later success",
    )
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "reward_vs_credit_assignment.png")]


def _plot_action_type_distribution(decisions: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """How the mix of chosen action types (scan / exploit / privilege
    escalation / finish / ...) shifts over training -- e.g. a policy that
    starts by scanning almost everything and gradually shifts toward
    exploiting/escalating as it learns is a qualitatively different
    learning curve than raw return alone can show.
    """
    if "selected_action_type" not in decisions.columns:
        return []
    x_col = "rollout" if "rollout" in decisions.columns and decisions["rollout"].notna().any() else None
    if x_col is None:
        return []
    fractions = (
        decisions.groupby(x_col)["selected_action_type"]
        .value_counts(normalize=True)
        .unstack("selected_action_type")
        .fillna(0.0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(fractions))
    for action_type in fractions.columns:
        ax.bar(fractions.index, fractions[action_type], bottom=bottom, label=action_type, alpha=0.85)
        bottom += fractions[action_type].to_numpy()
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Fraction of decisions")
    ax.set_ylim(0, _RATE_YLIM[1])
    ax.set_title("Action-type distribution over training")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize="small")
    ax.grid(alpha=0.3)
    return [_save(fig, plots_dir / "action_type_distribution.png")]


def _plot_finish_efficiency(episodes: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Objective/FINISH timing (spec section 5), restricted to
    successful-FINISH episodes: when the objective first became satisfied
    (``steps_to_goal``) vs. when the agent actually selected FINISH
    (``finish_step``) vs. the gap between them (``finish_delay_steps``) --
    distinct from episode_efficiency.png's "steps to objective" panel,
    which covers every objective-reached episode (FINISH not required)
    and doesn't show the finish-timing side at all.
    """
    if "finish_step" not in episodes.columns or "finish_delay_steps" not in episodes.columns:
        return []
    train, _eval_, x_col = _split_train_eval(episodes)
    x_label = "Rollout" if x_col == "rollout" else "Episode"
    successful = train[train["goal_success"] == True]  # noqa: E712
    if successful.empty:
        return []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for column, label, color in (
        ("steps_to_goal", "steps to objective", "tab:blue"),
        ("finish_step", "FINISH step", "tab:orange"),
    ):
        x, mean, band = _grouped_mean_and_band(successful, x_col, column)
        ax1.plot(x, mean, label=label, color=color, marker=".")
        ax1.fill_between(x, (mean - band).clip(min=0), mean + band, color=color, alpha=0.15)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Environment step")
    _set_title_with_note(ax1, "Steps to objective vs. FINISH step", "Successful-FINISH episodes only" + _SMOOTHING_NOTE)
    ax1.legend()
    ax1.grid(alpha=0.3)

    x, mean, band = _grouped_mean_and_band(successful, x_col, "finish_delay_steps")
    ax2.plot(x, mean, color="tab:green", marker=".")
    ax2.fill_between(x, (mean - band).clip(min=0), mean + band, color="tab:green", alpha=0.2)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("FINISH step minus steps to objective")
    _set_title_with_note(ax2, "FINISH delay after objective satisfied", "Successful-FINISH episodes only" + _SMOOTHING_NOTE)
    ax2.grid(alpha=0.3)
    return [_save(fig, plots_dir / "finish_efficiency.png")]


def _plot_assisted_policy_influence(decisions: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Base vs. final policy confidence, for accepted advice only (see
    :func:`_accepted_advice_rows`) -- does consultation actually sharpen
    (lower entropy) or reshape (lower selected-action probability at the
    base policy, higher at the final one) the decision, on top of
    advice_influence.png's beta/top-action-change view.
    """
    if "final_policy_entropy" not in decisions.columns:
        return []
    accepted = _accepted_advice_rows(decisions)
    if accepted.empty:
        return []

    window = max(1, min(20, len(accepted)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(
        range(len(accepted)), accepted["base_policy_entropy"].rolling(window, min_periods=1).mean(),
        label="base entropy", color="tab:gray",
    )
    ax1.plot(
        range(len(accepted)), accepted["final_policy_entropy"].rolling(window, min_periods=1).mean(),
        label="final entropy", color="tab:purple",
    )
    ax1.set_xlabel("Queried decision (in collection order)")
    ax1.set_ylabel(f"Entropy (rolling, window={window})")
    ax1.set_title("Base vs. final policy entropy\n(queried decisions)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(
        range(len(accepted)), accepted["selected_action_base_probability"].rolling(window, min_periods=1).mean(),
        label="base probability", color="tab:gray",
    )
    ax2.plot(
        range(len(accepted)), accepted["selected_action_final_probability"].rolling(window, min_periods=1).mean(),
        label="final probability", color="tab:purple",
    )
    ax2.set_xlabel("Queried decision (in collection order)")
    ax2.set_ylabel(f"P(selected action) (rolling, window={window})")
    ax2.set_ylim(*_RATE_YLIM)
    ax2.set_title("Selected-action probability shift\n(queried decisions)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    return [_save(fig, plots_dir / "assisted_policy_influence.png")]
