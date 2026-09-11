"""Post-hoc metric analysis over a completed run's CSVs: ROOT AUC,
State-N (visible-target) occupancy/FINISH-probability curves, and the
established multi-signal collapse definition -- used by the Optuna study
system (:mod:`marla.optuna_study`) to score a trial, and reusable by any
research script that wants these numbers without reimplementing them
again.

Pure functions over already-written ``episodes.csv``/``decisions.csv``/
``rollouts.csv``/``updates.csv`` DataFrames -- never touches a live
environment or policy, never re-derives anything CSV-schema-specific that
:mod:`marla.metrics.writer` doesn't already write.

"State N" here is the vectorized equivalent of
:func:`marla.learning.ppo.visible_target_state`, applied to
``decisions.csv``'s own ``has_visible_sensitive_target_before_action`` /
``all_visible_sensitive_targets_rooted_before_action`` columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify_visible_target_state(decisions: pd.DataFrame) -> pd.Series:
    """"N" (no sensitive target currently visible) / "I" (>=1 visible,
    not all ROOT) / "C" (>=1 visible, all currently-visible ones ROOT) --
    see :func:`marla.learning.ppo.visible_target_state`'s own docstring
    for the exact semantics this mirrors.
    """
    has_target = decisions["has_visible_sensitive_target_before_action"].astype(bool)
    # NaN/null in all_visible_sensitive_targets_rooted_before_action is
    # expected exactly when has_target is False (there is no visible
    # target to check "all rooted" for) -- filled with False first since
    # its value is irrelevant whenever has_target is False anyway (the
    # state is "N" regardless of what this column says).
    all_rooted = decisions["all_visible_sensitive_targets_rooted_before_action"].fillna(False).astype(bool)
    state = np.where(~has_target, "N", np.where(all_rooted, "C", "I"))
    return pd.Series(state, index=decisions.index)


def _training_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    if "is_eval" in episodes.columns:
        return episodes[episodes["is_eval"] == False]  # noqa: E712
    return episodes


def per_rollout_root_curve(episodes: pd.DataFrame, rollouts: pd.DataFrame) -> pd.DataFrame:
    """One row per rollout: ``mean_targets_rooted``/``fraction_with_any_root``,
    from TRAINING (non-eval) episodes only, joined to each rollout's
    cumulative ``environment_steps_total``.
    """
    train_ep = _training_episodes(episodes)
    if train_ep.empty or "sensitive_targets_with_root_final" not in train_ep.columns:
        return pd.DataFrame(columns=["rollout", "mean_targets_rooted", "fraction_with_any_root", "environment_steps_total"])
    grouped = train_ep.groupby("rollout")["sensitive_targets_with_root_final"].agg(
        mean_targets_rooted="mean", fraction_with_any_root=lambda s: (s >= 1).mean()
    )
    grouped = grouped.reset_index()
    steps = rollouts[["rollout", "environment_steps_total"]]
    return grouped.merge(steps, on="rollout", how="inner").sort_values("rollout").reset_index(drop=True)


def trapz_auc(curve: pd.DataFrame, y_col: str, x_col: str = "environment_steps_total") -> float | None:
    """Trapezoidal AUC of ``y_col`` over ``x_col``, normalized by the
    x-span so the result is comparable across runs with different step
    budgets (mean value over the covered range, not a raw integral).
    ``None`` (never 0.0 or NaN) when there are fewer than 2 points or a
    zero/negative span.
    """
    if y_col not in curve.columns or x_col not in curve.columns:
        return None
    df = curve[[x_col, y_col]].dropna()
    if len(df) < 2:
        return None
    x, y = df[x_col].to_numpy(dtype=float), df[y_col].to_numpy(dtype=float)
    span = x[-1] - x[0]
    if span <= 0:
        return None
    return float(np.trapz(y, x) / span)


def root_auc(episodes: pd.DataFrame, rollouts: pd.DataFrame) -> float | None:
    """The primary tuning metric (spec: ROOT AUC for sensitive-target
    ROOT progress across environment steps) -- trapezoidal AUC of
    per-rollout mean targets rooted.
    """
    return trapz_auc(per_rollout_root_curve(episodes, rollouts), "mean_targets_rooted")


def per_rollout_state_n_curve(decisions: pd.DataFrame, rollouts: pd.DataFrame) -> pd.DataFrame:
    """One row per rollout: ``occupancy_N`` (fraction of ALL decisions in
    State N) and ``p_selected_finish_N`` (fraction of State-N decisions
    that selected FINISH -- ``None`` for a rollout with zero State-N
    decisions, never a fabricated 0.0).
    """
    if decisions.empty:
        return pd.DataFrame(columns=["rollout", "occupancy_N", "p_selected_finish_N", "environment_steps_total"])
    d = decisions.copy()
    d["state"] = classify_visible_target_state(d)
    rows = []
    for rollout, group in d.groupby("rollout"):
        n_group = group[group["state"] == "N"]
        rows.append(
            {
                "rollout": rollout,
                "occupancy_N": float((group["state"] == "N").mean()),
                "p_selected_finish_N": float((n_group["selected_action_type"] == "finish").mean()) if len(n_group) else None,
            }
        )
    curve = pd.DataFrame(rows)
    steps = rollouts[["rollout", "environment_steps_total"]]
    return curve.merge(steps, on="rollout", how="inner").sort_values("rollout").reset_index(drop=True)


def per_rollout_episode_length(episodes: pd.DataFrame) -> pd.DataFrame:
    train_ep = _training_episodes(episodes)
    if train_ep.empty:
        return pd.DataFrame(columns=["rollout", "mean_episode_length"])
    return (
        train_ep.groupby("rollout")["environment_steps"]
        .mean()
        .reset_index()
        .rename(columns={"environment_steps": "mean_episode_length"})
    )


def detect_collapse(
    episodes: pd.DataFrame, decisions: pd.DataFrame, rollouts: pd.DataFrame, updates: pd.DataFrame
) -> dict:
    """The established multi-signal collapse definition (never a single
    arbitrary threshold -- see research/diagnostics/*'s prior collapse-
    investigation phases): evaluated over the LAST ~25% of a run's
    rollouts (at least 1). Requires at least 3 of 5 corroborating signals:
    P(FINISH|N) high, State-N occupancy near saturation, mean episode
    length near 1, action entropy collapsed, ROOT capture disappears.
    """
    state_curve = per_rollout_state_n_curve(decisions, rollouts)
    length_curve = per_rollout_episode_length(episodes)
    root_curve = per_rollout_root_curve(episodes, rollouts)

    if state_curve.empty:
        return {"collapsed": False, "n_corroborating_signals": 0, "signals": {}, "reason": "no decisions data"}

    all_rollouts = sorted(state_curve["rollout"].unique())
    n_late = max(1, int(round(len(all_rollouts) * 0.25)))
    late_rollouts = set(all_rollouts[-n_late:])

    late_state = state_curve[state_curve["rollout"].isin(late_rollouts)]
    late_length = length_curve[length_curve["rollout"].isin(late_rollouts)] if not length_curve.empty else length_curve
    late_root = root_curve[root_curve["rollout"].isin(late_rollouts)] if not root_curve.empty else root_curve

    late_p_finish_n = late_state["p_selected_finish_N"].mean()
    late_occupancy_n = late_state["occupancy_N"].mean()
    late_mean_length = late_length["mean_episode_length"].mean() if not late_length.empty else None
    late_mean_targets_rooted = late_root["mean_targets_rooted"].mean() if not late_root.empty else None

    late_action_entropy = None
    if "action_entropy" in updates.columns and "rollout" in updates.columns and not updates.empty:
        late_updates = updates[updates["rollout"].isin(late_rollouts)]
        if not late_updates.empty:
            late_action_entropy = late_updates["action_entropy"].mean()

    def _finite(x: float | None) -> float | None:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)

    late_p_finish_n = _finite(late_p_finish_n)
    late_occupancy_n = _finite(late_occupancy_n)
    late_mean_length = _finite(late_mean_length)
    late_mean_targets_rooted = _finite(late_mean_targets_rooted)
    late_action_entropy = _finite(late_action_entropy)

    signals = {
        "p_finish_n_high": late_p_finish_n is not None and late_p_finish_n > 0.8,
        "occupancy_n_saturated": late_occupancy_n is not None and late_occupancy_n > 0.85,
        "episode_length_collapsed": late_mean_length is not None and late_mean_length < 3.0,
        "entropy_collapsed": late_action_entropy is not None and late_action_entropy < 0.5,
        "no_root_capture": late_mean_targets_rooted is not None and late_mean_targets_rooted < 0.05,
    }
    n_signals = sum(1 for v in signals.values() if v)

    return {
        "collapsed": n_signals >= 3,
        "n_corroborating_signals": n_signals,
        "signals": signals,
        "late_p_finish_state_N": late_p_finish_n,
        "late_occupancy_state_N": late_occupancy_n,
        "late_mean_episode_length": late_mean_length,
        "late_action_entropy": late_action_entropy,
        "late_mean_targets_rooted": late_mean_targets_rooted,
    }
