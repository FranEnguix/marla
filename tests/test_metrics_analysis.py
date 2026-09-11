"""marla.metrics.analysis -- ROOT AUC and the multi-signal collapse
definition, tested against small synthetic DataFrames (no real training
run needed -- these are pure functions over already-written CSVs).
"""

from __future__ import annotations

import pandas as pd
import pytest

from marla.metrics.analysis import (
    classify_visible_target_state,
    detect_collapse,
    per_rollout_episode_length,
    per_rollout_root_curve,
    per_rollout_state_n_curve,
    root_auc,
    trapz_auc,
)


def test_classify_visible_target_state_matches_ppo_visible_target_state_semantics():
    decisions = pd.DataFrame(
        {
            "has_visible_sensitive_target_before_action": [False, True, True],
            "all_visible_sensitive_targets_rooted_before_action": [False, False, True],
        }
    )
    states = classify_visible_target_state(decisions)
    assert list(states) == ["N", "I", "C"]


def test_trapz_auc_of_a_constant_curve_equals_the_constant():
    curve = pd.DataFrame({"environment_steps_total": [0, 1000, 2000], "y": [0.5, 0.5, 0.5]})
    assert trapz_auc(curve, "y") == pytest.approx(0.5)


def test_trapz_auc_none_on_insufficient_points():
    assert trapz_auc(pd.DataFrame({"environment_steps_total": [0], "y": [1.0]}), "y") is None
    assert trapz_auc(pd.DataFrame(columns=["environment_steps_total", "y"]), "y") is None


def test_root_auc_reflects_rising_root_capture():
    episodes = pd.DataFrame(
        {
            "rollout": [1, 1, 2, 2, 3, 3],
            "is_eval": [False] * 6,
            "sensitive_targets_with_root_final": [0, 0, 1, 1, 2, 2],
        }
    )
    rollouts = pd.DataFrame({"rollout": [1, 2, 3], "environment_steps_total": [1000, 2000, 3000]})
    auc = root_auc(episodes, rollouts)
    assert auc is not None
    assert 0.5 < auc < 1.5  # rises 0 -> 1 -> 2, AUC of the mean curve should land near 1


def test_per_rollout_state_n_curve_p_finish_is_none_when_no_state_n_decisions():
    decisions = pd.DataFrame(
        {
            "rollout": [1, 1],
            "has_visible_sensitive_target_before_action": [True, True],
            "all_visible_sensitive_targets_rooted_before_action": [False, False],
            "selected_action_type": ["exploit", "os_scan"],
        }
    )
    rollouts = pd.DataFrame({"rollout": [1], "environment_steps_total": [1000]})
    curve = per_rollout_state_n_curve(decisions, rollouts)
    assert curve.loc[0, "occupancy_N"] == 0.0
    assert curve.loc[0, "p_selected_finish_N"] is None


def test_detect_collapse_flags_the_canonical_collapse_pattern():
    """A tiny synthetic run matching the historically-observed seed919
    collapse pattern: State-N occupancy saturates, P(FINISH|N) -> 1,
    episodes collapse to length 1, entropy collapses, no more ROOT
    capture in the late window.
    """
    rollouts = pd.DataFrame({"rollout": list(range(1, 9)), "environment_steps_total": [i * 512 for i in range(1, 9)]})
    # Early rollouts: healthy. Late rollouts (7, 8): collapsed.
    decisions_rows = []
    for rollout in range(1, 9):
        collapsed = rollout >= 7
        n_steps = 20
        for _ in range(n_steps):
            decisions_rows.append(
                {
                    "rollout": rollout,
                    "has_visible_sensitive_target_before_action": not collapsed,
                    "all_visible_sensitive_targets_rooted_before_action": False,
                    "selected_action_type": "finish" if collapsed else "exploit",
                }
            )
    decisions = pd.DataFrame(decisions_rows)

    episodes_rows = []
    for rollout in range(1, 9):
        collapsed = rollout >= 7
        for _ in range(5):
            episodes_rows.append(
                {
                    "rollout": rollout,
                    "is_eval": False,
                    "environment_steps": 1 if collapsed else 40,
                    "sensitive_targets_with_root_final": 0 if collapsed else 1,
                }
            )
    episodes = pd.DataFrame(episodes_rows)

    updates_rows = [{"rollout": r, "action_entropy": 0.01 if r >= 7 else 3.5} for r in range(1, 9) for _ in range(4)]
    updates = pd.DataFrame(updates_rows)

    result = detect_collapse(episodes, decisions, rollouts, updates)
    assert result["collapsed"] is True
    assert result["n_corroborating_signals"] >= 3
    assert result["signals"]["p_finish_n_high"] is True
    assert result["signals"]["occupancy_n_saturated"] is True
    assert result["signals"]["episode_length_collapsed"] is True


def test_detect_collapse_does_not_flag_a_healthy_run():
    rollouts = pd.DataFrame({"rollout": list(range(1, 9)), "environment_steps_total": [i * 512 for i in range(1, 9)]})
    decisions_rows = []
    for rollout in range(1, 9):
        for i in range(20):
            decisions_rows.append(
                {
                    "rollout": rollout,
                    "has_visible_sensitive_target_before_action": i % 2 == 0,
                    "all_visible_sensitive_targets_rooted_before_action": False,
                    "selected_action_type": "exploit" if i % 5 != 0 else "os_scan",
                }
            )
    decisions = pd.DataFrame(decisions_rows)

    episodes_rows = []
    for rollout in range(1, 9):
        for _ in range(5):
            episodes_rows.append(
                {
                    "rollout": rollout,
                    "is_eval": False,
                    "environment_steps": 45,
                    "sensitive_targets_with_root_final": 1,
                }
            )
    episodes = pd.DataFrame(episodes_rows)
    updates = pd.DataFrame([{"rollout": r, "action_entropy": 3.8} for r in range(1, 9) for _ in range(4)])

    result = detect_collapse(episodes, decisions, rollouts, updates)
    assert result["collapsed"] is False


def test_detect_collapse_handles_empty_decisions_gracefully():
    result = detect_collapse(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert result["collapsed"] is False


def test_per_rollout_episode_length_excludes_eval_episodes():
    episodes = pd.DataFrame(
        {
            "rollout": [1, 1, 1],
            "is_eval": [False, False, True],
            "environment_steps": [10, 20, 999],
        }
    )
    curve = per_rollout_episode_length(episodes)
    assert curve.loc[curve["rollout"] == 1, "mean_episode_length"].iloc[0] == 15.0
