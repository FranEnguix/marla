"""Regression tests for plot titles/subtitles/legends/missing-data
behavior -- metadata and rendering-path checks, never fragile
pixel-perfect image comparisons (see ``tests/test_plots.py``'s own
``_FigureCapture`` pattern, reused here).
"""

from __future__ import annotations

import pandas as pd
import pytest

import marla.metrics.plots as plots_module
from marla.metrics.plots import (
    _MIN_DECISIONS_PER_REMAINING_TARGET_COUNT,
    _MIN_RESOURCE_SAMPLES_FOR_TIMESERIES,
    _plot_episode_efficiency,
    _plot_finish_probability_by_remaining_targets,
    _plot_resource_cpu,
    _plot_resource_ram,
    _plot_reward_vs_credit_assignment,
)


def _tmp_dir(tmp_path):
    d = tmp_path / "plots"
    d.mkdir()
    return d


class _FigureCapture:
    """See tests/test_plots.py's own copy of this helper -- duplicated
    (not imported) so this file has no import-order dependency on that
    one; keeping both in sync is a small price for that independence.
    """

    def __init__(self, monkeypatch):
        self.figs: list = []
        real_save = plots_module._save

        def spy(fig, path):
            self.figs.append(fig)
            return real_save(fig, path)

        monkeypatch.setattr(plots_module, "_save", spy)

    @property
    def last_axes(self):
        return self.figs[-1].axes


def _all_text(ax) -> str:
    """Every piece of text matplotlib actually holds for this axes --
    title, any note annotations, x/y labels -- concatenated, so a single
    ``in`` check can look for a forbidden word regardless of which of
    those it might have ended up in.
    """
    parts = [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
    parts += [t.get_text() for t in ax.texts]
    if ax.get_legend() is not None:
        parts += [t.get_text() for t in ax.get_legend().get_texts()]
    return " ".join(parts)


# --- finish_probability_by_remaining_targets -------------------------------


def _remaining_targets_decisions(counts: dict[int, int]) -> pd.DataFrame:
    rows = []
    for remaining, n in counts.items():
        for i in range(n):
            rows.append(
                {
                    "sensitive_targets_remaining_before_action": remaining,
                    "base_finish_probability": 0.1 + 0.01 * i,
                }
            )
    return pd.DataFrame(rows)


def test_finish_probability_by_remaining_targets_has_short_title(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    decisions = _remaining_targets_decisions({0: 10, 1: 10, 2: 10})
    written = _plot_finish_probability_by_remaining_targets(decisions, _tmp_dir(tmp_path))
    assert written

    ax = capture.last_axes[0]
    title = ax.get_title()
    assert title == "FINISH probability vs. remaining targets"
    assert len(title) < 50, "main title should be short -- caveats belong in the note, not the title"


def test_finish_probability_by_remaining_targets_never_says_bucket(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    decisions = _remaining_targets_decisions({0: 10, 1: 10})
    _plot_finish_probability_by_remaining_targets(decisions, _tmp_dir(tmp_path))
    ax = capture.last_axes[0]
    assert "bucket" not in _all_text(ax).lower()


def test_finish_probability_by_remaining_targets_note_explains_filter_and_diagnostic_status(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    decisions = _remaining_targets_decisions({0: 10, 1: 10})
    _plot_finish_probability_by_remaining_targets(decisions, _tmp_dir(tmp_path))
    ax = capture.last_axes[0]
    note = " ".join(t.get_text() for t in ax.texts)
    assert "diagnostic" in note.lower()
    assert str(_MIN_DECISIONS_PER_REMAINING_TARGET_COUNT) in note
    assert "decisions are shown" in note.lower()


def test_finish_probability_by_remaining_targets_xlabel_has_no_bucket_jargon(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    decisions = _remaining_targets_decisions({0: 10})
    _plot_finish_probability_by_remaining_targets(decisions, _tmp_dir(tmp_path))
    ax = capture.last_axes[0]
    assert ax.get_xlabel() == "Sensitive targets remaining without ROOT"


def test_finish_probability_by_remaining_targets_below_threshold_is_dropped():
    decisions = _remaining_targets_decisions({0: _MIN_DECISIONS_PER_REMAINING_TARGET_COUNT - 1})
    assert _plot_finish_probability_by_remaining_targets(decisions, None) == []  # type: ignore[arg-type]


def test_finish_probability_by_remaining_targets_single_category_still_intentional(tmp_path, monkeypatch):
    """A single kept remaining-target count must not look like a
    truncated/blank plot -- widened xlim around the one point."""
    capture = _FigureCapture(monkeypatch)
    decisions = _remaining_targets_decisions({3: _MIN_DECISIONS_PER_REMAINING_TARGET_COUNT})
    written = _plot_finish_probability_by_remaining_targets(decisions, _tmp_dir(tmp_path))
    assert written
    ax = capture.last_axes[0]
    xlim = ax.get_xlim()
    assert xlim[0] < 3 < xlim[1]
    assert (xlim[1] - xlim[0]) >= 2


# --- reward_vs_credit_assignment -------------------------------------------


def test_reward_vs_credit_assignment_legend_never_says_bare_other(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    decisions = pd.DataFrame(
        {
            "training_reward": [-1.0, -1.0, 1.0, 0.5, -0.2],
            "gae_advantage": [2.0, -0.5, 1.0, -1.0, 0.3],
        }
    )
    written = _plot_reward_vs_credit_assignment(decisions, _tmp_dir(tmp_path))
    assert written
    ax = capture.last_axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "other" not in [label.lower() for label in labels]
    assert "All other decisions" in labels
    assert "Negative reward, positive advantage" in labels


def test_reward_vs_credit_assignment_has_short_title_and_explanatory_note(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    decisions = pd.DataFrame({"training_reward": [-1.0, 1.0], "gae_advantage": [2.0, 1.0]})
    _plot_reward_vs_credit_assignment(decisions, _tmp_dir(tmp_path))
    ax = capture.last_axes[0]
    assert ax.get_title() == "Reward vs. GAE advantage"
    note = " ".join(t.get_text() for t in ax.texts)
    assert "one training decision" in note.lower()


# --- episode_efficiency's "steps to objective" panel -----------------------


def _episodes_df(n_rollouts: int = 10, with_objective_reached: bool = True) -> pd.DataFrame:
    rows = []
    for rollout in range(1, n_rollouts + 1):
        for is_eval in (False,):
            rows.append(
                {
                    "rollout": rollout,
                    "is_eval": is_eval,
                    "goal_success": rollout % 3 == 0,
                    "environment_steps": 10 + rollout,
                    "steps_to_goal": float(5 + rollout) if rollout % 3 == 0 else None,
                    **({"objective_reached": rollout % 2 == 0} if with_objective_reached else {}),
                }
            )
    return pd.DataFrame(rows)


def test_steps_to_objective_panel_has_short_title(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    written = _plot_episode_efficiency(_episodes_df(), _tmp_dir(tmp_path))
    assert written
    ax3 = capture.last_axes[2]
    title = ax3.get_title()
    assert title == "Steps to objective"
    assert len(title) < 30


def test_steps_to_objective_panel_note_mentions_finish_not_required(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    _plot_episode_efficiency(_episodes_df(), _tmp_dir(tmp_path))
    ax3 = capture.last_axes[2]
    note = " ".join(t.get_text() for t in ax3.texts)
    assert "reached the objective" in note.lower()
    assert "finish is not required" in note.lower()


def test_episode_length_panel_title_is_short(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    _plot_episode_efficiency(_episodes_df(), _tmp_dir(tmp_path))
    ax2 = capture.last_axes[1]
    assert ax2.get_title() == "Episode length"


def test_episode_efficiency_panel_titles_never_overflow_reasonable_length(tmp_path, monkeypatch):
    """Every panel's own main title (not the note) must stay short -- the
    actual visual-overlap bug this whole file exists to guard against was
    caused by cramming the caveats into the title itself."""
    capture = _FigureCapture(monkeypatch)
    _plot_episode_efficiency(_episodes_df(), _tmp_dir(tmp_path))
    for ax in capture.last_axes:
        assert len(ax.get_title()) <= 45, ax.get_title()


def test_steps_to_objective_panel_shows_insufficient_panel_when_never_reached(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    episodes = pd.DataFrame(
        [
            {"rollout": 1, "is_eval": False, "goal_success": False, "environment_steps": 10, "steps_to_goal": None},
            {"rollout": 2, "is_eval": False, "goal_success": False, "environment_steps": 12, "steps_to_goal": None},
        ]
    )
    written = _plot_episode_efficiency(episodes, _tmp_dir(tmp_path))
    assert written
    ax3 = capture.last_axes[2]
    note_text = " ".join(t.get_text() for t in ax3.texts)
    assert "never reached" in note_text.lower()
    assert ax3.get_legend() is None


# --- resource plots: missing/insufficient-data behavior --------------------


def _resources_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [1000.0 + i for i in range(n)],
            "phase": ["rollout_collection"] * n,
            "cpu_process_pct": [50.0 + i for i in range(n)],
            "cpu_system_pct": [10.0 + i for i in range(n)],
            "ram_rss_mb": [500.0 + i for i in range(n)],
            "ram_system_used_mb": [2000.0 + i for i in range(n)],
            "torch_cuda_allocated_mb": [100.0 + i for i in range(n)],
            "torch_cuda_reserved_mb": [150.0 + i for i in range(n)],
        }
    )


@pytest.mark.parametrize("n", [0, 1, _MIN_RESOURCE_SAMPLES_FOR_TIMESERIES - 1])
def test_resource_ram_plot_shows_insufficient_panel_below_threshold(tmp_path, monkeypatch, n):
    """n=0 (the column exists but every value is null -- e.g. RAM
    telemetry present but RSS itself unavailable) still gets an honest
    "insufficient" panel, consistent with n=1/n=2: a present-but-empty
    column is a real (if unusual) case, not the "column doesn't exist at
    all" case, which is the ONLY one that skips the file entirely (see
    the column-existence check at the top of _plot_resource_ram)."""
    capture = _FigureCapture(monkeypatch)
    resources = _resources_df(n) if n > 0 else pd.DataFrame(columns=_resources_df(1).columns)
    written = _plot_resource_ram(resources, _tmp_dir(tmp_path))
    assert written
    ax = capture.last_axes[0]
    assert not ax.lines, "no line should be drawn when samples are insufficient"
    note_text = " ".join(t.get_text() for t in ax.texts)
    assert "insufficient" in note_text.lower()
    assert f"n={n}" in note_text


def test_resource_ram_plot_renders_real_series_at_threshold(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    resources = _resources_df(_MIN_RESOURCE_SAMPLES_FOR_TIMESERIES)
    written = _plot_resource_ram(resources, _tmp_dir(tmp_path))
    assert written
    ax = capture.last_axes[0]
    assert len(ax.lines) >= 1
    assert not any("insufficient" in t.get_text().lower() for t in ax.texts)
    assert ax.get_ylabel() == "RAM (MB)"


def test_resource_ram_plot_renders_with_many_points(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    resources = _resources_df(500)
    written = _plot_resource_ram(resources, _tmp_dir(tmp_path))
    assert written
    ax = capture.last_axes[0]
    assert len(ax.lines[0].get_xdata()) == 500


def test_resource_cpu_plot_process_pct_ylabel_says_can_exceed_100(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    resources = _resources_df(_MIN_RESOURCE_SAMPLES_FOR_TIMESERIES)
    _plot_resource_cpu(resources, _tmp_dir(tmp_path))
    ax1, ax2 = capture.last_axes
    assert "can exceed 100%" in ax1.get_ylabel()
    assert ax1.get_ylim() != (0, 105)  # process CPU is never clamped
    assert ax2.get_ylim() == (0.0, 105.0)  # system CPU is clamped to [0, 105]


def test_resource_cpu_plot_insufficient_data_both_panels_say_so(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    resources = _resources_df(1)
    written = _plot_resource_cpu(resources, _tmp_dir(tmp_path))
    assert written
    for ax in capture.last_axes:
        note_text = " ".join(t.get_text() for t in ax.texts)
        assert "insufficient" in note_text.lower()
