"""Plot data-aggregation logic (spec section 8) -- unit-testing the pure
pandas filtering/aggregation a plot is built from, not just "did a PNG get
written" (a PNG can be produced from wrong data just as easily as right
data).
"""

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import numpy as np

import marla.metrics.plots as plots_module
from marla.metrics.plots import (
    _accepted_advice_rows,
    _critic_quality_r_squared,
    _plot_advice_influence,
    _plot_critic_quality,
    _plot_decision_diagnostics,
)


def _decisions_row(**overrides):
    row = {
        "queried": False,
        "response_status": None,
        "beta": None,
        "advice_changed_top_action": None,
        "base_policy_entropy": 1.0,
        "final_policy_entropy": 1.0,
        "selected_action_base_probability": 0.5,
        "selected_action_final_probability": 0.5,
    }
    row.update(overrides)
    return row


def test_accepted_advice_rows_excludes_rejected_queries():
    """Regression: a schema-rejected query has a real, non-null beta
    (forced to exactly 0.0, see learning/decision.py) and a deterministic
    advice_changed_top_action=False (final_logits == base_logits by
    construction) -- both previously slipped through a `beta.notna()` /
    `queried == True` filter and were mislabeled as "accepted advice" in
    advice_influence.png / assisted_policy_influence.png, diluting the
    trust-weight and top-action-change signal toward whatever the schema
    rejection rate happened to be.
    """
    decisions = pd.DataFrame(
        [
            _decisions_row(queried=False),  # not queried at all
            _decisions_row(queried=True, response_status="schema_rejected", beta=0.0, advice_changed_top_action=False),
            _decisions_row(queried=True, response_status="accepted", beta=0.7, advice_changed_top_action=True),
            _decisions_row(queried=True, response_status="accepted", beta=0.3, advice_changed_top_action=False),
        ]
    )

    accepted = _accepted_advice_rows(decisions)

    assert len(accepted) == 2
    assert set(accepted["response_status"]) == {"accepted"}
    assert sorted(accepted["beta"]) == [0.3, 0.7]


def test_accepted_advice_rows_empty_for_baseline_variant():
    decisions = pd.DataFrame([_decisions_row(), _decisions_row()])  # never queried, no response_status column values set
    assert _accepted_advice_rows(decisions).empty


def test_accepted_advice_rows_tolerates_missing_response_status_column():
    # An older decisions.csv schema, or any DataFrame missing the column
    # entirely, must not raise -- just report nothing to plot.
    decisions = pd.DataFrame([{"queried": True, "beta": 0.5}])
    assert _accepted_advice_rows(decisions).empty


# --- Mathtext labels: subscripts/Greek letters must actually render, not ---
# --- show up as literal underscore text (e.g. "V_t" instead of V with a  ---
# --- subscript t) -- both the correct raw-string mathtext markup and a   ---
# --- real (headless) render pass, since a malformed mathtext expression  ---
# --- raises at *render* time, not at label-assignment time.              ---


def _tmp_dir(tmp_path):
    d = tmp_path / "plots"
    d.mkdir()
    return d


class _FigureCapture:
    """Captures the last figure/axes passed through ``plots._save`` before
    it gets closed, so a test can inspect what a plotting function actually
    set on real Matplotlib Text objects (``get_title()``/``get_xlabel()``/
    ...) -- not a second, independently-typed copy of the expected string,
    which would keep passing even if the production code regressed.
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


def test_critic_quality_uses_mathtext_subscripts_and_renders(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    decisions = pd.DataFrame(
        {"critic_value": [0.1, 0.5, -0.2, 1.0], "return_target": [0.2, 0.4, -0.1, 0.9]}
    )
    written = _plot_critic_quality(decisions, _tmp_dir(tmp_path))
    assert written and written[0].is_file() and written[0].stat().st_size > 0

    fig = capture.figs[-1]
    ax = capture.last_axes[0]
    # Real Text objects the function actually set -- $...$ must be present
    # (mathtext delimiters), not the literal "V_t"/"R_t" the bug report
    # named. get_xlabel()/get_ylabel()/the suptitle return the raw string
    # as set (mathtext isn't resolved until draw time), so this checks the
    # source string directly while still exercising the real call path.
    assert "$V_t$" in ax.get_ylabel()
    assert "$R_t$" in ax.get_xlabel()
    suptitle_text = fig._suptitle.get_text()
    assert "$V_t$" in suptitle_text and "$R_t$" in suptitle_text
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("$V_t = R_t$" in label for label in legend_labels)


def test_normalized_entropy_label_uses_mathtext(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    decisions = pd.DataFrame(
        {
            "base_policy_entropy": [0.5, 0.6, 0.4],
            "base_top_two_margin": [0.1, 0.2, 0.05],
            "legal_action_count": [3, 4, 5],
        }
    )
    written = _plot_decision_diagnostics(decisions, _tmp_dir(tmp_path))
    assert written and written[0].is_file()

    _ax1, ax2 = capture.last_axes[:2]
    assert ax2.get_ylabel() == r"Normalized entropy $H / \log N$"


def test_advice_influence_beta_label_uses_mathtext_and_renders(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    decisions = pd.DataFrame(
        {
            "response_status": ["accepted", "accepted", "schema_rejected"],
            "beta": [0.3, 0.7, 0.0],
            "advice_changed_top_action": [True, False, False],
        }
    )
    written = _plot_advice_influence(decisions, _tmp_dir(tmp_path))
    assert written and written[0].is_file()

    ax1 = capture.last_axes[0]
    assert "$\\beta$" in ax1.get_ylabel()
    assert "$\\beta$" in ax1.get_title()


# --- critic_quality.png: numerical correctness, density-aware rendering, ---
# --- and honest full-range/central-range handling of wide-tailed R_t.    ---


def test_critic_quality_r_squared_matches_manual_calculation():
    r = pd.Series([1.0, 2.0, 3.0, 4.0])
    v = pd.Series([1.1, 1.9, 3.2, 3.8])
    expected = 1.0 - ((r - v) ** 2).sum() / ((r - r.mean()) ** 2).sum()
    assert _critic_quality_r_squared(r, v) == expected


def test_critic_quality_r_squared_is_none_for_zero_variance_return_target():
    # R^2 is undefined (0/0), not 0.0, when every R_t is identical.
    r = pd.Series([2.0, 2.0, 2.0])
    v = pd.Series([1.0, 2.0, 3.0])
    assert _critic_quality_r_squared(r, v) is None


def test_critic_quality_suptitle_reports_correct_mae_rmse_bias_and_r2(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    v = pd.Series([0.0, 1.0, 2.0, 10.0])
    r = pd.Series([1.0, 1.0, 1.0, 1.0])
    decisions = pd.DataFrame({"critic_value": v, "return_target": r})
    _plot_critic_quality(decisions, _tmp_dir(tmp_path))

    err = v - r
    expected_mae = err.abs().mean()
    expected_rmse = float(np.sqrt((err**2).mean()))
    expected_bias = err.mean()
    suptitle = capture.figs[-1]._suptitle.get_text()
    assert f"n=4 decisions" in suptitle
    assert f"MAE={expected_mae:.3f}" in suptitle
    assert f"RMSE={expected_rmse:.3f}" in suptitle
    assert f"bias(V-R)={expected_bias:+.3f}" in suptitle
    # R_t has zero variance here -- R^2 must read "n/a", never a fabricated
    # number (see test_critic_quality_r_squared_is_none_for_zero_variance_return_target).
    assert "R^2=n/a" in suptitle


def test_critic_quality_title_clarifies_return_target_is_not_ground_truth(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    decisions = pd.DataFrame({"critic_value": [0.1, 0.2, 0.3], "return_target": [1.0, 2.0, 3.0]})
    _plot_critic_quality(decisions, _tmp_dir(tmp_path))
    suptitle = capture.figs[-1]._suptitle.get_text().lower()
    assert "gae" in suptitle
    assert "not ground truth" in suptitle


def test_critic_quality_returns_empty_for_no_paired_data(tmp_path):
    assert _plot_critic_quality(pd.DataFrame({"critic_value": [], "return_target": []}), _tmp_dir(tmp_path)) == []


def test_critic_quality_returns_empty_when_columns_missing(tmp_path):
    assert _plot_critic_quality(pd.DataFrame({"other": [1, 2]}), _tmp_dir(tmp_path)) == []


def test_critic_quality_handles_single_decision(tmp_path, monkeypatch):
    # n=1: no variance, no percentile spread -- must render, not crash.
    capture = _FigureCapture(monkeypatch)
    decisions = pd.DataFrame({"critic_value": [0.4], "return_target": [1.0]})
    written = _plot_critic_quality(decisions, _tmp_dir(tmp_path))
    assert written and written[0].is_file()
    assert "n=1 decisions" in capture.figs[-1]._suptitle.get_text()


def test_critic_quality_uses_scatter_below_hexbin_threshold(tmp_path, monkeypatch):
    # Fewer than 20 points: a hexbin over a handful of points would draw a
    # few large, mostly-empty hexagons -- the fallback must be a plain
    # scatter (no colorbar) instead.
    capture = _FigureCapture(monkeypatch)
    decisions = pd.DataFrame({"critic_value": list(range(10)), "return_target": list(range(10))})
    _plot_critic_quality(decisions, _tmp_dir(tmp_path))
    fig = capture.figs[-1]
    assert len(fig.axes) == 2  # no colorbar axes appended (hexbin's mappable would add one per panel)
    ax_full = fig.axes[0]
    assert len(ax_full.collections) == 1  # exactly the scatter's own PathCollection, no hexbin PolyCollection
    assert type(ax_full.collections[0]).__name__ == "PathCollection"
    assert len(ax_full.lines) >= 1  # the V_t = R_t identity line


def test_critic_quality_uses_hexbin_at_or_above_threshold(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    decisions = pd.DataFrame({"critic_value": list(range(25)), "return_target": list(range(25))})
    _plot_critic_quality(decisions, _tmp_dir(tmp_path))
    fig = capture.figs[-1]
    # A colorbar axes is appended per hexbin panel -> more than the base 2 axes.
    assert len(fig.axes) == 4


def test_critic_quality_zoom_panel_reports_correct_outlier_count(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    # 100 tightly-clustered points plus 2 extreme outliers far outside the
    # 5th-95th percentile window.
    clustered = list(np.linspace(0.0, 1.0, 100))
    r_values = clustered + [50.0, -50.0]
    v_values = clustered + [0.5, 0.5]
    decisions = pd.DataFrame({"critic_value": v_values, "return_target": r_values})
    _plot_critic_quality(decisions, _tmp_dir(tmp_path))
    # Read whichever axes carries the zoom panel's own title -- data axes
    # only (colorbar axes, if any, never carry a title).
    titles = [a.get_title() for a in capture.figs[-1].axes if a.get_title()]
    assert any("2 decision(s) outside this view" in t for t in titles)


def test_critic_quality_full_range_axis_limits_cover_both_series_extremes(tmp_path, monkeypatch):
    capture = _FigureCapture(monkeypatch)
    decisions = pd.DataFrame({"critic_value": [-5.0, 0.0, 3.0], "return_target": [0.0, 1.0, 20.0]})
    _plot_critic_quality(decisions, _tmp_dir(tmp_path))
    ax_full = capture.figs[-1].axes[0]
    xlo, xhi = ax_full.get_xlim()
    ylo, yhi = ax_full.get_ylim()
    # Full-range panel must not clip either series' true extremes.
    assert xlo <= -5.0 and xhi >= 20.0
    assert ylo <= -5.0 and yhi >= 20.0
