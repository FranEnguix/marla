"""Plot data-aggregation logic (spec section 8) -- unit-testing the pure
pandas filtering/aggregation a plot is built from, not just "did a PNG get
written" (a PNG can be produced from wrong data just as easily as right
data).
"""

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import marla.metrics.plots as plots_module
from marla.metrics.plots import (
    _accepted_advice_rows,
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

    ax = capture.last_axes[0]
    # Real Text objects the function actually set -- $...$ must be present
    # (mathtext delimiters), not the literal "V_t"/"R_t" the bug report
    # named. get_xlabel()/get_ylabel()/get_title() return the raw string
    # as set (mathtext isn't resolved until draw time), so this checks the
    # source string directly while still exercising the real call path.
    assert "$V_t$" in ax.get_ylabel()
    assert "$R_t$" in ax.get_xlabel()
    assert "$V_t$" in ax.get_title() and "$R_t$" in ax.get_title()
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
