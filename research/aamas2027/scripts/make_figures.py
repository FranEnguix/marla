#!/usr/bin/env python
"""Generates the 4 required paper figures (PDF+PNG) from research/aamas2027/aggregate/*.csv.

Run analyze.py first. Figures degrade gracefully (skip + warn) when their
source aggregate file is empty or missing seeds -- never fabricates a band
from fewer seeds than actually ran.

Scope (manifest.yaml v3): no ALWAYS_QUERY/PLAN_MAKER_ONLY/BETA_ZERO/
BETA_ONE reference point anywhere -- every figure compares only PPO_ONLY,
MARLA_FULL (as trained, i.e. MARLA_FULL_NORMAL at eval time), and
MARLA_FULL_NO_QUERY, per the resource-constrained revision.

Color: one fixed hue per condition, reused identically across every figure
-- taken from a validated categorical order (this program's dataviz skill
reference palette). Two measures with different units are always drawn as
separate panels, never a dual-axis overlay.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from analyze import bootstrap_ci  # noqa: E402

# Fixed condition -> color, reused across every figure in this script.
COLORS = {
    "PPO_ONLY": "#2a78d6",              # slot 1, blue
    "MARLA_FULL": "#eb6834",            # slot 2, orange
    "MARLA_FULL_NORMAL": "#eb6834",     # same identity as MARLA_FULL's own trained behavior
    "MARLA_FULL_NO_QUERY": "#eda100",   # slot 4, yellow
}
GRID_KW = dict(color="#888888", alpha=0.25, linewidth=0.6)


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_figures] wrote {name}.pdf / {name}.png")


def _style(ax) -> None:
    ax.grid(True, **GRID_KW)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# --- Figure 1: learning curves --------------------------------------------


def figure_1_learning_curve(agg_dir: Path, out_dir: Path) -> None:
    rows = read_csv(agg_dir / "learning_curve_bands.csv")
    if not rows:
        print("[figure_1] learning_curve_bands.csv is empty -- skipped")
        return

    fig, (ax_success, ax_return) = plt.subplots(1, 2, figsize=(9, 3.6))
    by_condition: dict[str, list[dict]] = {}
    for r in rows:
        by_condition.setdefault(r["condition"], []).append(r)

    for condition, points in by_condition.items():
        points.sort(key=lambda r: int(r["checkpoint_environment_steps"]))
        steps = [int(p["checkpoint_environment_steps"]) for p in points]
        color = COLORS.get(condition, "#888888")
        n_seeds = int(points[0]["num_seeds"]) if points else 0

        s_point = [float(p["goal_success_point"]) for p in points]
        s_lo = [float(p["goal_success_ci_low"]) for p in points]
        s_hi = [float(p["goal_success_ci_high"]) for p in points]
        ax_success.plot(steps, s_point, color=color, marker=".", label=f"{condition} (n={n_seeds})")
        ax_success.fill_between(steps, s_lo, s_hi, color=color, alpha=0.15, linewidth=0)

        r_point = [float(p["benchmark_return_point"]) for p in points]
        r_lo = [float(p["benchmark_return_ci_low"]) for p in points]
        r_hi = [float(p["benchmark_return_ci_high"]) for p in points]
        ax_return.plot(steps, r_point, color=color, marker=".", label=condition)
        ax_return.fill_between(steps, r_lo, r_hi, color=color, alpha=0.15, linewidth=0)

    for ax, title, ylabel in (
        (ax_success, "A. Deterministic goal success", "goal success rate"),
        (ax_return, "B. Deterministic benchmark return", "mean benchmark return"),
    ):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training environment steps")
        ax.set_ylabel(ylabel)
        _style(ax)
    ax_success.set_ylim(-0.02, 1.02)
    ax_success.legend(frameon=False, fontsize=8)
    fig.suptitle("Figure 1 -- Learning: PPO_ONLY vs MARLA_FULL, 95% bootstrap bands across training seeds", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "figure1_learning_curve")


# --- Figure 2: final ID + OOD performance ---------------------------------


def figure_2_final_performance(agg_dir: Path, out_dir: Path) -> None:
    rows = read_csv(agg_dir / "id_and_ood_eval.csv")
    if not rows:
        print("[figure_2] id_and_ood_eval.csv is empty -- skipped (run the evaluation harness + analyze.py first)")
        return

    conditions = [c for c in ("PPO_ONLY", "MARLA_FULL_NORMAL", "MARLA_FULL_NO_QUERY") if any(r["condition"] == c for r in rows)]
    scenarios = [s for s in ("ID", "OOD_dmz_three_subnets", "OOD_user_three_subnets") if any(r["scenario"] == s for r in rows)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    metrics = [
        ("goal_success_rate", "goal success rate", axes[0]),
        ("mean_benchmark_return", "mean benchmark return", axes[1]),
        ("mean_steps_to_goal", "mean steps to goal (successes only)", axes[2]),
    ]
    width = 0.8 / max(len(conditions), 1)
    x = list(range(len(scenarios)))
    for metric_key, ylabel, ax in metrics:
        for i, condition in enumerate(conditions):
            heights, errs_lo, errs_hi = [], [], []
            for scenario in scenarios:
                values = [float(r[metric_key]) for r in rows if r["condition"] == condition and r["scenario"] == scenario and r.get(metric_key)]
                if not values:
                    heights.append(0.0)
                    errs_lo.append(0.0)
                    errs_hi.append(0.0)
                    continue
                point, lo, hi = bootstrap_ci(values) if len(values) > 1 else (values[0], values[0], values[0])
                heights.append(point)
                errs_lo.append(point - lo)
                errs_hi.append(hi - point)
            offset = (i - (len(conditions) - 1) / 2) * width
            positions = [xi + offset for xi in x]
            ax.bar(positions, heights, width=width, color=COLORS.get(condition, "#888888"), label=condition)
            ax.errorbar(positions, heights, yerr=[errs_lo, errs_hi], fmt="none", ecolor="#333333", elinewidth=1, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=7, rotation=15)
        ax.set_ylabel(ylabel)
        _style(ax)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Figure 2 -- Final ID/OOD performance: PPO_ONLY vs MARLA_FULL vs MARLA_FULL_NO_QUERY", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "figure2_final_id_ood_performance")


# --- Figure 3: consultation / query-gate behavior -------------------------


def figure_3_consultation_behavior(agg_dir: Path, out_dir: Path) -> None:
    rows = read_csv(agg_dir / "consultation_behavior.csv")
    rate_rows = read_csv(agg_dir / "query_rate_over_training.csv")
    if not rows and not rate_rows:
        print("[figure_3] consultation_behavior.csv and query_rate_over_training.csv both empty -- skipped")
        return

    fig, (ax_hist, ax_scatter, ax_rate) = plt.subplots(1, 3, figsize=(13, 3.6))

    if rows:
        queried_h = [float(r["normalized_base_policy_entropy"]) for r in rows if r["queried"] == "True" and r["normalized_base_policy_entropy"]]
        not_queried_h = [float(r["normalized_base_policy_entropy"]) for r in rows if r["queried"] == "False" and r["normalized_base_policy_entropy"]]
        bins = [i / 20 for i in range(21)]
        ax_hist.hist(not_queried_h, bins=bins, color=COLORS["PPO_ONLY"], alpha=0.6, density=True, label=f"not queried (n={len(not_queried_h)})")
        ax_hist.hist(queried_h, bins=bins, color=COLORS["MARLA_FULL"], alpha=0.6, density=True, label=f"queried (n={len(queried_h)})")
        ax_hist.legend(frameon=False, fontsize=7)

        xs = [float(r["normalized_base_policy_entropy"]) for r in rows if r["normalized_base_policy_entropy"]]
        ys = [float(r["query_probability"]) for r in rows if r["normalized_base_policy_entropy"]]
        ax_scatter.scatter(xs, ys, s=4, alpha=0.25, color=COLORS["MARLA_FULL"])
    ax_hist.set_xlabel("normalized base-policy entropy H/log(N)")
    ax_hist.set_ylabel("density")
    ax_hist.set_title("A. Entropy: queried vs not", fontsize=10)
    ax_scatter.set_xlabel("normalized base-policy entropy H/log(N)")
    ax_scatter.set_ylabel("query probability")
    ax_scatter.set_title("B. Query probability vs entropy", fontsize=10)

    if rate_rows:
        by_seed: dict[str, list[dict]] = {}
        for r in rate_rows:
            by_seed.setdefault(r["training_seed"], []).append(r)
        for seed, points in by_seed.items():
            points.sort(key=lambda r: int(r["environment_steps"]))
            ax_rate.plot(
                [int(p["environment_steps"]) for p in points],
                [float(p["actual_query_rate"]) for p in points],
                marker=".", color=COLORS["MARLA_FULL"], alpha=0.6, label=f"seed {seed}",
            )
        ax_rate.legend(frameon=False, fontsize=7)
    ax_rate.set_xlabel("training environment steps")
    ax_rate.set_ylabel("actual query rate (per rollout)")
    ax_rate.set_ylim(0, 1.02)
    ax_rate.set_title("C. Query rate over training", fontsize=10)

    for ax in (ax_hist, ax_scatter, ax_rate):
        _style(ax)
    fig.suptitle("Figure 3 -- Query-gate/consultation behavior (MARLA_FULL training runs)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "figure3_consultation_behavior")


# --- Figure 4: advice / trust behavior ------------------------------------


def figure_4_advice_trust_behavior(agg_dir: Path, out_dir: Path) -> None:
    beta_training_rows = read_csv(agg_dir / "beta_over_training.csv")
    decision_rows = read_csv(agg_dir / "advice_trust_decisions.csv")
    episode_rows = read_csv(agg_dir / "consultations_per_successful_episode.csv")
    if not beta_training_rows and not decision_rows:
        print("[figure_4] beta_over_training.csv and advice_trust_decisions.csv both empty -- skipped")
        return

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    ax_beta_training, ax_beta_dist, ax_agreement, ax_consultations = axes

    if beta_training_rows:
        by_seed: dict[str, list[dict]] = {}
        for r in beta_training_rows:
            by_seed.setdefault(r["training_seed"], []).append(r)
        for seed, points in by_seed.items():
            points.sort(key=lambda r: int(r["environment_steps"]))
            ax_beta_training.plot(
                [int(p["environment_steps"]) for p in points], [float(p["mean_beta"]) for p in points],
                marker=".", color=COLORS["MARLA_FULL"], alpha=0.6, label=f"seed {seed}",
            )
        ax_beta_training.legend(frameon=False, fontsize=7)
    ax_beta_training.set_xlabel("training environment steps")
    ax_beta_training.set_ylabel("mean beta (per rollout)")
    ax_beta_training.set_title("A. Trust (beta) over training", fontsize=9)

    if decision_rows:
        betas = [float(r["beta"]) for r in decision_rows if r.get("beta")]
        ax_beta_dist.hist(betas, bins=20, color=COLORS["MARLA_FULL"], alpha=0.7)
        agreements = [float(r["base_plan_maker_top1_agreement"]) for r in decision_rows if r.get("base_plan_maker_top1_agreement")]
        changed = [1.0 if r["advice_changed_top_action"] == "True" else 0.0 for r in decision_rows if r.get("advice_changed_top_action") in ("True", "False")]
        agreement_rate = sum(agreements) / len(agreements) if agreements else None
        changed_rate = sum(changed) / len(changed) if changed else None
        labels, heights = [], []
        if agreement_rate is not None:
            labels.append(f"PPO-PM\ntop-1 agree\n(n={len(agreements)})")
            heights.append(agreement_rate)
        if changed_rate is not None:
            labels.append(f"advice changed\ntop action\n(n={len(changed)})")
            heights.append(changed_rate)
        ax_agreement.bar(range(len(heights)), heights, color=COLORS["MARLA_FULL"])
        ax_agreement.set_xticks(range(len(labels)))
        ax_agreement.set_xticklabels(labels, fontsize=7)
        ax_agreement.set_ylim(0, 1.05)
    ax_beta_dist.set_xlabel("beta (accepted-advice decisions)")
    ax_beta_dist.set_ylabel("count")
    ax_beta_dist.set_title("B. Beta distribution", fontsize=9)
    ax_agreement.set_title("C. Agreement / advice influence", fontsize=9)

    if episode_rows:
        seeds = [r["training_seed"] for r in episode_rows]
        values = [float(r["mean_consultations_per_successful_episode"]) for r in episode_rows]
        ax_consultations.bar(range(len(seeds)), values, color=COLORS["MARLA_FULL"])
        ax_consultations.set_xticks(range(len(seeds)))
        ax_consultations.set_xticklabels([f"seed {s}" for s in seeds], fontsize=7)
    ax_consultations.set_ylabel("consultations / successful episode")
    ax_consultations.set_title("D. Consultation cost of success", fontsize=9)

    for ax in axes:
        _style(ax)
    fig.suptitle("Figure 4 -- Advice/trust behavior (MARLA_FULL, already-collected queries only)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "figure4_advice_trust_behavior")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agg-dir", type=Path, default=HERE / "aggregate")
    parser.add_argument("--out-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()

    figure_1_learning_curve(args.agg_dir, args.out_dir)
    figure_2_final_performance(args.agg_dir, args.out_dir)
    figure_3_consultation_behavior(args.agg_dir, args.out_dir)
    figure_4_advice_trust_behavior(args.agg_dir, args.out_dir)


if __name__ == "__main__":
    main()
