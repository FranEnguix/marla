#!/usr/bin/env python
"""Generates the 4 required paper figures (PDF+PNG) from research/aamas2027/aggregate/*.csv.

Run analyze.py first. Figures degrade gracefully (skip + warn) when their
source aggregate file is empty or missing seeds -- never fabricates a band
from fewer seeds than actually ran.

Color: one fixed hue per condition, reused identically across every figure
(never re-cycled per-panel) -- taken from a validated categorical order
(this program's dataviz skill reference palette) so adjacent series stay
distinguishable under common color-vision deficiencies. Two measures with
different units (success rate, return) are always drawn as separate panels,
never a dual-axis overlay.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent

# Fixed condition -> color, reused across every figure in this script.
COLORS = {
    "PPO_ONLY": "#2a78d6",              # slot 1, blue
    "MARLA_FULL": "#eb6834",            # slot 2, orange
    "MARLA_FULL_NORMAL": "#eb6834",     # same identity as MARLA_FULL's own trained behavior
    "MARLA_FULL_ALWAYS_QUERY": "#1baf7a",  # slot 3, aqua
    "MARLA_FULL_NO_QUERY": "#eda100",      # slot 4, yellow
    "PLAN_MAKER_ONLY": "#e87ba4",          # slot 5, magenta
    "MARLA_FULL_BETA_ZERO": "#4a3aa7",     # slot 7, violet
    "MARLA_FULL_BETA_ONE": "#e34948",      # slot 8, red
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


def figure_1_learning_curve(agg_dir: Path, out_dir: Path) -> None:
    rows = read_csv(agg_dir / "learning_curve_bands.csv")
    if not rows:
        print("[figure_1] learning_curve_bands.csv is empty -- skipped (run analyze.py after training runs exist)")
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
        ax.grid(True, **GRID_KW)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    ax_success.set_ylim(-0.02, 1.02)
    ax_success.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Figure 1 -- Learning: PPO_ONLY vs MARLA_FULL, 95% bootstrap bands across training seeds",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, out_dir, "figure1_learning_curve")


def figure_2_final_id_performance(agg_dir: Path, out_dir: Path) -> None:
    rows = read_csv(agg_dir / "final_id_eval.csv")
    if not rows:
        print("[figure_2] final_id_eval.csv is empty -- skipped (run the evaluation harness + analyze.py first)")
        return

    by_condition: dict[str, list[dict]] = {}
    for r in rows:
        by_condition.setdefault(r["condition"], []).append(r)

    order = [c for c in COLORS if c in by_condition and c != "MARLA_FULL"]  # MARLA_FULL itself has no eval-harness output; MARLA_FULL_NORMAL does
    fig, ax = plt.subplots(figsize=(7, 3.6))
    xs, heights, errs_lo, errs_hi, colors, labels = [], [], [], [], [], []
    for i, condition in enumerate(order):
        seed_rows = by_condition[condition]
        values = [float(r["goal_success_rate"]) for r in seed_rows]
        n = len(values)
        point = sum(values) / n
        if n > 1:
            import sys

            sys.path.insert(0, str(HERE))
            from analyze import bootstrap_ci

            point, lo, hi = bootstrap_ci(values)
        else:
            lo, hi = point, point
        xs.append(i)
        heights.append(point)
        errs_lo.append(point - lo)
        errs_hi.append(hi - point)
        colors.append(COLORS.get(condition, "#888888"))
        labels.append(f"{condition}\n(n={n})")

    ax.bar(xs, heights, color=colors, width=0.6)
    ax.errorbar(xs, heights, yerr=[errs_lo, errs_hi], fmt="none", ecolor="#333333", elinewidth=1, capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("goal success rate (final ID eval)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", **GRID_KW)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Figure 2 -- Final ID performance, 95% bootstrap CI across training seeds", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "figure2_final_id_performance")


def figure_3_ood_generalization(agg_dir: Path, out_dir: Path) -> None:
    rows = read_csv(agg_dir / "ood_eval.csv")
    if not rows:
        print("[figure_3] ood_eval.csv is empty -- skipped")
        return

    import sys

    sys.path.insert(0, str(HERE))
    from analyze import bootstrap_ci

    scenarios = sorted({r["scenario"] for r in rows}, key=lambda s: (s != "ID", s))
    conditions = sorted({r["condition"] for r in rows})

    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.35
    x = list(range(len(scenarios)))
    for i, condition in enumerate(conditions):
        heights, errs_lo, errs_hi = [], [], []
        for scenario in scenarios:
            values = [float(r["goal_success_rate"]) for r in rows if r["condition"] == condition and r["scenario"] == scenario]
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
    ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel("goal success rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", **GRID_KW)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Figure 3 -- ID vs OOD generalization, 95% bootstrap CI across training seeds", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "figure3_ood_generalization")


def figure_4_query_gate_behavior(agg_dir: Path, out_dir: Path) -> None:
    rows = read_csv(agg_dir / "consultation_behavior.csv")
    if not rows:
        print("[figure_4] consultation_behavior.csv is empty -- skipped")
        return

    queried_h = [float(r["normalized_base_policy_entropy"]) for r in rows if r["queried"] == "True" and r["normalized_base_policy_entropy"]]
    not_queried_h = [float(r["normalized_base_policy_entropy"]) for r in rows if r["queried"] == "False" and r["normalized_base_policy_entropy"]]

    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(9, 3.6))
    bins = [i / 20 for i in range(21)]
    ax_hist.hist(not_queried_h, bins=bins, color=COLORS["PPO_ONLY"], alpha=0.6, density=True, label=f"not queried (n={len(not_queried_h)})")
    ax_hist.hist(queried_h, bins=bins, color=COLORS["MARLA_FULL"], alpha=0.6, density=True, label=f"queried (n={len(queried_h)})")
    ax_hist.set_xlabel("normalized base-policy entropy H/log(N)")
    ax_hist.set_ylabel("density")
    ax_hist.legend(frameon=False, fontsize=8)
    ax_hist.set_title("A. Entropy: queried vs not", fontsize=10)

    xs = [float(r["normalized_base_policy_entropy"]) for r in rows if r["normalized_base_policy_entropy"]]
    ys = [float(r["query_probability"]) for r in rows if r["normalized_base_policy_entropy"]]
    ax_scatter.scatter(xs, ys, s=4, alpha=0.25, color=COLORS["MARLA_FULL"])
    ax_scatter.set_xlabel("normalized base-policy entropy H/log(N)")
    ax_scatter.set_ylabel("query probability")
    ax_scatter.set_title("B. Query probability vs entropy", fontsize=10)

    for ax in (ax_hist, ax_scatter):
        ax.grid(True, **GRID_KW)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Figure 4 -- Query-gate behavior (MARLA_FULL training runs)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "figure4_query_gate_behavior")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agg-dir", type=Path, default=HERE / "aggregate")
    parser.add_argument("--out-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()

    figure_1_learning_curve(args.agg_dir, args.out_dir)
    figure_2_final_id_performance(args.agg_dir, args.out_dir)
    figure_3_ood_generalization(args.agg_dir, args.out_dir)
    figure_4_query_gate_behavior(args.agg_dir, args.out_dir)


if __name__ == "__main__":
    main()
