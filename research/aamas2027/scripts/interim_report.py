#!/usr/bin/env python
"""Interim feasibility report for MARLA_FULL seed 101's Stage A training run
(research/aamas2027 v4). Two parts:

1. ~1k-step-binned MARLA_FULL training diagnostics, computed entirely from
   decisions.csv/updates.csv -- no new Plan Maker calls. decisions.csv rows
   are written in strict chronological training order, one row per real
   environment step (eval episodes never produce StepRecords -- see
   AUDIT.md), so ROW INDEX doubles as cumulative environment-step count;
   this is what the binning below uses, not a stored column (none exists).
2. PPO stability diagnostics per bin, from updates.csv, with automatic
   flagging of the catastrophic failure modes this report exists to check
   for (not "any noise" -- see THRESHOLDS below).

Usage:
    python research/aamas2027/scripts/interim_report.py \\
        --run-dir runs/aamas2027_marla_full/marla-full-seed-101 \\
        --bin-size 1024
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Deliberately conservative -- flags only the catastrophic patterns this
# report is asked to check for, not ordinary training noise.
THRESHOLDS = {
    "kl_catastrophic": 1.0,  # naive-estimator KL this large indicates the policy moved far outside the trust region
    "clip_fraction_catastrophic": 0.9,  # almost every sample clipped -- the step size is badly miscalibrated
    "grad_norm_catastrophic": 1e4,  # exploding gradients
    "query_rate_collapse_low": 0.01,  # essentially never queries after the first bin
    "query_rate_collapse_high": 0.99,  # essentially always queries (indistinguishable from ALWAYS_QUERY)
    "advice_influence_negligible": 0.02,  # advice_changed_top_action rate this low across the WHOLE run suggests consultation never affects decisions
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str) -> float | None:
    v = row.get(key)
    return None if v in (None, "") else float(v)


def _b(row: dict, key: str) -> bool | None:
    v = row.get(key)
    return None if v in (None, "") else v == "True"


def bin_decisions(decisions: list[dict], bin_size: int) -> list[dict]:
    bins: dict[int, list[dict]] = {}
    for idx, row in enumerate(decisions, start=1):  # 1-based: row idx == cumulative environment step
        bin_idx = (idx - 1) // bin_size
        bins.setdefault(bin_idx, []).append(row)

    cumulative_calls = 0
    cumulative_wall_clock_s = 0.0
    results = []
    for bin_idx in sorted(bins):
        rows = bins[bin_idx]
        step_range = (bin_idx * bin_size + 1, min((bin_idx + 1) * bin_size, len(decisions)))
        queried_rows = [r for r in rows if _b(r, "queried")]
        accepted_rows = [r for r in queried_rows if r.get("response_status") == "accepted"]
        not_queried_rows = [r for r in rows if not _b(r, "queried")]

        normalized_entropies = []
        for r in rows:
            legal_count = int(r["legal_action_count"])
            entropy = _f(r, "base_policy_entropy")
            if entropy is not None and legal_count > 1:
                normalized_entropies.append(entropy / math.log(legal_count))
        queried_entropies = []
        not_queried_entropies = []
        for r in rows:
            legal_count = int(r["legal_action_count"])
            entropy = _f(r, "base_policy_entropy")
            if entropy is None or legal_count <= 1:
                continue
            ne = entropy / math.log(legal_count)
            (queried_entropies if _b(r, "queried") else not_queried_entropies).append(ne)

        latencies = [_f(r, "response_latency_ms") / 1000 for r in accepted_rows if r.get("response_latency_ms")]
        betas = [_f(r, "beta") for r in accepted_rows if r.get("beta")]
        agreements = [
            1.0 if r["base_top_action_id"] == r["plan_maker_top_action_id"] else 0.0
            for r in accepted_rows if r.get("base_top_action_id") and r.get("plan_maker_top_action_id")
        ]
        changed = [1.0 if _b(r, "advice_changed_top_action") else 0.0 for r in rows if _b(r, "advice_changed_top_action") is not None]
        query_probs = [_f(r, "query_probability") for r in rows if r.get("query_probability")]

        distinct_episodes = len({r["episode_id"] for r in rows})
        cumulative_calls += len(queried_rows)
        cumulative_wall_clock_s += sum(_f(r, "response_latency_ms") or 0.0 for r in queried_rows) / 1000

        def _mean(values):
            return sum(values) / len(values) if values else None

        def _pctl(values, p):
            if not values:
                return None
            s = sorted(values)
            return s[min(int(len(s) * p), len(s) - 1)]

        results.append(
            {
                "bin_index": bin_idx,
                "environment_step_range": f"{step_range[0]}-{step_range[1]}",
                "num_decisions": len(rows),
                "actual_query_rate": len(queried_rows) / len(rows) if rows else None,
                "mean_query_probability": _mean(query_probs),
                "normalized_entropy_all": _mean(normalized_entropies),
                "normalized_entropy_queried": _mean(queried_entropies),
                "normalized_entropy_not_queried": _mean(not_queried_entropies),
                "mean_beta": _mean(betas),
                "ppo_plan_maker_top1_agreement": _mean(agreements),
                "advice_changed_top_action_rate": _mean(changed),
                "consultations_per_episode_approx": len(queried_rows) / distinct_episodes if distinct_episodes else None,
                "plan_maker_latency_mean_s": _mean(latencies),
                "plan_maker_latency_p50_s": _pctl(latencies, 0.50),
                "plan_maker_latency_p95_s": _pctl(latencies, 0.95),
                "cumulative_plan_maker_calls": cumulative_calls,
                "cumulative_plan_maker_wall_clock_hours": round(cumulative_wall_clock_s / 3600, 3),
            }
        )
    return results


def bin_updates(updates: list[dict], bin_size: int) -> list[dict]:
    results = []
    by_step_bin: dict[int, list[dict]] = {}
    for row in updates:
        steps = int(row["environment_steps"])
        bin_idx = (steps - 1) // bin_size
        by_step_bin.setdefault(bin_idx, []).append(row)

    for bin_idx in sorted(by_step_bin):
        rows = by_step_bin[bin_idx]

        def _mean(key):
            values = [_f(r, key) for r in rows if r.get(key) not in (None, "")]
            return sum(values) / len(values) if values else None

        def _max_abs(key):
            values = [abs(_f(r, key)) for r in rows if r.get(key) not in (None, "")]
            return max(values) if values else None

        results.append(
            {
                "bin_index": bin_idx,
                "num_updates": len(rows),
                "mean_approximate_kl": _mean("approximate_kl"),
                "max_abs_approximate_kl": _max_abs("approximate_kl"),
                "mean_clip_fraction": _mean("clip_fraction"),
                "mean_explained_variance": _mean("explained_variance"),
                "mean_action_entropy": _mean("action_entropy"),
                "mean_query_entropy": _mean("query_entropy"),
                "mean_gradient_norm": _mean("gradient_norm"),
                "max_gradient_norm": _max_abs("gradient_norm"),
                "mean_policy_loss": _mean("policy_loss"),
                "mean_value_loss": _mean("value_loss"),
            }
        )
    return results


def check_catastrophic_flags(decision_bins: list[dict], update_bins: list[dict]) -> list[str]:
    flags = []

    # Numerical instability
    for b in update_bins:
        if b["max_abs_approximate_kl"] is not None and b["max_abs_approximate_kl"] > THRESHOLDS["kl_catastrophic"]:
            flags.append(f"bin {b['bin_index']}: max |approximate_kl|={b['max_abs_approximate_kl']:.2f} > {THRESHOLDS['kl_catastrophic']} (catastrophic policy shift)")
        if b["mean_clip_fraction"] is not None and b["mean_clip_fraction"] > THRESHOLDS["clip_fraction_catastrophic"]:
            flags.append(f"bin {b['bin_index']}: mean clip_fraction={b['mean_clip_fraction']:.2f} > {THRESHOLDS['clip_fraction_catastrophic']} (step size badly miscalibrated)")
        if b["max_gradient_norm"] is not None and b["max_gradient_norm"] > THRESHOLDS["grad_norm_catastrophic"]:
            flags.append(f"bin {b['bin_index']}: max gradient_norm={b['max_gradient_norm']:.1f} > {THRESHOLDS['grad_norm_catastrophic']} (exploding gradients)")
        for key in ("mean_policy_loss", "mean_value_loss"):
            v = b[key]
            if v is not None and (math.isnan(v) or math.isinf(v)):
                flags.append(f"bin {b['bin_index']}: {key} is {v} (NaN/Inf -- numerical divergence)")

    # Query-gate collapse: check EVERY bin after the first, not just the last
    # (a mid-run collapse that later "recovers" on paper by chance is still
    # worth flagging, not silently averaged away).
    non_first_bins = [b for b in decision_bins if b["bin_index"] > 0]
    if non_first_bins and all(
        b["actual_query_rate"] is not None and b["actual_query_rate"] < THRESHOLDS["query_rate_collapse_low"]
        for b in non_first_bins
    ):
        flags.append(f"query rate < {THRESHOLDS['query_rate_collapse_low']} in every bin after the first -- pathological immediate query-gate collapse to near-zero")
    if non_first_bins and all(
        b["actual_query_rate"] is not None and b["actual_query_rate"] > THRESHOLDS["query_rate_collapse_high"]
        for b in non_first_bins
    ):
        flags.append(f"query rate > {THRESHOLDS['query_rate_collapse_high']} in every bin after the first -- pathological collapse to always-query")

    # Consultation never affecting decisions
    all_changed = [b["advice_changed_top_action_rate"] for b in decision_bins if b["advice_changed_top_action_rate"] is not None]
    if all_changed and sum(all_changed) / len(all_changed) < THRESHOLDS["advice_influence_negligible"]:
        flags.append(f"mean advice_changed_top_action_rate={sum(all_changed)/len(all_changed):.3f} < {THRESHOLDS['advice_influence_negligible']} across the whole run -- consultation may never be meaningfully affecting decisions")

    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bin-size", type=int, default=1024)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    decisions = read_csv(run_dir / "decisions.csv")
    updates = read_csv(run_dir / "updates.csv")

    decision_bins = bin_decisions(decisions, args.bin_size)
    update_bins = bin_updates(updates, args.bin_size)

    print(f"=== MARLA_FULL training diagnostics, {run_dir.name}, {len(decisions)} total decisions, bin size {args.bin_size} ===\n")
    for b in decision_bins:
        print(f"--- bin {b['bin_index']} (steps {b['environment_step_range']}) ---")
        for k, v in b.items():
            if k in ("bin_index", "environment_step_range"):
                continue
            print(f"  {k}: {v}")
        print()

    print("=== PPO stability diagnostics ===\n")
    for b in update_bins:
        print(f"--- bin {b['bin_index']} ({b['num_updates']} updates) ---")
        for k, v in b.items():
            if k in ("bin_index", "num_updates"):
                continue
            print(f"  {k}: {v}")
        print()

    print("=== Catastrophic-failure check ===")
    flags = check_catastrophic_flags(decision_bins, update_bins)
    if flags:
        print(f"{len(flags)} FLAG(S) RAISED:")
        for f in flags:
            print(f"  - {f}")
    else:
        print("No catastrophic-failure patterns detected (numerical instability, query-gate collapse, or negligible advice influence).")


if __name__ == "__main__":
    main()
