#!/usr/bin/env python
"""Aggregates completed research/aamas2027 runs into research/aamas2027/aggregate/*.csv.

The unit of independent replication is TRAINING SEED, not episode (Phase 10).
Every aggregate number bootstraps across seed-level summaries; per-checkpoint
Wilson intervals are kept only as a labeled, separate descriptive quantity,
never substituted for the across-seed bootstrap. Never selects a "best"
seed/checkpoint/window -- every seed listed in manifest.yaml's
training_seeds is used, unconditionally, for every condition that has run.

Usage:
    python research/aamas2027/analyze.py [--manifest manifest.yaml]

Missing runs are reported and skipped, not fabricated -- see each
function's "found N/M seeds" printout. Re-run any time; never mutates raw
run directories, only writes into aggregate/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # research/aamas2027/analyze.py -> repo root
# manifest.yaml's run_dir values (e.g. "runs/aamas2027_ppo_only/...") are
# relative to the repo root, matching how `marla run` itself writes and
# reports them -- not relative to this script's own directory.


# --- statistics --------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Descriptive within-checkpoint uncertainty ONLY -- never a substitute
    for the across-seed bootstrap below (Phase 10's explicit distinction)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((center - margin) / denom, (center + margin) / denom)


def bootstrap_ci(
    values: list[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(point_estimate, ci_low, ci_high) via the percentile bootstrap over
    ``values`` -- one value per training seed, never per episode."""
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    if len(values) == 1:
        # A single seed has no resampling variance to estimate -- report the
        # point value with an explicitly degenerate (zero-width) interval
        # rather than a fabricated one. Callers must treat this as "not a
        # real CI," not silently plot it as if it were.
        return (values[0], values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    point = sum(values) / n
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return (point, means[max(lo_idx, 0)], means[min(hi_idx, n_boot - 1)])


def iqm(values: list[float]) -> float:
    """Interquartile mean: robust to outlier seeds/scenarios, standard in
    the RL-evaluation literature (Agarwal et al. 2021) for aggregating
    across a small number of runs/tasks."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    n = len(ordered)
    lo = n / 4
    hi = 3 * n / 4
    kept = [v for i, v in enumerate(ordered) if lo - 0.5 <= i < hi - 0.5] or ordered
    return sum(kept) / len(kept)


def probability_of_improvement(a_values: list[float], b_values: list[float], n_boot: int = 10000, seed: int = 0) -> float:
    """P(a > b), estimated by paired bootstrap resampling of both seed-level
    value lists together (assumes a_values/b_values share the same seed
    order -- callers must guarantee this)."""
    if not a_values or not b_values or len(a_values) != len(b_values):
        return float("nan")
    rng = random.Random(seed)
    n = len(a_values)
    wins = 0
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a_mean = sum(a_values[i] for i in idx) / n
        b_mean = sum(b_values[i] for i in idx) / n
        wins += a_mean > b_mean
    return wins / n_boot


# --- manifest / IO -------------------------------------------------------


def load_manifest(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_scenario_manifest(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as fh:
        return {row["scenario_name"]: row["ID_or_OOD"] for row in csv.DictReader(fh)}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _b(row: dict, key: str) -> bool | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return value in ("True", "true", "1")


# --- Figure 1 data: learning curves (trainable conditions only) ---------


def learning_curve(manifest: dict, out_dir: Path) -> None:
    rows_out: list[dict[str, Any]] = []
    for condition in ("PPO_ONLY", "MARLA_FULL"):
        cond_cfg = manifest["conditions"][condition]
        found = 0
        for seed, run_info in cond_cfg["runs"].items():
            run_dir = REPO_ROOT / run_info["run_dir"]
            episodes = read_csv_rows(run_dir / "episodes.csv")
            updates = read_csv_rows(run_dir / "updates.csv")
            if not episodes or not updates:
                continue
            found += 1
            eval_rows = [r for r in episodes if _b(r, "is_eval")]
            by_rollout: dict[int, list[dict]] = defaultdict(list)
            for r in eval_rows:
                if r.get("rollout"):
                    by_rollout[int(r["rollout"])].append(r)
            # environment_steps at a given rollout boundary = rollout_index *
            # rollout_steps, read off config.yaml rather than assumed.
            config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
            rollout_steps = config["policy"]["ppo"]["rollout_steps"]
            for rollout_idx, rows in sorted(by_rollout.items()):
                successes = [1.0 if _b(r, "goal_success") else 0.0 for r in rows]
                returns = [_f(r, "benchmark_return") for r in rows]
                rows_out.append(
                    {
                        "condition": condition,
                        "training_seed": seed,
                        "checkpoint_environment_steps": rollout_idx * rollout_steps,
                        "num_eval_episodes": len(rows),
                        "mean_goal_success": sum(successes) / len(successes),
                        "mean_benchmark_return": sum(returns) / len(returns),
                    }
                )
        print(f"[learning_curve] {condition}: found {found}/{len(cond_cfg['runs'])} training seed(s)")

    out_path = out_dir / "learning_curve.csv"
    _write_csv(out_path, rows_out)
    print(f"[learning_curve] wrote {len(rows_out)} row(s) to {out_path}")


def learning_curve_bands(out_dir: Path) -> None:
    """Bootstraps across training seeds at each shared checkpoint_environment_steps."""
    rows = read_csv_rows(out_dir / "learning_curve.csv")
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["condition"], int(r["checkpoint_environment_steps"]))].append(r)

    band_rows = []
    for (condition, steps), group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        success_values = [float(g["mean_goal_success"]) for g in group]
        return_values = [float(g["mean_benchmark_return"]) for g in group]
        s_point, s_lo, s_hi = bootstrap_ci(success_values)
        r_point, r_lo, r_hi = bootstrap_ci(return_values)
        band_rows.append(
            {
                "condition": condition,
                "checkpoint_environment_steps": steps,
                "num_seeds": len(group),
                "goal_success_point": s_point, "goal_success_ci_low": s_lo, "goal_success_ci_high": s_hi,
                "benchmark_return_point": r_point, "benchmark_return_ci_low": r_lo, "benchmark_return_ci_high": r_hi,
            }
        )
    out_path = out_dir / "learning_curve_bands.csv"
    _write_csv(out_path, band_rows)
    print(f"[learning_curve_bands] wrote {len(band_rows)} row(s) to {out_path}")


# --- Figure 2 data: final ID + OOD performance ---------------------------
# Restricted to the three REQUIRED conditions (manifest.yaml v3):
# PPO_ONLY, MARLA_FULL_NORMAL (learned gate, no forced queries),
# MARLA_FULL_NO_QUERY (the main inference ablation, zero Plan Maker calls).
# ALWAYS_QUERY/BETA_ZERO/BETA_ONE/PLAN_MAKER_ONLY are postponed -- see
# manifest.yaml's postponed_pending_feasibility.

REQUIRED_EVAL_CONDITIONS = ("PPO_ONLY", "MARLA_FULL_NORMAL", "MARLA_FULL_NO_QUERY")
SCENARIO_LABELS = ("ID", "OOD_dmz_three_subnets", "OOD_user_three_subnets")


def id_and_ood_eval(manifest: dict, out_dir: Path, eval_raw_dir: Path) -> None:
    rows_out = []
    for condition in REQUIRED_EVAL_CONDITIONS:
        for scenario_label in SCENARIO_LABELS:
            found = 0
            # ID eval is written to <condition>/seed-<seed>/ (no scenario
            # suffix, see README.md); OOD eval to <condition>__<label>/seed-<seed>/.
            subdir_name = condition if scenario_label == "ID" else f"{condition}__{scenario_label}"
            for seed in manifest["training_seeds"]:
                episodes_path = eval_raw_dir / subdir_name / f"seed-{seed}" / "episodes.csv"
                episodes = read_csv_rows(episodes_path)
                if not episodes:
                    continue
                found += 1
                successes = [1.0 if _b(r, "goal_success") else 0.0 for r in episodes]
                returns = [_f(r, "benchmark_return") for r in episodes]
                steps_to_goal = [_f(r, "steps_to_goal") for r in episodes if r.get("steps_to_goal")]
                consultations = [_f(r, "consultation_count") or 0.0 for r in episodes]
                rows_out.append(
                    {
                        "condition": condition,
                        "scenario": scenario_label,
                        "training_seed": seed,
                        "num_episodes": len(episodes),
                        "goal_success_rate": sum(successes) / len(successes),
                        "goal_success_wilson_low": wilson_interval(int(sum(successes)), len(successes))[0],
                        "goal_success_wilson_high": wilson_interval(int(sum(successes)), len(successes))[1],
                        "mean_benchmark_return": sum(returns) / len(returns),
                        "mean_steps_to_goal": (sum(steps_to_goal) / len(steps_to_goal)) if steps_to_goal else None,
                        "mean_consultations_per_episode": sum(consultations) / len(consultations),
                    }
                )
            print(f"[id_and_ood_eval] {condition} / {scenario_label}: found {found}/{len(manifest['training_seeds'])} seed(s)")
    out_path = out_dir / "id_and_ood_eval.csv"
    _write_csv(out_path, rows_out)
    print(f"[id_and_ood_eval] wrote {len(rows_out)} row(s) to {out_path}")


# --- Figure 3 data: consultation / query-gate behavior --------------------


def query_rate_over_training(manifest: dict, out_dir: Path) -> None:
    """Per-rollout actual_query_rate already computed by trainer.py --
    zero new computation needed, just reading an existing updates.csv
    column across MARLA_FULL's training seeds."""
    rows_out = []
    for seed, run_info in manifest["conditions"]["MARLA_FULL"]["runs"].items():
        run_dir = REPO_ROOT / run_info["run_dir"]
        updates = read_csv_rows(run_dir / "updates.csv")
        if not updates:
            print(f"[query_rate_over_training] seed {seed}: no updates.csv found, skipped")
            continue
        seen_steps = set()
        for u in updates:
            steps = int(u["environment_steps"])
            if steps in seen_steps or not u.get("actual_query_rate"):
                continue  # several PPO updates share one rollout's aggregates; one point per rollout
            seen_steps.add(steps)
            rows_out.append(
                {"training_seed": seed, "environment_steps": steps, "actual_query_rate": u["actual_query_rate"]}
            )
    out_path = out_dir / "query_rate_over_training.csv"
    _write_csv(out_path, rows_out)
    print(f"[query_rate_over_training] wrote {len(rows_out)} row(s) to {out_path}")


def consultation_behavior(manifest: dict, out_dir: Path) -> None:
    """Per-decision data for Figure 3 (entropy/query-probability) -- every
    field here is a direct read or one-line derivation from an existing
    decisions.csv column; no new Plan Maker calls, no new instrumentation."""
    rows_out = []
    for seed, run_info in manifest["conditions"]["MARLA_FULL"]["runs"].items():
        run_dir = REPO_ROOT / run_info["run_dir"]
        decisions = read_csv_rows(run_dir / "decisions.csv")
        if not decisions:
            print(f"[consultation_behavior] seed {seed}: no decisions.csv found, skipped")
            continue
        for r in decisions:
            legal_count = int(r["legal_action_count"])
            entropy = _f(r, "base_policy_entropy")
            normalized_entropy = None if entropy is None or legal_count <= 1 else entropy / math.log(legal_count)
            rows_out.append(
                {
                    "training_seed": seed,
                    "environment_step": r["environment_step"],
                    "queried": r["queried"],
                    "query_probability": r["query_probability"],
                    "normalized_base_policy_entropy": normalized_entropy,
                    "base_top_two_margin": r.get("base_top_two_margin"),
                    "reward_this_step": r.get("training_reward"),
                }
            )
    out_path = out_dir / "consultation_behavior.csv"
    _write_csv(out_path, rows_out)
    print(f"[consultation_behavior] wrote {len(rows_out)} row(s) to {out_path}")


# --- Figure 4 data: advice / trust behavior -------------------------------


def beta_over_training(manifest: dict, out_dir: Path) -> None:
    """updates.csv's mean_beta is already a per-rollout aggregate -- read,
    not recomputed."""
    rows_out = []
    for seed, run_info in manifest["conditions"]["MARLA_FULL"]["runs"].items():
        run_dir = REPO_ROOT / run_info["run_dir"]
        updates = read_csv_rows(run_dir / "updates.csv")
        if not updates:
            continue
        seen_steps = set()
        for u in updates:
            steps = int(u["environment_steps"])
            if steps in seen_steps or not u.get("mean_beta"):
                continue
            seen_steps.add(steps)
            rows_out.append({"training_seed": seed, "environment_steps": steps, "mean_beta": u["mean_beta"]})
    _write_csv(out_dir / "beta_over_training.csv", rows_out)
    print(f"[beta_over_training] wrote {len(rows_out)} row(s)")


def advice_trust_behavior(manifest: dict, out_dir: Path) -> None:
    """Per-decision beta distribution, PPO/Plan-Maker top-1 agreement,
    advice_changed_top_action fraction -- all direct decisions.csv reads.
    Per-episode consultations-per-successful-episode from episodes.csv."""
    decision_rows_out = []
    episode_summary_rows = []
    for seed, run_info in manifest["conditions"]["MARLA_FULL"]["runs"].items():
        run_dir = REPO_ROOT / run_info["run_dir"]
        decisions = read_csv_rows(run_dir / "decisions.csv")
        episodes = read_csv_rows(run_dir / "episodes.csv")
        if not decisions:
            continue
        for r in decisions:
            if not r.get("queried") == "True":
                continue
            agreement = None
            if r.get("base_top_action_id") and r.get("plan_maker_top_action_id"):
                agreement = 1.0 if r["base_top_action_id"] == r["plan_maker_top_action_id"] else 0.0
            decision_rows_out.append(
                {
                    "training_seed": seed,
                    "beta": r.get("beta"),
                    "alpha": r.get("alpha"),
                    "base_plan_maker_top1_agreement": agreement,
                    "advice_changed_top_action": r.get("advice_changed_top_action"),
                    "selected_action_base_rank": r.get("selected_action_base_rank"),
                    "selected_action_plan_maker_rank": r.get("selected_action_plan_maker_rank"),
                }
            )
        successful_episodes = [e for e in episodes if _b(e, "goal_success") and not _b(e, "is_eval")]
        if successful_episodes:
            consultations = [_f(e, "consultation_count") or 0.0 for e in successful_episodes]
            episode_summary_rows.append(
                {
                    "training_seed": seed,
                    "num_successful_episodes": len(successful_episodes),
                    "mean_consultations_per_successful_episode": sum(consultations) / len(consultations),
                }
            )
    _write_csv(out_dir / "advice_trust_decisions.csv", decision_rows_out)
    _write_csv(out_dir / "consultations_per_successful_episode.csv", episode_summary_rows)
    print(f"[advice_trust_behavior] wrote {len(decision_rows_out)} decision row(s), {len(episode_summary_rows)} episode-summary row(s)")


def beta_stratified_advice_influence(manifest: dict, out_dir: Path) -> None:
    """Observational, NOT a beta=0/1 counterfactual: decisions.csv does not
    persist the raw per-action base-logit/advice vectors needed to
    recompute what a forced beta would have selected (only summary fields
    -- top-action IDs, ranks, the actual beta/alpha that occurred). This
    instead buckets already-recorded decisions by the beta value that
    actually occurred and reports advice_changed_top_action rate per
    bucket -- zero new Plan Maker calls, but explicitly an observed
    correlation across MARLA_FULL's own natural beta distribution, not a
    causal forced-beta comparison. See manifest.yaml's
    postponed_pending_feasibility entry for MARLA_FULL_BETA_ZERO/ONE."""
    buckets = {"low_beta_[0,0.33)": [], "mid_beta_[0.33,0.67)": [], "high_beta_[0.67,1]": []}
    for seed, run_info in manifest["conditions"]["MARLA_FULL"]["runs"].items():
        run_dir = REPO_ROOT / run_info["run_dir"]
        decisions = read_csv_rows(run_dir / "decisions.csv")
        for r in decisions:
            beta = _f(r, "beta")
            changed = _b(r, "advice_changed_top_action")
            if beta is None or changed is None:
                continue
            key = "low_beta_[0,0.33)" if beta < 0.33 else "mid_beta_[0.33,0.67)" if beta < 0.67 else "high_beta_[0.67,1]"
            buckets[key].append(1.0 if changed else 0.0)
    rows_out = [
        {
            "beta_bucket": bucket,
            "num_decisions": len(values),
            "advice_changed_top_action_rate": (sum(values) / len(values)) if values else None,
        }
        for bucket, values in buckets.items()
    ]
    _write_csv(out_dir / "beta_stratified_advice_influence.csv", rows_out)
    print(f"[beta_stratified_advice_influence] wrote {len(rows_out)} row(s)")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.yaml")
    parser.add_argument("--out-dir", type=Path, default=HERE / "aggregate")
    parser.add_argument("--eval-raw-dir", type=Path, default=HERE / "raw" / "eval")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    learning_curve(manifest, args.out_dir)
    learning_curve_bands(args.out_dir)
    id_and_ood_eval(manifest, args.out_dir, args.eval_raw_dir)
    query_rate_over_training(manifest, args.out_dir)
    consultation_behavior(manifest, args.out_dir)
    beta_over_training(manifest, args.out_dir)
    advice_trust_behavior(manifest, args.out_dir)
    beta_stratified_advice_influence(manifest, args.out_dir)


if __name__ == "__main__":
    main()
