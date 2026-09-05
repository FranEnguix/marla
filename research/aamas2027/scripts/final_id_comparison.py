#!/usr/bin/env python
"""Builds the interim feasibility report's 3-condition ID comparison table
(goal success, benchmark return, steps-to-goal, episode length, finish
reasons) from evaluate_checkpoint.py's output directories. Explicitly
prints a single-seed caveat -- this compares ONE paired seed (101), not
evidence of superiority, per the explicit instruction not to over-interpret
a single seed.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    successes = [1.0 if r["goal_success"] == "True" else 0.0 for r in episodes]
    returns = [float(r["benchmark_return"]) for r in episodes]
    lengths = [float(r["environment_steps"]) for r in episodes]
    steps_to_goal = [float(r["steps_to_goal"]) for r in episodes if r.get("steps_to_goal")]
    finish_reasons = Counter(r["finish_reason"] for r in episodes)
    return {
        "n_episodes": n,
        "goal_success_rate": sum(successes) / n if n else None,
        "mean_benchmark_return": sum(returns) / n if n else None,
        "mean_episode_length": sum(lengths) / n if n else None,
        "mean_steps_to_goal_successes_only": (sum(steps_to_goal) / len(steps_to_goal)) if steps_to_goal else None,
        "n_successes_with_steps_to_goal": len(steps_to_goal),
        "finish_reasons": dict(finish_reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-raw-dir", type=Path, default=REPO_ROOT / "research/aamas2027/raw/eval")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--conditions", nargs="+", default=["PPO_ONLY", "MARLA_FULL_NORMAL", "MARLA_FULL_NO_QUERY"]
    )
    args = parser.parse_args()

    print(f"=== Final ID evaluation comparison, seed {args.seed} ===")
    print(
        "CAVEAT: single paired seed. This is a feasibility signal, not evidence "
        "of superiority -- do not use this alone to conclude one method is better.\n"
    )

    rows = {}
    for condition in args.conditions:
        path = args.eval_raw_dir / condition / f"seed-{args.seed}" / "episodes.csv"
        if not path.is_file():
            print(f"[{condition}] no episodes.csv found at {path} -- skipped")
            continue
        episodes = read_csv(path)
        rows[condition] = summarize(episodes)

    header = ["condition", "n_episodes", "goal_success_rate", "mean_benchmark_return", "mean_episode_length", "mean_steps_to_goal (successes only)", "n_successes"]
    print(" | ".join(header))
    print("-" * 100)
    for condition, s in rows.items():
        print(
            f"{condition} | {s['n_episodes']} | {s['goal_success_rate']:.3f} | "
            f"{s['mean_benchmark_return']:.3f} | {s['mean_episode_length']:.1f} | "
            f"{s['mean_steps_to_goal_successes_only'] if s['mean_steps_to_goal_successes_only'] is not None else 'n/a'} | "
            f"{s['n_successes_with_steps_to_goal']}"
        )
    print()
    for condition, s in rows.items():
        print(f"[{condition}] finish_reason breakdown: {s['finish_reasons']}")


if __name__ == "__main__":
    main()
