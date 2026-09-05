#!/usr/bin/env python
"""CLI wrapper around ``marla.evaluation.checkpoint_eval`` -- evaluates one
run's checkpoint under one named condition (see manifest.yaml's
`conditions` section) and writes episodes.csv/decisions.csv-shaped output.

No gradient step is ever taken; see src/marla/evaluation/checkpoint_eval.py
and research/aamas2027/VALIDATION.md checks 9-10.

Usage:
    python research/aamas2027/scripts/evaluate_checkpoint.py \\
        --run-dir runs/aamas2027_marla_full/marla-full-seed-101 \\
        --condition MARLA_FULL_BETA_ZERO \\
        --seed-start 5001 --num-episodes 20 \\
        --out-dir research/aamas2027/raw/eval/MARLA_FULL_BETA_ZERO/seed-101 \\
        [--scenario /abs/path/to/other_scenario.v2.yaml] \\
        [--cache research/aamas2027/raw/advisory_cache.jsonl] \\
        [--training-seed 101] [--id-or-ood ID]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from marla.evaluation.checkpoint_eval import evaluate_checkpoint  # noqa: E402
from marla.evaluation.overrides import (  # noqa: E402
    ALWAYS_QUERY,
    BETA_ONE,
    BETA_ZERO,
    NO_QUERY,
    NORMAL,
    PLAN_MAKER_ONLY,
    EvaluationOverrides,
)
from marla.learning.recurrent_policy import RecurrentPolicy  # noqa: E402
from marla.metrics.writer import _decision_row  # noqa: E402
from marla.runtime.device import resolve_device  # noqa: E402

# Named conditions this program actually evaluates (manifest.yaml's
# `conditions` section is the authoritative description; this dict is just
# the mechanical query_mode/beta_override/action_selection each one maps
# to). PPO_ONLY/MARLA_FULL themselves need no overrides at all (their own
# trained/learned behavior, overrides=None).
NAMED_CONDITIONS: dict[str, EvaluationOverrides | None] = {
    "PPO_ONLY": None,
    "MARLA_FULL_NORMAL": NORMAL,
    "MARLA_FULL_NO_QUERY": NO_QUERY,
    "MARLA_FULL_ALWAYS_QUERY": ALWAYS_QUERY,
    "MARLA_FULL_BETA_ZERO": BETA_ZERO,
    "MARLA_FULL_BETA_ONE": BETA_ONE,
    "PLAN_MAKER_ONLY": PLAN_MAKER_ONLY,
}


def _episode_row(summary, condition: str, training_seed: int, id_or_ood: str, checkpoint_environment_steps: int) -> dict:
    return {
        "run_id": summary.run_id,
        "condition": condition,
        "training_seed": training_seed,
        "evaluation_seed": summary.seed,
        "id_or_ood": id_or_ood,
        "checkpoint_environment_steps": checkpoint_environment_steps,
        "goal_success": summary.goal_success,
        "benchmark_return": summary.nasimemu_return,
        "training_return": summary.training_return,
        "environment_steps": summary.environment_steps,
        "steps_to_goal": summary.steps_to_goal,
        "finish_reason": summary.finish_reason,
        "episode_seconds": summary.episode_seconds,
        "consultation_count": summary.consultation_count,
        "consultation_cost": summary.consultation_cost,
        "is_eval": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=sorted(NAMED_CONDITIONS), required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scenario", type=str, default=None, help="Absolute path; defaults to the run's own training scenario")
    parser.add_argument("--id-or-ood", choices=["ID", "OOD"], default="ID")
    parser.add_argument("--training-seed", type=int, required=True, help="The training seed this checkpoint/scaffold corresponds to")
    parser.add_argument("--checkpoint-environment-steps", type=int, default=None, help="Defaults to the checkpoint's own recorded environment_steps")
    parser.add_argument("--cache", type=Path, default=None, help="Advisory-response cache path (shared across ablations on the same checkpoint)")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--fresh-scaffold-seed", type=int, default=None,
        help="PLAN_MAKER_ONLY only: build a freshly-initialized, never-trained policy with this torch seed instead of loading --run-dir's checkpoint.pt",
    )
    args = parser.parse_args()

    overrides = NAMED_CONDITIONS[args.condition]
    policy_override = None
    if args.fresh_scaffold_seed is not None:
        from marla.evaluation.checkpoint_eval import build_policy, load_run_config

        config = load_run_config(args.run_dir)
        resolved = resolve_device(args.device)
        policy_override = build_policy(
            config, consultation_enabled=True, device=resolved.torch_device,
            checkpoint_path=None, seed=args.fresh_scaffold_seed,
        )

    result = asyncio.run(
        evaluate_checkpoint(
            run_dir=args.run_dir,
            seed_start=args.seed_start,
            num_episodes=args.num_episodes,
            overrides=overrides,
            scenario_path=args.scenario,
            device=args.device,
            policy_override=policy_override,
            advisory_cache_path=args.cache,
        )
    )

    checkpoint_steps = args.checkpoint_environment_steps
    if checkpoint_steps is None:
        import torch

        checkpoint_steps = int(torch.load(args.run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)["environment_steps"])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = [
        _episode_row(s, args.condition, args.training_seed, args.id_or_ood, checkpoint_steps) for s in result.summaries
    ]
    with (args.out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(episode_rows[0].keys()) if episode_rows else [])
        writer.writeheader()
        for row in episode_rows:
            writer.writerow(row)

    decision_fieldnames = None
    with (args.out_dir / "decisions.csv").open("w", newline="", encoding="utf-8") as fh:
        for record in result.records:
            row = _decision_row(None, record)  # config unused by _decision_row's body
            is_baseline = args.condition == "PPO_ONLY"
            row["condition"] = args.condition
            row["training_seed"] = args.training_seed
            row["query_mode"] = "n/a" if is_baseline else (overrides.query_mode if overrides is not None else "learned")
            row["trust_mode"] = (
                "n/a" if is_baseline
                else "learned" if overrides is None or overrides.beta_override is None
                else f"fixed:{overrides.beta_override}"
            )
            row["advice_transformation"] = "n/a" if is_baseline else (overrides.advice_transformation if overrides is not None else "identity")
            if decision_fieldnames is None:
                decision_fieldnames = list(row.keys())
                writer = csv.DictWriter(fh, fieldnames=decision_fieldnames)
                writer.writeheader()
            writer.writerow(row)

    manifest_entry = {
        "run_dir": str(args.run_dir),
        "condition": args.condition,
        "training_seed": args.training_seed,
        "seed_start": args.seed_start,
        "num_episodes": args.num_episodes,
        "scenario": result.scenario,
        "id_or_ood": args.id_or_ood,
        "checkpoint_environment_steps": checkpoint_steps,
        "cache_hits": result.cache_hits,
        "cache_misses": result.cache_misses,
    }
    (args.out_dir / "eval_metadata.json").write_text(json.dumps(manifest_entry, indent=2), encoding="utf-8")
    print(f"Wrote {len(episode_rows)} episode(s), {len(result.records)} decision(s) to {args.out_dir}/")
    print(f"Advisory cache: {result.cache_hits} hit(s), {result.cache_misses} miss(es)")


if __name__ == "__main__":
    main()
