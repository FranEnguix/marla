# research/aamas2027

A scoped-down experimental pipeline built on top of MARLA (see repo root
`README.md`/`docs/`). Start with `AUDIT.md` (what the codebase actually does
and its limitations), then `manifest.yaml` (the exact experiment matrix:
conditions, seeds, scenario, git SHA, config hashes), then this file for
exact reproduction commands. `VALIDATION.md` documents every check run
before any long training job.

**Status distinctions used throughout this directory**: *implemented*
(code exists and is validated), *launched* (a run has been started against
real compute), *completed* (finished, artifacts exist on disk). Never
conflated -- see the final summary given alongside this deliverable set for
which experiments are currently in which state.

**Current status (manifest.yaml v4 -- staged training)**: 50k/20k/15k-step
MARLA_FULL training is off the table for this paper's timeline (real pilot
measurement: ~13.5 days/seed at 50k). Training now proceeds in stages --
Stage A is 5,000 steps, extending to Stage B (10,000, via `marla run
--resume`, never from scratch) only with explicit approval after Stage A's
interim report. `runs/aamas2027_ppo_only/ppo-only-seed-101` already holds
a real, verified-matching 5,000-step Stage A result (reused as-is).
`runs/aamas2027_marla_full/marla-full-seed-101` holds nothing on disk (the
pilot measurement pass was killed before writing anything) -- Stage A for
MARLA_FULL has not been launched. See `VALIDATION.md` and the final
summary for the launch decision.

## Layout

```
AUDIT.md                 repository audit (Phase 1)
VALIDATION.md             validation checks actually run (Phase 12)
manifest.yaml              conditions x seeds x scenario x git SHA x config hashes x eval seed sets
scenario_manifest.csv       ID + OOD scenario inventory
configs/                    the 6 primary training configs (PPO_ONLY/MARLA_FULL x 3 seeds)
scripts/
  evaluate_checkpoint.py     CLI: evaluate one checkpoint under one named condition (no gradient step)
  make_figures.py             generates the 4 required figures from aggregate/*.csv
raw/                          README pointing at the true runs/<experiment>/<run-id>/ directories (not copied)
aggregate/                    analyze.py's joined/derived CSVs
figures/                      the 4 required figures, PDF+PNG
tables/                       compact CSV/LaTeX table(s)
analyze.py                    aggregation + bootstrap statistics (Phase 10)
```

## Reproduce: one baseline (PPO_ONLY) training run

```bash
export MARLA_RL_ORCHESTRATOR_PASSWORD=changeme
marla run research/aamas2027/configs/ppo_only_seed101.yaml
marla summarize runs/aamas2027_ppo_only/ppo-only-seed-101
```

## Reproduce: one MARLA training run

Needs the local-lm extra and a real ~3GB model download on first use:

```bash
pip install -e ".[local-lm]"
export MARLA_RL_ORCHESTRATOR_PASSWORD=changeme
export MARLA_GATEKEEPER_PASSWORD=changeme
export MARLA_PLAN_MAKER_1_PASSWORD=changeme
marla run research/aamas2027/configs/marla_full_seed101.yaml
marla summarize runs/aamas2027_marla_full/marla-full-seed-101
```

## Extending to Stage B (10,000 steps) -- requires explicit approval first

Never restarts from scratch: `--resume` loads policy/optimizer/RNG state
and continues episode-seed progression from the Stage A checkpoint (see
`tests/test_checkpoint_resume.py` for the confirmed-working end-to-end
test). Create a Stage B config that is identical to the Stage A one except
`ppo.total_environment_steps: 10000` and a different `run_id`, then:

```bash
marla run --resume runs/aamas2027_marla_full/marla-full-seed-101/checkpoint.pt \
  research/aamas2027/configs/marla_full_seed101_stage_b.yaml
```

`marla run --resume` computes the remaining rollouts itself (target minus
the checkpoint's own `environment_steps`) and refuses to run if the
checkpoint already meets or exceeds the target.

## Required evaluation suite (manifest.yaml v4)

Restricted to exactly three conditions after a real pilot measured Plan
Maker consultation latency (~73s mean, ~109s p95) -- see `VALIDATION.md`
and `manifest.yaml`'s `postponed_pending_feasibility` section.
`MARLA_FULL_ALWAYS_QUERY`/`PLAN_MAKER_ONLY`/`MARLA_FULL_BETA_ZERO`/
`MARLA_FULL_BETA_ONE` are implemented in `src/marla/evaluation` and
`scripts/evaluate_checkpoint.py --condition` accepts them, but they are
**not** part of the required suite below and should not be launched
without confirming their cost first (ALWAYS_QUERY/PLAN_MAKER_ONLY call the
Plan Maker on every step -- infeasible at the measured latency for any
non-trivial episode count).

## Reproduce: final ID checkpoint evaluation (MARLA_FULL_NORMAL)

`MARLA_FULL_NORMAL` uses the checkpoint's own learned query gate normally
-- no forced extra queries:

```bash
python research/aamas2027/scripts/evaluate_checkpoint.py \
  --run-dir runs/aamas2027_marla_full/marla-full-seed-101 \
  --condition MARLA_FULL_NORMAL \
  --seed-start 5001 --num-episodes 5 \
  --training-seed 101 --id-or-ood ID \
  --cache research/aamas2027/raw/advisory_cache.jsonl \
  --out-dir research/aamas2027/raw/eval/MARLA_FULL_NORMAL/seed-101
```

For `PPO_ONLY` (no Plan Maker, `--condition PPO_ONLY` implies `overrides=None`):

```bash
python research/aamas2027/scripts/evaluate_checkpoint.py \
  --run-dir runs/aamas2027_ppo_only/ppo-only-seed-101 \
  --condition PPO_ONLY --seed-start 5001 --num-episodes 5 \
  --training-seed 101 --id-or-ood ID \
  --out-dir research/aamas2027/raw/eval/PPO_ONLY/seed-101
```

## Reproduce: OOD evaluation (harder-DMZ scenario)

```bash
python research/aamas2027/scripts/evaluate_checkpoint.py \
  --run-dir runs/aamas2027_marla_full/marla-full-seed-101 \
  --condition MARLA_FULL_NORMAL \
  --scenario "$(pwd)/NASimEmu/scenarios/sm_entry_dmz_three_subnets.v2.yaml" \
  --seed-start 6001 --num-episodes 3 \
  --training-seed 101 --id-or-ood OOD \
  --cache research/aamas2027/raw/advisory_cache.jsonl \
  --out-dir research/aamas2027/raw/eval/MARLA_FULL_NORMAL__OOD_dmz_three_subnets/seed-101
```

Repeat with `sm_entry_user_three_subnets.v2.yaml` and output directory
suffix `OOD_user_three_subnets` for the second OOD scenario, and with
`--condition PPO_ONLY` (no `--cache` needed) for the PPO_ONLY baseline.

## Reproduce: the main inference ablation (MARLA_FULL_NO_QUERY)

Forces q=0 for the whole episode -- **zero Plan Maker calls**, the trained
PPO policy otherwise unchanged. This is an evaluation-time ablation of a
MARLA-trained checkpoint, not an independently trained baseline:

```bash
python research/aamas2027/scripts/evaluate_checkpoint.py \
  --run-dir runs/aamas2027_marla_full/marla-full-seed-101 \
  --condition MARLA_FULL_NO_QUERY \
  --seed-start 5001 --num-episodes 5 \
  --training-seed 101 --id-or-ood ID \
  --out-dir research/aamas2027/raw/eval/MARLA_FULL_NO_QUERY/seed-101
```

(No `--cache` needed -- this condition never calls the Plan Maker.)

## Aggregation and figures

```bash
python research/aamas2027/analyze.py
python research/aamas2027/scripts/make_figures.py
```

Both skip (with a printed `found N/M seeds` line) whatever hasn't run yet
rather than fabricating a result from partial data. Re-run any time --
neither script mutates a raw run directory.

## Raw run directories

Not copied into this tree (would duplicate/stale multi-GB run directories).
`manifest.yaml`'s `conditions.<NAME>.runs.<seed>.run_dir` is the
authoritative path for every trained run, e.g.
`runs/aamas2027_ppo_only/ppo-only-seed-101/`. Evaluation-harness output
(final ID/OOD/ablation episodes+decisions) lives under
`research/aamas2027/raw/eval/<condition>/seed-<seed>/`.
