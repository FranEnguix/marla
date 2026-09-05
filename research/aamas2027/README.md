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

## Reproduce: final checkpoint evaluation (NORMAL condition, 20 ID seeds)

```bash
python research/aamas2027/scripts/evaluate_checkpoint.py \
  --run-dir runs/aamas2027_marla_full/marla-full-seed-101 \
  --condition MARLA_FULL_NORMAL \
  --seed-start 5001 --num-episodes 20 \
  --training-seed 101 --id-or-ood ID \
  --cache research/aamas2027/raw/advisory_cache.jsonl \
  --out-dir research/aamas2027/raw/eval/MARLA_FULL_NORMAL/seed-101
```

For `PPO_ONLY` (no Plan Maker, `--condition PPO_ONLY` implies `overrides=None`):

```bash
python research/aamas2027/scripts/evaluate_checkpoint.py \
  --run-dir runs/aamas2027_ppo_only/ppo-only-seed-101 \
  --condition PPO_ONLY --seed-start 5001 --num-episodes 20 \
  --training-seed 101 --id-or-ood ID \
  --out-dir research/aamas2027/raw/eval/PPO_ONLY/seed-101
```

## Reproduce: OOD evaluation (10 seeds, harder-DMZ scenario)

```bash
python research/aamas2027/scripts/evaluate_checkpoint.py \
  --run-dir runs/aamas2027_marla_full/marla-full-seed-101 \
  --condition MARLA_FULL_NORMAL \
  --scenario "$(pwd)/NASimEmu/scenarios/sm_entry_dmz_three_subnets.v2.yaml" \
  --seed-start 6001 --num-episodes 10 \
  --training-seed 101 --id-or-ood OOD \
  --cache research/aamas2027/raw/advisory_cache.jsonl \
  --out-dir research/aamas2027/raw/eval/MARLA_FULL_NORMAL__OOD_dmz_three_subnets/seed-101
```

Repeat with `sm_entry_user_three_subnets.v2.yaml` and output directory
suffix `OOD_user_three_subnets` for the second OOD scenario.

## Reproduce: a trust/advice ablation (BETA_ZERO)

```bash
python research/aamas2027/scripts/evaluate_checkpoint.py \
  --run-dir runs/aamas2027_marla_full/marla-full-seed-101 \
  --condition MARLA_FULL_BETA_ZERO \
  --seed-start 5001 --num-episodes 20 \
  --training-seed 101 --id-or-ood ID \
  --cache research/aamas2027/raw/advisory_cache.jsonl \
  --out-dir research/aamas2027/raw/eval/MARLA_FULL_BETA_ZERO/seed-101
```

Pass the SAME `--cache` path across every ablation on the same checkpoint
(NORMAL/NO_QUERY/ALWAYS_QUERY/BETA_ZERO/BETA_ONE) so paired ablations reuse
identical Plan Maker responses wherever their trajectories haven't yet
diverged (see AUDIT.md / `src/marla/evaluation/advisory_cache.py`).

For `PLAN_MAKER_ONLY` (a freshly-initialized scaffold, no `checkpoint.pt`
needed):

```bash
python research/aamas2027/scripts/evaluate_checkpoint.py \
  --run-dir runs/aamas2027_marla_full/marla-full-seed-101 \
  --condition PLAN_MAKER_ONLY --fresh-scaffold-seed 101 \
  --seed-start 5001 --num-episodes 20 \
  --training-seed 101 --id-or-ood ID \
  --cache research/aamas2027/raw/advisory_cache.jsonl \
  --out-dir research/aamas2027/raw/eval/PLAN_MAKER_ONLY/seed-101
```

(`--run-dir` here only supplies `config.yaml` for policy/scenario shape --
`PLAN_MAKER_ONLY` never loads that run's `checkpoint.pt`; any MARLA_FULL
run's config works.)

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
