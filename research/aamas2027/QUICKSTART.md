# AAMAS 2027 Reproduction Quickstart

Last verified with git commit: `6c7839530f2a1b5a75f8b1884ff2c61804cc21d2`

For the full experimental reasoning, see `AUDIT.md` (what the codebase
does and its limitations), `manifest.yaml` (the exact condition/seed/scenario
matrix), and `VALIDATION.md` (checks run before any long training job).
This page is deliberately shorter and command-oriented.

## 1. What this reproduces

Three conditions, all on the scenario `NASimEmu/scenarios/sm_entry_dmz_two_subnets.v2.yaml`:

- **PPO_ONLY** -- recurrent PPO alone, no Plan Maker. Independently trained.
- **MARLA_FULL** -- the same PPO policy plus a learned query gate (when to
  consult) and a learned trust head (how much to weigh the advice).
  Independently trained, paired with PPO_ONLY on the same seeds.
- **MARLA_FULL_NO_QUERY** -- **not** a third trained agent. It is a
  frozen, already-trained MARLA_FULL checkpoint evaluated with the query
  gate forced to `q=0` for the whole episode. It never calls the Plan
  Maker and never takes a gradient step.

Paper-scale seeds are **101, 202, 303** (paired: the same seed trains both
PPO_ONLY and MARLA_FULL). The current checked-in training budget is
**5,000 environment steps per seed** (`manifest.yaml`'s `staged_training.stage_a`
-- a real pilot measurement found the originally planned 50,000-step
budget would take on the order of two weeks per MARLA_FULL seed on the
available hardware; see Section 15 and `VALIDATION.md`). If a later
revision of `manifest.yaml`/the configs under `configs/` changes this
budget, that checked-in value is authoritative, not this page.

## 2. Hardware expectations

- **PPO_ONLY is cheap.** A full 5,000-step run takes a few minutes.
- **MARLA_FULL's runtime is dominated by Plan Maker inference**, not by
  PPO or the NASimEmu simulation.
- The reference setup used for all measurements in this repository is a
  **single NVIDIA T4** with the Plan Maker running
  **Qwen/Qwen2.5-1.5B-Instruct** locally (`transformers`, bfloat16).
- Representative measured Plan Maker consultation latency: **mean 73.3s,
  p50 91.5s, p95 108.7s** (`manifest.yaml`'s `plan_maker_profiling` /
  `pilot_measurements`), ranging from ~8s (few legal actions, early in an
  episode) to >100s (many legal actions, later in an episode).
- **A full MARLA_FULL seed can therefore take many hours.** Exact runtime
  depends heavily on the *learned* query rate (not fixed in advance) and
  on your hardware -- do not treat any number on this page as a guarantee.

## 3. Installation

```bash
git clone git@github.com:FranEnguix/marla.git
cd marla
```

MARLA requires **Python 3.10.x exactly** (`>=3.10,<3.11` -- checked at CLI
startup). From the repository root, with a 3.10 interpreter active:

```bash
pip install -e ".[dev,local-lm]"
```

`dev` gives you `pytest`; `local-lm` gives you the real local Plan Maker
backend (`transformers`, `accelerate`) needed for MARLA_FULL. This
research program also needs NASimEmu, vendored inside this repository at
`NASimEmu/` (no separate install step).

Verify the install and which device/dependency versions were resolved:

```bash
marla version
```

Verify CUDA is actually visible to PyTorch (MARLA_FULL is impractical on
CPU given Section 2's latencies):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Run the test suite (full suite includes real local-model integration
tests and takes several minutes):

```bash
pytest -q
```

## 4. Required environment variables

Local-mode agents authenticate to an embedded XMPP server with these
passwords (any value works locally -- **never commit real credentials**;
the examples below use an obvious placeholder):

| Variable | Needed for |
|---|---|
| `MARLA_RL_ORCHESTRATOR_PASSWORD` | PPO_ONLY and MARLA_FULL (every run) |
| `MARLA_GATEKEEPER_PASSWORD` | MARLA_FULL only |
| `MARLA_PLAN_MAKER_1_PASSWORD` | MARLA_FULL only |

```bash
export MARLA_RL_ORCHESTRATOR_PASSWORD=changeme
export MARLA_GATEKEEPER_PASSWORD=changeme
export MARLA_PLAN_MAKER_1_PASSWORD=changeme
```

## 5. Reproduce one PPO baseline run

```bash
export MARLA_RL_ORCHESTRATOR_PASSWORD=changeme
marla run research/aamas2027/configs/ppo_only_seed101.yaml
```

Output lands in `runs/aamas2027_ppo_only/ppo-only-seed-101/`:

| File | Contents |
|---|---|
| `config.yaml` | The fully resolved configuration this run actually used |
| `metadata.json` | git commit, `config_hash`, dependency versions, seed, device |
| `episodes.csv` | One row per episode (training + periodic eval) |
| `decisions.csv` | One row per environment step/compound decision |
| `updates.csv` | One row per PPO gradient update |
| `summary.json` | Aggregate run summary |
| `checkpoint.pt` | Final policy/optimizer state |

Summarize and plot it:

```bash
marla summarize runs/aamas2027_ppo_only/ppo-only-seed-101
```

## 6. Reproduce one MARLA_FULL run

```bash
export MARLA_RL_ORCHESTRATOR_PASSWORD=changeme
export MARLA_GATEKEEPER_PASSWORD=changeme
export MARLA_PLAN_MAKER_1_PASSWORD=changeme
marla run research/aamas2027/configs/marla_full_seed101.yaml
```

**This is slow.** Every step the learned query gate decides to consult
invokes a real local LLM generation call (Section 2). Output lands in
`runs/aamas2027_marla_full/marla-full-seed-101/`, same file layout as
Section 5.

### Resuming instead of restarting

`marla run --resume` loads a checkpoint's policy/optimizer state and
**restores torch's global RNG state and continues episode-seed
progression** (rather than repeating the same episode sequence), then
trains only the remaining steps to the *new* config's
`policy.ppo.total_environment_steps`:

```bash
marla run --resume runs/aamas2027_marla_full/marla-full-seed-101/checkpoint.pt \
  research/aamas2027/configs/marla_full_seed101_stage_b.yaml
```

(`marla_full_seed101_stage_b.yaml` is not checked in as of this commit --
create it by copying the Stage A config and changing
`policy.ppo.total_environment_steps` and `experiment.run_id`. Extending
past the checked-in budget is a deliberate decision documented in
`manifest.yaml`'s `staged_training`, not something this page decides for
you.)

## 7. Run the paper-scale paired seeds

**No batch launcher script exists in this repository as of this commit.**
`research/aamas2027/scripts/` contains per-checkpoint evaluation and
analysis tools (Sections 8-10), not a training launcher. Run the six
configs directly:

```bash
export MARLA_RL_ORCHESTRATOR_PASSWORD=changeme
marla run research/aamas2027/configs/ppo_only_seed101.yaml
marla run research/aamas2027/configs/ppo_only_seed202.yaml
marla run research/aamas2027/configs/ppo_only_seed303.yaml

export MARLA_GATEKEEPER_PASSWORD=changeme
export MARLA_PLAN_MAKER_1_PASSWORD=changeme
marla run research/aamas2027/configs/marla_full_seed101.yaml
marla run research/aamas2027/configs/marla_full_seed202.yaml
marla run research/aamas2027/configs/marla_full_seed303.yaml
```

The three `marla_full_*` runs are the expensive ones (Section 2) and are
run **sequentially** here deliberately -- `manifest.yaml` records that
these were launched one at a time on a single GPU, not in parallel. As of
this commit, only seed 101 has completed for either condition; seeds 202
and 303 have not yet been run.

## 8. Final ID evaluation

Fixed evaluation seeds (`manifest.yaml`'s `evaluation_seed_sets.final_id`):
**5 seeds starting at 5001** (5001-5005), identical across all three
conditions.

```bash
python research/aamas2027/scripts/evaluate_checkpoint.py \
  --run-dir runs/aamas2027_ppo_only/ppo-only-seed-101 \
  --condition PPO_ONLY --seed-start 5001 --num-episodes 5 \
  --training-seed 101 --id-or-ood ID \
  --out-dir research/aamas2027/raw/eval/PPO_ONLY/seed-101

python research/aamas2027/scripts/evaluate_checkpoint.py \
  --run-dir runs/aamas2027_marla_full/marla-full-seed-101 \
  --condition MARLA_FULL_NORMAL --seed-start 5001 --num-episodes 5 \
  --training-seed 101 --id-or-ood ID \
  --cache research/aamas2027/raw/advisory_cache.jsonl \
  --out-dir research/aamas2027/raw/eval/MARLA_FULL_NORMAL/seed-101

python research/aamas2027/scripts/evaluate_checkpoint.py \
  --run-dir runs/aamas2027_marla_full/marla-full-seed-101 \
  --condition MARLA_FULL_NO_QUERY --seed-start 5001 --num-episodes 5 \
  --training-seed 101 --id-or-ood ID \
  --out-dir research/aamas2027/raw/eval/MARLA_FULL_NO_QUERY/seed-101
```

`MARLA_FULL_NO_QUERY` loads the same MARLA_FULL checkpoint, forces `q=0`
for the whole episode, makes **zero** Plan Maker calls, and never takes a
gradient step -- no `--cache` is needed since it never consults. Repeat
all three commands for seeds 202/303 once those checkpoints exist.

## 9. OOD evaluation

The two OOD scenarios actually configured in `manifest.yaml`
(`evaluation_seed_sets.final_ood_*`, verified loadable/steppable against
the trained architecture in `scenario_manifest.csv`): **3 seeds starting
at 6001** (6001-6003) for each.

```bash
python research/aamas2027/scripts/evaluate_checkpoint.py \
  --run-dir runs/aamas2027_marla_full/marla-full-seed-101 \
  --condition MARLA_FULL_NORMAL \
  --scenario "$(pwd)/NASimEmu/scenarios/sm_entry_dmz_three_subnets.v2.yaml" \
  --seed-start 6001 --num-episodes 3 \
  --training-seed 101 --id-or-ood OOD \
  --cache research/aamas2027/raw/advisory_cache.jsonl \
  --out-dir research/aamas2027/raw/eval/MARLA_FULL_NORMAL__OOD_dmz_three_subnets/seed-101

python research/aamas2027/scripts/evaluate_checkpoint.py \
  --run-dir runs/aamas2027_marla_full/marla-full-seed-101 \
  --condition MARLA_FULL_NORMAL \
  --scenario "$(pwd)/NASimEmu/scenarios/sm_entry_user_three_subnets.v2.yaml" \
  --seed-start 6001 --num-episodes 3 \
  --training-seed 101 --id-or-ood OOD \
  --cache research/aamas2027/raw/advisory_cache.jsonl \
  --out-dir research/aamas2027/raw/eval/MARLA_FULL_NORMAL__OOD_user_three_subnets/seed-101
```

Repeat with `--condition PPO_ONLY --run-dir runs/aamas2027_ppo_only/ppo-only-seed-101`
(no `--cache` needed) for the PPO_ONLY baseline on each scenario.

## 10. Generate the reported analyses/figures

```bash
# Cross-run aggregation (bootstrap CIs across training seeds) -> research/aamas2027/aggregate/
python research/aamas2027/analyze.py

# The 4 paper figures (PDF+PNG) -> research/aamas2027/figures/
python research/aamas2027/scripts/make_figures.py

# Per-seed diagnostics, no aggregation across seeds:
python research/aamas2027/scripts/interim_report.py \
  --run-dir runs/aamas2027_marla_full/marla-full-seed-101

python research/aamas2027/scripts/final_id_comparison.py --seed 101
```

`analyze.py` and `make_figures.py` both skip (printing `found N/M seeds`)
whatever hasn't been run yet rather than fabricating a result, and never
mutate a raw run directory. **`research/aamas2027/tables/` currently has
no generator script** -- it exists as a target directory but nothing in
this repository populates it as of this commit.

## 11. Key metrics

- **`goal_success`** -- whether the episode ended with every sensitive
  host at root access.
- **`benchmark_return`** -- the NASimEmu task return (identical to
  `nasimemu_return`). **This is the metric to compare across conditions.**
- **`training_return`** -- `benchmark_return` minus consultation-cost
  deductions (only differs from `benchmark_return` for
  MARLA_FULL/MARLA_FULL_NORMAL). This is what PPO actually optimizes, not
  a task-performance number to compare baseline vs. assisted on.
- **`steps_to_goal`** -- environment steps to success, only defined for
  successful episodes.
- **query_rate** -- fraction of decisions where the query gate consulted
  the Plan Maker (`decisions.csv`'s `queried` column; `updates.csv`'s
  `actual_query_rate` is the same thing aggregated per rollout).
- **consultations_per_successful_episode** -- Plan Maker calls divided by
  successful episodes, a cost-of-success measure.
- **normalized policy entropy** -- `H(pi) / log(legal_action_count)`,
  comparable across states with different numbers of legal actions.
- **`beta`** -- the learned per-step trust weight applied to Plan Maker
  advice (0 = ignore it, 1 = full weight).
- **`advice_changed_top_action`** -- whether accepting the Plan Maker's
  advice changed which action the base policy would otherwise have
  picked.
- **Plan Maker latency** -- `decisions.csv`'s `response_latency_ms`,
  end-to-end round trip including any schema-correction retries.
- **PPO approximate KL / clip fraction / explained variance** --
  `updates.csv` columns; standard PPO health diagnostics (see Section 15
  and `docs/metrics.rst` for interpretation).

## 12. Reproducibility metadata

Every run directory (Section 5) preserves everything needed to identify
exactly what produced it:

- **Resolved config**: `config.yaml`
- **Git SHA**: `metadata.json`'s `git_commit`
- **Config hash**: `metadata.json`'s `config_hash` (also independently
  recomputable from `config.yaml`)
- **Dependency versions**: `metadata.json`'s `dependency_versions`
  (torch, torch_geometric, spade, pydantic, typer) and `marla_version`
- **Training seed**: `metadata.json`'s `seed`, and `config.yaml`'s
  `experiment.seed`
- **Model name / prompt version / knowledge version**: `metadata.json`'s
  `plan_maker` block (MARLA_FULL runs only; `null` for PPO_ONLY)
- **Checkpoint**: `checkpoint.pt` in the same directory
- **Evaluation seeds**: not per-run metadata -- these are fixed constants
  in `research/aamas2027/manifest.yaml`'s `evaluation_seed_sets`, shared
  across every checkpoint by design

## 13. Minimal reproduction vs full reproduction

### Minimal sanity reproduction (fits in well under an hour)

```bash
export MARLA_RL_ORCHESTRATOR_PASSWORD=changeme
marla run research/aamas2027/configs/ppo_only_seed101.yaml
marla summarize runs/aamas2027_ppo_only/ppo-only-seed-101
```

This alone exercises the full training/metrics pipeline, but never touches
the Plan Maker (PPO_ONLY has no Gatekeeper/Plan Maker at all). Training a
real MARLA_FULL checkpoint cannot be shortcut below roughly an hour given
Section 2's latencies. If a MARLA_FULL checkpoint is already on disk (e.g.
`runs/aamas2027_marla_full/marla-full-seed-101/`), a reviewer with limited
compute can instead sanity-check the assisted path cheaply by running only
the `MARLA_FULL_NO_QUERY` evaluation (Section 8) -- it loads that
checkpoint but makes zero Plan Maker calls -- and, budget permitting, the
`MARLA_FULL_NORMAL` evaluation at a reduced `--num-episodes` (e.g. `2`) to
see a handful of real consultations without committing to the full 5.

### Paper reproduction

1. Section 7 (all six 5,000-step training runs)
2. Section 8 (final ID evaluation, all three conditions, all three seeds)
3. Section 9 (OOD evaluation, both scenarios, all three seeds)
4. Section 10 (aggregation and figures)

## 14. Expected output checklist

- [ ] `runs/aamas2027_ppo_only/ppo-only-seed-<seed>/` exists, `status: completed` in `metadata.json`
- [ ] `runs/aamas2027_marla_full/marla-full-seed-<seed>/` exists, `status: completed`
- [ ] `checkpoint.pt` present in each
- [ ] `summary.json` present in each
- [ ] `research/aamas2027/raw/eval/<condition>/seed-<seed>/episodes.csv` and `decisions.csv` exist for each evaluated condition
- [ ] `research/aamas2027/aggregate/*.csv` populated (non-empty) after `analyze.py`
- [ ] `research/aamas2027/figures/figure{1,2,3,4}_*.{png,pdf}` exist after `make_figures.py`

## 15. Known runtime caveat

**MARLA_FULL's wall-clock cost is dominated by local language-model
consultation latency and by the learned query frequency, not by PPO or
the simulator.** A real, controlled profiling pass
(`research/aamas2027/scripts/profile_plan_maker.py`,
`manifest.yaml`'s `plan_maker_profiling`) confirmed there is no fixable
bottleneck behind the measured ~73s mean latency -- no CPU fallback, the
model is loaded once, its KV cache is active, and it already stops
generating almost exactly when a valid response is complete. This is a
genuine cost of running a ~1.5B-parameter model on a single T4 for
variable-length structured output, not an implementation defect. Do not
assume a full MARLA_FULL run will finish quickly, and do not treat the
per-seed hour estimates anywhere in this repository as guaranteed --
`manifest.yaml`'s `pilot_measurements` are the actual basis for them, and
the real, observed query rate can only be known after a run completes.
