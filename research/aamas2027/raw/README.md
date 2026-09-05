# raw/

Deliberately does not contain copies of run directories -- see
`../README.md`'s "Raw run directories" section. Trained-run data lives at
the paths recorded in `../manifest.yaml` (e.g.
`runs/aamas2027_ppo_only/ppo-only-seed-101/`, relative to the repo root).

This directory holds only:

- `eval/<condition>/seed-<seed>/` -- output of
  `scripts/evaluate_checkpoint.py` (episodes.csv, decisions.csv,
  eval_metadata.json) for the final ID/OOD/ablation evaluation passes.
  Populated by running the commands in `../README.md`, not committed here
  in advance of those runs actually happening.
- `advisory_cache.jsonl` -- the content-addressed Plan Maker response cache
  shared across ablations on the same checkpoint (see
  `src/marla/evaluation/advisory_cache.py`). Safe to delete; it only ever
  saves redundant Plan Maker calls, never affects correctness.
