# Validation (Phase 12)

Audited commit for the checks below: see `manifest.yaml`'s `git_sha` (update
after the final commit that ran these checks). Every check here was
actually executed; none are asserted from reading the code alone.

## 1. Unit tests

Full suite: `306 passed, 1 skipped, 1 xfailed` (`pytest -q`, ~9 minutes,
includes real HF model integration tests). The skip/xfail are pre-existing
and unrelated to this program (a distributed-mode `pyjabber` limitation
documented in `README.md`). 30 of the 306 are new, added for this program:

- `tests/test_rollout.py` (+16): the `EvaluationOverrides` hook —
  `overrides=None` byte-identical to today's code; NO_QUERY never calls
  `consult_fn`; ALWAYS_QUERY always does; BETA_ZERO/BETA_ONE only affect
  `final_logits` when advice was actually obtained (a documented no-op
  otherwise); PLAN_MAKER_ONLY ignores the base policy for action choice and
  falls back to a seeded-reproducible random choice on `schema_rejected`;
  `advice_transformation != "identity"` raises `NotImplementedError`; same
  seed → identical initial observation across independently-constructed
  adapters (check 6); `nasimemu_reward` unaffected by `consultation.cost`
  (check 7).
- `tests/test_advisory_cache.py` (+5): key determinism, order-independence
  of `legal_action_ids`, content-sensitivity, on-disk persistence,
  idempotent `put`.
- `tests/test_direct_consult.py` (+6): accept-first-try, correct-then-accept,
  reject-after-max-revisions, cache hit avoids a second backend call
  (across different `episode_id`/`step`/`observation_id`, proving the cache
  key genuinely ignores those), different observation is a cache miss,
  backend exception routes through rejection rather than crashing.
- `tests/test_checkpoint_eval.py` (+6): config round-trip, correct episode
  count returned, **policy weights provably unchanged by evaluation**
  (check 10), scenario override actually changes the scenario used, a
  fresh scaffold policy needs no `checkpoint.pt` at all (PLAN_MAKER_ONLY's
  case), and a structural AST-based grep confirming no file in
  `src/marla/evaluation/` imports `torch.optim` or calls `.backward()`
  (check 9).
- `tests/test_local_backend.py` (+1): token counts match an independent
  tokenizer-based recount (check 8).
- `tests/test_config.py` (+1): regression test for a real bug found while
  building this program (see below).

**Bug found and fixed while building this program's test coverage**:
`marla.config.loader._is_secret_key` used raw substring matching
(`marker in lowered`), which redacted `max_new_tokens` into the string
`"***REDACTED***"` on any config that gets `redacted_config_dict`-dumped
(i.e. every `config.yaml` a real `marla run` ever wrote for an assisted
run) — "token" is a substring of "tokens". Reloading such a `config.yaml`
(exactly what the checkpoint-evaluation harness does) then failed Pydantic
validation with `int_parsing`. This would have silently broken on the very
first assisted-mode checkpoint this program tried to evaluate. Fixed by
matching whole underscore-separated segments instead of raw substrings
(`src/marla/config/loader.py`); regression test added.

**Pre-existing test fragility found and fixed**: `tests/test_rollout.py`'s
`test_collect_sets_bootstrap_value_when_stopped_mid_step_not_at_episode_end`
implicitly depended on whatever torch global RNG state happened to exist
when it ran (an unseeded `RecurrentPolicy()` construction), rather than
being hermetic — passed in isolation, failed once new, unrelated tests
that explicitly seed torch were added earlier in the same file. Fixed by
seeding it explicitly.

## 2. Tiny smoke experiments for every new mode

Real, non-mocked `marla run` executions (not unit tests) against
drastically reduced step budgets (1024 steps, `rollout_steps: 256` — 4
rollouts) on the actual `sm_entry_dmz_two_subnets.v2.yaml` ID scenario:

- **PPO_ONLY smoke** (`aamas2027_smoke_ppo_only/smoke-ppo-only-seed-101`):
  completed cleanly, 61 episodes, **goal_success mixed True/False (4/73
  successes across all episodes including eval passes)** — this is the
  concrete, empirical confirmation that switching the ID scenario away
  from `sm_entry_dmz_one_subnet.v2.yaml` fixed the vacuous-objective defect
  (AUDIT.md 7.1): a real policy is not trivially succeeding from step 0.
- **MARLA_FULL smoke** (`aamas2027_smoke_marla_full/smoke-marla-full-seed-101`):
  ran for real against the live local Qwen2.5-1.5B-Instruct backend on the
  sandbox's Tesla T4. Deliberately stopped partway (after ~27 of the
  planned 1024 steps) once it had produced enough real measurements to be
  worth reporting -- completing the full 1024-step smoke run at the
  observed rate would itself have become a multi-hour job, which is
  exactly the finding below. **This run is NOT a completed result**; it
  exists only to measure real Plan Maker latency/query-rate, and its
  checkpoint/metrics should not be treated as a trained MARLA_FULL policy.
  Measured, with a fresh (untrained) query gate:
  - 3 consultations completed cleanly, latencies **91.3s / 91.6s / 93.4s**
    (a 4th was in progress when stopped) -- all at 81 legal actions (a
    ~15400-character prompt), i.e. a late-episode observation on this
    scenario where most/all hosts have been discovered.
  - Query events at steps 14, 16, 21, 24 of the same episode -- **4
    queries in the first ~24 steps observed (~15-20% query rate)** for an
    untrained, randomly-initialized query gate.
  - **This is the critical, previously-unmeasured cost driver for this
    program**, reported in full in the final summary alongside this
    deliverable set: at ~92s/consultation and a ~15-20% query rate
    (expected to be an upper bound, since `consultation.cost: 0.10`
    specifically trains the gate to query less over time -- but early
    training, before the gate converges, could sustain a rate near this
    for a non-trivial fraction of a run), a full 50,000-step MARLA_FULL
    run could plausibly take many hours to multiple days of sequential
    wall-clock time on this single-GPU sandbox, dominated almost entirely
    by Plan Maker inference rather than PPO/environment compute. This was
    not knowable from code inspection alone and is exactly why this smoke
    test was run before proposing to launch the real pilot.

## 3/4. PPO_ONLY / MARLA_FULL semantics unchanged

No branch in `rollout.py`'s training-critical path was edited except the
new `overrides` parameter, which defaults to `None` and is never passed by
`marla.learning.trainer`/`runtime.local`/`runtime.distributed`. Verified
two ways: (a) the full pre-existing test suite (276 tests, unrelated to
this program) still passes unmodified after every change in this program;
(b) `test_overrides_none_matches_default_learned_behavior` directly diffs
`overrides=None` against an explicit all-default `EvaluationOverrides()`
instance and asserts identical `sampled_query` sequences.

## 5. Forced overrides affect only their intended component

Covered by the `tests/test_rollout.py` additions above: each override is
tested in isolation against a control, with the specific claim it makes
("never calls consult_fn," "only changes final_logits when advice exists,"
etc.) asserted directly rather than inferred.

## 6. Same evaluation seed -> same scenario realization across methods

Holds by construction: `NasimEmuAdapter.reset(seed)` is a pure function of
`(scenario, seed)`, independent of which policy calls it (AUDIT.md
section 8). `test_same_seed_produces_identical_initial_observation_across_independent_adapters`
verifies this directly for two independently-constructed adapters. The
evaluation harness (`checkpoint_eval.py`) always takes an explicit,
external `seed_start`, never derives it from anything policy- or
condition-specific.

## 7. `benchmark_return` unaffected by consultation cost

`nasimemu_return`/`benchmark_return` and `training_reward` are separate
accumulators (`rollout.py`: `training_reward = nasimemu_reward -
consultation_cost`, and `EpisodeSummary.nasimemu_return` only ever
accumulates `transition.nasimemu_reward`). Verified directly:
`test_nasimemu_reward_is_unaffected_by_consultation_cost` runs two
otherwise-identical collectors (same policy weights, same seed, same
forced ALWAYS_QUERY pattern) differing only in `consultation_cost` (0.0 vs
0.5) and asserts `nasimemu_reward` sequences are exactly equal while
`training_reward` sequences differ.

## 8. Token accounting

`test_local_backend_reports_token_counts_matching_the_tokenizer` recomputes
the expected input-token count independently via the same tokenizer and
asserts an exact match; output-token count is asserted `> 0` and `<=
max_new_tokens`; `total_tokens == input_tokens + output_tokens`. Threaded
end-to-end from `BackendResponse` through `AdvisoryResponsePayload`,
`ConsultationResult`, `StepRecord`, to three new (additive,
None-by-default) `decisions.csv` columns.

## 9. No evaluation performs optimizer steps

Structural, not just behavioral: `src/marla/evaluation/` contains no
`torch.optim` import anywhere (`test_evaluation_package_never_imports_torch_optim_or_calls_backward`
parses every file's AST and fails on either), and
`checkpoint_eval.evaluate_checkpoint` calls `load_checkpoint(...,
optimizer=None, ...)` -- there is no code path in this package capable of
constructing an optimizer or calling `.backward()`.

## 10. OOD evaluation does not retrain or reset checkpoint weights

`test_evaluate_checkpoint_never_mutates_the_loaded_policy_weights` snapshots
every parameter tensor before calling `evaluate_checkpoint(...,
policy_override=<that policy>)` and asserts bit-for-bit equality
(`torch.equal`) afterward. Combined with check 9 (no optimizer exists to
have stepped) and `evaluate_checkpoint` always constructing a *fresh*
`RecurrentPolicy` per call (never mutating a shared instance across
scenarios), this rules out both a gradient update and cross-scenario state
leakage.

## OOD scenario compatibility (Phase 6 pre-check)

Before writing `scenario_manifest.csv`'s OOD rows, both candidate OOD
scenarios were empirically loaded and stepped (20 steps each, untrained
policy, real `NasimEmuAdapter`+`RolloutCollector`, no errors):
`sm_entry_dmz_three_subnets.v2.yaml` (11 legal actions at step 0) and
`sm_entry_user_three_subnets.v2.yaml` (31 legal actions at step 0) both ran
cleanly, confirming the architecture-invariance argument in AUDIT.md
section 3/6 empirically rather than by inspection alone.

## Pilot run outcome (v2 -> v3 revision)

The 5,000-step pilot pair was actually launched: **PPO_ONLY seed 101
completed** (5000 steps, 197s wall-clock, 1510 episodes, 3.6% goal success
rate -- non-trivial and non-saturated, confirming the ID-scenario fix
works). **MARLA_FULL seed 101 was deliberately interrupted** after ~151 of
its planned 5000 steps, once it had produced a reliable latency/query-rate
sample (waiting for full completion at the observed rate would itself have
taken ~32 hours for just 5000 steps). Measured over 48 real consultations:

| | value |
|---|---|
| mean latency | 73.3s |
| p50 latency | 91.5s |
| p95 latency | 108.7s |
| min / max | 7.7s / 109.2s (bimodal: ~8s for ~11-action observations, ~90-109s for ~81-91-action observations) |
| query rate (untrained gate) | 31.8% (48/151 steps) |
| fraction of wall-clock on Plan Maker | 99.5% |

Extrapolated: a full 50,000-step MARLA_FULL run at this rate would need
roughly 325 hours (~13.5 days) of sequential wall-clock on this single-GPU
sandbox (two independent methods -- scaling the observed blended per-step
rate, and expected-calls x mean-latency -- agree to within 1%). **This is
the basis for the PILOT FIRST checklist verdict**:
task non-trivial (yes), initial success not saturated (yes, 3.6%), PPO
numerically stable (no NaN/divergence observed in the partial run), MARLA
actually queries (yes, 31.8%), **query rate low enough for the full
50k-step experiment to be feasible in this session (no)**.

Per this finding, the required evaluation/ablation suite was cut down
(manifest.yaml v3) to remove every experiment that calls the Plan Maker on
every or nearly every step (`MARLA_FULL_ALWAYS_QUERY`, `PLAN_MAKER_ONLY`)
and to descope `MARLA_FULL_BETA_ZERO`/`BETA_ONE` for this phase (kept
implemented, not launched). The required suite is now exactly `PPO_ONLY`,
`MARLA_FULL`, and the zero-Plan-Maker-call `MARLA_FULL_NO_QUERY` ablation.
Full reasoning, options, and recommendation: see the final summary
delivered alongside this revision.

## Status of this document

This file reflects validation and a real (deliberately interrupted) pilot
measurement performed **before** launching any full-scale (50k-step)
training run. No 50k-step run has completed as of this revision.
