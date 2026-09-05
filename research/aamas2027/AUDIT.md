# MARLA repository audit

Audited commit: `4e3c7ed411ee51890a7f857c9b0bb4b52c44aa3b`. Read-only inspection,
no code changed while writing this document. All file:line references were
verified against this commit; re-verify before trusting them against a later
one.

This document exists to ground every later engineering decision in what the
code actually does today, not in what a spec or README claims it does. Where
the two disagree, the code wins and the disagreement is called out.

## 1. Current training/evaluation pipeline

`marla run <config.yaml>` (`src/marla/cli.py:144-265`) is the only entry
point that trains. Local mode (`runtime/local.py`): builds one
`NasimEmuAdapter` bound permanently to `config.environment.scenario`, one
`RecurrentPolicy` + `Adam` optimizer (`learning/trainer.py:52-57`), then
`run_training_loop` (`trainer.py:60-200`) drives `num_rollouts =
ceil(total_environment_steps / rollout_steps)` iterations of:

1. `RolloutCollector.collect(rollout_steps)` — one compound decision (query
   gate → optional Plan Maker consultation → advice blend → action) per
   environment step, `rollout.py:194-317`.
2. `compute_gae(...)` on `training_reward` (NASimEmu reward minus
   consultation cost, not raw NASimEmu reward) — `trainer.py:114-122`.
3. `optimize(...)` — `ppo.py:244-269` — `epochs` shuffled passes over
   episode-bounded sequence chunks, one optimizer step per minibatch.
4. Optionally, every `eval_every_rollouts` rollouts, `eval_episodes`
   deterministic (greedy, `torch.no_grad()`) episodes via
   `run_evaluation_episodes` (`rollout.py:445-485`), tagged `is_eval=True`.

"Environment steps" == number of compound decisions == number of
`StepRecord`s, including `finish` (which never touches the simulator).
Eval-episode steps are **not** counted toward the training step budget.

**There is no `evaluate` command and no way to resume or evaluate a saved
checkpoint anywhere in `src/`.** `cli.py` defines exactly `init`, `run`,
`validate`, `summarize`, `version` (`cli.py:99,144,268,286,363`). `run`
accepts only `config_path`, `--agent`, `--debug` — no `--checkpoint`, no
`--scenario` override, no eval-only mode. `summarize` only reads
`summary.json`/`metadata.json`/CSVs and plots them; it never touches
`checkpoint.pt`. `load_checkpoint` (`learning/checkpoint.py:42-56`) is
called from **nowhere in `src/`**, only from `tests/test_checkpoint.py`.
Everything in Phases 3, 4, 6, 7 of the requested program that depends on
"load a checkpoint and run it somewhere else" is therefore new
infrastructure, not existing capability being reused.

## 2. Existing metrics

Four artifacts per run, all in `src/marla/metrics/writer.py`, written once at
the end of `marla run` by `write_run_artifacts` (`writer.py:379-424`):

- **`episodes.csv`** — one row per episode (training then eval), columns
  `run_id, variant, episode_id, seed, scenario, goal_success, nasimemu_return,
  training_return, benchmark_return, environment_steps, rl_decisions,
  steps_to_goal, episode_seconds, consultation_count, consultation_cost,
  schema_rejection_count, finish_reason, rollout, is_eval`
  (`writer.py:44-49`). `benchmark_return` is a literal duplicate of
  `nasimemu_return` (`writer.py:164`), not an independent quantity.
- **`decisions.csv`** — one row per compound decision, gated on
  `metrics.record_decisions` (default `true`), columns `run_id, episode_id,
  environment_step, observation_id, legal_action_count, query_probability,
  queried, consultation_cost, request_id, response_status,
  response_latency_ms, schema_revision_count, base_policy_entropy,
  base_top_two_margin, base_top_action_id, plan_maker_top_action_id,
  final_top_action_id, selected_action_id, selected_action_base_rank,
  selected_action_plan_maker_rank, beta, alpha, advice_changed_top_action,
  action_success, nasimemu_reward, training_reward, terminated, truncated,
  artifact_path` (`writer.py:51-58`). **Three columns are hard-coded empty
  forever**: `schema_revision_count`, `action_success`, `artifact_path`
  (`writer.py:117,131,136` — the Gatekeeper never reports its retry count
  back to the RL Orchestrator, and NASimEmu exposes no per-action success
  signal, so these are honest placeholders, not bugs).
- **`updates.csv`** — one row per PPO gradient step (not per rollout):
  `run_id, update, environment_steps, policy_loss, value_loss, query_entropy,
  action_entropy, approximate_kl, clip_fraction, explained_variance,
  gradient_norm, learning_rate, mean_beta, mean_query_probability,
  actual_query_rate, mean_nasimemu_reward, mean_training_reward,
  elapsed_training_seconds, checkpoint_id` (`writer.py:60-65`). All PPO
  diagnostics the research program asks for (Phase 8's PPO-diagnostics list)
  are **already computed for real** in `ppo.py:224-241`, not placeholders.
  `checkpoint_id` is hard-coded `None` (`trainer.py:166`) since there is only
  ever one checkpoint per run.
- **`summary.json`** — 28 aggregate keys including Wilson 95% CI on goal
  success rate, latency percentiles, advice-acceptance rate
  (`writer.py:270-313`).
- **`metadata.json`** — run-level provenance: `marla_version`,
  `python_version`, `dependency_versions`, `git_commit`, `config_hash`,
  `device_requested/resolved`, scenario path, Plan Maker
  `model_name/prompt_version/knowledge_version`, start/end time, status
  (`writer.py:317-362`). **No token counts, no dollar cost, no cache
  statistics anywhere in the codebase.**

`docs/metrics.rst` matches the code on every plot filename (15/15) and every
"columns include..." claim it makes; it simply doesn't enumerate several
implemented-but-undocumented fields (full diff is in the per-file audit
transcript this document is built from). Nothing is documented but
unimplemented.

15 plots exist today (`metrics/plots.py`, listed in `generate_plots()`);
several of Phase 11's required figures (learning curves across training
seeds with uncertainty bands, generalization across scenarios, the
performance-consultation frontier, trust/advice-robustness comparison) do
not exist because they need cross-run aggregation, which nothing in the
current single-run `summarize` command does.

## 3. Existing ability to load a checkpoint and evaluate it on another scenario

**Architecturally possible, operationally absent.**

- `learning/checkpoint.py` saves exactly `policy_state_dict,
  optimizer_state_dict, update_count, environment_steps, config_hash`
  (`checkpoint.py:30-39`). No RNG state, no scenario identity, no git SHA, no
  policy hyperparameters, no `consultation_enabled` flag, no format version.
  `config_hash` is stored but **never compared against anything** at load
  time — there is no compatibility check at all beyond PyTorch's own
  strict-by-default `load_state_dict` (which will simply throw a shape/key
  mismatch error if you build the wrong policy shape, e.g. loading an
  assisted checkpoint's `query_gate.*`/`trust_head.*`/`advice_scale.*`
  weights into a baseline-constructed policy).
- `run_evaluation_episodes(policy, adapter, run_id, num_episodes, seed_start,
  consultation_enabled, consultation_cost, consult_fn)`
  (`learning/rollout.py:445-485`) is the one genuinely reusable evaluation
  primitive: `policy.eval()`, `torch.no_grad()`, deterministic (argmax)
  action/query selection, and — crucially — it already takes `adapter` as a
  parameter, so it is scenario-agnostic at the function level. Its own
  docstring is explicit that today's only caller (the between-rollout eval
  callback) reuses the *training* scenario and is "not a genuine held-out
  generalization test."
- The policy architecture is confirmed **fully size-invariant** across every
  scenario in `NASimEmu/scenarios/`: node features are a fixed 12-dim
  derived vector never keyed on scenario vocabulary
  (`environment/graph.py:28-64`), the action encoder uses 7 fixed action
  *types* plus a 2-dim parameter-presence indicator, never per-name
  embeddings (`learning/action_encoder.py:18-28`), and the GraphSAGE +
  mean-pool encoder is node-count agnostic by construction
  (`learning/graph_encoder.py`). A checkpoint trained on one scenario loads
  and runs, mechanically, against any other scenario in the repo with zero
  shape changes required.
- What's missing is pure wiring: a way to (a) reconstruct a `RecurrentPolicy`
  with the right `consultation_enabled` flag, (b) `load_checkpoint` into it,
  (c) build a second `NasimEmuAdapter` on a different scenario, (d) call
  `run_evaluation_episodes` against it, deterministically, with no optimizer
  step ever taken. None of this exists as a CLI command, library function, or
  script today — it has to be built (see manifest/scripts design).

## 4. How consultation mode, query probability, beta, alpha, and consultation cost are implemented

- `consultation.mode` is a strict `Literal["disabled", "learned"]`
  (`config/models.py:108`). There is **no third value** for "always query,"
  "random query," or "entropy-threshold query" — Conditions C, E, F (Phase 3)
  do not exist as trainable configs today and require a schema change.
- **Query probability** `p_t^q = sigmoid(MLP([z_t, H_t^0, Δ_t^0, N_t, κ]))`
  is a genuinely learned network output (`learning/query_gate.py:28-43`,
  driven by `decision.compute_query_probability`, `decision.py:27-34`) — a
  function of the base policy's recurrent state, base entropy, top-two
  logit margin, the **raw unnormalized** legal-action count, and the
  configured consultation cost. It is sampled via `Bernoulli` at training
  time, thresholded at `>= 0.5` at deterministic eval time
  (`rollout.py:349-353`).
- **Consultation cost** (`consultation.cost`, default `0.0`) is subtracted
  from `training_reward` (never from `nasimemu_reward`/`benchmark_return`)
  exactly when `sampled_query` is `True` (`rollout.py:221-223, 365`). It also
  enters the query gate as a raw feature `κ` (`decision.py:33`), so the gate
  can in principle learn cost-sensitivity, not just observe it after the
  fact via the reward.
- **beta (trust)** = `sigmoid(MLP([z_t, S(c_t), A_t]))`, a genuinely learned,
  per-step network output in `(0,1)` (`learning/advice.py:97-112`), where
  `S(c)` is a 5-dim summary of the Plan Maker's clipped-logit-normalized
  scores and `A_t` is a 2-dim non-differentiable agreement/correlation
  feature against the base logits.
- **alpha (advice scale)** = `softplus(alpha_bar)`, a **single learned global
  scalar** (`nn.Parameter`, not per-step, not per-action), initialized so
  `alpha = softplus(0) ≈ 0.693` (`advice.py:115-123`).
- **Exact blend**: `final_logits = base_logits + beta * alpha *
  normalized_advice` (`learning/recurrent_policy.py:182`) — a residual
  additive adjustment in logit space. Rejected advice (schema_rejected)
  forces `beta = 0` as a constant (not the trust head's output), so the
  resulting distribution is numerically identical to the base policy while
  still reporting a real `alpha` for bookkeeping (`decision.py:66-72`).
- **The query decision is trained by PPO on a *joint* log-probability, not a
  separate REINFORCE head.** `joint_log_prob = log(q_t if queried else
  1-q_t) + log softmax(final_logits)[a_t]` (`decision.py:84-104`) is the
  *only* quantity PPO's clipped ratio uses (`ppo.py:186,190,197`); there is
  no separate advantage or baseline for the query decision. This has a
  precise, important consequence for Condition C (documented in full in
  §7 below and in the manifest): forcing the query probability to a fixed
  constant only produces a valid PPO update if it is forced **identically at
  both collection time (`rollout.py:346`) and replay time (`ppo.py:118`)** —
  both call the same `decision.compute_query_probability` choke point, so a
  single override there keeps them consistent by construction. Forcing it at
  only one site silently breaks the PPO ratio (the "old" and "new" joint
  log-probs would stop referring to the same distribution). No such override
  exists today.

## 5. Exact model-generation configuration and whether token counts are available

- Generation is **greedy, not sampled**: `do_sample=False` in the one and
  only `generate()` call (`models/local_backend.py:107-112`). No
  `temperature`, `top_p`, `top_k`, or `GenerationConfig` object is passed
  anywhere — with `do_sample=False` these would be ignored by HF regardless.
- No explicit RNG seeding for the LM in this file (no `torch.manual_seed`,
  no `transformers.set_seed`); Plan Maker determinism rests entirely on
  greedy decoding over fixed weights.
- `max_new_tokens` is a **floor**, not a fixed value: `max(configured_floor,
  estimated_minimum)`, where the estimate is a closed-form function of the
  number and token-length of the legal action IDs
  (`local_backend.py:80,83-88`).
- **Token counts do not exist anywhere in the codebase.** `BackendResponse`
  has exactly two fields, `raw_text` and `latency_ms`
  (`models/plan_maker_backend.py:16-19`). `input_length =
  inputs["input_ids"].shape[-1]` is computed in `local_backend.py:105` but
  used only to slice the generated output, then discarded — never recorded.
  There is no `input_tokens`/`output_tokens`/`usage` field on any message
  schema, `StepRecord`, or CSV column. **Adding Phase 8's
  `plan_maker_input_tokens`/`output_tokens`/`total_tokens` fields is new
  work end to end**: a new `BackendResponse` field populated from
  `input_length` and `output_ids.shape[-1] - input_length` (both already
  computed in `local_backend.py`, just not surfaced), threaded through
  `AdvisoryResponsePayload` → `ConsultationResult` → `StepRecord` → a new
  `decisions.csv` column.
- **Latency actually recorded is end-to-end round-trip time on the RL
  Orchestrator side** (`agents/advisory_client.py:116,119`), including XMPP
  transport and every correction retry — not pure model inference time. A
  separate, more precise `inference_latency_ms` is measured around the
  `generate()` call itself (`local_backend.py:91,116`) but is **currently
  computed and then discarded** — it reaches the response payload
  (`plan_maker.py:198`) but nothing downstream reads it.

## 6. Which NASimEmu scenarios are checkpoint-compatible for zero-shot evaluation

**All ten `NASimEmu/scenarios/*.v2.yaml` files are architecturally
checkpoint-compatible** (see §3) — MARLA's own graph/action encoding never
depends on `address_space_bounds` or per-scenario vocabulary, unlike the
bundled NASimEmu-agents baseline, whose raw observation width is `s.shape[1]
+ 1` and *does* change with `address_space_bounds` (`(5,10)` for the
`sm_*`/`md_*` scenarios vs `(10,10)` for `corp`/`uni` — confirmed by
instantiating both: observation width 30 vs 35). This is a genuine,
verified difference in robustness between MARLA's architecture and the
external baseline, worth stating as a positive result in its own right.

Full scenario inventory (`NASimEmu/scenarios/`, all procedurally generated
per-episode — subnet sizes, IDs, and sensitive-host assignment are re-rolled
on every `reset()`):

| Scenario | Subnets | Hosts/episode | Entry | Sensitive-host probability |
|---|---|---|---|---|
| `sm_entry_dmz_one_subnet.v2.yaml` | 2 | 3–6 | DMZ | **0.0 / 0.0 — see §7, critical finding** |
| `sm_entry_dmz_two_subnets.v2.yaml` | 3 | 5–11 | DMZ | service 0.7 |
| `sm_entry_dmz_three_subnets.v2.yaml` | 4 | 6–14 | DMZ | service 0.7, db 1.0 |
| `sm_entry_user_three_subnets.v2.yaml` | 4 | 6–14 | user subnet | service 0.7, db 1.0 |
| `md_entry_dmz_one_subnet.v2.yaml` | 2 | 7–11 | DMZ | **0.0 / 0.0 — same defect** |
| `md_entry_dmz_two_subnets.v2.yaml` | 3 | 13–21 | DMZ | service 0.7 |
| `md_entry_dmz_three_subnets.v2.yaml` | 4 | 17–27 | DMZ | service 0.7, db 1.0 |
| `md_entry_user_three_subnets.v2.yaml` | 4 | 17–27 | user subnet | service 0.7, db 1.0 |
| `corp.v2.yaml` | 8 | 9–36 | private_wifi | hr .3, employees .3, mgmt .5, db .3 |
| `uni.v2.yaml` | 9 | 13–50 | private_wifi | db .5, hpc .2, backup .9 |

No scenario file declares a `step_limit`; MARLA enforces its own
`max_episode_steps` (100 in both example configs) independent of NASimEmu.
Full scenario_manifest.csv (Phase 6) is derived from this table plus the
`ID_or_OOD` designation decided in §7/manifest.yaml.

## 7. Implementation limitations that would make a requested experiment scientifically invalid

These are the findings this audit exists to surface. None have been silently
worked around.

### 7.1 CRITICAL — the production training scenario has a vacuous objective

`examples/baseline.yaml` and `examples/assisted.yaml` both train on
`sm_entry_dmz_one_subnet.v2.yaml`. That scenario's DMZ and user subnets are
both configured with **sensitive-host probability 0.0**
(`sm_entry_dmz_one_subnet.v2.yaml:12-14`; identically for
`md_entry_dmz_one_subnet.v2.yaml:12-14`). Given
`nasim/envs/network.py:214-218`'s `all_sensitive_hosts_compromised()`
vacuously returns `True` over an empty sensitive-host set, **the objective is
already satisfied at `t=0`, every episode, before any action is taken.**
Verified empirically by instantiating the environment directly: `sensitive:
{}`, `goal_at_reset: True`.

Consequence: `finish` on step 1 always earns the full `completion_reward`
(+1.0) and never the premature-finish penalty. `goal_success_rate` and
`benchmark_return` on this scenario cannot distinguish a good policy from a
trivial one, because *every* policy that ever learns to call `finish`
achieves the ceiling — this is precisely the primary dependent variable the
paper's claim rests on, on precisely the scenario the example configs use.
Training PPO_ONLY, MARLA_FULL, and ALWAYS_CONSULT on this scenario as
Phase 6 literally instructs ("train on the current production training
scenario **unless inspection reveals a reason not to**") would produce a
headline learning curve and generalization table that cannot show the
claimed effect (learning when to consult, trading off cost against
performance) even if the effect is real, because the ceiling is reachable
for free from step one.

**Decision taken (documented here, not silently): the primary training/ID
scenario for this research program is changed from
`sm_entry_dmz_one_subnet.v2.yaml` to `sm_entry_dmz_two_subnets.v2.yaml`** —
the smallest scenario with a genuine, non-vacuous objective (service subnet
sensitive-host probability 0.7), same entry topology (DMZ), same action-type
vocabulary (10 actions), same size-invariant architecture applies unchanged.
Every PPO/architecture hyperparameter from the production example configs is
kept exactly as-is (Phase 2's instruction); only `environment.scenario`
changes. This is recorded explicitly in `manifest.yaml` and
`scenario_manifest.csv` so it is traceable, not silently assumed. If this
decision is unwanted, the alternative is to keep
`sm_entry_dmz_one_subnet.v2.yaml` and reinterpret the whole program as
measuring cost/query behavior under a free win — which contradicts the
stated claim, so it was not chosen.

### 7.2 CRITICAL — training-seed reproducibility is not actually guaranteed on the real run path

`torch.manual_seed` is called in exactly one place in `src/marla/`:
`learning/trainer.py:216`, inside `run_baseline_training`, a
convenience wrapper used only by tests
(`grep` confirms `runtime/local.py`/`runtime/distributed.py` — the only
paths `marla run` actually takes — never seed torch). Consequences:
- Policy weight initialization is **not** controlled by `experiment.seed` on
  a real `marla run`.
- All `torch.distributions.Categorical`/`Bernoulli` sampling (every
  stochastic action and query decision during training) draws from an
  **unseeded** global torch RNG.
- `reproducibility.deterministic_torch` (default `True`,
  `config/models.py:174-175`) is **dead configuration** — grep confirms it
  is never read anywhere in `src/marla`. No `torch.use_deterministic_algorithms`,
  no cuDNN determinism setting exists.
- Environment-level determinism is fine: `NasimEmuAdapter.reset(seed)` does
  seed Python's `random`/`numpy` globals before generating the scenario
  instance (`nasimemu_adapter.py:73-86`), and PPO's minibatch shuffling uses
  a private `random.Random` instance (`trainer.py:95`), not the global one.

This directly undermines Phase 2's central premise — "use the SAME paired
training seeds for all primary trainable conditions" only produces
paired, reproducible comparisons if `experiment.seed` actually controls
everything stochastic in a training run. **This is treated as a bug and
fixed** (not worked around): `runtime/local.py`/`runtime/distributed.py` now
call `torch.manual_seed(config.experiment.seed)` before constructing the
policy, mirroring what the test-only `run_baseline_training` already did,
and `reproducibility.deterministic_torch` is wired to
`torch.use_deterministic_algorithms(...)` when true. This is an additive
reproducibility fix, not an algorithm change — it changes *which* random
draws happen (making them a deterministic function of `experiment.seed`
instead of process-startup entropy), not the PPO/advice/query-gate math
itself. See VALIDATION.md for the check that this fix doesn't otherwise
alter baseline/assisted semantics.

### 7.3 No cross-scenario / checkpoint evaluation infrastructure

Documented fully in §3. Everything in Phases 3(C-F)/4/6/7 that requires
loading a checkpoint, overriding its query/trust/advice behavior, or running
it on a different scenario needs new code: a `marla evaluate`-style CLI path
or script, an `EvaluationOverrides`-style hook threaded through
`RolloutCollector`, and a small, explicit checkpoint compatibility check
(matching `consultation_enabled` at minimum, since PyTorch's own
`strict=True` load will otherwise fail with a confusing tensor-name error
rather than an actionable message).

### 7.4 No periodic checkpointing

Documented fully in §1/§3. Only one checkpoint is ever saved, at the very
end of a run (`writer.py:410-424`). Phase 7's periodic (0, 10k, ..., 100k)
evaluation points cannot be reconstructed after the fact from a finished
run's single final checkpoint. **Approach taken**: since the existing
between-rollout evaluation callback (`trainer.py:169-193`,
`run_evaluation_episodes`) already runs deterministic episodes at fixed
rollout intervals and records them into `episodes.csv` tagged with
`is_eval=True` and `rollout=<index>`, Phase 7's learning curves are produced
by **using that existing callback** (with `eval_episodes` set to the
required periodic seed-set size and `eval_every_rollouts` tuned so a rollout
boundary lands close to each 10k-step mark) rather than by adding new
periodic checkpoint files. This reuses existing, already-validated code
instead of adding a new persistence mechanism, at the cost of the periodic
points being *interpolated to the nearest rollout boundary* rather than
landing on the exact step count — `rollout_steps=512` means rollout
boundaries are at multiples of 512, so "10k steps" in practice means the
rollout ending at step 10240 (the 20th rollout), not exactly step 10000.
This is recorded in the manifest, not silently rounded away.

### 7.5 No fair external RL baseline without modifying the vendored agent

Documented fully in the NASimEmu-agents audit. Three material differences
make a naive head-to-head invalid: (1) MARLA pays ±1.0 for `finish`,
NASimEmu-agents' terminal action always pays exactly `0.0`
(`NASimEmu-agents/.../env.py:179` vs `finish.py:17-23`); (2)
NASimEmu-agents runs with `step_limit` enforced *inside* the env with
auto-reset and `random_init=True`, MARLA enforces its own external step
budget with `random_init=False`; (3) NASimEmu-agents' standard configuration
augments observations with the previous action as a one-hot block
(`augment_with_action=True` in every shipped run), which MARLA does not do
identically (MARLA feeds the previous action's *learned embedding* through
the GRU instead — related, not the same signal). None of these can be
reconciled by wrapper code alone without editing the vendored agent or the
shared `nasimemu/env.py`. Per Phase 9's own instruction ("if a fair
comparison cannot be implemented without materially changing the original
agent, document that rather than forcing an invalid comparison"), this is
documented rather than forced; see manifest.yaml's `external_baseline`
section for the precise, minimal set of upstream changes that *would* make
it fair, left unimplemented pending an explicit decision to touch vendored
code.

### 7.6 Consultation-cost validator asymmetry (minor, noted not fixed)

`ConsultationConfig`'s `cost >= 0` check only fires when `mode == "learned"`
(`config/models.py:112-124`); a `disabled` config could carry a negative
`consultation.cost` with no error (though it would have no effect, since
`RolloutCollector` never reads it when `consultation_enabled=False`). Noted
as a latent validation gap, not touched, since it has no behavioral
consequence for any experiment in this program and touching validators
unrelated to the research task risks scope creep on "preserve the existing
algorithm."

### 7.7 Compute reality

Single machine, one Tesla T4 GPU, 8 vCPUs, 31GB RAM — no cluster, no
multi-GPU parallelism available in this environment. Every assisted
(MARLA_FULL/ALWAYS_CONSULT/PLAN_MAKER_ONLY/E/F) run performs real local
`transformers` inference (Qwen2.5-1.5B-Instruct, greedy decoding) on
whichever steps are queried, on the same GPU used for policy
training/replay. Runs in this program are therefore necessarily sequential,
not parallel, and assisted runs are substantially slower wall-clock than
PPO_ONLY runs of the same step budget. This is why Phase 13's execution
priority order exists, and why deliverables distinguish "implemented" (code
exists and is validated) from "launched" (a run/sweep has been started
against real compute in the background) from "completed" (finished and its
artifacts exist on disk) — see `VALIDATION.md` and the final response for
which experiments are in which state at any given time.
