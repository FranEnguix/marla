Configuration reference
=========================

Every MARLA experiment is described by one YAML file, validated against the
schema in :mod:`marla.config.models` (Pydantic, ``extra="forbid"`` --
unknown keys fail fast rather than being silently ignored). This page
documents every field and its accepted values; ``examples/baseline.yaml``
and ``examples/assisted.yaml`` are the same schema with inline comments,
production-realistic settings, and worked examples of both variants.

Top level
---------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Type / values
     - Notes
   * - ``schema_version``
     - ``"1.0"``
     - Only supported value today.
   * - ``experiment``
     - see `experiment`_
     -
   * - ``execution``
     - see `execution`_
     -
   * - ``device``
     - ``cpu`` \| ``gpu`` \| ``auto`` (default)
     - See `Device resolution`_.
   * - ``xmpp``
     - see `xmpp`_
     -
   * - ``environment``
     - see `environment`_
     -
   * - ``objective``
     - see `objective`_
     -
   * - ``policy``
     - see `policy`_
     -
   * - ``consultation``
     - see `consultation`_
     -
   * - ``rl_orchestrator``
     - see `Agent identities`_
     -
   * - ``gatekeeper``
     - see `Agent identities`_
     - Optional.
   * - ``agents``
     - list, see `Plan Maker agents`_
     - Default ``[]``.
   * - ``metrics``
     - see `metrics`_
     - Has defaults.
   * - ``reproducibility``
     - see `reproducibility`_
     - Has defaults.
   * - ``carbon``
     - see `carbon`_
     - Has defaults; ``enabled: false`` unless turned on explicitly.

experiment
~~~~~~~~~~

- ``name`` (str, required): experiment name; also the default run
  directory grouping under ``metrics.output_directory``.
- ``run_id`` (str, optional): unique ID for this specific run.
  **Mandatory** when ``execution.mode: distributed`` (every process in the
  same distributed run must share it); optional for local runs (falls back
  to ``name``).
- ``phase``: ``training`` (default) or ``evaluation``. Reserved for a
  future evaluation-only run mode; not yet consumed by the runtime.
- ``seed`` (int, required): base random seed. Episode seeds increment from
  this value as training progresses (see :class:`marla.learning.rollout.RolloutCollector`).

execution
~~~~~~~~~

- ``mode``: ``local`` or ``distributed``.

  - ``local``: every configured agent runs in one process, one shared
    ``asyncio`` event loop, over SPADE's embedded XMPP server. Zero
    external setup; ``xmpp.server`` can just be ``localhost``.
  - ``distributed``: one ``marla run ... --agent <alias-or-jid>`` process
    per agent (or group of agents), connecting to a real, externally
    reachable XMPP server. Requires ``experiment.run_id``. See
    :doc:`architecture`'s distributed-mode section.

xmpp
~~~~

- ``server`` (str, required): the XMPP domain/host every agent connects
  to. ``localhost`` for local mode's embedded server; a real server
  (Prosody, ejabberd, ...) for distributed mode, or for local *assisted*
  runs that need better reliability than the embedded server currently
  provides (see the repository README's known-limitations section).

environment
~~~~~~~~~~~

- ``mode``: ``simulation`` (only supported value).
- ``scenario`` (str, required): a NASimEmu scenario, in one of three forms
  (see :mod:`marla.scenarios.uri`):

  - ``marla://<filename>``, e.g.
    ``marla://sm_entry_user_three_subnets.solvable.v2.yaml`` -- a
    MARLA-owned, pre-validated scenario packaged with MARLA itself.
    Resolves identically regardless of the current working directory, the
    config file's location, or how MARLA is installed (source checkout,
    editable install, or a built wheel) -- never relative to a filesystem
    path. This is what ``marla init`` and every ``examples/``/
    ``research/aamas2027/configs/`` config use, and what a saved run
    directory's ``config.yaml`` keeps (never rewritten to a materialized
    path).
  - A value ending in ``.yaml`` -- a path to a static scenario file,
    resolved relative to the config file's directory if not absolute.
  - Any other string -- the name of a procedurally generated benchmark
    NASimEmu resolves at runtime.
- ``max_episode_steps`` (int > 0, required): episode truncation limit.
  NASimEmu itself never internally terminates a non-FINISH action (see
  :mod:`marla.environment.nasimemu_adapter`), so this is the only thing
  bounding an episode that never reaches FINISH.

objective
~~~~~~~~~

- ``type``: ``capture_target`` (only supported value) -- NASimEmu's native
  goal (all sensitive/value hosts compromised).
- ``description`` (str, required): free-text objective description, sent
  to the Plan Maker verbatim as context.
- ``completion_reward`` (float, default ``1.0``): reward for FINISH when
  the objective is satisfied.
- ``premature_finish_penalty`` (float, default ``-1.0``): fixed component
  of the reward for FINISH when the objective isn't satisfied.
- ``premature_finish_penalty_per_remaining_target`` (float, default
  ``0.0``): additional reward per sensitive/value target not yet at ROOT
  access (USER access does not count) at the moment of a premature FINISH.
  The two terms add: effective premature-FINISH reward is
  ``premature_finish_penalty + premature_finish_penalty_per_remaining_target
  * remaining_sensitive_targets``. Leaving this at its default ``0.0``
  preserves the old fixed-penalty-only behavior exactly. For a *pure*
  proportional penalty with no fixed component (e.g. -20 per remaining
  target), set ``premature_finish_penalty: 0.0`` alongside a non-zero
  per-target value.

policy
~~~~~~

- ``algorithm``: ``recurrent_ppo`` (only supported value).
- ``graph_encoder``:

  - ``type``: ``graphsage`` (only supported value).
  - ``hidden_size`` (int > 0): GraphSAGE layer width.
  - ``layers`` (int > 0): number of ``SAGEConv`` layers.
- ``action_encoder``:

  - ``hidden_size`` (int > 0): per-action embedding width.
  - ``action_type_embedding_size`` (int > 0): width of the learned
    embedding for the action-type categorical (service scan, exploit, ...).
- ``recurrent``:

  - ``hidden_size`` (int > 0): GRU hidden state width.
  - ``sequence_length`` (int > 0): max steps per truncated-BPTT chunk
    during PPO replay (see :mod:`marla.learning.ppo`); a chunk never spans
    an episode boundary regardless of this value.
  - ``visible_target_progress`` (bool, default ``True``): feed the
    3-dimensional visible-only sensitive-target-progress summary
    (:func:`marla.environment.visible_facts.compute_target_progress`) into
    the recurrent core alongside the graph embedding. See :doc:`metrics`'s
    ``decisions.csv`` section for the exact definition and anti-leak
    argument.
  - ``visible_subnet_exploration`` (bool, default ``False``): feed the
    2-dimensional visible-only known-subnet-exploration summary
    (``[fraction_known_subnets_scanned, any_known_subnet_unscanned]``,
    :func:`marla.environment.visible_facts.compute_exploration_progress`)
    into the recurrent core, appended after target progress, **and**
    include the per-subnet-node ``subnet_scan_completed`` GraphSAGE
    feature -- the same flag gates both, so disabling it never leaves the
    exploration signal reachable through the graph embedding either. Off
    by default to keep existing configs/checkpoints at version-3-equivalent
    behavior; see :doc:`metrics` for the full definitions and
    ``research/diagnostics/ppo_learnability/run_representation_ablation.py``
    for the v2/v3-target/v3-full ablation this flag, together with
    ``visible_target_progress``, exists to make possible. Changing either
    flag changes the recurrent core's/GraphSAGE's input width, so a saved
    checkpoint's flags must match the loading policy's exactly
    (:func:`marla.learning.checkpoint.load_checkpoint` rejects a mismatch
    before touching any state).
- ``ppo``:

  - ``total_environment_steps`` (int > 0): overall training budget, a
    GLOBAL count across every environment stream (see ``num_envs``
    below) -- the number of PPO updates run is
    ``ceil(total_environment_steps / (num_envs * steps_per_env))``.
  - ``num_envs`` (int >= 1, default ``1``): number of INDEPENDENT
    environment streams collected, under the same frozen policy, before
    one PPO update runs. Each stream gets its own environment instance,
    its own recurrent-hidden-state chain, and its own deterministically-
    derived RNG seed range; GAE is computed independently per stream
    before combining. ``num_envs=1`` is a single-stream collector with no
    other behavioral change. See
    ``research/diagnostics/multi_env_ablation/README.md`` for why
    ``num_envs=4`` is the current provisional research baseline, and
    :mod:`marla.learning.rollout` for the exact seed-derivation scheme.
  - ``steps_per_env`` (int > 0): environment transitions collected per
    PPO update, PER environment stream -- the effective batch size across
    all streams is ``num_envs * steps_per_env``.
  - ``epochs`` (int > 0): PPO passes over each collected batch.
  - ``minibatch_sequences`` (int > 0): sequence chunks per gradient step.
  - ``gamma`` (0 < float <= 1): discount factor.
  - ``gae_lambda`` (0 <= float <= 1): GAE lambda.
  - ``clip_epsilon`` (float > 0): PPO clipping range.
  - ``value_coefficient`` (float >= 0): value-loss weight in the total loss.
  - ``query_entropy_coefficient`` (float >= 0): entropy bonus weight for
    the query gate (assisted variant only; harmless if set for baseline).
  - ``action_entropy_coefficient`` (float >= 0): entropy bonus weight for
    the action distribution.
  - ``max_grad_norm`` (float > 0): gradient clipping norm.
  - ``optimizer``: only ``adam`` is supported -- an unrecognized ``type``
    is rejected rather than silently falling back to something else.
    ``learning_rate`` and ``scheduler`` live here (not directly under
    ``ppo``) since they are optimizer-level, not PPO-algorithm-level,
    concerns.

    - ``type``: ``adam`` (only supported value; never AdamW).
    - ``learning_rate`` (float > 0, required): Adam's initial learning
      rate.
    - ``eps`` (float > 0, default ``1.0e-5``): Adam's numerical-stability
      denominator term. Larger than torch's default (``1e-8``): PPO's
      advantage-scaled policy gradient is noisier than typical
      supervised-learning gradients, and this value (also used by OpenAI
      Baselines / CleanRL's PPO) improves numerical stability.
    - ``scheduler`` (required, no default -- every config must state its
      schedule explicitly): a PyTorch-native LR scheduler
      (:mod:`marla.learning.lr_scheduler`), stepped exactly once per
      COMPLETED PPO update (never per minibatch/epoch). Horizon-dependent
      types (``linear``/``cosine``) derive their horizon from
      ``ceil(total_environment_steps / (num_envs * steps_per_env)) - 1``
      -- the actual number of ``.step()`` calls the run will make, so the
      schedule reaches its endpoint exactly at the final PPO update
      regardless of batch size. Supported ``type`` values:

      - ``constant``: no scheduler -- the optimizer's ``learning_rate`` is
        used unchanged for the whole run.
      - ``linear``: decays from ``learning_rate`` to
        ``end_factor * learning_rate`` (required field, ``0 <= end_factor
        <= 1``), reaching ``end_factor`` exactly at the final update.
      - ``cosine``: cosine annealing from ``learning_rate`` down to
        ``eta_min`` (default ``0.0``), reaching it exactly at the final
        update.
      - ``step``: multiplies by ``gamma`` (required, ``0 < gamma <= 1``)
        every ``step_size`` (required, int > 0) PPO updates.
      - ``exponential``: multiplies by ``gamma`` (required,
        ``0 < gamma <= 1``) every PPO update.

      The actual current rate, scheduler type, and scheduler step count
      are recorded per update in ``updates.csv`` (``learning_rate``/
      ``scheduler_type``/``scheduler_step`` columns) and the rate is
      plotted in ``marla summarize``'s ``learning_rate.png``
      (:doc:`metrics`). Scheduler state (not just the optimizer's) is
      saved in every checkpoint and restored exactly on
      ``marla run --resume``.
  - ``critic_refinement_epochs`` (int >= 0, default ``0``): experimental.
    Extra value-head-only PPO passes run immediately AFTER the normal
    ``epochs``-pass actor+critic update, using the SAME rollout/GAE return
    targets (never Monte Carlo, oracle, or future-trajectory data) -- see
    :func:`marla.learning.ppo.critic_refinement`. Every parameter except
    the critic head's own two tensors is frozen for the duration of these
    extra passes, so the actor and shared backbone (``GraphEncoder``/
    ``ActionEncoder``/``RecurrentCore``) never change during refinement --
    only ``Critic.value_head`` does. Does not overload ``epochs``, which
    remains the actor+critic joint-update count, unchanged. Default ``0``
    is an exact behavioral no-op: every existing config/checkpoint is
    unaffected, and this remains ``0`` in every production/paper config
    unless a future experiment justifies changing it.

consultation
~~~~~~~~~~~~

- ``mode``: ``disabled`` (baseline: no Gatekeeper, no Plan Maker, no
  ``agents``/``gatekeeper`` may be configured) or ``learned`` (assisted:
  ``gatekeeper`` and exactly one ``agents[]`` entry with
  ``role: plan_maker`` are required).
- ``cost`` (float >= 0, required when ``mode: learned``): per-consultation
  reward penalty, subtracted from the training reward (not the reported
  benchmark/NASimEmu reward) every time the query gate decides to consult.
- ``max_schema_revisions`` (int >= 0, required when ``mode: learned``):
  how many correction attempts the Gatekeeper gives the Plan Maker before
  giving up and reporting ``schema_rejected`` back to the RL Orchestrator.

Agent identities
~~~~~~~~~~~~~~~~

``rl_orchestrator`` and ``gatekeeper`` share the same shape:

- ``alias`` (str, required): short human-readable name, unique across all
  configured agents.
- ``jid`` (str, required): full XMPP JID, unique across all configured
  agents.
- ``password_env`` (str, optional but required in practice to actually
  connect): name of an environment variable holding the XMPP password --
  never the password itself. ``marla run`` reads this at startup and fails
  fast if the named variable isn't set.

Plan Maker agents
~~~~~~~~~~~~~~~~~~

``agents`` is a list; the first release supports exactly one entry with
``role: plan_maker`` when ``consultation.mode: learned``:

- ``alias`` / ``jid`` / ``password_env``: as above.
- ``role``: ``plan_maker`` (only supported value).
- ``model``:

  - ``backend``: ``local`` (a real Hugging Face ``transformers`` model,
    loaded and run on this machine -- needs the ``local-lm`` extra, see
    :doc:`installation`) or ``remote`` (not implemented in this release;
    fails loudly and immediately with an explanatory error, see
    :mod:`marla.models.remote_backend`).
  - ``name`` (str, required): a Hugging Face model ID (``local`` backend)
    or an implementation-defined identifier (``remote`` backend).
  - ``max_new_tokens`` (int, default ``512``): a *floor* on generation
    length, not a fixed value -- the local backend also estimates a
    per-request minimum from the actual number and length of legal action
    IDs and uses whichever is larger (see
    :class:`marla.models.local_backend.LocalTransformersBackend`).
- ``prompt_version`` (str, required): free-text version tag, echoed back in
  every advisory response for reproducibility bookkeeping.
- ``knowledge``:

  - ``path`` (str, required): ``package://marla/<path-inside-the-marla-package>``
    or a plain filesystem path (see :mod:`marla.knowledge.retriever`).
  - ``version`` (str, required): must match the loaded knowledge file's own
    ``version`` field, or the run refuses to start.

metrics
~~~~~~~

- ``output_directory`` (str, default ``"runs"``): where run directories are
  written; see :doc:`metrics`.
- ``record_decisions`` (bool, default ``true``): whether to write
  ``decisions.csv`` (one row per environment step). Disable for a very
  long run where per-step granularity isn't needed and the file size
  matters.
- ``eval_episodes`` (int >= 0, default ``0``): number of deterministic
  (greedy) evaluation episodes to run between training rollouts -- an
  ``EvalCallback``-style measurement of the current policy without
  exploration noise, on a fixed set of held-out seeds. ``0`` disables it
  entirely (no extra cost). See :doc:`metrics` and
  :func:`marla.learning.rollout.run_evaluation_episodes`.
- ``eval_every_rollouts`` (int > 0, default ``1``): run the evaluation pass
  every this-many rollouts (only meaningful when ``eval_episodes > 0``).
- ``resource_monitoring``: CPU/RAM/GPU telemetry
  (:mod:`marla.monitoring.resources`), enabled by default -- computational
  cost is a first-class research metric, not an opt-in extra.

  - ``enabled`` (bool, default ``true``).
  - ``sampling_interval_seconds`` (float > 0, default ``1.0``): how often
    the background sampling thread records CPU/RAM/GPU usage, tagged with
    the training loop's current phase (``rollout_collection`` /
    ``ppo_update`` / ``evaluation``). Measured overhead at the default
    interval is small (~2% wall-clock on a short CPU smoke workload; see
    ``research/RESEARCH_READINESS.md``). Writes ``resources.csv`` (one row
    per sample) and ``resource_summary.json`` (phase-level mean/p95/max)
    into the run directory.

reproducibility
~~~~~~~~~~~~~~~~

- ``deterministic_torch`` (bool, default ``true``): reserved for wiring up
  ``torch.use_deterministic_algorithms`` and related settings.

carbon
~~~~~~

Per-agent energy/CO2-equivalent emissions tracking
(:mod:`marla.monitoring.carbon`), backed by `CodeCarbon
<https://github.com/mlco2/codecarbon>`_ -- an OPTIONAL dependency (the
``carbon`` extra, ``pip install -e ".[carbon]"``); ``marla run`` fails
loudly and specifically at startup if ``carbon.enabled: true`` but
CodeCarbon isn't installed, rather than silently skipping tracking.

- ``enabled`` (bool, default ``false``): unlike
  ``metrics.resource_monitoring``, this defaults OFF, since CodeCarbon is
  an optional dependency and every existing config should keep working
  without it installed.
- ``tracking_mode``: ``"process"`` (default) or ``"machine"`` -- isolates
  CPU/RAM energy estimates to the MARLA process vs. the whole machine.
  **GPU power is always measured at the device level regardless of this
  setting** (an NVML limitation, not a MARLA one) -- run agents strictly
  sequentially, never concurrently, on a shared GPU if you need correct
  per-agent carbon attribution.
- ``measure_power_secs`` (float > 0, default ``1.0``): CodeCarbon's own
  background hardware-power sampling interval.
- ``country_iso_code`` / ``region`` / ``cloud_provider`` / ``cloud_region``
  (str, optional): when ANY of these is set, tracking uses an offline
  tracker with exactly this location, no network call. Leaving all four
  unset does *not* mean "no location" -- it means "let CodeCarbon
  auto-resolve it via a one-time geo-IP lookup", which is the normal,
  default path. This lookup is unrelated to emissions-data upload: MARLA
  never uploads any emissions data anywhere, regardless of this setting
  (CSV-only local output; see :doc:`metrics`).

Every number CodeCarbon (and MARLA's own carbon summary) reports is a
**best-effort estimate of CO2-equivalent emissions**, not an exact
physical measurement -- see :doc:`metrics` for units, per-agent output
files, and how it's surfaced in ``marla summarize``.

Device resolution
------------------

Every locally-executed model (the RL Orchestrator's policy, and a
``model.backend: local`` Plan Maker) resolves ``device`` independently,
following the same rule (see :mod:`marla.runtime.device`):

- ``cpu``: always CPU.
- ``gpu``: must successfully initialize CUDA (availability *and* a real
  tensor allocation) or the process fails fast, before the agent ever
  reports ready -- never a silent fallback to CPU.
- ``auto``: CUDA if it initializes successfully, else CPU.

In distributed mode, each process resolves its own device -- the Plan
Maker can run on different hardware than the RL Orchestrator.

Secrets
-------

MARLA never stores an actual password in configuration, only the *name* of
an environment variable holding it (``password_env``). ``marla run``
writes a redacted copy of the resolved configuration to every run
directory's ``config.yaml``; the redaction (see
:func:`marla.config.loader.redacted_config_dict`) only touches
secret-*shaped* keys that aren't already an ``_env`` reference, since
there's nothing else to redact by design.
