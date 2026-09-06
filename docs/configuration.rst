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
- ``scenario`` (str, required): a NASimEmu scenario. A value ending in
  ``.yaml`` is a path to a static scenario file (resolved relative to the
  config file's directory if not absolute); any other string is the name
  of a procedurally generated benchmark NASimEmu resolves at runtime.
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
- ``ppo``:

  - ``total_environment_steps`` (int > 0): overall training budget; the
    number of rollouts run is ``ceil(total_environment_steps / rollout_steps)``.
  - ``rollout_steps`` (int > 0): environment steps collected per
    rollout/PPO-update cycle.
  - ``epochs`` (int > 0): PPO passes over each collected rollout.
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
  - ``learning_rate`` (float > 0): Adam's initial learning rate.
  - ``learning_rate_schedule`` (``constant`` \| ``linear``, default
    ``linear``): ``constant`` keeps ``learning_rate`` unchanged for the
    whole run. ``linear`` decays it linearly to ``0`` as
    ``completed_environment_steps / total_environment_steps`` goes from
    ``0`` to ``1`` (clamped, so it never goes negative), evaluated once
    per rollout rather than once per PPO minibatch -- see
    :mod:`marla.learning.lr_schedule`. Older YAML omitting this field still
    loads (it defaults to ``linear``, same as new configs). The actual
    current rate is always recorded per update in ``updates.csv`` and
    plotted in ``marla summarize``'s ``learning_rate.png`` (:doc:`metrics`).
  - ``optimizer``: only ``adam`` is supported -- an unrecognized ``type``
    is rejected rather than silently falling back to something else.

    - ``type``: ``adam`` (only supported value; never AdamW).
    - ``eps`` (float > 0, default ``1.0e-5``): Adam's numerical-stability
      denominator term. Larger than torch's default (``1e-8``): PPO's
      advantage-scaled policy gradient is noisier than typical
      supervised-learning gradients, and this value (also used by OpenAI
      Baselines / CleanRL's PPO) improves numerical stability. Older YAML
      omitting ``optimizer`` entirely still loads with this default.

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

reproducibility
~~~~~~~~~~~~~~~~

- ``deterministic_torch`` (bool, default ``true``): reserved for wiring up
  ``torch.use_deterministic_algorithms`` and related settings.

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
