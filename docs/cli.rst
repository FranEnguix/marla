Command-line reference
========================

MARLA is invoked as ``marla <command> ...`` (or ``python -m marla <command> ...``).
Every command validates the target configuration before doing anything
else and exits with a non-zero status on any failure, printing a specific
reason.

``marla init``
---------------

.. code-block:: text

   marla init [DIRECTORY] [--force]

Generates starter ``baseline.yaml``/``assisted.yaml`` configs in
``DIRECTORY`` (default ``experiment_templates/``), scaled down for a fast
first run. Looks for a ``NASimEmu/scenarios/`` directory near the current
directory to fill in ``environment.scenario`` automatically; prints a
warning and leaves a placeholder if it can't find one. Refuses to
overwrite existing files unless ``--force`` is given.

``marla run``
--------------

.. code-block:: text

   marla run CONFIG_PATH [--agent ALIAS_OR_JID ...] [--debug] [--resume CHECKPOINT_PATH]

Runs an experiment.

- ``CONFIG_PATH``: path to the experiment YAML file.
- ``--agent``: repeatable; alias or JID of an agent to start *in this
  process*. Required (at least one) when ``execution.mode: distributed``;
  rejected when ``execution.mode: local`` (all configured agents always
  start together in local mode).
- ``--debug``: the Plan Maker writes every query it receives and every
  response it generates to ``debug/<run-id>/<NNNN>_<request-id>_query.txt``
  / ``..._response.txt`` (sequence-numbered, since a correction retry
  reuses the same ``request_id``). Only meaningful for assisted-mode runs.
- ``--resume``: local mode only. Loads policy/optimizer/RNG state from a
  prior run's ``checkpoint.pt`` (its final checkpoint) or ``checkpoint_last.pt``
  (kept up to date at every rollout boundary during training -- the one to
  use after an interrupted run that never reached a normal shutdown) and
  continues training toward *this*
  config's ``policy.ppo.total_environment_steps`` instead of starting
  over -- only that field (and typically ``run_id``, for a distinct output
  directory) is expected to differ from the config that produced the
  checkpoint; every other hyperparameter should match. The number of
  rollouts run is computed as the remainder to the new target (refuses to
  run if the checkpoint already meets or exceeds it). Episode seeds
  continue forward from where the checkpoint left off rather than
  repeating the original run's sequence, and the resumed run's own
  ``environment_steps``/``update`` numbering in ``updates.csv`` is
  cumulative (includes the steps/updates already done before the resume),
  so it stays a meaningful x-axis when stitched with the original run's
  data. A checkpoint saved before this existed still loads (episode-seed
  continuation and RNG-state restoration are both best-effort: absent for
  an old checkpoint, in which case episode seeds restart from
  ``experiment.seed`` and the RNG stream is unseeded from that point).
  This is separate from *representation* compatibility, which is not
  best-effort: every checkpoint records the policy architecture's
  ``policy_representation_version`` (:mod:`marla.learning.action_encoder`)
  at save time, and loading one saved under a different version (e.g. one
  saved before the action/target compatibility features existed) raises
  :class:`marla.learning.checkpoint.PolicyRepresentationMismatchError`
  loudly, before any weights are touched, rather than silently partially
  loading a shape-incompatible ``ActionEncoder``.

Before any agent -- RL Orchestrator, Gatekeeper, Plan Maker, or the
embedded XMPP server -- is started, ``marla run`` resolves the configured
scenario and runs a **scenario solvability preflight check** (see
:ref:`scenario-solvability`): every NASimEmu realization the scenario file
can legally produce must have at least one valid path to ROOT on every
sensitive host, under the experiment's own ``objective.type``. On success
this prints two quiet lines (``Validating scenario solvability...`` /
``Scenario solvability: PASSED``) and continues normally. On failure it
prints a full diagnostic (the structural reason, an example unsolvable
host configuration, and the exact ``marla scenario repair ...`` command to
fix it), exits non-zero, and -- this is a hard rule, not a convenience --
**no MARLA agent of any kind is started**. ``marla run`` never repairs a
scenario itself; see :ref:`scenario-repair`.

Exit code ``1`` on: invalid configuration, an unresolvable device request
(``device: gpu`` without working CUDA), a proven-unsolvable (or
unprovable) scenario, a missing ``--agent``/local-mode ``--agent`` misuse,
or the run itself ending in failure (a construction error, an
unrecoverable training exception, or -- in distributed mode -- a required
participant disconnecting, including mid-training). On any failure after
agents were constructed, the usual run artifacts are still written to the
run directory with ``status: failed``, so partial diagnostics are never
lost.

Press :kbd:`Ctrl+C` once for a graceful stop (finishes the current step,
writes every artifact with ``status: stopped_by_user``); a second
:kbd:`Ctrl+C` force-quits immediately.

.. _scenario-solvability:

``marla scenario check``
--------------------------

.. code-block:: text

   marla scenario check SCENARIO [--objective capture_target] [--json] [--verbose]

Runs the same universal solvability analysis ``marla run`` uses as its
preflight, standalone, without loading a MARLA experiment config or
starting anything. Useful to check a scenario before writing a config
against it, or in CI.

``SCENARIO`` accepts either a filesystem path or a ``marla://<filename>``
reference (see :mod:`marla.scenarios.uri`), resolved through the same
authoritative resolver ``marla run``'s preflight uses -- e.g. ``marla
scenario check marla://sm_entry_user_three_subnets.solvable.v2.yaml``.

Prints scenario format/version, the objective checked against (defaults to
MARLA's current ``capture_target`` semantics -- there is no other objective
type yet, but the flag exists so a future one doesn't need a new command),
whether every realization is provably solvable, the possible sensitive-host
locations and count range, network-reachability status, host-rootability
status, and -- for a failing scenario -- one entry per *distinct structural
failure class* (never one entry per equivalent host permutation) plus the
exact ``marla scenario repair`` command to fix it. Exit code ``1`` unless
the result is ``PROVEN_SOLVABLE``; ``UNKNOWN`` (a scenario shape the
checker cannot yet reason about exhaustively) is treated as a failure too,
never silently as success.

``--json`` prints the same information as a single JSON object instead
(``status``, ``universally_solvable``, ``failure_classes``,
``network_issues``, ...) -- for tests and automation; the checker itself
never depends on any particular text formatting, only on this typed result.

.. _scenario-repair:

``marla scenario repair``
---------------------------

.. code-block:: text

   marla scenario repair SCENARIO [--output PATH] [--overwrite]

``SCENARIO`` accepts a filesystem path or a ``marla://<filename>``
reference, resolved the same way ``marla scenario check`` does.

Analyzes ``SCENARIO``; if it is already universally solvable, reports
that no repair is needed and does nothing. Otherwise, derives and applies
the smallest structural change needed (see :ref:`repair-policy`), writes a
**new** file -- by default ``<name>.solvable.v2.yaml`` next to the source,
never the source file itself, and never overwriting an existing output
unless ``--overwrite`` is given -- and only ever reports success after
re-loading that file through NASimEmu's own real scenario loader *and*
re-running this same checker on it, confirming ``PROVEN_SOLVABLE``. If
either check fails, the candidate file is removed and the command exits
non-zero rather than leaving anyone with a file believed to be fixed but
not verified.

``marla validate``
--------------------

.. code-block:: text

   marla validate CONFIG_PATH

Runs full schema + cross-cutting validation (:mod:`marla.config.loader`,
:mod:`marla.config.validation`) without starting any agent or touching the
network. Useful in CI or before a long run.

``marla summarize``
----------------------

.. code-block:: text

   marla summarize RUN_DIRECTORY

Where ``RUN_DIRECTORY`` is ``<metrics.output_directory>/<experiment-name>/<run-id>``
(printed by ``marla run`` on completion). Prints the run's aggregate
statistics from ``summary.json`` and writes plots to
``RUN_DIRECTORY/plots/`` -- see :doc:`metrics` for what's read and what
each plot shows. Safe to re-run at any time; it only reads existing
artifacts and (re)writes the ``plots/`` directory.

``marla compare``
--------------------

.. code-block:: text

   marla compare RUN_DIRECTORY RUN_DIRECTORY [RUN_DIRECTORY ...] [--output DIR] [--label TEXT ...]

Compares wall-clock/resource/energy metrics **across** two or more runs
(e.g. several seeds of the same configuration) -- never a single run's
own time series, which stays in ``marla summarize``. Each run contributes
one scalar per metric (its own total training time, mean CPU%, total
energy, ...); for each metric with at least one value across the given
runs, writes one ``compare_<metric>.png`` to ``--output`` (default: a
``compare_plots/`` directory next to the first run) showing every run's
own value as a point, plus a mean +/- std marker when 2 or more runs have
that metric. Never a box-and-whisker plot: this project's runs are
typically compared in small numbers (e.g. 3 seeds), too few for box-plot
quartiles to mean anything (see :mod:`marla.metrics.compare_plots`'s own
docstring). ``--label`` (repeatable, one per run directory, in the same
order) overrides the default ``<parent-dir>/<run-id>`` label -- useful
when directory names alone don't say what varies between the runs (e.g.
different algorithms, not just different seeds).

``marla version``
--------------------

.. code-block:: text

   marla version

Prints MARLA's own version and the resolved versions of its key
dependencies (torch, torch_geometric, spade, pydantic, typer, nasimemu,
codecarbon, optuna -- ``"not installed"`` for the latter two if the
``carbon``/``optuna`` extras weren't installed) -- the same information
recorded in every run's ``metadata.json``.

``marla optimize``
---------------------

.. code-block:: text

   marla optimize STUDY_CONFIG_PATH

Requires the ``optuna`` extra (``pip install -e ".[optuna]"``). Runs (or
resumes) a persistent Optuna hyperparameter study against a *study config*
YAML -- a separate format from a normal MARLA experiment config (see
:mod:`marla.optuna_study.config`), naming a base experiment config for
every fixed parameter plus study-level settings:

- ``study_name`` / ``storage`` (an Optuna RDB storage URL, e.g.
  ``sqlite:///study.db`` -- relative paths resolve against the study
  config file's own directory).
- ``sampler_seed`` / ``n_startup_trials``: TPE sampler configuration.
  ``sampler_seed`` must not be a paper seed (101/202/303).
- ``n_completed_trials_target``: the study runs until this many trials
  reach Optuna's ``COMPLETE`` state (failed/pruned trials never count).
- ``base_config``: the MARLA experiment config supplying every fixed
  (non-tuned) hyperparameter.
- ``tuning_seeds`` / ``holdout_seed``: every trial is evaluated on ALL
  ``tuning_seeds``; ``holdout_seed`` must be disjoint from them and from
  the paper seeds, and is never touched by the study itself.
- ``tuning_total_environment_steps`` / ``finalist_total_environment_steps``:
  training horizon during the search vs. for a fresh, non-resumed
  finalist confirmation run.
- ``runs_root``: where per-trial run directories are written.
- ``max_consecutive_infrastructure_failures`` (default ``5``): the study
  stops (rather than retrying indefinitely) after this many consecutive
  trials fail for the same infrastructure reason. An infrastructure
  failure is never silently converted into a numerical objective value.

Safe to interrupt (:kbd:`Ctrl+C`, a process kill, or a host restart) and
rerun with the same study config -- Optuna's SQLite-backed persistence
(``load_if_exists=True``) resumes the same study, re-running only what
never reached ``COMPLETE``, never starting a fresh study by accident.
Runs strictly sequentially (``n_jobs=1``): required both for GPU-carbon
attribution (see the ``carbon`` config's tracking-mode caveat in
:doc:`configuration`) and for reproducible timing.

``marla study status``
--------------------------

.. code-block:: text

   marla study status STUDY_CONFIG_PATH

Prints how many trials are ``COMPLETE``/``FAIL``/``RUNNING``/``WAITING``
in the study named by ``STUDY_CONFIG_PATH``, and progress toward
``n_completed_trials_target``, without running anything. Safe to call at
any time, including while ``marla optimize`` is running elsewhere.

``marla study summarize``
-----------------------------

.. code-block:: text

   marla study summarize STUDY_CONFIG_PATH

Prints a compact one-row-per-trial comparison table: state, stability,
worst-seed/mean ROOT AUC, the sampled hyperparameters, and per-trial
carbon/energy/wall-clock totals. A trial that never reached ``COMPLETE``
prints only its state, with a pointer to inspect its stored attributes
for the reason.
