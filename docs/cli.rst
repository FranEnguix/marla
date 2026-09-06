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

   marla scenario check SCENARIO_PATH [--objective capture_target] [--json] [--verbose]

Runs the same universal solvability analysis ``marla run`` uses as its
preflight, standalone, without loading a MARLA experiment config or
starting anything. Useful to check a scenario before writing a config
against it, or in CI.

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

   marla scenario repair SCENARIO_PATH [--output PATH] [--overwrite]

Analyzes ``SCENARIO_PATH``; if it is already universally solvable, reports
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

``marla version``
--------------------

.. code-block:: text

   marla version

Prints MARLA's own version and the resolved versions of its key
dependencies (torch, torch_geometric, spade, pydantic, typer, nasimemu) --
the same information recorded in every run's ``metadata.json``.
