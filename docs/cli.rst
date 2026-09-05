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
  prior run's ``checkpoint.pt`` and continues training toward *this*
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

Exit code ``1`` on: invalid configuration, an unresolvable device request
(``device: gpu`` without working CUDA), a missing ``--agent``/local-mode
``--agent`` misuse, or the run itself ending in failure (a construction
error, an unrecoverable training exception, or -- in distributed mode -- a
required participant disconnecting, including mid-training). On any
failure after agents were constructed, the usual run artifacts are still
written to the run directory with ``status: failed``, so partial
diagnostics are never lost.

Press :kbd:`Ctrl+C` once for a graceful stop (finishes the current step,
writes every artifact with ``status: stopped_by_user``); a second
:kbd:`Ctrl+C` force-quits immediately.

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
