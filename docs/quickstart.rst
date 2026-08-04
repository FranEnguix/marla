Quickstart
==========

Generate starter configs
-------------------------

.. code-block:: bash

   marla init

This writes ``experiment_templates/baseline.yaml`` and
``experiment_templates/assisted.yaml``, scaled down for a fast first run
(seconds to a couple of minutes, not a real research run -- see
``examples/`` for production-realistic settings). If it can find a
``NASimEmu/scenarios/`` directory near your current directory, it fills in
``environment.scenario`` automatically; otherwise it prints a warning and
leaves a placeholder you'll need to fix by hand.

Run the baseline variant
--------------------------

No Gatekeeper, no Plan Maker, no external setup beyond NASimEmu itself:

.. code-block:: bash

   export MARLA_RL_ORCHESTRATOR_PASSWORD=changeme
   marla run experiment_templates/baseline.yaml

``execution.mode: local`` runs every configured agent (here, just the RL
Orchestrator) in one process, communicating over SPADE's embedded XMPP
server -- nothing external to install or configure.

Run the assisted variant
--------------------------

Adds a Gatekeeper (schema validator) and a Plan Maker (a small local
language model giving advisory confidence scores):

.. code-block:: bash

   pip install -e ".[local-lm]"   # once, for the local Plan Maker backend
   export MARLA_RL_ORCHESTRATOR_PASSWORD=changeme
   export MARLA_GATEKEEPER_PASSWORD=changeme
   export MARLA_PLAN_MAKER_1_PASSWORD=changeme
   marla run experiment_templates/assisted.yaml

The first run downloads the configured model (see
``experiment_templates/assisted.yaml``'s ``agents[0].model.name``) from the
Hugging Face Hub, then loads it onto ``device`` (CPU by default in the
generated template -- see :doc:`configuration`). Every environment step,
the RL Orchestrator's learned query gate decides whether this step is worth
a consultation; only queried steps actually invoke the Plan Maker.

Pass ``--debug`` to have the Plan Maker dump every query it receives and
every response it generates to ``debug/<run-id>/`` -- useful for
diagnosing a model that isn't producing the expected JSON shape (see
:mod:`marla.agents.plan_maker`).

Stopping a run early
----------------------

Press :kbd:`Ctrl+C` once to request a graceful stop: the run finishes its
current step (including any in-flight Plan Maker consultation), writes
every metrics artifact as usual with ``status: stopped_by_user``, and exits
cleanly. Press :kbd:`Ctrl+C` a second time to force-quit immediately if the
graceful stop doesn't return promptly enough.

Summarize a run
------------------

.. code-block:: bash

   marla summarize runs/<experiment-name>/<run-id>

Prints the run's key aggregate statistics (episode count, goal success
rate, mean return, consultation stats if any) and writes a set of plots to
``runs/<experiment-name>/<run-id>/plots/`` -- see :doc:`metrics` for what
each one shows.

Validate a config without running it
---------------------------------------

.. code-block:: bash

   marla validate experiment_templates/assisted.yaml

Runs the full Pydantic schema validation plus MARLA's cross-cutting checks
(does the scenario file actually exist, are agent aliases/JIDs unique,
etc. -- see :mod:`marla.config.validation`) without starting any agent.

Next steps
------------

- :doc:`configuration` walks through every experiment YAML field.
- :doc:`architecture` explains what each agent actually does and how a
  consultation round-trip works.
- The ``examples/`` directory has heavily-commented, production-realistic
  baseline and assisted configs.
