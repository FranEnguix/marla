MARLA documentation
====================

MARLA (Multi-Agent Reinforcement Learning Architecture) is a research
platform for augmenting a reinforcement-learning offensive-security agent
with an advisory, language-model-based *Plan Maker*.

Concretely: a recurrent-PPO agent plays `NASimEmu
<https://github.com/jaromiru/NASimEmu>`_ scenarios directly. In the
**assisted** variant, it can also learn *when* to pause and ask a frozen,
schema-validated Plan Maker for advice on the current set of legal actions,
then learns *how much* to trust that advice via a residual, per-step
confidence-weighted adjustment to its own policy.

There is no "coordinator" agent and no shared blackboard: the RL
Orchestrator drives the whole experiment lifecycle itself, and every piece
of advisory traffic is validated by a Gatekeeper agent before it can affect
a decision.

.. note::
   This documentation describes the implementation as it exists in this
   repository. For the full design rationale and the acceptance criteria
   the implementation is checked against, see
   ``MARLA_complete_implementation_specification.md`` at the repository
   root and the ``architecture_design_documents/`` folder.

Where to start
--------------

- New to MARLA? Start with :doc:`installation` and :doc:`quickstart`.
- Writing or editing an experiment YAML? See :doc:`configuration`.
- Want to understand *how* the agents talk to each other? See
  :doc:`architecture`.
- Looking for a specific CLI flag? See :doc:`cli`.
- Wondering what ``marla summarize`` plots mean? See :doc:`metrics`.
- Digging into a specific module? See the :doc:`api/index`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   configuration
   architecture
   cli
   metrics
   api/index

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
