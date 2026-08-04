Architecture
============

There is no Coordinator agent and no shared blackboard. The RL Orchestrator
drives the whole experiment lifecycle itself, and the Gatekeeper is the
only path advisory traffic can take between the RL Orchestrator and the
Plan Maker.

Components
----------

RL Orchestrator (:mod:`marla.agents.orchestrator`)
    The only agent that owns NASimEmu and writes central experiment
    metrics. Runs the recurrent PPO policy (:mod:`marla.learning.recurrent_policy`),
    coordinates startup/shutdown (:class:`~marla.agents.orchestrator.OrchestratorLifecycleBehaviour`),
    and -- in the assisted variant -- sends ``ADVISORY_REQUEST`` messages to
    the Gatekeeper mid-training via a real ``await`` on the shared event
    loop (see `The consultation round trip`_ below).

Gatekeeper (:mod:`marla.agents.gatekeeper`)
    The authoritative schema validator for advisory traffic. All Plan Maker
    traffic routes through it: ``RL Orchestrator -> Gatekeeper -> Plan
    Maker`` and back. It validates message correlation and payload shape
    (never final NASimEmu actions), and holds only a transient
    pending-request registry (:class:`~marla.agents.gatekeeper.PendingRequest`)
    -- no cross-run memory.

Plan Maker (:mod:`marla.agents.plan_maker`)
    Frozen and advisory: no PPO gradients, no cross-run memory, no hidden
    simulator access. It only ever sees what the Gatekeeper forwards -- the
    current visible observation, current legal action descriptions, the
    configured objective, and the static versioned knowledge base
    (:mod:`marla.knowledge.retriever`) -- and returns per-action confidence
    scores in ``[0, 1]``.

NASimEmu Adapter (:mod:`marla.environment.nasimemu_adapter`)
    The single point of contact between MARLA and NASimEmu. Nothing
    outside :mod:`marla.environment` imports ``nasimemu`` directly.
    Converts the visible observation into a fixed-width graph
    (:mod:`marla.environment.graph`) and enumerates the current legal
    action set with stable, semantically meaningful IDs
    (:mod:`marla.environment.actions`) -- advisory correlation with the
    Plan Maker always uses these IDs, never positional/vector order.

The consultation round trip
-----------------------------

Every environment step in the assisted variant is a *compound decision*
(spec section 16), computed once per step:

1. The recurrent policy scores every legal action (:meth:`marla.learning.recurrent_policy.RecurrentPolicy.step`)
   and the learned **query gate** (:mod:`marla.learning.query_gate`)
   computes a probability of consulting the Plan Maker *this step*, as a
   function of the recurrent state, base-policy entropy, top-two-action
   margin, candidate-set size, and the configured consultation cost --
   never a fixed schedule or a rule.
2. That probability is sampled (training) or thresholded at 0.5
   (deterministic evaluation, see :func:`marla.learning.rollout.run_evaluation_episodes`).
   Most steps are *not* queried -- consultation is on-demand, not every
   step -- which is exactly what the learned query gate is for: to learn
   when consulting is worth its configured cost.
3. If queried, :func:`marla.agents.advisory_client.send_advisory_request`
   sends an ``ADVISORY_REQUEST`` to the Gatekeeper and ``await``\ s the
   correlated ``ADVISORY_RESPONSE`` -- **indefinitely, with no elapsed
   timeout** (spec sections 5, 10, 14). This is deliberate: a real model
   doing real inference can legitimately take anywhere from milliseconds
   to minutes, and there is no principled fixed cutoff. The only things
   that end the wait are the response itself, or (in distributed mode) a
   detected disconnect of the Gatekeeper -- see `Failure handling`_.
4. The Gatekeeper (:class:`~marla.agents.gatekeeper.AdvisoryRequestBehaviour`)
   validates the request and forwards it verbatim to the Plan Maker.
5. The Plan Maker (:class:`~marla.agents.plan_maker.AdvisoryRequestHandler`)
   retrieves applicable knowledge rules (deterministic, not learned --
   :func:`marla.knowledge.retriever.retrieve_rules`), builds a prompt
   (:mod:`marla.models.prompt`), and generates a response through the
   configured backend (:mod:`marla.models.local_backend` or
   :mod:`marla.models.remote_backend`).
6. The Gatekeeper (:class:`~marla.agents.gatekeeper.AdvisoryResponseBehaviour`)
   validates the response: correct schema, matching ``run_id``/``request_id``,
   and *exact* coverage of the requested legal action IDs (no more, no
   fewer -- see :func:`marla.messaging.advisory_validation.validate_advisory_response_body`).
   A failing response triggers a ``CORRECTION_REQUEST`` back to the Plan
   Maker with the specific reason and a worked example, up to
   ``consultation.max_schema_revisions`` times, before giving up and
   reporting ``schema_rejected``.
7. Back at the RL Orchestrator: an accepted response's scores are clipped,
   converted to log-odds, and z-score normalized
   (:mod:`marla.learning.advice`), then blended into the base policy's
   logits as a residual adjustment scaled by a **learned per-step trust
   coefficient** (beta) and a **learned global scale** (alpha) -- a
   rejected response forces beta to 0, which is numerically identical to
   ignoring the advice while still training the trust head on a real
   (if unused) alpha. The final action is sampled from this adjusted
   distribution.

Every one of these fields -- whether this step was queried, the request/response
IDs, response latency, beta/alpha, whether accepting the advice actually
changed the top-ranked action -- is recorded per-step in ``decisions.csv``;
see :doc:`metrics`.

Execution modes
-----------------

Local (:mod:`marla.runtime.local`)
    Every configured agent runs in one process, one shared ``asyncio``
    event loop, communicating over SPADE's embedded XMPP server
    (``pyjabber``) -- still real SPADE messages, never direct Python calls
    between agents. Zero external setup.

Distributed (:mod:`marla.runtime.distributed`)
    One ``marla run ... --agent <alias-or-jid>`` process per agent (or
    group of agents), connecting to a real, externally reachable XMPP
    server. Every process loads the identical configuration and shares
    ``experiment.run_id``; only the process hosting ``rl_orchestrator``
    owns NASimEmu and writes central metrics.

Failure handling
------------------

A required participant (Gatekeeper, Plan Maker) disconnecting is a real
failure, not a silent hang -- in distributed mode, both agents subscribe to
presence for their upstream peer
(:func:`marla.agents.lifecycle_behaviours.make_disconnect_detector`), and
the RL Orchestrator subscribes to presence for every required participant.
This is checked both:

- **Before training starts**, while waiting for every required participant
  to report ``READY`` (:func:`marla.runtime.lifecycle.wait_for_all_ready`).
- **During training**, including while an advisory request is in flight --
  the training loop races against a live watch for a disconnect event, so
  a Gatekeeper or Plan Maker that dies mid-consultation fails the run
  (raising a clear error) instead of hanging forever (see
  :meth:`marla.agents.orchestrator.OrchestratorLifecycleBehaviour._run_training_watching_for_failure`).

A disconnect reported *after* the run has already sent ``STOP_EXPERIMENT``
is expected teardown, not a failure -- both sides arm an "expect a
disconnect" flag right before disconnecting on purpose, precisely to avoid
a normal end-of-run race being mistaken for a crash.

Local mode skips presence-based disconnect detection entirely (there is
only one process; a dead peer there is this process crashing, which fails
the whole run trivially) -- the same-process ``asyncio.to_thread``/keepalive
concerns are handled separately (see
:func:`marla.agents.lifecycle_behaviours.disable_reconnect_on_missed_ping`'s
docstring for the keepalive-vs-blocking-inference interaction this
resolves).

No blackboard
--------------

Every agent's state is local to that agent's process. The only
cross-agent communication is the SPADE messages themselves (envelope +
JSON payload, see :mod:`marla.messaging.schemas`); there is no shared
mutable store any agent reads or writes outside of that. The Gatekeeper's
``pending`` dict and the RL Orchestrator's ``pending`` futures dict are
each private, transient, per-process correlation state -- not a shared
resource.
