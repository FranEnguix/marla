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
    simulator access. It only ever sees what the Gatekeeper forwards -- and,
    since subnet-scoped consultation (see below), that is deliberately a
    SUBSET of the current step's information, not everything the policy
    itself can see: detailed host facts and candidate actions for exactly
    ONE subnet, plus a compact whole-network progress summary and the
    global ``finish`` action.

    The local per-host facts it receives (:func:`marla.environment.consultation_scope.build_scoped_observation`,
    itself built from :func:`marla.environment.observation_summary.build_observation_summary`)
    list *specific* confirmed facts per host -- which services, processes,
    and OS are actually known to be present, not just how many. This is
    what lets the prompt ask it to cross-reference a given exploit/privesc
    action's own required service/OS/process (carried in that action's
    ``parameters``) against the specific target host's confirmed facts,
    rather than scoring actions from vague aggregate counts alone.
    NASimEmu's own partial-observability model only ever reveals *positive*
    facts and never reverts one to unknown, so an absent name always means
    "not yet confirmed," never "confirmed absent" -- the prompt says this
    explicitly, and nothing in the observation claims otherwise.

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
3. If queried, :class:`~marla.learning.rollout.RolloutCollector` deterministically
   routes the consultation to exactly ONE subnet (see `Subnet-scoped
   consultation`_ below) and builds the scoped request/observation
   *before* calling into the consultation transport.
   :func:`marla.agents.advisory_client.send_advisory_request` then sends
   an ``ADVISORY_REQUEST`` to the Gatekeeper and ``await``\ s the
   correlated ``ADVISORY_RESPONSE`` -- **indefinitely, with no elapsed
   timeout** (spec sections 5, 10, 14). This is deliberate: a real model
   doing real inference can legitimately take anywhere from milliseconds
   to minutes, and there is no principled fixed cutoff. The only things
   that end the wait are the response itself, or (in distributed mode) a
   detected disconnect of the Gatekeeper -- see `Failure handling`_.
4. The Gatekeeper (:class:`~marla.agents.gatekeeper.AdvisoryRequestBehaviour`)
   validates the request and forwards it verbatim to the Plan Maker. It
   requires no knowledge of the global candidate set at all -- only exact
   coverage of whatever action IDs the request actually carries (see step
   6), which is already the SCOPED subset by the time it reaches here.
5. The Plan Maker (:class:`~marla.agents.plan_maker.AdvisoryRequestHandler`)
   retrieves applicable knowledge rules (deterministic, not learned --
   :func:`marla.knowledge.retriever.retrieve_rules`), builds a prompt
   (:mod:`marla.models.prompt`), and generates a response through the
   configured backend (:mod:`marla.models.local_backend` or
   :mod:`marla.models.remote_backend`).
6. The Gatekeeper (:class:`~marla.agents.gatekeeper.AdvisoryResponseBehaviour`)
   validates the response: correct schema, matching ``run_id``/``request_id``,
   and *exact* coverage of the requested (scoped) action IDs (no more, no
   fewer -- see :func:`marla.messaging.advisory_validation.validate_advisory_response_body`).
   A failing response triggers a ``CORRECTION_REQUEST`` back to the Plan
   Maker with the specific reason and a worked example, up to
   ``consultation.max_schema_revisions`` times, before giving up and
   reporting ``schema_rejected``.
7. Back at the RL Orchestrator: an accepted response's scores (covering
   only the consulted subset) are clipped, converted to log-odds, and
   z-score normalized OVER THE CONSULTED SUBSET ONLY
   (:mod:`marla.learning.advice`), then scattered into a GLOBAL residual
   vector -- zero everywhere except the consulted indices -- and blended
   into the base policy's FULL logits as a residual adjustment scaled by a
   **learned per-step trust coefficient** (beta) and a **learned global
   scale** (alpha). A rejected response forces beta to 0, which is
   numerically identical to ignoring the advice while still training the
   trust head on a real (if unused) alpha. The final action is sampled
   from this adjusted distribution over the FULL GLOBAL action set --
   never restricted to the consulted subnet, so the policy can still
   select an action in a different subnet than the one it consulted about
   (the Plan Maker supplies local evidence, never a hard mask).

Every one of these fields -- whether this step was queried, the request/response
IDs, response latency, beta/alpha, whether accepting the advice actually
changed the top-ranked action, the consulted subnet and its action-count
reduction ratio -- is recorded per-step in ``decisions.csv``; see :doc:`metrics`.

Subnet-scoped consultation
-----------------------------

Without scoping, both the Plan Maker's prompt and the score vector it must
generate grow roughly linearly with every visible host and its action list
-- on a large explored network, that means an ever-larger prompt, an
ever-larger generated JSON object, and ever-larger Plan Maker latency, none
of which the RL policy's own action space is limited by. Subnet-scoped
consultation decouples the two:

- **PPO's action space stays global.** :meth:`~marla.learning.recurrent_policy.RecurrentPolicy.step`
  still scores every currently legal action across every visible subnet,
  exactly as before -- nothing about candidate-action construction, the
  query gate, or PPO_ONLY changes (spec invariants 1, 11).
- **Deterministic, unlearned routing.** When the query gate samples
  ``True``, :func:`marla.environment.consultation_scope.select_consulted_subnet`
  picks the subnet of the single highest-base-logit NON-FINISH candidate
  (``argmax`` over the policy's own already-computed logits -- no new
  learned head, no sampling, FINISH itself can never determine the
  route).
- **The Plan Maker sees one subnet.** :func:`~marla.environment.consultation_scope.select_consulted_actions`
  restricts the request to that subnet's actions plus the global
  ``finish`` action; :func:`~marla.environment.consultation_scope.build_scoped_observation`
  restricts the observation to a compact whole-network progress summary
  (sensitive-target/subnet-exploration counts -- never per-host detail)
  plus full per-host detail for that one subnet only. Both are built ONCE,
  by :class:`~marla.learning.rollout.RolloutCollector`, before the
  consultation transport is ever invoked -- so the real SPADE path and the
  in-process :class:`~marla.evaluation.direct_consult.DirectConsultant`
  (checkpoint evaluation) both consume an already-scoped request and can
  never define scoping differently from one another.
- **Advice is applied sparsely.** The Plan Maker's scores are normalized
  only over the consulted subset (never against the global candidate
  count -- that would make normalization depend on how many unrelated
  actions happen to exist elsewhere) and scattered into a global residual
  vector that is exactly zero for every unconsulted action. Adding
  arbitrary candidate actions in other subnets therefore never changes the
  consulted subset's own normalized advice.
- **PPO replay reuses the exact consultation experienced at collection
  time.** The consulted subnet, the consulted action indices, and the Plan
  Maker's scores for them are stored on each transition
  (:class:`~marla.learning.rollout.StepRecord`) and replayed verbatim
  during a PPO update -- the Plan Maker is never called again, and the
  route is never recomputed from the update's own (possibly different)
  parameters.

  **This is deliberately documented as a PIECEWISE_EXACT_SEMIGRADIENT
  design, not an unconditionally exact PPO policy ratio.** Routing
  (``consulted_subnet = f(theta, observation)``) is deterministic, but --
  unlike ``legal_action_descriptors``, which is exogenous and literally
  independent of theta -- it is a function OF theta, so a different theta
  can genuinely route differently. When the current parameters' own
  routing rule agrees with the stored route (the common case while updates
  stay small), replay is exact. When it would now disagree (a "route
  switch"), replay computes the log-probability of the stored action under
  a surrogate policy that keeps the collection-time external routing/
  consultation context frozen for that on-policy batch, rather than the
  probability the fully-redeployed current policy would assign -- a real,
  bounded, and measured approximation, not a silently accepted one. A
  dedicated update-level diagnostic (``route_switch_count``/
  ``route_switch_fraction``/``mean_routing_margin`` in ``updates.csv``,
  never used to alter training) reports exactly how often this happens.
  See :mod:`marla.learning.decision`'s module docstring for the full audit
  and the formal argument for why this is defensible, and
  ``research/aamas2027/PAPER_EXPERIMENTS.md`` for measured route-switch
  behavior.

**Trade-off**: the Plan Maker no longer compares candidate actions across
different subnets within one request -- it is a local expert on whichever
subnet the deterministic route selected, not a global comparator. The
global progress summary gives it enough whole-network context to reason
about ``finish`` sensibly without seeing every host's detail. A second,
related trade-off introduced by this same design: PPO replay's likelihood
ratio is exact only while the deterministic route stays unchanged from
collection to replay -- see the PIECEWISE_EXACT_SEMIGRADIENT note above.

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

Every message built via :func:`marla.messaging.builders.build_message` is
also recorded into that run's raw message-event log -- once as a "sent"
event at build time, and again as a "handled" event if and when the
intended MARLA behaviour actually accepts it for processing (see
:mod:`marla.messaging.telemetry` and :doc:`metrics`'s ``messages.csv``
section for why these are tracked as two separate claims, not one) --
reconstructable communication statistics without adding any new shared
state of its own.
