Metrics and plots
==================

Every run writes a self-contained directory:
``<metrics.output_directory>/<experiment-name>/<run-id>/`` (default
``runs/<experiment-name>/<run-id>/``), regardless of whether the run
completed, was stopped by the user, or failed. All of it is produced by
:mod:`marla.metrics.writer`.

Incremental persistence
--------------------------

Results are written *as training progresses*, not only at the very end:
``config.yaml`` and a provisional ``metadata.json`` (``status: "running"``)
exist before a single environment step runs
(:func:`marla.metrics.writer.initialize_run_directory`), and every CSV
below gets its header row at the same time. After each completed rollout,
that rollout's episode/decision/rollout/update rows are appended
immediately (:func:`marla.metrics.writer.append_rollout_metrics`), and
``checkpoint_last.pt`` is overwritten. Only ``summary.json``, the final
``metadata.json`` (with ``end_time``/final ``status``), and
``checkpoint.pt`` are written once, at shutdown
(:func:`marla.metrics.writer.finalize_run_directory`). This is what makes
an interrupted long run's results usable: a process killed hard mid-run
still leaves every completed rollout's data on disk, not just whatever
happened to be in memory at the moment of the crash.

This module also keeps memory bounded independent of run length: each
rollout's heavyweight per-step records (graph tensors, logits, ...) are
used only to compute GAE and run the PPO update, then immediately turned
into that rollout's compact ``decisions.csv`` rows and discarded -- nothing
resembling a full-run list of per-step tensors is ever retained.

Run directory contents
------------------------

``config.yaml``
    The fully resolved configuration, as actually used (not just what was
    in the YAML file -- defaults included), with secret-*shaped* keys
    redacted (there's nothing to redact by default: passwords are only
    ever referenced by environment variable name, never embedded).

``metadata.json``
    Run identity, resolved device, dependency versions, git commit,
    start/end time, and final ``status`` (``completed`` \|
    ``stopped_by_user`` \| ``failed``).

``episodes.csv``
    One row per episode -- both ordinary training episodes and, if
    ``metrics.eval_episodes > 0``, deterministic evaluation episodes
    (tagged via ``is_eval``). Columns include ``objective_reached``,
    ``successful_finish``, ``episode_success``, ``goal_success`` (see the
    success-semantics note below),
    ``nasimemu_return``/``training_return``/``benchmark_return``,
    ``environment_steps``, ``steps_to_goal`` (first step the objective
    became satisfied, whether or not/whenever FINISH was later selected),
    ``finish_step`` (the step FINISH was actually selected),
    ``finish_delay_steps`` (``finish_step - steps_to_goal``, only for a
    ``successful_finish`` episode where both exist), ``sensitive_targets_total``
    / ``sensitive_targets_with_root_final`` / ``sensitive_targets_remaining_final``
    (the simulator's real sensitive-target state at episode end, a direct
    query rather than a running/cached value -- correct however the episode
    ended, since FINISH itself performs no simulator action),
    ``consultation_count``/``consultation_cost``, ``schema_rejection_count``,
    ``finish_reason``, and ``rollout`` (which training rollout produced --
    or immediately followed, for an eval episode -- this row, so returns
    can be bucketed by training progress rather than raw episode index).

    **Success semantics -- three related but distinct fields, never
    conflated:**

    - ``objective_reached``: true the instant ``adapter.objective_satisfied()``
      first becomes true (all sensitive targets rooted), independent of
      FINISH entirely. Equivalent to ``steps_to_goal is not None``.
    - ``successful_finish``: true only if the policy *additionally* selected
      FINISH while the objective held. A timeout after the objective was
      reached is ``objective_reached=True`` but ``successful_finish=False``
      -- a real, important diagnostic case (Stage A undertraining can look
      very different from "never found a solution at all"), not a plain
      failure indistinguishable from never reaching the objective.
    - ``episode_success``: MARLA's protocol defines a successful episode as
      an explicit FINISH after the objective holds, so this is always
      identical to ``successful_finish`` -- kept as an explicitly-named
      synonym so a reader doesn't have to know that definition already.
    - ``goal_success``: **deprecated**, kept only for backward compatibility
      with existing analysis code. Identical to ``successful_finish``,
      never to ``objective_reached`` -- new code should read
      ``successful_finish``/``episode_success`` instead.

    ``steps_to_goal`` and ``finish_step`` are tracked independently: the
    agent can satisfy the objective and keep acting for several more steps
    before finally choosing FINISH (or never choose it at all, timing out
    instead) -- MARLA's fundamental rule that a successful episode requires
    an explicit FINISH is unchanged, ``successful_finish``/``episode_success``/
    ``goal_success`` all still require it; only ``objective_reached`` does not.

``decisions.csv``
    One row per environment step (omitted if
    ``metrics.record_decisions: false``). Identity/progress columns
    (``global_environment_step``, ``rollout``, ``episode_id``,
    ``environment_step``, ``observation_id``); action-space
    (``legal_action_count``, ``selected_action_id``,
    ``selected_action_type``); policy confidence, computed once from the
    logits available at that step -- never a stored full probability
    vector (``base_policy_entropy``, ``base_top_two_margin``,
    ``selected_action_base_probability``/``selected_action_final_probability``,
    their ranks, ``final_policy_entropy``); critic/credit assignment
    (``critic_value``, ``gae_advantage``, ``return_target``); rewards
    (``nasimemu_reward``, ``training_reward``, ``consultation_cost``);
    objective/episode state (``objective_satisfied``,
    ``objective_became_satisfied`` -- true exactly on the step the
    objective first becomes satisfied -- ``terminated``, ``truncated``);
    observable state change since the previous step, derived from a cheap
    scalar diff of the two ``HostVector`` snapshots, never a stored
    observation (``new_hosts_discovered``, ``new_subnets_discovered``,
    ``new_services_confirmed``, ``new_processes_confirmed``,
    ``access_gain`` -- 0/1/2 for none/USER/ROOT); sensitive-target progress,
    a direct query of the underlying simulator's current state rather than
    anything stored (``sensitive_targets_total``, ``sensitive_targets_with_root``,
    ``sensitive_targets_remaining`` -- USER access does not count as
    "with root"; feeds ``objective.premature_finish_penalty_per_remaining_target``);
    action/target compatibility (see :mod:`marla.environment.action_compatibility`
    -- the same helper drives the action encoder's own input features, so
    these can never drift from what the policy actually saw):
    ``selected_action_known_service_match``/``selected_action_known_process_match``/
    ``selected_action_known_os_match``/``selected_action_known_os_mismatch``
    (``null``, never a misleading ``False``, whenever the selected action
    has no such requirement at all), ``selected_action_visible_preconditions_status``
    (``confirmed_compatible``/``contradicted``/``unknown``/``not_applicable``),
    and the compatible-action probability mass the current policy places
    on ``confirmed_compatible`` actions before/after any consultation
    (``base_compatible_action_probability_mass``/
    ``final_compatible_action_probability_mass`` -- identical to each
    other for the baseline variant, since no advice ever perturbs the
    logits there); FINISH diagnostics -- see the labeled note below for
    which of these are diagnostic simulator truth vs. agent-visible
    quantities (``objective_satisfied_before_action``,
    ``sensitive_targets_remaining_before_action``,
    ``sensitive_targets_with_root_before_action``,
    ``base_finish_probability``/``final_finish_probability``,
    ``base_finish_rank``/``final_finish_rank`` -- 1 = highest
    probability/logit, ties broken by legal-action order --
    ``base_finish_is_argmax``/``final_finish_is_argmax``,
    ``finish_selected``); and, for the assisted variant only (``null`` in the baseline): whether/how this step was
    queried, the request/response status and latency, base vs. Plan Maker
    vs. final top action, ``beta``/``alpha``, and whether accepted advice
    changed the top action.

    .. important::

       ``objective_satisfied_before_action`` is a **different question**
       from ``objective_satisfied``/``objective_became_satisfied`` above,
       not a duplicate: the latter two describe the simulator state
       *after* this step's action was applied; ``objective_satisfied_before_action``
       (and its paired ``sensitive_targets_remaining_before_action``/
       ``sensitive_targets_with_root_before_action``) describe the state
       that existed **when the policy chose this step's action** --
       queried from the live simulator immediately before that action was
       applied, never inferred retrospectively from the transition. This
       is the field to condition on when asking "was FINISH the correct
       choice at this decision?" (see ``finish_probability_by_objective_state.png``
       below), and it is what the rollouts.csv aggregates below split on.

       These decision-time fields, like ``sensitive_targets_total``/
       ``sensitive_targets_with_root``/``sensitive_targets_remaining``
       themselves, are **diagnostic simulator truth**: a direct query of
       NASimEmu's real, fully-observable state, recorded here purely for
       evaluator/debug/scientific-reporting purposes. None of it is fed to
       the policy as a neural input -- persisting a diagnostic label in
       this CSV is not the same thing as using it for a decision. What the
       policy is actually allowed to see is defined entirely by
       :mod:`marla.environment.visible_facts`/
       :mod:`marla.environment.action_compatibility`, independent of
       anything recorded here.

       The recurrent policy *does* receive one new, genuinely visible
       global-progress input as of ``POLICY_REPRESENTATION_VERSION`` 3
       (the FINISH-learnability investigation, ``research/diagnostics/ppo_learnability/README.md``):
       :func:`marla.environment.visible_facts.compute_visible_progress`, a
       3-dimensional ``[has_visible_sensitive_target,
       fraction_visible_sensitive_targets_with_root,
       any_visible_sensitive_target_without_root]`` summary fed into
       ``RecurrentCore``'s ``x_t`` alongside the graph embedding. Built
       exclusively from confirmed-visible facts (a host's true sensitivity
       is revealed by NASimEmu's own observation model upon a successful
       exploit against it, never assumed) -- see that function's own
       docstring and ``tests/test_visible_progress.py`` for the exact
       anti-leak argument. This is *not* diagnostic-only like the fields
       above; it is a real (if intentionally narrow) policy input, kept
       separate from ``objective_satisfied_before_action`` and friends.

       As of ``POLICY_REPRESENTATION_VERSION`` 4 (the subnet-exploration
       investigation) there is a **second**, independently-ablatable
       visible-only progress signal alongside the target-progress one
       above: **observable subnet exploration progress**, gated by
       ``policy.recurrent.visible_subnet_exploration`` (default ``False``;
       ``visible_target_progress`` defaults ``True``, preserving version-3
       behavior unless a config opts in). Three terms, all defined purely
       from what the policy has actually discovered/attempted --  never
       from NASimEmu's true, hidden topology:

       - A **known subnet** is any subnet containing at least one
         currently-visible host (``EnvironmentState.host_addresses``); an
         undiscovered subnet is not "known unscanned," it simply does not
         exist in this vocabulary at all.
       - A known subnet is **successfully scanned** once a ``SubnetScan``
         action *originating from a host in it* has returned
         ``info["success"] == True`` at any point in the episode --
         tracked by MARLA itself (``EnvironmentState.successfully_scanned_subnets``,
         reset every episode), because NASimEmu's own ``subnet_graph`` is
         updated on every attempted scan regardless of success and so
         cannot answer this question. A successful scan that discovers
         zero new subnets still counts; a failed/rejected/precondition-failed
         scan never does.
       - The **known exploration frontier** is the set of known subnets not
         yet successfully scanned; ``known_exploration_frontier_remaining``
         is true iff that set is non-empty. Deliberately not called
         "network fully explored" when false -- that would overclaim: a
         false frontier flag means only that every *currently known*
         subnet has been scanned, not that every *true* subnet has been
         found. This is a strictly narrower, honest claim.

       These are computed once by :class:`marla.environment.visible_facts.VisibleNetworkExploration`
       and reused verbatim by the per-subnet-node GraphSAGE feature
       (``subnet_scan_completed``, the 13th node feature, never set on a
       host node and never encoding the subnet's own ID), the 2-dimensional
       recurrent-policy vector (``[fraction_known_subnets_scanned,
       any_known_subnet_unscanned]``, appended after target progress, never
       duplicating it), the decision/rollout metrics below, and
       :func:`marla.environment.observation_summary.build_observation_summary`'s
       Plan Maker-visible ``network_exploration`` block -- one authoritative
       source, not four independent reimplementations. Deliberately *not*
       one fixed feature per subnet (no ``subnet_1_scanned``,
       ``subnet_2_scanned``, ...): that would make the representation's
       width depend on scenario size, defeating the point of a
       fixed-width, scenario-independent progress summary. See
       ``tests/test_subnet_exploration.py`` for the anti-leak arguments
       (an unknown subnet has zero representational influence, proven by
       direct construction, not just by inspection) and the exact dynamic-
       discovery sequence this is built against.

``rollouts.csv``
    One row per completed rollout: ``environment_steps_total`` (the main
    training-progress x-axis), ``steps_collected``, ``episodes_finished``,
    reward/critic/advantage statistics (mean and, where meaningful, std)
    over that rollout's steps, query-gate aggregates (``mean_query_probability``,
    ``actual_query_rate``, ``mean_beta`` -- ``null`` in the baseline), and
    wall-clock timing (``collection_seconds``, ``optimization_seconds``,
    ``evaluation_seconds``). Every rollout-level aggregate lives here
    exactly once -- ``updates.csv`` no longer repeats them per minibatch.
    Also: this rollout's mean compatible-action probability mass
    (``mean_base_compatible_action_probability_mass``/
    ``mean_final_compatible_action_probability_mass``) and the fraction of
    decisions whose selected action's ``visible_preconditions_status`` was
    each of the four values (``selected_confirmed_compatible_rate``,
    ``selected_contradicted_rate``, ``selected_unknown_rate``,
    ``selected_not_applicable_rate`` -- one shared denominator, every
    decision in the rollout, so these four always sum to 1.0), built
    directly from that rollout's own ``decisions.csv`` rows rather than
    recomputed from ``records`` a second time. The same three rates are
    also reported with an **applicable-actions-only** denominator
    (``selected_confirmed_compatible_rate_applicable_actions``,
    ``selected_contradicted_rate_applicable_actions``,
    ``selected_unknown_rate_applicable_actions``) that excludes
    ``not_applicable`` selections (scans/FINISH) from the denominator
    entirely rather than folding them in -- these three sum to 1.0 over
    *applicable* decisions only. Both denominators are kept; neither
    silently replaces the other.

    FINISH diagnostics, partitioned by ``objective_satisfied_before_action``
    (the decision-time truth, not the post-action field --
    see decisions.csv's note above): ``mean_base_finish_probability_objective_reached``/
    ``mean_base_finish_probability_objective_not_reached`` (and the
    ``final_`` equivalents), ``base_finish_argmax_rate_objective_reached``/
    ``_objective_not_reached`` (and ``final_``), and
    ``finish_selected_rate_objective_reached``/``_objective_not_reached``.
    Each ``_objective_reached``/``_objective_not_reached`` pair shares that
    partition's own denominator (how many decisions in the rollout fell on
    that side of the split); a partition with zero decisions this rollout
    is ``null``, never ``0.0``.

    Subnet-exploration aggregates (all computed from the visible-only
    ``VisibleNetworkExploration`` facts described above, never true
    topology): ``mean_fraction_known_subnets_scanned``,
    ``mean_known_subnets_unscanned``, ``known_frontier_remaining_rate``
    (fraction of this rollout's decisions where a known, unscanned subnet
    still existed). Also two FINISH-conditional pairs restricted to
    decisions where ``has_visible_sensitive_target_before_action`` is true
    **and** ``all_visible_sensitive_targets_rooted_before_action`` is true
    (i.e. every *currently visible* sensitive target already has ROOT --
    "no visible sensitive target at all" is never folded into "complete"):
    ``mean_finish_probability_visible_complete_frontier_remaining`` and
    ``mean_finish_probability_visible_complete_no_frontier``, splitting
    that subset by ``known_exploration_frontier_remaining_before_action``.
    Their difference is :math:`\Delta_{\text{frontier}}` (no-frontier minus
    frontier-remaining) -- the main evidence for whether representing the
    exploration frontier helps FINISH discrimination, kept conceptually
    separate from :math:`\Delta_{\text{objective}}` (``mean_base_finish_probability_objective_reached``
    minus ``..._objective_not_reached`` above, using true hidden objective
    state): :math:`\Delta_{\text{frontier}}` asks "does the policy's FINISH
    decision track *what it has itself observed*", :math:`\Delta_{\text{objective}}`
    asks "does that observable reasoning actually correlate with the real,
    hidden, correct stopping point". A partition with zero qualifying
    decisions this rollout is ``null``.

``updates.csv``
    One row per PPO minibatch update: ``update``, ``rollout``, ``epoch``,
    ``minibatch`` (so a row maps back to exactly which PPO replay pass
    produced it), ``environment_steps``, losses, entropies,
    ``approximate_kl``, ``clip_fraction``, ``explained_variance``,
    ``gradient_norm``, and the actual current ``learning_rate`` (reflecting
    the configured schedule, see :doc:`configuration`).

``summary.json``
    Aggregate statistics computed from the above, without ever holding a
    full-run list of per-step records in memory (see "Incremental
    persistence" above): ``objective_reached_rate`` and
    ``successful_finish_rate`` are always reported as two separate numbers
    (each with its own 95% Wilson-score confidence interval) -- never
    collapsed into one, since they answer different questions (see
    ``episodes.csv``'s success-semantics note above). ``goal_success_rate``
    is kept, deprecated, identical to ``successful_finish_rate``. Also:
    premature-finish and timeout rates,
    mean/median return, mean/median ``finish_delay_steps`` and mean
    successful ``steps_to_goal``, action-type frequencies, consultation
    totals and rates (including queries-per-successful-episode and the
    Gatekeeper's advice-acceptance rate), Plan Maker latency
    (mean/median/p95/max), advice-changed-top-action rate, rollout timing
    totals, and (when eval episodes ran) their own separate aggregates.

``checkpoint_last.pt`` / ``checkpoint_best.pt`` / ``checkpoint.pt``
    ``checkpoint_last.pt`` is overwritten at every rollout boundary
    (including intra-run, not just at shutdown) -- the one to use for
    ``marla run --resume`` after an interrupted run. ``checkpoint_best.pt``
    exists only when ``metrics.eval_episodes > 0``, and is overwritten only
    when a periodic evaluation pass beats the running best (goal success
    rate primary, mean evaluation return as tie-breaker). ``checkpoint.pt``
    is written once at shutdown, identical to ``checkpoint_last.pt`` at
    that point -- kept for backward compatibility with tooling that looks
    for the old, single-checkpoint name. All three are only written once
    training actually started (absent if construction failed before that),
    and all three load with :func:`marla.learning.checkpoint.load_checkpoint`.

``plots/``
    Written by ``marla summarize`` (or directly via
    :func:`marla.metrics.plots.generate_plots`), not by ``marla run``
    itself -- see below.

A few spec-listed ``decisions.csv`` columns are not populated in this
release and are omitted from the schema entirely (not written empty):
``schema_revision_count`` (would need a wire-protocol change to report the
Gatekeeper's internal revision count back to the RL Orchestrator) and
``action_success``/``artifact_path`` (NASimEmu doesn't expose a distinct
success signal separate from reward, and no generic boolean was invented
in its place). See :mod:`marla.metrics.writer`'s module docstring for
details. Older run directories written before this schema existed may
still have these columns (harmlessly ignored) or be missing newer ones
entirely -- ``marla summarize`` tolerates both, degrading individual plots
rather than failing outright.

Plots
-----

``marla summarize`` renders whichever of the following the available CSVs
support -- a run with ``metrics.record_decisions: false``, or the baseline
variant, or a very short run, still gets everything its data supports
rather than failing the whole command over one missing file
(:func:`marla.metrics.plots.generate_plots`). Grouped below by the four
questions a MARLA run needs to answer: is the policy learning, is it any
good, is consultation actually helping, and is PPO training itself stable.

Is the RL policy learning?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``episode_returns.png``
    **The single most important training-progress plot.** Average
    *episodic return* over training -- environment return
    (``benchmark_return``, the correct metric for comparing baseline vs.
    assisted vs. other agents) and, whenever it actually differs, the
    cost-adjusted training return PPO optimizes (dotted). A large gap
    between the two means the agent is paying substantial consultation
    cost. Train (solid) vs. eval (dashed), both by rollout when available,
    with a shaded uncertainty band: the real cross-episode standard
    deviation where more than one episode shares a point (e.g. an eval
    checkpoint's batch), a rolling standard deviation across nearby points
    otherwise.

``reward_over_training.png``
    Mean *reward per step* over PPO updates -- the raw signal PPO actually
    trained on, distinct from the per-episode *return* above (the sum of
    this over a whole episode; episode return also depends on episode
    length, this doesn't). Plots ``mean_nasimemu_reward`` and, whenever it
    differs (assisted variant only), ``mean_training_reward``.

``training_dynamics.png``
    Action and query entropy over PPO updates -- should start high during
    early exploration and decrease as the agent commits to known-good
    paths, but a collapse to near-zero *before* performance improves is a
    warning sign, not a good one.

``policy_confidence.png``
    Base policy entropy and top-two-action margin over decisions, in
    collection order, plus action-count-*normalized* entropy
    (``H / log(legal_action_count)``, in ``[0, 1]``) -- raw entropy's
    range depends on how many legal actions exist, which grows as
    NASimEmu discovers more hosts within an episode, so the normalized
    version is what's actually comparable across different points in an
    episode or across episodes.

``compatible_action_probability.png``
    :math:`P(\text{action compatible with observed facts})` -- the mean
    policy probability mass placed on actions whose
    ``visible_preconditions_status`` is ``confirmed_compatible``
    (:mod:`marla.environment.action_compatibility`), over training
    environment steps. This is the direct evidence that the action/target
    compatibility fix is not just representable but actually *used*: it
    should rise as the actor learns to prefer confirmed-compatible actions
    over ones it cannot yet confirm are safe. Base and final-policy series
    both plotted (they coincide exactly for the baseline/PPO_ONLY variant,
    since no advice ever perturbs the logits there). Absent for a run
    written before these ``rollouts.csv`` columns existed.

``finish_probability_by_objective_state.png``
    :math:`P(\text{FINISH})`, conditioned on ``objective_satisfied_before_action``
    (was the true objective already satisfied *when the policy chose this
    step's action* -- decision-time truth, never the post-action field),
    over training environment steps. This is the direct, measured answer
    to "once MARLA's policy has already reached the true objective, does
    it assign high probability to FINISH?" -- never inferred indirectly
    from ``successful_finish_rate``. The ideal learned shape:
    :math:`P(\text{FINISH}\mid\text{objective reached}) \to 1` while
    :math:`P(\text{FINISH}\mid\text{objective not reached}) \to 0`. Base
    and final-policy series both plotted when they differ (assisted
    variant); for PPO_ONLY they coincide exactly. Absent for a run written
    before these ``rollouts.csv`` columns existed.

``finish_probability_by_remaining_targets.png``
    :math:`P(\text{FINISH})` (base policy) as a function of the number of
    true sensitive targets still missing ROOT *at decision time*
    (``sensitive_targets_remaining_before_action``) -- a diagnostic plot,
    not a paper result. Expected qualitative shape: ``remaining = 0`` ->
    high FINISH probability, ``remaining > 0`` -> low. A remaining-target
    count with too few observations this run is omitted rather than
    plotted as a fabricated stable estimate. Absent for a run written
    before these ``decisions.csv`` columns existed.

``finish_probability_by_visible_completion_and_frontier.png``
    :math:`P(\text{FINISH})` (base policy), restricted to decisions where
    every *currently visible* sensitive target already has ROOT, split by
    ``known_exploration_frontier_remaining_before_action`` -- "targets
    complete, exploration frontier remains" (orange) vs. "targets
    complete, no known frontier remains" (green), over training
    environment steps. Unlike ``finish_probability_by_objective_state.png``
    above, this conditions entirely on what the policy has *itself*
    observed, never on hidden simulator truth. The desirable shape is the
    no-frontier curve rising above the frontier-remaining curve (a growing
    :math:`\Delta_{\text{frontier}}`, see ``rollouts.csv`` above) --
    evidence the policy is learning to treat "nothing known left to
    explore" as part of what makes FINISH justified, not merely "every
    visible target is rooted." Skipped cleanly (no file written) for a run
    whose ``rollouts.csv`` predates these columns, or where neither
    partition ever had a qualifying decision.

``known_subnet_exploration_progress.png``
    ``mean_fraction_known_subnets_scanned`` and, when present,
    ``known_frontier_remaining_rate``, over training environment steps --
    a training-health diagnostic for the exploration representation
    itself (is the policy actually driving the known frontier toward
    zero, or is exploration effectively stalled), not a claim about true
    network coverage. y-axis fixed to ``[0, 1]``. Skipped cleanly for a
    run written before these ``rollouts.csv`` columns existed.

Is the learned policy successful and efficient?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``episode_efficiency.png``
    Three panels, train vs. eval: **goal success rate** (the fraction of
    episodes that captured every sensitive host -- arguably more
    understandable than any loss curve, and reported in ``summary.json``
    with a 95% confidence interval); episode length over *all* episodes
    (success and failure); and steps-to-goal over *successful* episodes
    only. Conflating the last two would be misleading -- an agent
    succeeding in 90% of episodes in 20 steps is better than one
    succeeding in 90% in 100 steps, and failed episodes shouldn't dilute
    that comparison.

``episode_outcomes.png``
    Stacked bar chart of *why* each rollout's episodes ended: successful
    completion, premature ``finish`` (chose to end before every sensitive
    host had root access), or timeout (hit ``max_episode_steps``). A low
    success rate has very different fixes depending on which of these
    dominates -- episode_efficiency.png's success-rate curve alone can't
    tell them apart. These three always sum to 1; there is no fourth
    "environment/protocol failure" outcome to report, since NASimEmu never
    internally terminates a non-``FINISH`` action and an absorbed
    action-space precondition failure is a failed *attempt*, not an
    episode-ending error.

Is Plan Maker consultation actually useful?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``consultation_activity.png``
    Consultations and Gatekeeper rejections per episode. Skipped entirely
    for the baseline variant.

``query_behavior.png``
    Mean query probability vs. actual query rate over rollouts (sourced
    from ``rollouts.csv``, not ``updates.csv``) -- a consistently high rate
    may mean the policy has learned to depend on the Plan Maker; a
    decreasing rate with stable performance suggests growing autonomy; a
    near-zero rate from the very start may mean the query gate collapsed
    prematurely. Skipped for the baseline variant.

``query_decision_analysis.png``
    Does the query gate actually ask under uncertainty? Splits base-policy
    entropy by whether that decision was queried (a real gap, queried
    higher than not, is the signature of a gate using uncertainty as
    intended), and scatters query probability against entropy directly.
    The relationship need not be monotonic -- the GRU state and
    consultation cost matter too -- but it should trend upward. Skipped
    for the baseline variant.

``gatekeeper_reliability.png``
    Accepted vs. rejected consultation outcomes over training. A high
    query rate isn't useful if many responses get rejected: the
    consultation cost is paid either way. (There is no elapsed-timeout or
    "Plan Maker unavailable" status to plot alongside these -- consultation
    waits indefinitely by design, and a dead peer fails the whole run
    instead of showing up as a per-decision outcome.)

``plan_maker_latency.png``
    Response latency distribution (with mean/median/p95/p99 marked) and
    its trend over training. Consultation has two costs -- the reward
    penalty (``consultation.cost``, see ``reward_over_training.png``) and
    real wall-clock time -- and a policy can look great on the former
    while being operationally slow.

``advice_influence.png``
    Mean trust weight (beta) and how often accepted advice changes the
    top-ranked action, over queried decisions -- the most direct read on
    "is consultation doing anything." High beta with a low change rate
    typically means advice agrees with the base policy rather than being
    ignored; low beta with a high change rate can still mean advice tips
    close calls despite modest overall trust. Note that "changed the top
    action" does not by itself prove the change was an *improvement* --
    confirming that rigorously needs a paired ablation (same checkpoint
    and seed, consultation on vs. off), which is outside what a single
    run's plots can show. Skipped when there's no accepted advice.

``assisted_policy_influence.png``
    Base vs. final policy entropy, and base vs. final probability of the
    *selected* action, over accepted-advice decisions -- a companion to
    ``advice_influence.png``'s beta/top-action-change view: does
    consultation actually sharpen (lower entropy) or reshape confidence in
    the chosen action, not just occasionally flip the argmax. Skipped when
    there's no accepted advice.

Is PPO training numerically stable?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ppo_losses.png``
    Policy/value loss, approximate KL, and explained variance over PPO
    updates. Don't judge policy quality from loss magnitude alone -- PPO
    loss isn't expected to decrease smoothly the way supervised loss does.
    Use this to spot divergence, extreme spikes, or critic instability;
    task return and success rate remain the primary performance metrics.
    Explained variance near 1 means the critic predicts returns well, near
    0 means it's little better than a constant, and negative means worse
    than that -- occasional negative dips are normal, persistently very
    negative values are not.

``gradient_and_clipping.png``
    Gradient norm (pre-clipping -- the logged value shows how much
    clipping is actually happening, not just its result) and PPO clip
    fraction over PPO updates. No universal ideal clip fraction; interpret
    it alongside KL and task performance.

``learning_rate.png``
    Actual learning rate over PPO updates, reflecting whichever scheduler
    is configured (``policy.ppo.optimizer.scheduler``, see
    :doc:`configuration` and :mod:`marla.learning.lr_scheduler`) -- flat
    for ``constant``, monotonically decreasing for ``linear``/``cosine``/
    ``exponential``, stepped for ``step``. Titled explicitly as constant
    when every value is identical, so that remains visible rather than
    assumed.

``rollout_timing.png``
    Wall-clock seconds per rollout, split into environment/rollout
    collection, PPO optimization, and periodic evaluation
    (``rollouts.csv``'s ``collection_seconds``/``optimization_seconds``/
    ``evaluation_seconds``). Plan Maker latency is never a separate term:
    it's already inside collection time (a consultation happens *during*
    collection), never inside optimization time (PPO replay never calls
    the Plan Maker).

``critic_quality.png``
    Scatter of the critic's predicted value (``critic_value``, at
    collection time) against the GAE return target it was trained toward
    (``return_target``), with a y=x reference line -- how well-calibrated
    the critic is, complementing ``ppo_losses.png``'s explained-variance
    summary statistic with the actual point cloud.

``resource_cpu_over_time.png`` / ``resource_ram_over_time.png`` / ``resource_gpu_over_time.png`` / ``resource_gpu_memory_over_time.png``
    CPU/RAM/GPU telemetry from ``resources.csv``
    (:mod:`marla.monitoring.resources`), x-axis is wall-clock seconds
    since this run's own first sample (resource samples are on a fixed
    ~1s cadence, not aligned to rollout/environment-step boundaries).
    CPU and RAM plots overlay process- and system-level series; the GPU
    utilization plot color-codes points by training phase
    (``rollout_collection``/``ppo_update``/``evaluation``) when available.
    All four are skipped entirely (not present, not empty) when
    ``metrics.resource_monitoring.enabled: false`` or the run predates
    this feature; the GPU plots are additionally skipped on a CPU-only
    machine with no NVML/CUDA data to show. See :doc:`configuration`'s
    ``resource_monitoring`` section and
    ``research/RESEARCH_READINESS.md`` for measured sampling overhead.

``reward_vs_credit_assignment.png``
    Immediate ``training_reward`` vs. the credit PPO actually assigned
    that step (``gae_advantage``), highlighting the negative-reward/
    positive-advantage quadrant: a step that cost something immediately
    (a paid consultation, a failed attempt) but that GAE still credits
    because it led to later success.

``action_type_distribution.png``
    Stacked bar chart of the fraction of decisions choosing each
    ``selected_action_type`` (scan / exploit / privilege escalation /
    finish / ...) per rollout -- shows *how* the policy's behavior shifts
    over training, not just whether return improves.

``finish_efficiency.png``
    Mean ``steps_to_goal`` and ``finish_step`` (successful episodes only),
    plus the gap between them (``finish_delay_steps``) -- distinct from
    ``episode_efficiency.png``'s steps-to-goal panel, this shows how much
    the agent lingers after the objective is already satisfied before
    finally selecting FINISH.

Periodic evaluation
----------------------

Setting ``metrics.eval_episodes`` above ``0`` runs that many deterministic
(greedy, no exploration) episodes with the current policy between
rollouts, on a **fixed** set of held-out seeds reused at every checkpoint
(an apples-to-apples comparison across training progress) -- see
:func:`marla.learning.rollout.run_evaluation_episodes`. This is an
``EvalCallback``-style measurement of the current policy's performance
without exploration noise, not a genuine train/test *generalization* split
(MARLA's config has a single ``environment.scenario``, not separate
train/test scenario sets). It's what gives ``episode_returns.png`` and
``episode_efficiency.png`` their eval series. It costs real wall-clock time
-- in assisted mode, each eval episode's queried steps still consult the
Plan Maker -- which is why it defaults to ``0`` (disabled).

Beyond a single run
----------------------

Everything above is computed from one run's own CSVs. A few genuinely
useful research-evaluation questions need more than that and are
deliberately not implemented as ``marla summarize`` plots, since they need
new infrastructure (multi-run aggregation, held-out scenarios, or repeated
runs with one setting swept), not just more plotting:

- **Consultation ablation**: does accepted advice actually *improve*
  return, versus just changing the chosen action? Needs the same
  checkpoint and scenario seed run with consultation enabled, disabled,
  and (optionally) forced-always-consult, then compared -- see
  ``advice_influence.png``'s note on this above.
- **Generalization**: MARLA's periodic evaluation reuses the training
  scenario on fixed seeds (see above) -- a genuine generalization test
  needs previously unseen seeds, and ideally unseen scenarios entirely.
- **Consultation-cost sensitivity**: how does success rate/return/query
  rate change as ``consultation.cost`` varies? Needs a sweep of separate
  runs, one per cost value.
- **Performance-cost frontier**: mean consultations (or latency) per
  episode on one axis, success rate/return on the other, across multiple
  runs or checkpoints -- shows whether an agent is trading query volume
  for performance efficiently.

If you build any of these, ``marla summarize``'s existing per-run
``summary.json`` files are the natural input to aggregate across.
