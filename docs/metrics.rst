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
    and, for the assisted variant only (``null`` in the baseline): whether/how this step was
    queried, the request/response status and latency, base vs. Plan Maker
    vs. final top action, ``beta``/``alpha``, and whether accepted advice
    changed the top action.

``rollouts.csv``
    One row per completed rollout: ``environment_steps_total`` (the main
    training-progress x-axis), ``steps_collected``, ``episodes_finished``,
    reward/critic/advantage statistics (mean and, where meaningful, std)
    over that rollout's steps, query-gate aggregates (``mean_query_probability``,
    ``actual_query_rate``, ``mean_beta`` -- ``null`` in the baseline), and
    wall-clock timing (``collection_seconds``, ``optimization_seconds``,
    ``evaluation_seconds``). Every rollout-level aggregate lives here
    exactly once -- ``updates.csv`` no longer repeats them per minibatch.

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
    Actual learning rate over PPO updates, reflecting whichever schedule is
    configured (``policy.ppo.learning_rate_schedule``, see
    :doc:`configuration`) -- a flat line for ``constant``, monotonically
    decreasing toward zero for ``linear``. Titled explicitly as constant
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
