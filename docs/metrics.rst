Metrics and plots
==================

Every run writes a self-contained directory:
``<metrics.output_directory>/<experiment-name>/<run-id>/`` (default
``runs/<experiment-name>/<run-id>/``), regardless of whether the run
completed, was stopped by the user, or failed. All of it is produced by
:mod:`marla.metrics.writer`.

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
    (tagged via ``is_eval``). Columns include ``goal_success``,
    ``nasimemu_return``/``training_return``/``benchmark_return``,
    ``environment_steps``, ``steps_to_goal``, ``consultation_count``/``consultation_cost``,
    ``schema_rejection_count``, ``finish_reason``, and ``rollout`` (which
    training rollout produced -- or immediately followed, for an eval
    episode -- this row, so returns can be bucketed by training progress
    rather than raw episode index).

``decisions.csv``
    One row per environment step (omitted if
    ``metrics.record_decisions: false``) -- the full compound-decision
    record: whether this step was queried, the request/response status and
    latency, base vs. Plan Maker vs. final top action and their ranks,
    ``beta``/``alpha``, whether accepted advice changed the top action, and
    the reward/termination outcome. This is the ground truth for
    understanding *when* and *how much* the query gate and trust head are
    actually doing.

``updates.csv``
    One row per PPO gradient step: losses, entropies, KL, clip fraction,
    explained variance, gradient norm, learning rate, and per-rollout
    aggregates (mean beta, mean/actual query rate).

``summary.json``
    Aggregate statistics computed from the above: episode/goal-success
    rate (with a 95% Wilson-score confidence interval -- more reliable
    than a normal approximation when the rate is near 0 or 1, which a
    short run's episode count often is), premature-finish and timeout
    rates, mean/median return, consultation totals and rates (including
    queries-per-successful-episode and the Gatekeeper's advice-acceptance
    rate), Plan Maker latency (mean/median/p95/max), advice-changed-top-
    action rate, and (when eval episodes ran) their own separate
    aggregates.

``checkpoint.pt``
    The final policy + optimizer state (only written once training
    actually started -- absent if construction failed before that).
    Load it with :func:`marla.learning.checkpoint.load_checkpoint`.

``plots/``
    Written by ``marla summarize`` (or directly via
    :func:`marla.metrics.plots.generate_plots`), not by ``marla run``
    itself -- see below.

A handful of spec-listed ``decisions.csv``/``updates.csv`` columns are
always written empty in this release (not populated, not removed, so the
schema stays stable): ``schema_revision_count``, ``action_success``,
``artifact_path``, and per-update ``checkpoint_id`` (there is one
end-of-run checkpoint, not one per update). See
:mod:`marla.metrics.writer`'s module docstring for exactly why each one is
unpopulated.

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
    Mean query probability vs. actual query rate over PPO updates -- a
    consistently high rate may mean the policy has learned to depend on
    the Plan Maker; a decreasing rate with stable performance suggests
    growing autonomy; a near-zero rate from the very start may mean the
    query gate collapsed prematurely. Skipped for the baseline variant.

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
    Learning rate over PPO updates. Currently always a flat line -- there
    is no learning-rate scheduler in this release -- but titled explicitly
    as constant rather than silently omitted, so that remains visible
    rather than assumed.

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
