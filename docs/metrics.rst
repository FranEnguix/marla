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
    rate, mean/median return, consultation totals and rates, mean Plan
    Maker latency, schema rejection rate, advice-changed-top-action rate,
    and (when eval episodes ran) their own separate aggregates.

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
(:func:`marla.metrics.plots.generate_plots`).

``episode_returns.png``
    Average reward (``benchmark_return``) over training. Train (solid,
    shaded uncertainty band) vs. eval (dashed), both by rollout when
    available. The band is the real cross-episode standard deviation at a
    given point when more than one episode shares it (e.g. an eval
    checkpoint's batch of episodes), and a rolling standard deviation
    across nearby points otherwise -- a smoothed learning curve with an
    honest uncertainty estimate, not just a raw noisy per-episode line.

``episode_efficiency.png``
    Same train/eval split and banding as above, for goal success rate and
    episode length (environment steps).

``consultation_activity.png``
    Consultations and Gatekeeper rejections per episode. Skipped entirely
    for the baseline variant (there's nothing to show).

``advice_influence.png``
    Mean trust weight (beta) and how often accepted advice actually
    changes the top-ranked action, both over queried decisions. This is
    the most direct answer to "is consultation doing anything." Skipped
    when there's no accepted advice to show (baseline, or a run where every
    consultation was rejected).

``ppo_losses.png``
    Policy/value loss, approximate KL, and explained variance over PPO
    updates.

``training_dynamics.png``
    Action and query entropy over PPO updates.

``gradient_and_clipping.png``
    Gradient norm and PPO clip fraction over PPO updates (twin axes).

``learning_rate.png``
    Learning rate over PPO updates. Currently always a flat line -- there
    is no learning-rate scheduler in this release -- but titled explicitly
    as constant rather than silently omitted, so that remains visible
    rather than assumed.

``query_behavior.png``
    Mean query probability vs. actual query rate over PPO updates.
    Skipped for the baseline variant (no query gate at all).

``policy_confidence.png``
    Base policy entropy and top-two-action margin over decisions, in
    collection order.

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
