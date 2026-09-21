import asyncio
import unittest.mock
from pathlib import Path

import pytest
import torch
import torch_geometric

from marla.config.loader import load_config
from marla.environment.action_compatibility import compute_compatibility_matrix
from marla.environment.actions import ActionDescriptor
from marla.environment.graph import NODE_FEATURE_DIM, GraphObservation
from marla.environment.nasimemu_adapter import EnvironmentState, NasimEmuAdapter, TransitionResult
from marla.environment.visible_facts import (
    assemble_visible_progress,
    extract_visible_host_facts,
    extract_visible_network_exploration,
)
from marla.evaluation.overrides import ALWAYS_QUERY, BETA_ONE, BETA_ZERO, NO_QUERY, PLAN_MAKER_ONLY, EvaluationOverrides
from marla.learning.recurrent_policy import RecurrentPolicy, RecurrentState
from marla.learning.rollout import EVAL_SEED_OFFSET, ConsultationResult, RolloutCollector, run_evaluation_episodes

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


def make_collector(max_episode_steps=6, seed=1, stop_event=None):
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO,
        max_episode_steps=max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    policy = RecurrentPolicy(config.policy)
    collector = RolloutCollector(adapter, policy, run_id="test-run", base_seed=seed, stop_event=stop_event)
    return collector, policy


@pytest.mark.asyncio
async def test_collect_returns_exactly_num_steps_records():
    collector, _policy = make_collector()
    records, _summaries = await collector.collect(10)
    assert len(records) == 10


@pytest.mark.asyncio
async def test_baseline_records_never_query_and_have_zero_consultation_cost():
    collector, _policy = make_collector()
    records, _summaries = await collector.collect(15)
    for r in records:
        assert r.sampled_query is False
        assert r.consultation_cost == 0.0
        assert r.training_reward == r.nasimemu_reward
        assert r.final_logits.equal(r.base_logits)


@pytest.mark.asyncio
async def test_episode_boundary_resets_gru_hidden_state():
    collector, policy = make_collector(max_episode_steps=3)
    records, summaries = await collector.collect(20)
    assert len(summaries) >= 1  # with max_episode_steps=3, episodes end quickly

    zero_z = torch.zeros_like(records[0].initial_gru_hidden_state)
    start_token = policy.recurrent_core.start_action_token

    for record in records:
        if record.environment_step == 0:
            assert torch.equal(record.initial_gru_hidden_state, zero_z)
            assert torch.equal(record.previous_action_embedding, start_token)


@pytest.mark.asyncio
async def test_bootstrap_value_set_exactly_when_needed():
    collector, _policy = make_collector(max_episode_steps=3)
    records, _summaries = await collector.collect(20)
    for r in records:
        if r.terminated:
            assert r.bootstrap_value is None
        elif r.truncated:
            assert r.bootstrap_value is not None
    # the last record of the window must carry a bootstrap value unless terminated
    if not records[-1].terminated:
        assert records[-1].bootstrap_value is not None


@pytest.mark.asyncio
async def test_collect_is_resumable_across_calls():
    # A random, untrained policy can legally select FINISH at any step, so an
    # episode's length is not guaranteed even with a large max_episode_steps;
    # what must hold is that collection state (episode id / step counters)
    # carries over correctly between separate collect() calls.
    collector, _policy = make_collector(max_episode_steps=100)
    first, _ = await collector.collect(5)
    second, _ = await collector.collect(5)

    assert len(first) == 5
    assert len(second) == 5

    last_of_first = first[-1]
    first_of_second = second[0]
    if first_of_second.episode_id == last_of_first.episode_id:
        assert first_of_second.environment_step == last_of_first.environment_step + 1
    else:
        assert first_of_second.environment_step == 0
        assert first_of_second.episode_id == last_of_first.episode_id + 1


@pytest.mark.asyncio
async def test_episode_ids_increment_across_resets():
    collector, _policy = make_collector(max_episode_steps=2)
    records, _summaries = await collector.collect(10)
    episode_ids = sorted({r.episode_id for r in records})
    assert episode_ids == list(range(1, len(episode_ids) + 1))


@pytest.mark.asyncio
async def test_consultation_enabled_requires_consult_fn():
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO,
        max_episode_steps=6,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    policy = RecurrentPolicy(config.policy, consultation_enabled=True)
    with pytest.raises(ValueError):
        RolloutCollector(adapter, policy, run_id="test-run", base_seed=1, consultation_enabled=True)


@pytest.mark.asyncio
async def test_collect_stops_early_when_stop_event_is_already_set():
    stop_event = asyncio.Event()
    stop_event.set()
    collector, _policy = make_collector(stop_event=stop_event)
    records, _summaries = await collector.collect(10)
    assert records == []


@pytest.mark.asyncio
async def test_collect_runs_to_completion_when_stop_event_is_never_set():
    stop_event = asyncio.Event()
    collector, _policy = make_collector(stop_event=stop_event)
    records, _summaries = await collector.collect(10)
    assert len(records) == 10


@pytest.mark.asyncio
async def test_collect_stops_partway_once_stop_event_is_set_mid_rollout():
    stop_event = asyncio.Event()
    collector, _policy = make_collector(stop_event=stop_event)

    # Advance a few steps manually, then request a stop and collect the rest
    # of the same window -- mirrors a Ctrl+C arriving mid-rollout.
    first, _ = await collector.collect(3)
    assert len(first) == 3
    stop_event.set()
    rest, _ = await collector.collect(7)
    assert rest == []


@pytest.mark.asyncio
async def test_collect_sets_bootstrap_value_when_stopped_mid_step_not_at_episode_end():
    # Regression test: a stop requested *during* a step's (blocking, real)
    # consultation makes that step's record the last one of the window, just
    # like reaching num_steps -- and GAE requires a bootstrap value for
    # whichever record is last in a non-terminated window (learning/gae.py).
    # Without this, a Ctrl+C landing mid-consultation crashed the run with
    # "bootstrap_values[0] is required for the last row of a rollout window".
    config = load_config(REPO_ROOT / "examples" / "assisted.yaml")
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO,
        max_episode_steps=50,  # long enough that this step won't be a natural episode end
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    # Seeded explicitly (this test previously relied on whatever torch
    # global RNG state happened to exist when it ran -- order-dependent and
    # not hermetic, exposed by adding unrelated tests earlier in this file
    # that also seed torch). The assertion below needs the random initial
    # policy's first action to not itself be a natural episode end, which a
    # fixed seed makes reproducible instead of incidental.
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy, consultation_enabled=True)
    stop_event = asyncio.Event()

    from marla.learning.rollout import ConsultationResult

    async def consult_fn(legal_actions, episode_id, step, source_observation_id, observation):
        # Simulate Ctrl+C arriving while awaiting the Plan Maker's response.
        stop_event.set()
        return ConsultationResult(status="schema_rejected", scores=None, request_id="request-1")

    collector = RolloutCollector(
        adapter, policy, run_id="test-run", base_seed=1,
        consultation_enabled=True, consultation_cost=0.1, consult_fn=consult_fn, stop_event=stop_event,
    )

    # Force every step to query, so the very first step exercises consult_fn.
    import unittest.mock

    with unittest.mock.patch("marla.learning.rollout.compute_query_probability", return_value=torch.tensor(1.0)):
        records, _summaries = await collector.collect(10)

    assert len(records) == 1
    assert records[0].terminated is False
    assert records[0].bootstrap_value is not None


# --- Regression coverage for the assisted recurrent bootstrap bug: ---------
# --- previous_query is a genuine GRU input (recurrent_core.py concatenates ---
# --- it into x_t), so _bootstrap_value's V(s_{t+1}) must be computed from ---
# --- the ACTUAL sampled_query at t -- never a hardcoded query=False, which ---
# --- would silently substitute a different recurrent history than the one ---
# --- normal live continuation (and PPO replay) would have used. See      ---
# --- RolloutCollector.collect()'s "next_rstate" comment for the invariant. -


def _build_assisted_collector(max_episode_steps: int, base_seed: int = 1, overrides=None):
    """A consultation-enabled collector with a deterministic seeded policy
    and a lightweight stub consult_fn (never the real Plan Maker) -- forces
    every consultation to resolve immediately with a fixed, schema-rejected
    result, since these tests only need sampled_query to be True, not a real
    accepted advisory response."""
    config = load_config(REPO_ROOT / "examples" / "assisted.yaml")
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO,
        max_episode_steps=max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy, consultation_enabled=True)

    async def consult_fn(legal_actions, episode_id, step, source_observation_id, observation):
        return ConsultationResult(status="schema_rejected", scores=None, request_id="request-1")

    collector = RolloutCollector(
        adapter, policy, run_id="test-run", base_seed=base_seed,
        consultation_enabled=True, consultation_cost=0.1, consult_fn=consult_fn, overrides=overrides,
    )
    return collector, policy


def _manual_next_rstate(policy, step_out, action_index, training_reward, query):
    return policy.advance_recurrent_state(step_out, action_index, training_reward, query=query)


def _manual_bootstrap_probe(collector, next_state, next_rstate):
    """Reimplements exactly what RolloutCollector._bootstrap_value does,
    independently, so the test doesn't just call the method under test and
    trust its own output -- this recomputes V(s_{t+1}) from scratch given an
    explicit next_rstate, the same way _bootstrap_value is documented to."""
    policy = collector._policy
    next_legal = collector._adapter.legal_actions(next_state)
    next_graph_obs = collector._adapter.to_pyg_data(
        next_state, include_subnet_scan_feature=policy.visible_subnet_exploration_enabled
    )
    next_facts = extract_visible_host_facts(next_state)
    next_compatibility = compute_compatibility_matrix(next_legal, next_facts)
    next_exploration = extract_visible_network_exploration(next_state)
    next_visible_progress = assemble_visible_progress(
        next_facts, next_exploration, policy.visible_target_progress_enabled, policy.visible_subnet_exploration_enabled,
    )
    with torch.no_grad():
        probe = policy.step(next_graph_obs, next_legal, next_rstate, next_compatibility, next_visible_progress)
    return float(probe.value)


@pytest.mark.asyncio
async def test_bootstrap_value_distinguishes_previous_query_true_from_false():
    """Core numeric proof: V(s_{t+1} | previous_query=True) must differ from
    V(s_{t+1} | previous_query=False) for this policy/state, and
    _bootstrap_value must actually reflect whichever next_rstate it is
    given -- not just "return something non-None" (the insufficient style
    of test this regression guards against)."""
    collector, policy = _build_assisted_collector(max_episode_steps=50)

    state = collector._adapter.reset(seed=1)
    rstate = policy.initial_recurrent_state()
    legal_actions = collector._adapter.legal_actions(state)
    graph_obs = collector._adapter.to_pyg_data(state, include_subnet_scan_feature=policy.visible_subnet_exploration_enabled)
    facts = extract_visible_host_facts(state)
    compatibility = compute_compatibility_matrix(legal_actions, facts)
    exploration = extract_visible_network_exploration(state)
    visible_progress = assemble_visible_progress(
        facts, exploration, policy.visible_target_progress_enabled, policy.visible_subnet_exploration_enabled,
    )
    step_out = policy.step(graph_obs, legal_actions, rstate, compatibility, visible_progress)

    action_index = 0
    training_reward = 0.5
    transition = collector._adapter.step(legal_actions[action_index])
    assert not transition.terminated and transition.state is not None, (
        "test precondition: the probed action must not end the episode, so there is a real s_{t+1} to bootstrap"
    )

    next_rstate_true = _manual_next_rstate(policy, step_out, action_index, training_reward, query=True)
    next_rstate_false = _manual_next_rstate(policy, step_out, action_index, training_reward, query=False)

    # Isolate the causal variable: only previous_query should differ between
    # the two constructed states -- both come from the same step_out/action/reward.
    assert next_rstate_true.previous_query == 1.0
    assert next_rstate_false.previous_query == 0.0
    assert torch.equal(next_rstate_true.z, next_rstate_false.z)
    assert torch.equal(next_rstate_true.previous_action_embedding, next_rstate_false.previous_action_embedding)
    assert next_rstate_true.previous_reward == next_rstate_false.previous_reward

    bootstrap_true = collector._bootstrap_value(transition.state, next_rstate_true)
    bootstrap_false = collector._bootstrap_value(transition.state, next_rstate_false)

    # Independent recomputation (not just re-calling the method under test).
    expected_true = _manual_bootstrap_probe(collector, transition.state, next_rstate_true)
    expected_false = _manual_bootstrap_probe(collector, transition.state, next_rstate_false)

    assert bootstrap_true == pytest.approx(expected_true)
    assert bootstrap_false == pytest.approx(expected_false)
    assert bootstrap_true != bootstrap_false, (
        "V(s_{t+1}) must be sensitive to previous_query for this test to be meaningful -- "
        "if this ever starts failing, previous_query may have stopped being a real GRU input"
    )


@pytest.mark.asyncio
async def test_collect_rollout_cutoff_bootstrap_uses_actual_sampled_query_true():
    """Regression for the bug itself: at a rollout-window cutoff (the last
    record of the window, not terminated, not truncated -- i.e.
    is_last_in_window) with sampled_query=True, the bootstrap value must be
    computed from a next_rstate with previous_query=True, matching what
    normal live continuation would have installed. Fails on the buggy
    implementation, which hardcoded query=False here."""
    collector, policy = _build_assisted_collector(max_episode_steps=50, overrides=ALWAYS_QUERY)

    captured_next_rstate = []
    real_bootstrap = RolloutCollector._bootstrap_value

    def spy(self, next_state, next_rstate):
        captured_next_rstate.append(next_rstate)
        return real_bootstrap(self, next_state, next_rstate)

    with unittest.mock.patch.object(RolloutCollector, "_bootstrap_value", spy):
        records, _summaries = await collector.collect(3)

    assert records[-1].sampled_query is True
    assert records[-1].terminated is False, "test precondition: the window must end mid-episode, not on FINISH"
    assert records[-1].bootstrap_value is not None
    assert len(captured_next_rstate) == 1

    used_rstate = captured_next_rstate[0]
    assert used_rstate.previous_query == 1.0, (
        "bootstrap must use previous_query=True (the actual sampled_query), not a hardcoded False"
    )

    # Numerically distinguish: the buggy (query=False) counterfactual must
    # differ from the actual recorded bootstrap_value, not merely "differ in
    # principle" -- reconstruct it explicitly and compare.
    counterfactual_false_rstate = RecurrentState(
        z=used_rstate.z, previous_action_embedding=used_rstate.previous_action_embedding,
        previous_reward=used_rstate.previous_reward, previous_query=0.0,
    )
    # The record's own stored decisions.csv-equivalent fields don't retain
    # the raw next_state, so recompute the counterfactual against the same
    # underlying environment state the real bootstrap just probed -- the
    # collector's own _adapter is already positioned there (episode continues).
    counterfactual_bootstrap = _manual_bootstrap_probe(collector, collector._state, counterfactual_false_rstate)

    assert records[-1].bootstrap_value == pytest.approx(_manual_bootstrap_probe(collector, collector._state, used_rstate))
    assert records[-1].bootstrap_value != pytest.approx(counterfactual_bootstrap)


@pytest.mark.asyncio
async def test_collect_truncation_boundary_bootstrap_uses_actual_sampled_query_true():
    """Same invariant as the rollout-cutoff test above, for a time-limit
    truncation boundary specifically (transition.truncated=True,
    transition.terminated=False) -- truncation is a separate code path in
    NasimEmuAdapter.step() (step_idx >= max_episode_steps) from the rollout-
    window cutoff, and both feed the same _bootstrap_value call site."""
    collector, policy = _build_assisted_collector(max_episode_steps=1, overrides=ALWAYS_QUERY)

    captured_next_rstate = []
    real_bootstrap = RolloutCollector._bootstrap_value

    def spy(self, next_state, next_rstate):
        captured_next_rstate.append(next_rstate)
        return real_bootstrap(self, next_state, next_rstate)

    with unittest.mock.patch.object(RolloutCollector, "_bootstrap_value", spy):
        records, _summaries = await collector.collect(1)

    assert records[0].sampled_query is True
    assert records[0].truncated is True
    assert records[0].terminated is False
    assert records[0].bootstrap_value is not None
    assert len(captured_next_rstate) == 1

    used_rstate = captured_next_rstate[0]
    assert used_rstate.previous_query == 1.0

    counterfactual_false_rstate = RecurrentState(
        z=used_rstate.z, previous_action_embedding=used_rstate.previous_action_embedding,
        previous_reward=used_rstate.previous_reward, previous_query=0.0,
    )
    # At truncation the episode ends, so collector._state was reset to None
    # by the time collect() returns -- probe against a fresh reset of the
    # same seed instead (deterministic, same scenario instance either way
    # would work; what matters here is only that both probes share the
    # identical next_state, isolating previous_query as the sole variable).
    next_state_for_probe = collector._adapter.reset(seed=1)
    actual_bootstrap = _manual_bootstrap_probe(collector, next_state_for_probe, used_rstate)
    counterfactual_bootstrap = _manual_bootstrap_probe(collector, next_state_for_probe, counterfactual_false_rstate)
    assert actual_bootstrap != pytest.approx(counterfactual_bootstrap)


@pytest.mark.asyncio
async def test_collect_baseline_bootstrap_uses_previous_query_false():
    """Guard against accidentally forcing True globally: PPO_ONLY (baseline,
    consultation disabled) must still bootstrap with previous_query=False --
    which is correct there because sampled_query is always False on that
    path (rollout.py's _decide() short-circuits to sampled_query=False
    whenever consultation_enabled is False), so False is the actual history,
    not a hardcoded assumption."""
    collector, _policy = make_collector(max_episode_steps=3)

    captured_next_rstate = []
    real_bootstrap = RolloutCollector._bootstrap_value

    def spy(self, next_state, next_rstate):
        captured_next_rstate.append(next_rstate)
        return real_bootstrap(self, next_state, next_rstate)

    with unittest.mock.patch.object(RolloutCollector, "_bootstrap_value", spy):
        records, _summaries = await collector.collect(20)

    boundary_records = [r for r in records if not r.terminated and r.bootstrap_value is not None]
    assert boundary_records, "test precondition: at least one non-terminal boundary must occur"
    for r in boundary_records:
        assert r.sampled_query is False
    assert captured_next_rstate  # _bootstrap_value was actually exercised
    for rstate in captured_next_rstate:
        assert rstate.previous_query == 0.0


@pytest.mark.asyncio
async def test_bootstrap_and_continuation_share_the_same_next_rstate():
    """Structural-reuse guard for the preferred fix design: next_rstate is
    computed exactly once per non-terminated transition and reused for both
    the bootstrap probe and (when the episode continues) normal live
    continuation -- never two independently-constructed RecurrentStates that
    could drift apart again. Verified via a window boundary that is
    is_last_in_window (bootstrap fires) but not terminated/truncated (the
    episode continues into the collector's own self._rstate for the next
    collect() call)."""
    collector, policy = _build_assisted_collector(max_episode_steps=50, overrides=ALWAYS_QUERY)

    captured_next_rstate = []
    real_bootstrap = RolloutCollector._bootstrap_value

    def spy(self, next_state, next_rstate):
        captured_next_rstate.append(next_rstate)
        return real_bootstrap(self, next_state, next_rstate)

    with unittest.mock.patch.object(RolloutCollector, "_bootstrap_value", spy):
        records, _summaries = await collector.collect(3)

    assert records[-1].terminated is False and records[-1].truncated is False, (
        "test precondition: the window must end mid-episode (bootstrap fires from is_last_in_window "
        "alone) so the SAME next_rstate is also installed for continuation"
    )
    assert len(captured_next_rstate) == 1

    used_for_bootstrap = captured_next_rstate[0]
    used_for_continuation = collector._rstate
    assert used_for_continuation is not None

    assert torch.equal(used_for_bootstrap.z, used_for_continuation.z)
    assert torch.equal(used_for_bootstrap.previous_action_embedding, used_for_continuation.previous_action_embedding)
    assert used_for_bootstrap.previous_reward == used_for_continuation.previous_reward
    assert used_for_bootstrap.previous_query == used_for_continuation.previous_query


@pytest.mark.asyncio
async def test_deterministic_collector_is_reproducible_across_runs():
    # A deterministic (greedy) collector must always pick the same action
    # for the same policy weights/state -- unlike the stochastic training
    # path, two independent collect() calls from the same seed must produce
    # an identical action sequence.
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)

    def make_deterministic_collector():
        adapter = NasimEmuAdapter(
            scenario=SMALL_SCENARIO, max_episode_steps=6,
            completion_reward=config.objective.completion_reward,
            premature_finish_penalty=config.objective.premature_finish_penalty,
        )
        return RolloutCollector(adapter, policy, run_id="test-run", base_seed=1, deterministic=True)

    first, _ = await make_deterministic_collector().collect(10)
    second, _ = await make_deterministic_collector().collect(10)

    assert [r.selected_action_index for r in first] == [r.selected_action_index for r in second]


@pytest.mark.asyncio
async def test_run_evaluation_episodes_tags_summaries_as_eval_and_restores_train_mode():
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO, max_episode_steps=6,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    policy = RecurrentPolicy(config.policy)

    summaries = await run_evaluation_episodes(
        policy=policy, adapter=adapter, run_id="test-run", num_episodes=2, seed_start=EVAL_SEED_OFFSET,
    )

    assert len(summaries) == 2
    assert all(s.is_eval for s in summaries)
    assert policy.training is True  # eval() must not leak into subsequent training


@pytest.mark.asyncio
async def test_run_evaluation_episodes_uses_fixed_seeds_regardless_of_training_progress():
    # The same eval seed range must be reused every checkpoint (an
    # apples-to-apples fixed test set), independent of how far training has
    # progressed -- unlike training episode seeds, which always increment.
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)

    def make_adapter():
        return NasimEmuAdapter(
            scenario=SMALL_SCENARIO, max_episode_steps=6,
            completion_reward=config.objective.completion_reward,
            premature_finish_penalty=config.objective.premature_finish_penalty,
        )

    first = await run_evaluation_episodes(
        policy=policy, adapter=make_adapter(), run_id="test-run", num_episodes=2, seed_start=EVAL_SEED_OFFSET,
    )
    second = await run_evaluation_episodes(
        policy=policy, adapter=make_adapter(), run_id="test-run", num_episodes=2, seed_start=EVAL_SEED_OFFSET,
    )
    assert [s.seed for s in first] == [s.seed for s in second]


# --- EvaluationOverrides (research/aamas2027) -------------------------------
# These test the *default-None* guarantee (overrides=None is byte-identical
# to today's code, i.e. trainer.py's path is unaffected) plus each ablation
# in isolation. None of this is reachable from marla.learning.trainer.


async def _default_rejecting_consult_fn(legal_actions, episode_id, step, source_observation_id, observation):
    return ConsultationResult(status="schema_rejected", scores=None, request_id="unused")


def make_assisted_collector(overrides=None, max_episode_steps=50, seed=1, consult_fn=None, policy_seed=None):
    config = load_config(REPO_ROOT / "examples" / "assisted.yaml")
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO,
        max_episode_steps=max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    if policy_seed is not None:
        # Isolates a comparison to "overrides differ" alone -- without this,
        # two independently-constructed policies get different random
        # initial weights purely from torch's global RNG having advanced
        # between the two constructions, which has nothing to do with
        # overrides at all.
        torch.manual_seed(policy_seed)
    policy = RecurrentPolicy(config.policy, consultation_enabled=True)
    collector = RolloutCollector(
        adapter, policy, run_id="test-run", base_seed=seed,
        consultation_enabled=True, consultation_cost=0.1,
        consult_fn=consult_fn or _default_rejecting_consult_fn,
        deterministic=True, overrides=overrides,
    )
    return collector, policy


def _accepting_consult_fn_factory(call_counter: list):
    """Always accepts, scoring the first legal action highest and the rest
    descending -- a predictable ranking to assert PLAN_MAKER_ONLY/BETA
    behavior against."""

    async def consult_fn(legal_actions, episode_id, step, source_observation_id, observation):
        call_counter.append(1)
        n = len(legal_actions)
        scores = {a.action_id: (n - i) / n for i, a in enumerate(legal_actions)}
        return ConsultationResult(status="accepted", scores=scores, request_id=f"request-{len(call_counter)}")

    return consult_fn


@pytest.mark.asyncio
async def test_overrides_none_matches_default_learned_behavior():
    # overrides=None must be indistinguishable from not passing the
    # parameter at all -- this is the guarantee the whole hook design rests
    # on (trainer.py never passes overrides).
    collector_a, _ = make_assisted_collector(overrides=None, seed=7, policy_seed=99)
    collector_b, _ = make_assisted_collector(overrides=EvaluationOverrides(), seed=7, policy_seed=99)
    records_a, _ = await collector_a.collect(15)
    records_b, _ = await collector_b.collect(15)
    assert [r.sampled_query for r in records_a] == [r.sampled_query for r in records_b]


@pytest.mark.asyncio
async def test_no_query_never_calls_consult_fn():
    calls: list = []
    collector, _ = make_assisted_collector(overrides=NO_QUERY, consult_fn=_accepting_consult_fn_factory(calls))
    records, _ = await collector.collect(20)
    assert all(r.sampled_query is False for r in records)
    assert all(r.plan_maker_response_status is None for r in records)
    assert calls == []


@pytest.mark.asyncio
async def test_always_query_calls_consult_fn_every_step():
    calls: list = []
    collector, _ = make_assisted_collector(overrides=ALWAYS_QUERY, consult_fn=_accepting_consult_fn_factory(calls))
    records, _ = await collector.collect(10)
    assert all(r.sampled_query is True for r in records)
    assert len(calls) == 10


@pytest.mark.asyncio
async def test_beta_zero_leaves_final_logits_equal_to_base_logits():
    # query_mode="always" forced here (not just BETA_ZERO's bare
    # beta_override=0.0) so this test doesn't depend on a freshly
    # random-initialized gate happening to cross the 0.5 query threshold --
    # "does the gate decide to query" is a separate, already-tested concern.
    overrides = EvaluationOverrides(query_mode="always", beta_override=BETA_ZERO.beta_override)
    calls: list = []
    collector, _ = make_assisted_collector(overrides=overrides, consult_fn=_accepting_consult_fn_factory(calls))
    records, _ = await collector.collect(10)
    queried = [r for r in records if r.sampled_query and r.plan_maker_validation_status == "accepted"]
    assert queried, "expected at least one accepted consultation to test beta=0 against"
    for r in queried:
        assert torch.allclose(r.final_logits, r.base_logits)
        assert r.beta == 0.0


@pytest.mark.asyncio
async def test_beta_one_applies_full_advice_residual():
    overrides = EvaluationOverrides(query_mode="always", beta_override=BETA_ONE.beta_override)
    calls: list = []
    collector, _ = make_assisted_collector(overrides=overrides, consult_fn=_accepting_consult_fn_factory(calls))
    records, _ = await collector.collect(10)
    queried = [r for r in records if r.sampled_query and r.plan_maker_validation_status == "accepted"]
    assert queried
    for r in queried:
        assert r.beta == 1.0
        # base logit for the top-ranked action is boosted relative to the
        # worst-ranked one, since advice was strictly descending by rank.
        assert not torch.allclose(r.final_logits, r.base_logits)


@pytest.mark.asyncio
async def test_beta_override_is_a_noop_when_advice_was_not_obtained():
    # NO_QUERY combined with a beta override: nothing was ever queried, so
    # there is no advice to apply -- this must not raise or fabricate advice.
    collector, _ = make_assisted_collector(overrides=EvaluationOverrides(query_mode="never", beta_override=1.0))
    records, _ = await collector.collect(10)
    for r in records:
        assert r.sampled_query is False
        assert torch.equal(r.final_logits, r.base_logits)


@pytest.mark.asyncio
async def test_plan_maker_only_ignores_base_policy_for_action_choice():
    calls: list = []
    collector, _ = make_assisted_collector(
        overrides=PLAN_MAKER_ONLY, consult_fn=_accepting_consult_fn_factory(calls)
    )
    records, _ = await collector.collect(10)
    assert len(calls) == 10  # PLAN_MAKER_ONLY forces query_mode="always"
    for r in records:
        # The fake consult_fn always scores the first legal action highest.
        selected_id = r.legal_action_descriptors[r.selected_action_index].action_id
        assert selected_id == r.legal_action_descriptors[0].action_id


@pytest.mark.asyncio
async def test_plan_maker_only_falls_back_to_seeded_random_choice_on_rejection():
    async def rejecting_consult_fn(legal_actions, episode_id, step, source_observation_id, observation):
        return ConsultationResult(status="schema_rejected", scores=None, request_id="request-1")

    overrides = EvaluationOverrides(
        query_mode="always", action_selection="plan_maker_argmax", fallback_rng_seed=42
    )
    collector_a, _ = make_assisted_collector(overrides=overrides, consult_fn=rejecting_consult_fn, seed=3)
    collector_b, _ = make_assisted_collector(overrides=overrides, consult_fn=rejecting_consult_fn, seed=3)
    records_a, _ = await collector_a.collect(8)
    records_b, _ = await collector_b.collect(8)
    # Same fallback_rng_seed -> the same fallback action sequence, even
    # though nothing was ever accepted (fully reproducible fallback).
    assert [r.selected_action_index for r in records_a] == [r.selected_action_index for r in records_b]


@pytest.mark.asyncio
async def test_advice_transformation_other_than_identity_is_not_implemented():
    with pytest.raises(NotImplementedError):
        EvaluationOverrides(advice_transformation="shuffled")


# --- VALIDATION.md check 6: same eval seed -> same scenario realization ----
# across independently-constructed adapters/policies (i.e. across methods).


def test_same_seed_produces_identical_initial_observation_across_independent_adapters():
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")

    def make_adapter():
        return NasimEmuAdapter(
            scenario=SMALL_SCENARIO, max_episode_steps=10,
            completion_reward=config.objective.completion_reward,
            premature_finish_penalty=config.objective.premature_finish_penalty,
        )

    state_a = make_adapter().reset(seed=777)
    state_b = make_adapter().reset(seed=777)
    graph_a = make_adapter().to_pyg_data(state_a, include_subnet_scan_feature=True)
    graph_b = make_adapter().to_pyg_data(state_b, include_subnet_scan_feature=True)

    assert torch.equal(graph_a.data.x, graph_b.data.x)
    assert torch.equal(graph_a.data.edge_index, graph_b.data.edge_index)
    assert graph_a.node_key_to_index == graph_b.node_key_to_index


# --- VALIDATION.md check 7: benchmark_return (nasimemu_reward) is unaffected
# by consultation.cost -- only training_reward should differ. ---------------


@pytest.mark.asyncio
async def test_nasimemu_reward_is_unaffected_by_consultation_cost():
    calls: list = []

    def make_collector(cost):
        config = load_config(REPO_ROOT / "examples" / "assisted.yaml")
        adapter = NasimEmuAdapter(
            scenario=SMALL_SCENARIO, max_episode_steps=10,
            completion_reward=config.objective.completion_reward,
            premature_finish_penalty=config.objective.premature_finish_penalty,
        )
        torch.manual_seed(1234)  # identical policy weights across both runs
        policy = RecurrentPolicy(config.policy, consultation_enabled=True)
        return RolloutCollector(
            adapter, policy, run_id="test-run", base_seed=5,
            consultation_enabled=True, consultation_cost=cost,
            consult_fn=_accepting_consult_fn_factory(calls),
            deterministic=True, overrides=ALWAYS_QUERY,  # force identical query pattern regardless of cost
        )

    low_cost_collector = make_collector(cost=0.0)
    high_cost_collector = make_collector(cost=0.5)
    low_records, _ = await low_cost_collector.collect(10)
    high_records, _ = await high_cost_collector.collect(10)

    assert [r.nasimemu_reward for r in low_records] == [r.nasimemu_reward for r in high_records]
    assert [r.training_reward for r in low_records] != [r.training_reward for r in high_records]


# --- Objective/FINISH timing (spec section 5) -------------------------------
# steps_to_goal (first step the objective becomes satisfied) and finish_step
# (when FINISH is selected) are tracked independently -- these tests drive a
# fully scripted fake adapter (not a real NASimEmu scenario, whose action
# outcomes are not controllable step-by-step) to pin down exact off-by-one
# behavior, not just plumbing/existence.


class _ScriptedAdapter:
    """A minimal NasimEmuAdapter substitute: scripted legal action and
    objective_satisfied() per step index, everything else a fixed stub.
    """

    max_episode_steps = 50

    def __init__(self, actions: list, objective_satisfied: list, max_steps: int = 50):
        self._actions = actions
        self._objective_satisfied = objective_satisfied
        self._max_steps = max_steps
        self._step_idx = 0
        self._last_step_idx = 0
        x = torch.zeros((1, NODE_FEATURE_DIM))
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        self._data = torch_geometric.data.Data(x=x, edge_index=edge_index)

    def reset(self, seed=None):
        self._step_idx = 0
        # host_rows must be a real (if empty) array, matching
        # EnvironmentState's actual contract -- compute_state_delta zips it
        # against host_addresses eagerly, which raises on None regardless
        # of host_addresses being empty (no host addresses are ever used in
        # this fixture, so an empty array is otherwise inert).
        return EnvironmentState(
            raw_observation=None, host_rows=torch.empty(0), host_addresses=[], subnet_graph=set(), step_idx=0
        )

    def legal_actions(self, state):
        return [self._actions[min(self._step_idx, len(self._actions) - 1)]]

    def to_pyg_data(self, state, include_subnet_scan_feature=True):
        return GraphObservation(data=self._data, node_key_to_index={"host-0-0": 0})

    def objective_satisfied(self):
        return self._objective_satisfied[min(self._last_step_idx, len(self._objective_satisfied) - 1)]

    def sensitive_target_status(self):
        # Not exercised by this fixture's assertions; a fixed stub is
        # enough to satisfy RolloutCollector.collect()'s call.
        return (0, 0)

    def step(self, action):
        idx = self._step_idx
        self._last_step_idx = idx
        self._step_idx += 1
        if action.is_finish:
            return TransitionResult(state=None, nasimemu_reward=1.0, terminated=True, truncated=False, info={})
        truncated = self._step_idx >= self._max_steps
        new_state = EnvironmentState(
            raw_observation=None, host_rows=torch.empty(0), host_addresses=[], subnet_graph=set(), step_idx=idx + 1
        )
        return TransitionResult(state=new_state, nasimemu_reward=0.0, terminated=False, truncated=truncated, info={})


def _scripted_policy():
    from marla.config.models import (
        ActionEncoderConfig,
        ConstantSchedulerConfig,
        GraphEncoderConfig,
        OptimizerConfig,
        PolicyConfig,
        PPOConfig,
        RecurrentConfig,
    )

    policy_config = PolicyConfig(
        graph_encoder=GraphEncoderConfig(hidden_size=8, layers=1),
        action_encoder=ActionEncoderConfig(hidden_size=8, action_type_embedding_size=4),
        recurrent=RecurrentConfig(hidden_size=8, sequence_length=4),
        ppo=PPOConfig(
            total_environment_steps=10, steps_per_env=2, epochs=1, minibatch_sequences=1,
            gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, value_coefficient=0.5,
            query_entropy_coefficient=0.01, action_entropy_coefficient=0.01, max_grad_norm=0.5,
            optimizer=OptimizerConfig(learning_rate=0.0003, scheduler=ConstantSchedulerConfig()),
        ),
    )
    return RecurrentPolicy(policy_config)


_SCAN = ActionDescriptor(action_id="scan", action_type="service_scan", target_key="host-0-0", parameters={})
_FINISH_ACTION = ActionDescriptor(action_id="finish", action_type="finish", target_key=None, parameters={}, is_finish=True)


@pytest.mark.asyncio
async def test_steps_to_goal_and_finish_step_are_tracked_independently():
    # scan, scan, scan (objective becomes satisfied on this 3rd action),
    # scan, scan, FINISH (the 6th action) -- 3 extra actions after the goal.
    objective_satisfied = [False, False, True, True, True, True]
    actions = [_SCAN, _SCAN, _SCAN, _SCAN, _SCAN, _FINISH_ACTION]
    adapter = _ScriptedAdapter(actions, objective_satisfied)
    collector = RolloutCollector(adapter, _scripted_policy(), run_id="t", base_seed=1, deterministic=True)

    records, summaries = await collector.collect(6)

    became_satisfied_at = [r.environment_step for r in records if r.objective_became_satisfied]
    assert became_satisfied_at == [2]  # fires exactly once, at the 0-indexed step where it flips
    assert [r.objective_satisfied for r in records] == [False, False, True, True, True, True]

    summary = summaries[0]
    assert summary.goal_success is True
    assert summary.objective_reached is True
    assert summary.successful_finish is True
    assert summary.episode_success is True
    assert summary.steps_to_goal == 3
    assert summary.finish_step == 6
    assert summary.finish_delay_steps == 3
    assert summary.environment_steps == 6
    assert summary.finish_reason == "finish"


@pytest.mark.asyncio
async def test_premature_finish_has_no_steps_to_goal_or_finish_delay():
    # Objective never satisfied; FINISH selected anyway on the 2nd action.
    adapter = _ScriptedAdapter([_SCAN, _FINISH_ACTION], [False, False])
    collector = RolloutCollector(adapter, _scripted_policy(), run_id="t", base_seed=1, deterministic=True)

    _records, summaries = await collector.collect(2)
    summary = summaries[0]

    assert summary.goal_success is False
    assert summary.objective_reached is False  # never satisfied at all -- not merely un-finished
    assert summary.successful_finish is False
    assert summary.episode_success is False
    assert summary.steps_to_goal is None
    assert summary.finish_step == 2  # FINISH was still selected -- tracked regardless of success
    assert summary.finish_delay_steps is None
    assert summary.finish_reason == "finish"


@pytest.mark.asyncio
async def test_goal_satisfied_on_first_action_then_immediate_finish_gives_minimal_delay():
    adapter = _ScriptedAdapter([_SCAN, _FINISH_ACTION], [True, True])
    collector = RolloutCollector(adapter, _scripted_policy(), run_id="t", base_seed=1, deterministic=True)

    _records, summaries = await collector.collect(2)
    summary = summaries[0]

    assert summary.steps_to_goal == 1
    assert summary.finish_step == 2
    assert summary.finish_delay_steps == 1  # FINISH itself is the one delay step
    assert summary.objective_reached is True
    assert summary.successful_finish is True
    assert summary.episode_success is True


@pytest.mark.asyncio
async def test_timeout_without_finish_leaves_steps_to_goal_and_finish_step_none():
    adapter = _ScriptedAdapter([_SCAN, _SCAN, _SCAN], [False, False, False], max_steps=2)
    collector = RolloutCollector(adapter, _scripted_policy(), run_id="t", base_seed=1, deterministic=True)

    _records, summaries = await collector.collect(2)
    summary = summaries[0]

    assert summary.goal_success is False
    assert summary.objective_reached is False  # never satisfied at all -- see the
    # sibling test right below for the "reached, then timed out anyway" case
    # this must NOT be confused with.
    assert summary.successful_finish is False
    assert summary.episode_success is False
    assert summary.steps_to_goal is None
    assert summary.finish_step is None
    assert summary.finish_delay_steps is None
    assert summary.finish_reason == "truncated"


@pytest.mark.asyncio
async def test_objective_reached_then_timeout_is_not_confused_with_never_reached():
    """Regression test (spec section 26): the single most important
    diagnostic case this whole objective_reached/successful_finish split
    exists for. The objective becomes satisfied mid-episode (step 2) but
    the policy never selects FINISH afterward and the episode times out --
    this must be reported as "the objective WAS reached, just not
    finished", not collapsed into the same bucket as an episode that never
    got anywhere near the objective at all (the previous test). Under an
    implementation that only tracked a FINISH-gated `goal_success` boolean
    with no independent objective_reached field, these two genuinely
    different episodes would have been indistinguishable in episodes.csv.
    """
    # scan (not yet satisfied), scan (becomes satisfied here, step 2),
    # scan, scan (truncates at max_steps=4) -- no FINISH ever selected.
    objective_satisfied = [False, True, True, True]
    adapter = _ScriptedAdapter([_SCAN, _SCAN, _SCAN, _SCAN], objective_satisfied, max_steps=4)
    collector = RolloutCollector(adapter, _scripted_policy(), run_id="t", base_seed=1, deterministic=True)

    _records, summaries = await collector.collect(4)
    summary = summaries[0]

    assert summary.objective_reached is True  # the key assertion this test exists for
    assert summary.successful_finish is False
    assert summary.episode_success is False
    assert summary.goal_success is False  # deprecated alias, same value as successful_finish
    assert summary.steps_to_goal == 2
    assert summary.finish_step is None  # FINISH was never selected
    assert summary.finish_delay_steps is None
    assert summary.finish_reason == "truncated"
