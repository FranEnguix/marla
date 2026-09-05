import asyncio
from pathlib import Path

import pytest
import torch

from marla.config.loader import load_config
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.evaluation.overrides import ALWAYS_QUERY, BETA_ONE, BETA_ZERO, NO_QUERY, PLAN_MAKER_ONLY, EvaluationOverrides
from marla.learning.recurrent_policy import RecurrentPolicy
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
    graph_a = make_adapter().to_pyg_data(state_a)
    graph_b = make_adapter().to_pyg_data(state_b)

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
