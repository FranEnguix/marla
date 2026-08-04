import asyncio
from pathlib import Path

import pytest
import torch

from marla.config.loader import load_config
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import RolloutCollector

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
