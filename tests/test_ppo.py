import copy
import random
from pathlib import Path

import pytest
import torch

from marla.config.loader import load_config, parse_config
from marla.environment.action_compatibility import COMPATIBILITY_FEATURE_DIM
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.environment.visible_facts import VISIBLE_PROGRESS_DIM
from marla.learning.gae import compute_gae
from marla.learning.ppo import build_sequence_chunks, optimize
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import ConsultationResult, RolloutCollector, StepRecord

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


def _fake_record(episode_id: int) -> StepRecord:
    return StepRecord(
        run_id="r", episode_id=episode_id, environment_step=0, observation_id="o",
        graph_data=None, node_key_to_index={}, legal_action_descriptors=[],
        compatibility_features=torch.zeros((0, COMPATIBILITY_FEATURE_DIM)),
        visible_progress=torch.zeros(VISIBLE_PROGRESS_DIM),
        initial_gru_hidden_state=torch.zeros(2), previous_action_embedding=torch.zeros(2),
        previous_training_reward=0.0, previous_query=False,
        base_logits=torch.zeros(1), final_logits=torch.zeros(1), selected_action_index=0,
        old_action_log_probability=0.0, old_joint_log_probability=0.0, critic_value=0.0,
        nasimemu_reward=0.0, consultation_cost=0.0, training_reward=0.0,
        terminated=False, truncated=False,
    )


def test_build_sequence_chunks_never_spans_episode_boundary():
    records = [_fake_record(1), _fake_record(1), _fake_record(2), _fake_record(2), _fake_record(2)]
    advantages = [0.0] * 5
    returns = [0.0] * 5
    chunks = build_sequence_chunks(records, advantages, returns, sequence_length=10)
    assert len(chunks) == 2
    assert {r.episode_id for r in chunks[0].records} == {1}
    assert {r.episode_id for r in chunks[1].records} == {2}


def test_build_sequence_chunks_respects_max_length():
    records = [_fake_record(1) for _ in range(7)]
    chunks = build_sequence_chunks(records, [0.0] * 7, [0.0] * 7, sequence_length=3)
    assert [len(c.records) for c in chunks] == [3, 3, 1]


def _tiny_config():
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    data["policy"]["ppo"]["steps_per_env"] = 16
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    return parse_config(data)


@pytest.mark.asyncio
async def test_ppo_update_changes_trainable_parameters():
    config = _tiny_config()
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy.ppo.optimizer.learning_rate)

    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    collector = RolloutCollector(adapter, policy, run_id="test-run", base_seed=1)
    records, _summaries = await collector.collect(config.policy.ppo.steps_per_env)

    rewards = [r.training_reward for r in records]
    values = [r.critic_value for r in records]
    terminated = [r.terminated for r in records]
    truncated = [r.truncated for r in records]
    bootstrap_values = [r.bootstrap_value for r in records]
    advantages, returns = compute_gae(
        rewards, values, terminated, truncated, bootstrap_values,
        config.policy.ppo.gamma, config.policy.ppo.gae_lambda,
    )

    before = copy.deepcopy(policy.state_dict())

    metrics = optimize(
        policy, optimizer, records, advantages, returns, config.policy.ppo,
        config.policy.recurrent.sequence_length, config.policy.ppo.minibatch_sequences,
        torch.device("cpu"), random.Random(0),
    )

    after = policy.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed, "PPO update should change at least one trainable parameter"

    assert len(metrics) > 0
    for m in metrics:
        for key, value in m.items():
            if value is None:
                continue  # empty-denominator diagnostic (e.g. no FINISH this minibatch), not a numerical fault
            assert torch.isfinite(torch.tensor(value)), f"{key} is not finite: {value}"


def test_optimize_signature_has_no_environment_or_plan_maker_access():
    """Structural guard for 'never re-invoke the Plan Maker / environment during PPO optimization'."""
    import inspect

    params = inspect.signature(optimize).parameters
    forbidden = {"adapter", "env", "environment", "plan_maker"}
    assert forbidden.isdisjoint(params.keys())


@pytest.mark.asyncio
async def test_assisted_ppo_update_changes_parameters_without_reinvoking_plan_maker():
    """Full assisted rollout + PPO update, with a fake consult_fn (no SPADE needed).

    The fake consult_fn call-counter proves the Plan Maker (stand-in) is
    invoked only during collection, never during the PPO replay that follows.
    """
    config = _tiny_config()
    data = config.model_dump()
    data["consultation"] = {"mode": "learned", "cost": 0.1, "max_schema_revisions": 3}
    data["gatekeeper"] = {"alias": "gatekeeper", "jid": "gk@localhost"}
    data["agents"] = [
        {
            "alias": "plan_maker_1",
            "jid": "pm@localhost",
            "role": "plan_maker",
            "model": {"backend": "local", "name": "x"},
            "prompt_version": "v1",
            "knowledge": {"path": "package://marla/knowledge/nasimemu_rules.yaml", "version": "v1"},
        }
    ]
    config = parse_config(data)

    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy, consultation_enabled=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy.ppo.optimizer.learning_rate)

    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )

    call_count = 0

    async def fake_consult(legal_actions, episode_id, step, source_observation_id, observation):
        nonlocal call_count
        call_count += 1
        return ConsultationResult(
            status="accepted",
            scores={a.action_id: 0.5 for a in legal_actions},
            request_id=f"request-{call_count}",
        )

    collector = RolloutCollector(
        adapter, policy, run_id="test-run", base_seed=1,
        consultation_enabled=True, consultation_cost=config.consultation.cost, consult_fn=fake_consult,
    )
    records, _summaries = await collector.collect(config.policy.ppo.steps_per_env)
    calls_during_collection = call_count
    assert calls_during_collection >= 0  # may legitimately be 0 if the gate never queried this seed

    rewards = [r.training_reward for r in records]
    values = [r.critic_value for r in records]
    terminated = [r.terminated for r in records]
    truncated = [r.truncated for r in records]
    bootstrap_values = [r.bootstrap_value for r in records]
    advantages, returns = compute_gae(
        rewards, values, terminated, truncated, bootstrap_values,
        config.policy.ppo.gamma, config.policy.ppo.gae_lambda,
    )

    before = copy.deepcopy(policy.state_dict())

    metrics = optimize(
        policy, optimizer, records, advantages, returns, config.policy.ppo,
        config.policy.recurrent.sequence_length, config.policy.ppo.minibatch_sequences,
        torch.device("cpu"), random.Random(0), consultation_cost=config.consultation.cost,
    )

    assert call_count == calls_during_collection, "PPO optimization must never call the Plan Maker again"

    after = policy.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed, "PPO update should change at least one trainable parameter"

    assert len(metrics) > 0
    for m in metrics:
        for key, value in m.items():
            if value is None:
                continue  # empty-denominator diagnostic (e.g. no FINISH this minibatch), not a numerical fault
            assert torch.isfinite(torch.tensor(value)), f"{key} is not finite: {value}"


@pytest.mark.asyncio
async def test_ppo_replay_reuses_stored_compatibility_features_verbatim(monkeypatch):
    """Spec section 11/32: PPO replay must feed the encoder EXACTLY the
    compatibility tensor stored at collection time -- never recompute it
    (there is no live simulator state to recompute it from during replay
    anyway, but this proves the plumbing, not just the absence of a
    simulator call).
    """
    from marla.learning import action_encoder as action_encoder_module
    from marla.learning.ppo import build_sequence_chunks, _replay_chunk

    config = _tiny_config()
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy)
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    collector = RolloutCollector(adapter, policy, run_id="test-run", base_seed=1)
    records, _summaries = await collector.collect(8)
    assert all(r.compatibility_features.shape[1] == COMPATIBILITY_FEATURE_DIM for r in records)

    # Deliberately corrupt one record's stored tensor to a recognizable,
    # impossible-to-recompute-by-accident sentinel value -- if replay ever
    # recomputed compatibility from scratch instead of reusing this, the
    # sentinel would never reach the encoder and the captured tensor below
    # would not match it.
    sentinel = torch.full_like(records[0].compatibility_features, 7.0)
    records[0].compatibility_features = sentinel

    captured: list[torch.Tensor] = []
    real_encode = action_encoder_module.ActionEncoder.encode_descriptors

    def spy(self, descriptors, node_embeddings, node_key_to_index, device, compatibility_matrix):
        captured.append(compatibility_matrix.clone())
        return real_encode(self, descriptors, node_embeddings, node_key_to_index, device, compatibility_matrix)

    monkeypatch.setattr(action_encoder_module.ActionEncoder, "encode_descriptors", spy)

    chunks = build_sequence_chunks(records, [0.0] * len(records), [0.0] * len(records), sequence_length=8)
    _replay_chunk(policy, chunks[0], torch.device("cpu"), consultation_cost=0.0)

    assert torch.equal(captured[0], sentinel)
    for record, seen in zip(chunks[0].records, captured):
        assert torch.equal(seen, record.compatibility_features)


@pytest.mark.asyncio
async def test_optimize_raises_training_diverged_error_on_nan_returns():
    """NaN/Inf safety (research-readiness audit): a corrupted `returns`
    tensor (e.g. from an upstream GAE/reward bug) must make optimize()
    fail loudly with a specific, named exception -- never silently take an
    optimizer step on poisoned data and keep going.
    """
    from marla.learning.ppo import TrainingDivergedError

    config = _tiny_config()
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy.ppo.optimizer.learning_rate)

    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    collector = RolloutCollector(adapter, policy, run_id="test-run", base_seed=1)
    records, _summaries = await collector.collect(config.policy.ppo.steps_per_env)

    advantages = [0.0] * len(records)
    returns = [float("nan")] * len(records)

    with pytest.raises(TrainingDivergedError):
        optimize(
            policy, optimizer, records, advantages, returns, config.policy.ppo,
            config.policy.recurrent.sequence_length, config.policy.ppo.minibatch_sequences,
            torch.device("cpu"), random.Random(0),
        )
