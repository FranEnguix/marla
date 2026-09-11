from pathlib import Path

import pytest
import torch

from marla.config.loader import load_config
from marla.environment.action_compatibility import compute_compatibility_matrix
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.environment.visible_facts import compute_visible_progress, extract_visible_host_facts
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.scenarios.uri import resolve_scenario_reference

REPO_ROOT = Path(__file__).resolve().parent.parent


def _compatibility(adapter, state, legal):
    return compute_compatibility_matrix(legal, extract_visible_host_facts(state))


def _progress(state):
    return compute_visible_progress(extract_visible_host_facts(state))


@pytest.fixture
def baseline_policy_and_adapter():
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    scenario = resolve_scenario_reference(config.environment.scenario, (REPO_ROOT / "examples"))
    adapter = NasimEmuAdapter(
        scenario=scenario,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    policy = RecurrentPolicy(config.policy)
    return policy, adapter


def test_initial_recurrent_state_shapes(baseline_policy_and_adapter):
    policy, _adapter = baseline_policy_and_adapter
    rstate = policy.initial_recurrent_state()
    assert rstate.z.shape == (policy.recurrent_core.hidden_size,)
    assert rstate.previous_action_embedding.shape == (policy.recurrent_core.action_embedding_size,)
    assert rstate.previous_reward == 0.0
    assert rstate.previous_query == 0.0


def test_gru_reset_is_independent_of_prior_episode(baseline_policy_and_adapter):
    policy, adapter = baseline_policy_and_adapter
    state = adapter.reset(seed=1)
    rstate = policy.initial_recurrent_state()

    for _ in range(3):
        legal = adapter.legal_actions(state)
        graph_obs = adapter.to_pyg_data(state, include_subnet_scan_feature=policy.visible_subnet_exploration_enabled)
        out = policy.step(graph_obs, legal, rstate, _compatibility(adapter, state, legal), _progress(state))
        idx = int(torch.argmax(out.base_probs))
        result = adapter.step(legal[idx])
        rstate = policy.advance_recurrent_state(out, idx, result.nasimemu_reward, query=False)
        if result.terminated or result.truncated:
            break
        state = result.state

    fresh_rstate = policy.initial_recurrent_state()
    assert torch.equal(fresh_rstate.z, torch.zeros_like(fresh_rstate.z))
    assert torch.equal(fresh_rstate.previous_action_embedding, policy.recurrent_core.start_action_token)


def test_step_output_shapes_and_probability_normalization(baseline_policy_and_adapter):
    policy, adapter = baseline_policy_and_adapter
    state = adapter.reset(seed=2)
    rstate = policy.initial_recurrent_state()

    for _ in range(5):
        legal = adapter.legal_actions(state)
        graph_obs = adapter.to_pyg_data(state, include_subnet_scan_feature=policy.visible_subnet_exploration_enabled)
        out = policy.step(graph_obs, legal, rstate, _compatibility(adapter, state, legal), _progress(state))

        n = len(legal)
        assert out.base_logits.shape == (n,)
        assert out.base_probs.shape == (n,)
        assert out.action_embeddings.shape[0] == n
        assert torch.isfinite(out.base_logits).all()
        assert torch.allclose(out.base_probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert out.value.dim() == 0

        idx = int(torch.argmax(out.base_probs))
        result = adapter.step(legal[idx])
        rstate = policy.advance_recurrent_state(out, idx, result.nasimemu_reward, query=False)
        if result.terminated or result.truncated:
            break
        state = result.state


def test_step_rejects_empty_action_list(baseline_policy_and_adapter):
    policy, adapter = baseline_policy_and_adapter
    state = adapter.reset(seed=1)
    graph_obs = adapter.to_pyg_data(state, include_subnet_scan_feature=policy.visible_subnet_exploration_enabled)
    rstate = policy.initial_recurrent_state()
    empty_compatibility = compute_compatibility_matrix([], {})
    empty_progress = compute_visible_progress({})
    with pytest.raises(ValueError):
        policy.step(graph_obs, [], rstate, empty_compatibility, empty_progress)


def test_parameter_count_is_finite_and_positive(baseline_policy_and_adapter):
    policy, _adapter = baseline_policy_and_adapter
    total = sum(p.numel() for p in policy.parameters())
    assert total > 0
