import torch

from marla.config.loader import parse_config
from marla.learning.decision import compute_final_decision, compute_joint_log_probability, compute_query_probability
from marla.learning.recurrent_policy import PolicyStepOutput, RecurrentPolicy
from tests.conftest import minimal_config_dict


def make_assisted_policy():
    data = minimal_config_dict("scenario-name-without-yaml-suffix")
    config = parse_config(data)
    return RecurrentPolicy(config.policy, consultation_enabled=True)


def fake_step_output(num_actions=3, hidden_size=32, action_hidden_size=32):
    z = torch.randn(hidden_size)
    base_logits = torch.randn(num_actions)
    base_probs = torch.softmax(base_logits, dim=-1)
    action_embeddings = torch.randn(num_actions, action_hidden_size)
    value = torch.randn(())
    return PolicyStepOutput(z=z, base_logits=base_logits, base_probs=base_probs, action_embeddings=action_embeddings, value=value)


def test_compute_query_probability_is_valid_probability():
    policy = make_assisted_policy()
    step_out = fake_step_output()
    prob = compute_query_probability(policy, step_out, num_actions=3, kappa=0.1)
    assert prob.dim() == 0
    assert 0.0 <= prob.item() <= 1.0


def test_compute_final_decision_not_queried_returns_base_logits_and_none_beta():
    policy = make_assisted_policy()
    step_out = fake_step_output()
    decision = compute_final_decision(policy, step_out, sampled_query=False, plan_maker_confidence=None)
    assert torch.equal(decision.final_logits, step_out.base_logits)
    assert decision.beta is None
    assert decision.alpha is None
    assert decision.normalized_advice is None


def test_compute_final_decision_queried_and_rejected_forces_beta_zero():
    policy = make_assisted_policy()
    step_out = fake_step_output()
    decision = compute_final_decision(policy, step_out, sampled_query=True, plan_maker_confidence=None)
    assert torch.equal(decision.final_logits, step_out.base_logits)
    assert decision.beta is not None
    assert decision.beta.item() == 0.0
    assert decision.alpha is not None  # still reported, just unused
    assert decision.normalized_advice is None


def test_compute_final_decision_queried_and_accepted_applies_residual():
    policy = make_assisted_policy()
    step_out = fake_step_output(num_actions=3)
    confidence = torch.tensor([0.9, 0.1, 0.5])
    decision = compute_final_decision(policy, step_out, sampled_query=True, plan_maker_confidence=confidence)
    assert decision.beta is not None
    assert 0.0 <= decision.beta.item() <= 1.0
    assert decision.alpha is not None
    assert decision.alpha.item() > 0.0
    assert decision.normalized_advice is not None
    assert decision.normalized_advice.shape == (3,)
    # final logits should differ from base logits since a real residual was applied
    # (unless beta happens to be exactly 0 or normalized_advice exactly 0, astronomically unlikely here)
    assert not torch.equal(decision.final_logits, step_out.base_logits)


def test_joint_log_probability_not_queried_matches_base_log_prob():
    final_logits = torch.tensor([1.0, 2.0, 0.5])
    query_probability = torch.tensor(0.3)
    result = compute_joint_log_probability(query_probability, sampled_query=False, final_logits=final_logits, selected_action_index=1)

    expected_action_log_prob = torch.log_softmax(final_logits, dim=-1)[1]
    expected_query_log_prob = torch.log(1 - query_probability)

    assert torch.isclose(result.action_log_prob, expected_action_log_prob, atol=1e-5)
    assert torch.isclose(result.query_log_prob, expected_query_log_prob, atol=1e-5)
    assert torch.isclose(result.joint_log_prob, expected_action_log_prob + expected_query_log_prob, atol=1e-5)


def test_joint_log_probability_queried_uses_query_probability_directly():
    final_logits = torch.tensor([1.0, 2.0, 0.5])
    query_probability = torch.tensor(0.7)
    result = compute_joint_log_probability(query_probability, sampled_query=True, final_logits=final_logits, selected_action_index=0)

    expected_query_log_prob = torch.log(query_probability)
    assert torch.isclose(result.query_log_prob, expected_query_log_prob, atol=1e-5)
