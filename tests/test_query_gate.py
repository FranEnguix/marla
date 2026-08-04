import torch

from marla.learning.query_gate import QueryGate, compute_base_entropy, compute_top_two_margin


def test_base_entropy_uniform_distribution():
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    entropy = compute_base_entropy(probs)
    assert torch.isclose(entropy, torch.log(torch.tensor(4.0)), atol=1e-5)


def test_base_entropy_degenerate_distribution_is_near_zero():
    probs = torch.tensor([1.0, 0.0, 0.0])
    entropy = compute_base_entropy(probs)
    assert entropy.item() < 1e-4


def test_top_two_margin_single_action_is_zero():
    logits = torch.tensor([3.0])
    assert compute_top_two_margin(logits).item() == 0.0


def test_top_two_margin_computes_difference():
    logits = torch.tensor([1.0, 5.0, 3.0])
    margin = compute_top_two_margin(logits)
    assert torch.isclose(margin, torch.tensor(2.0))  # 5 - 3


def test_query_gate_output_is_valid_probability():
    gate = QueryGate(recurrent_hidden_size=8)
    z = torch.randn(8)
    prob = gate(
        z,
        entropy=torch.tensor(0.5),
        margin=torch.tensor(1.0),
        num_actions=torch.tensor(5.0),
        kappa=torch.tensor(0.1),
    )
    assert prob.dim() == 0
    assert 0.0 <= prob.item() <= 1.0
    assert torch.isfinite(prob)
