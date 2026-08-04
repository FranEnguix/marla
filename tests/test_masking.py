import torch

from marla.learning.masking import masked_entropy, masked_log_softmax, masked_softmax


def test_masked_softmax_sums_to_one_over_valid_entries_only():
    logits = torch.tensor([[1.0, 2.0, 999.0], [0.5, -0.5, 999.0]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    probs = masked_softmax(logits, mask)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0, 1.0]), atol=1e-5)
    assert torch.allclose(probs[:, 2], torch.zeros(2), atol=1e-6)


def test_masked_softmax_matches_plain_softmax_when_fully_valid():
    logits = torch.randn(3, 4)
    mask = torch.ones(3, 4, dtype=torch.bool)
    expected = torch.softmax(logits, dim=-1)
    actual = masked_softmax(logits, mask)
    assert torch.allclose(expected, actual, atol=1e-5)


def test_masked_log_softmax_is_very_negative_for_padded_entries():
    logits = torch.zeros(1, 3)
    mask = torch.tensor([[True, True, False]])
    log_probs = masked_log_softmax(logits, mask)
    assert log_probs[0, 2] < -1e6


def test_masked_entropy_ignores_padding():
    # two identical valid logits + one masked-out entry should have the
    # same entropy as a plain 2-way uniform distribution: log(2)
    logits = torch.tensor([[0.0, 0.0, 5.0]])
    mask = torch.tensor([[True, True, False]])
    log_probs = masked_log_softmax(logits, mask)
    entropy = masked_entropy(log_probs, mask)
    assert torch.allclose(entropy, torch.tensor([torch.log(torch.tensor(2.0))]), atol=1e-4)
