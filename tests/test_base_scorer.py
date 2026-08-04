import torch
from torch_geometric.utils import softmax as scatter_softmax

from marla.learning.base_scorer import BaseActionScorer


def test_output_shape_matches_action_count():
    scorer = BaseActionScorer(recurrent_hidden_size=8, action_hidden_size=6, scorer_hidden_size=10)
    z = torch.randn(2, 8)
    action_embeddings = torch.randn(5, 6)
    action_to_sample = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    logits = scorer(z, action_embeddings, action_to_sample)
    assert logits.shape == (5,)
    assert torch.isfinite(logits).all()


def test_variable_length_candidate_sets_softmax_sums_to_one_per_sample():
    """Each sample's candidate set can have a different size (dynamic action set)."""
    scorer = BaseActionScorer(recurrent_hidden_size=4, action_hidden_size=3, scorer_hidden_size=5)
    z = torch.randn(3, 4)
    # sample 0 has 2 candidates, sample 1 has 1, sample 2 has 4
    action_to_sample = torch.tensor([0, 0, 1, 2, 2, 2, 2], dtype=torch.long)
    action_embeddings = torch.randn(7, 3)

    logits = scorer(z, action_embeddings, action_to_sample)
    probs = scatter_softmax(logits, action_to_sample)

    for sample in range(3):
        mask = action_to_sample == sample
        assert torch.allclose(probs[mask].sum(), torch.tensor(1.0), atol=1e-5)


def test_single_candidate_gets_probability_one():
    scorer = BaseActionScorer(recurrent_hidden_size=4, action_hidden_size=3, scorer_hidden_size=5)
    z = torch.randn(1, 4)
    action_to_sample = torch.tensor([0], dtype=torch.long)
    action_embeddings = torch.randn(1, 3)

    logits = scorer(z, action_embeddings, action_to_sample)
    probs = scatter_softmax(logits, action_to_sample)
    assert torch.allclose(probs, torch.tensor([1.0]))
