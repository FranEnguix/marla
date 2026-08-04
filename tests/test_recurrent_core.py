import torch

from marla.learning.recurrent_core import RecurrentCore


def make_core():
    return RecurrentCore(graph_embedding_size=6, action_embedding_size=4, hidden_size=8)


def test_initial_hidden_state_is_zero():
    core = make_core()
    z0 = core.initial_hidden_state(batch_size=3, device=torch.device("cpu"))
    assert z0.shape == (3, 8)
    assert torch.all(z0 == 0)


def test_initial_previous_action_embedding_is_the_learned_start_token():
    core = make_core()
    tokens = core.initial_previous_action_embedding(batch_size=2, device=torch.device("cpu"))
    assert tokens.shape == (2, 4)
    assert torch.allclose(tokens[0], core.start_action_token)
    assert torch.allclose(tokens[1], core.start_action_token)


def test_forward_shape_batched():
    core = make_core()
    batch = 3
    graph_emb = torch.randn(batch, 6)
    prev_action_emb = torch.randn(batch, 4)
    prev_reward = torch.zeros(batch)
    prev_query = torch.zeros(batch)
    prev_hidden = torch.zeros(batch, 8)

    z = core(graph_emb, prev_action_emb, prev_reward, prev_query, prev_hidden)
    assert z.shape == (batch, 8)
    assert torch.isfinite(z).all()


def test_gru_reset_produces_identical_state_regardless_of_prior_episode():
    """Episode boundaries must never leak hidden state (spec section 17)."""
    core = make_core()
    z0_a = core.initial_hidden_state(1, torch.device("cpu"))
    token_a = core.initial_previous_action_embedding(1, torch.device("cpu"))

    # simulate an arbitrary number of steps within a (discarded) episode
    z = z0_a.clone()
    for _ in range(5):
        z = core(torch.randn(1, 6), torch.randn(1, 4), torch.ones(1), torch.ones(1), z)

    # starting a new episode must give back exactly the same initial state
    z0_b = core.initial_hidden_state(1, torch.device("cpu"))
    token_b = core.initial_previous_action_embedding(1, torch.device("cpu"))
    assert torch.equal(z0_a, z0_b)
    assert torch.equal(token_a, token_b)
    assert not torch.equal(z, z0_b)  # sanity: mid-episode state actually did move
