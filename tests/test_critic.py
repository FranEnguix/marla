import torch

from marla.learning.critic import Critic


def test_critic_output_shape():
    critic = Critic(hidden_size=8)
    z = torch.randn(4, 8)
    value = critic(z)
    assert value.shape == (4,)
    assert torch.isfinite(value).all()
