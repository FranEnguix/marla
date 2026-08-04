"""Value critic: a linear head over the GRU recurrent state."""

from __future__ import annotations

from torch import Tensor, nn


class Critic(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, z: Tensor) -> Tensor:
        """``z``: ``(B, hidden_size)`` -> ``(B,)``."""
        return self.value_head(z).squeeze(-1)
