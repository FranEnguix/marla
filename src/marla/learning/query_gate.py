"""Learned binary query gate (spec section 14).

The query decision happens *before* the Plan Maker is consulted: it is a
function of the base policy alone (recurrent state, base entropy, top-two
margin, candidate-set size, and the configured consultation cost).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def compute_base_entropy(base_probs: Tensor) -> Tensor:
    """H_t^0 = -sum_i pi^0(a_i) log pi^0(a_i)."""
    log_probs = torch.log(base_probs.clamp_min(1e-12))
    return -(base_probs * log_probs).sum()


def compute_top_two_margin(base_logits: Tensor) -> Tensor:
    """Delta_t^0 = b_(1) - b_(2); zero when only one legal action exists."""
    if base_logits.numel() < 2:
        return torch.zeros((), device=base_logits.device, dtype=base_logits.dtype)
    top2 = torch.topk(base_logits, k=2).values
    return top2[0] - top2[1]


class QueryGate(nn.Module):
    """p_t^q = sigma(f_q[z_t, H_t^0, Delta_t^0, N_t, kappa])."""

    def __init__(self, recurrent_hidden_size: int, mlp_hidden_size: int = 32):
        super().__init__()
        input_dim = recurrent_hidden_size + 4  # + entropy, margin, num_actions, kappa
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_size), nn.Tanh(), nn.Linear(mlp_hidden_size, 1)
        )

    def forward(self, z: Tensor, entropy: Tensor, margin: Tensor, num_actions: Tensor, kappa: Tensor) -> Tensor:
        """All of ``entropy``/``margin``/``num_actions``/``kappa`` are 0-dim tensors."""
        features = torch.stack([entropy, margin, num_actions, kappa]).to(dtype=z.dtype, device=z.device)
        x = torch.cat([z, features], dim=-1)
        logit = self.mlp(x).squeeze(-1)
        return torch.sigmoid(logit)
