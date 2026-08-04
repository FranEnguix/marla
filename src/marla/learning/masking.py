"""Masked softmax/entropy utilities for padded, fixed-size PPO minibatches.

Live rollout collection scores a single instance's legal actions directly
(no padding needed). PPO replay (Milestone 4) batches multiple timesteps
with different candidate-action counts into fixed-size tensors, padding
short rows; padded entries must be masked out of softmax, log-probability,
and entropy computations (spec section 13).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_NEG_INF = -1e9


def masked_log_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """``mask``: ``True`` = valid/real entry, ``False`` = padding."""
    masked_logits = logits.masked_fill(~mask.bool(), _NEG_INF)
    return F.log_softmax(masked_logits, dim=dim)


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    return masked_log_softmax(logits, mask, dim=dim).exp()


def masked_entropy(log_probs: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    probs = log_probs.exp()
    contribution = torch.where(mask.bool(), probs * log_probs, torch.zeros_like(log_probs))
    return -contribution.sum(dim=dim)
