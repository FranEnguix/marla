"""Advice processing and learned trust (spec section 15).

Confidence scores from the Plan Maker are not assumed calibrated
probabilities: they are clipped, converted to log-odds, and z-score
normalized before being used as a residual adjustment to the base logits,
scaled by a learned per-step trust coefficient and a single learned global
scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

_EPSILON = 1e-6


def clip_and_logit(confidence: Tensor) -> Tensor:
    """c_i -> u_i = logit(clip(c_i, eps, 1-eps))."""
    clipped = confidence.clamp(_EPSILON, 1 - _EPSILON)
    return torch.log(clipped / (1 - clipped))


def normalize_advice(log_odds: Tensor) -> Tensor:
    """z-score normalize log-odds; all-zero when variance is ~0 (spec section 15)."""
    std = log_odds.std(unbiased=False)
    if std < _EPSILON:
        return torch.zeros_like(log_odds)
    mean = log_odds.mean()
    return (log_odds - mean) / (std + _EPSILON)


@dataclass
class AdviceSummary:
    mean: Tensor
    std: Tensor
    max: Tensor
    top1_minus_top2: Tensor
    entropy: Tensor


def compute_advice_summary(log_odds: Tensor) -> AdviceSummary:
    """S(c) = [mean, std, max, top1-top2, H(softmax(u))]."""
    probs = torch.softmax(log_odds, dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    entropy = -(probs * log_probs).sum()

    if log_odds.numel() >= 2:
        top2 = torch.topk(log_odds, k=2).values
        top1_minus_top2 = top2[0] - top2[1]
    else:
        top1_minus_top2 = torch.zeros((), device=log_odds.device, dtype=log_odds.dtype)

    return AdviceSummary(
        mean=log_odds.mean(),
        std=log_odds.std(unbiased=False),
        max=log_odds.max(),
        top1_minus_top2=top1_minus_top2,
        entropy=entropy,
    )


def summary_to_tensor(summary: AdviceSummary) -> Tensor:
    return torch.stack([summary.mean, summary.std, summary.max, summary.top1_minus_top2, summary.entropy])


def compute_agreement_features(base_logits: Tensor, advice_log_odds: Tensor) -> Tensor:
    """[top-action agreement (0/1), correlation between base logits and PM evidence].

    Correlation is 0 when undefined (zero variance in either vector) --
    spec section 15: "If correlation is undefined, use zero."

    Fixed, non-differentiable features for the trust head: the Plan Maker's
    output is external data, not a trainable parameter, so there is no
    gradient PPO could usefully take through this comparison -- ``base_logits``
    keeps its own graph connection for the *outer* residual sum in
    :meth:`RecurrentPolicy.apply_advice`, this is a separate detached copy.
    """
    device = base_logits.device
    base_logits = base_logits.detach()
    advice_log_odds = advice_log_odds.detach()
    agree = float(torch.argmax(base_logits) == torch.argmax(advice_log_odds))

    if base_logits.numel() < 2:
        correlation = 0.0
    else:
        base_centered = base_logits - base_logits.mean()
        advice_centered = advice_log_odds - advice_log_odds.mean()
        denom = torch.sqrt((base_centered**2).sum()) * torch.sqrt((advice_centered**2).sum())
        correlation = 0.0 if denom < _EPSILON else float((base_centered * advice_centered).sum() / denom)

    return torch.tensor([agree, correlation], dtype=base_logits.dtype, device=device)


class TrustHead(nn.Module):
    """beta_t = sigma(f_beta[z_t, S(c_t), A_t])."""

    SUMMARY_DIM = 5
    AGREEMENT_DIM = 2

    def __init__(self, recurrent_hidden_size: int, mlp_hidden_size: int = 32):
        super().__init__()
        input_dim = recurrent_hidden_size + self.SUMMARY_DIM + self.AGREEMENT_DIM
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_size), nn.Tanh(), nn.Linear(mlp_hidden_size, 1)
        )

    def forward(self, z: Tensor, summary: Tensor, agreement: Tensor) -> Tensor:
        x = torch.cat([z, summary.to(z.dtype), agreement.to(z.dtype)], dim=-1)
        return torch.sigmoid(self.mlp(x).squeeze(-1))


class AdviceScale(nn.Module):
    """alpha = softplus(alpha_bar); a single learned global scalar (not per-step)."""

    def __init__(self):
        super().__init__()
        self.alpha_bar = nn.Parameter(torch.zeros(()))

    def forward(self) -> Tensor:
        return torch.nn.functional.softplus(self.alpha_bar)


@dataclass
class AdvisedOutput:
    final_logits: Tensor
    beta: Tensor
    alpha: Tensor
    normalized_advice: Tensor
