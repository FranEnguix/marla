"""Compound-decision math shared by live rollout collection and PPO replay.

Both call sites need *exactly* the same computation: at collection time to
record the "old" probabilities, and at replay time to recompute "new"
probabilities under updated parameters for the PPO ratio. Sharing this
module is what makes that guarantee structural rather than a matter of
remembering to keep two implementations in sync.

Replay never samples: it reuses the stored ``sampled_query`` flag, the
stored ``selected_action_index``, and (critically) the exact stored Plan
Maker confidence vector -- never recomputing or re-fetching any of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from marla.learning.query_gate import compute_base_entropy, compute_top_two_margin
from marla.learning.recurrent_policy import PolicyStepOutput, RecurrentPolicy

_EPSILON = 1e-12


def compute_query_probability(policy: RecurrentPolicy, step_output: PolicyStepOutput, num_actions: int, kappa: float) -> Tensor:
    """p_t^q, before sampling (spec section 14)."""
    device = step_output.z.device
    entropy = compute_base_entropy(step_output.base_probs)
    margin = compute_top_two_margin(step_output.base_logits)
    num_actions_t = torch.tensor(float(num_actions), device=device)
    kappa_t = torch.tensor(float(kappa), device=device)
    return policy.query_gate(step_output.z, entropy, margin, num_actions_t, kappa_t)


@dataclass
class FinalDecision:
    final_logits: Tensor
    beta: Tensor | None  # None only when not queried at all
    alpha: Tensor | None
    normalized_advice: Tensor | None


def compute_final_decision(
    policy: RecurrentPolicy,
    step_output: PolicyStepOutput,
    sampled_query: bool,
    plan_maker_confidence: Tensor | None,
) -> FinalDecision:
    """Base logits, or the advised residual adjustment when queried and accepted.

    Rejected advice forces beta=0 (spec section 15), which makes the
    resulting distribution numerically identical to the base policy while
    still reporting a real (if unused) alpha for metrics.
    """
    if sampled_query and plan_maker_confidence is not None:
        advised = policy.apply_advice(step_output.z, step_output.base_logits, plan_maker_confidence)
        return FinalDecision(
            final_logits=advised.final_logits,
            beta=advised.beta,
            alpha=advised.alpha,
            normalized_advice=advised.normalized_advice,
        )

    if sampled_query:  # queried but rejected (schema_rejected)
        return FinalDecision(
            final_logits=step_output.base_logits,
            beta=torch.zeros((), device=step_output.z.device),
            alpha=policy.advice_scale(),
            normalized_advice=None,
        )

    return FinalDecision(final_logits=step_output.base_logits, beta=None, alpha=None, normalized_advice=None)


@dataclass
class JointLogProbability:
    joint_log_prob: Tensor
    query_log_prob: Tensor
    action_log_prob: Tensor


def compute_joint_log_probability(
    query_probability: Tensor, sampled_query: bool, final_logits: Tensor, selected_action_index: int
) -> JointLogProbability:
    """log p_t = log pi^q(q_t|z_t) + (1-q_t) log pi^0(a_t) + q_t log pi^PM(a_t) (spec section 16).

    The single ``final_logits`` softmax already *is* pi^0 when not queried
    (or when advice was rejected, since beta=0) and pi^PM when queried and
    accepted, so one formula covers both branches of the spec's piecewise
    definition without a special case.
    """
    query_prob_used = query_probability if sampled_query else (1 - query_probability)
    query_log_prob = torch.log(query_prob_used.clamp_min(_EPSILON))

    log_probs_all = torch.log(torch.softmax(final_logits, dim=-1).clamp_min(_EPSILON))
    action_log_prob = log_probs_all[selected_action_index]

    return JointLogProbability(
        joint_log_prob=query_log_prob + action_log_prob,
        query_log_prob=query_log_prob,
        action_log_prob=action_log_prob,
    )
