"""Compound-decision math shared by live rollout collection and PPO replay.

Both call sites need *exactly* the same computation: at collection time to
record the "old" probabilities, and at replay time to recompute "new"
probabilities under updated parameters for the PPO ratio. Sharing this
module is what makes that guarantee structural rather than a matter of
remembering to keep two implementations in sync.

Replay never samples: it reuses the stored ``sampled_query`` flag, the
stored ``selected_action_index``, and (critically) the exact stored Plan
Maker confidence vector -- never recomputing or re-fetching any of them.

**PPO-ratio consistency of subnet routing (subnet-scoped consultation,
design audit -- required before this feature could be implemented at
all)**: the joint log-probability this module computes is::

    log p_t = log pi^q(q_t|z_t) + (1-q_t) log pi^0(a_t) + q_t log pi^PM(a_t)

Every term here already comes from a RANDOM VARIABLE that was actually
sampled at collection time (``q_t``, ``a_t``) and whose realized value is
frozen/replayed verbatim, with only its *probability under new parameters*
recomputed for the importance-sampling ratio -- this was already true
before subnet-scoped consultation existed for ``selected_action_index``
and ``legal_action_descriptors`` (which actions exist AT ALL is itself a
frozen, replayed context, never resampled or recomputed from a live
environment during replay).

Subnet-scoped consultation adds exactly one new piece of per-decision
context: ``consulted_subnet`` (and the ``consulted_action_indices`` it
implies against the frozen candidate set). By explicit design (spec:
"do NOT add a learned subnet-selection head", "do NOT add an unaccounted
stochastic subnet action"), this is a DETERMINISTIC function of
already-accounted context -- ``argmax`` over the collection-time base
logits, restricted to non-FINISH candidates -- never a sampled decision of
its own. A deterministic function of already-accounted random variables
contributes no probability term of its own to a joint log-probability
(there is nothing to integrate over); it is exactly the same kind of
"frozen context" ``legal_action_descriptors`` already was. Consequently:

- ``consulted_subnet``/``consulted_action_indices``/the Plan Maker's
  scores for them are stored on each :class:`~marla.learning.rollout.StepRecord`
  and REPLAYED VERBATIM, never recomputed from replay's own (possibly
  different, under updated parameters) base logits, and the Plan Maker is
  NEVER called again during replay.
- ``softmax(final_logits)`` (used for both ``pi^0`` and ``pi^PM`` via the
  single ``final_logits`` -- see :func:`compute_joint_log_probability`)
  still integrates to 1 over the FULL global action set regardless of
  which subset received a sparse advice residual, so no normalization
  inconsistency is introduced by scoping.
- Because routing is frozen rather than recomputed, a PPO update CANNOT
  silently "discover" that a different subnet would now route differently
  under new parameters and apply the stored advice to the wrong indices --
  the stored ``consulted_action_indices`` are the authoritative target for
  scattering the stored scores, always validated against the replayed
  candidate set (see ``learning/ppo.py``'s replay-time assertions).

This was audited as a hard correctness gate before implementation began
(not an afterthought): if routing could not be defended as consistent with
the existing ratio definition this way, the intended next step was to STOP
and report the smallest principled alternative rather than silently
approximate it. It passed the audit; no second learned head or extra
stochastic term was introduced.
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
    consulted_indices: Tensor | None = None,
) -> FinalDecision:
    """Base logits, or the sparse advised residual adjustment when queried and accepted.

    ``consulted_indices`` (a LongTensor of positions into ``base_logits``)
    is required exactly when ``plan_maker_confidence`` is not None --
    subnet-scoped consultation (spec sections 2-10) always scores a
    SUBSET of the global candidates (that subnet's actions plus FINISH),
    never the full action vector, so there is no longer a "confidence
    covers every action" case to fall back to.

    Rejected advice forces beta=0 (spec section 15), which makes the
    resulting distribution numerically identical to the base policy while
    still reporting a real (if unused) alpha for metrics. No sparse
    residual is constructed at all in this branch -- final_logits is
    exactly base_logits, globally (spec section 12).
    """
    if sampled_query and plan_maker_confidence is not None:
        assert consulted_indices is not None, "consulted_indices is required whenever plan_maker_confidence is given"
        advised = policy.apply_advice(step_output.z, step_output.base_logits, plan_maker_confidence, consulted_indices)
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
