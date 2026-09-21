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
design audit)**: the joint log-probability this module computes is::

    log p_t = log pi^q(q_t|z_t) + (1-q_t) log pi^0(a_t) + q_t log pi^PM(a_t)

**Classification: PIECEWISE_EXACT_SEMIGRADIENT** (not an unconditionally
exact PPO policy ratio -- see the two-tier argument below and
``research/aamas2027/PAPER_EXPERIMENTS.md``'s "Routing likelihood-ratio
classification" section for the paper-facing writeup).

An earlier version of this docstring claimed reusing the stored route was
"exactly the same kind of frozen context ``legal_action_descriptors``
already was." That claim is imprecise and has been corrected after a
targeted audit. ``legal_action_descriptors`` (which actions exist at all)
is EXOGENOUS: a function of the environment/observation only, literally
independent of theta under every possible parameter value -- replaying it
verbatim is trivially exact for any theta. ``consulted_subnet = f(theta,
observation) = subnet(argmax_i base_logit_i, i != FINISH)`` is instead
PARAMETER-DEPENDENT: it is a deterministic function of theta, but a
*different* theta can genuinely produce a *different* value. "Deterministic"
and "parameter-independent" are not the same property, and only the second
one gives an unconditional exactness guarantee. Conflating them was the
error.

Formally, let ``s_old = f(theta_old, o_t)`` (the subnet actually consulted
at collection) and ``s_new = f(theta_new, o_t)`` (what the SAME deterministic
rule would pick under the parameters live at replay time). Two cases:

- **s_new == s_old** (no route switch): replaying the stored
  ``consulted_action_indices``/Plan Maker scores at ``theta_new`` computes
  EXACTLY the log-probability the fully-redeployed ``theta_new`` policy
  would assign to the stored compound action -- the consulted index set is
  identical either way (a pure function of the unchanged route and the
  already-exogenous candidate set), and the Plan Maker's response would be
  unchanged too (its request depends only on the subnet and the unchanged
  observation, not on theta directly, under the existing assumption --
  already true before this feature -- that the configured backend is a
  deterministic function of its request; genuine backend sampling
  temperature is a separate, pre-existing simplification this feature does
  not introduce or resolve). This branch is EXACT.
- **s_new != s_old** (route switch): the fully-redeployed ``theta_new``
  policy would have consulted a DIFFERENT subnet, gotten different Plan
  Maker evidence, and produced a different final-action distribution than
  what replay computes. Replay instead computes the log-probability of the
  stored action under a SURROGATE policy that keeps the collection-time
  external routing/consultation context frozen and only re-evaluates the
  actor/critic/trust-head parameters conditioned on it. This is NOT the
  probability the actual current policy would assign -- it is a genuine,
  identifiable target-policy mismatch, not a missing probability term
  (routing has zero entropy on both sides; there is no stochastic quantity
  being silently dropped -- see :mod:`marla.learning.ppo`'s route-switch
  diagnostic for why this is issue "target-policy mismatch", not issue
  "missing stochastic term").

This is a defensible, explicit design -- NOT silently accepted -- under the
following reading: the on-policy PPO batch treats "which subnet was
consulted, and what the Plan Maker said about it" as part of the frozen
STATE/CONTEXT for that collected transition (like the observation itself),
refreshed only when the next rollout is collected, rather than as a
continuously-current property of the live policy. PPO already tolerates
old-vs-new policy drift within a trust region (that is what the clip
epsilon is for); this adds one more source of same-batch drift, bounded by
the SAME trust region, and empirically measurable -- see
:mod:`marla.learning.ppo`'s ``route_switch_count``/``route_switch_fraction``/
``mean_routing_margin`` update-level diagnostics. It is not tuned or hidden:
``learning/ppo.py``'s replay loop measures, for every queried transition,
whether the CURRENT parameters' own deterministic routing rule would now
disagree with the stored route, purely for reporting.

Consequently:

- ``consulted_subnet``/``consulted_action_indices``/the Plan Maker's
  scores for them are stored on each :class:`~marla.learning.rollout.StepRecord`
  and REPLAYED VERBATIM, never recomputed from replay's own (possibly
  different, under updated parameters) base logits, and the Plan Maker is
  NEVER called again during replay -- this is unconditionally true and
  unconditionally intentional, regardless of which branch above applies.
- ``softmax(final_logits)`` (used for both ``pi^0`` and ``pi^PM`` via the
  single ``final_logits`` -- see :func:`compute_joint_log_probability`)
  still integrates to 1 over the FULL global action set regardless of
  which subset received a sparse advice residual, so no normalization
  inconsistency is introduced by scoping, in either branch.
- The stored ``consulted_action_indices`` are always validated against the
  replayed candidate set (see ``learning/ppo.py``'s replay-time
  assertions) -- indices are never silently wrong, even when the ROUTE
  they encode has become stale relative to current parameters.

**Hard empirical confirmation**: at ``theta == theta_old`` (before any
optimizer step touches the policy), ``s_new`` trivially equals ``s_old``
for every transition (nothing about theta has changed), so replay must
reproduce ``old_joint_log_probability`` exactly and report zero route
switches -- proven by
``tests/test_route_switch_diagnostics.py::test_replay_at_theta_old_reproduces_old_log_prob_and_has_zero_route_switches``.
A short real multi-update PPO diagnostic probe in the same file shows route
switching is NOT a negligible edge case once parameters move: in one
400-step/4-epoch diagnostic run (non-paper seed, ``examples/baseline.yaml``'s
default PPO hyperparameters, not trial-22's), route-switch fraction stayed
at 0.0 for the first several updates and then rose into the 15-80% range
per update as clip fraction/approximate KL grew -- see that test's printed
report and ``research/aamas2027/PAPER_EXPERIMENTS.md`` for the numbers.
This was audited as a hard correctness gate before implementation began
(not an afterthought): had routing been indefensible under this reading,
the intended next step was to STOP and report the smallest principled
alternative rather than silently approximate it. No second learned head or
extra stochastic term was introduced.
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
