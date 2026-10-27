"""Compound-decision math shared by live rollout collection and PPO replay.

Both call sites need *exactly* the same computation: at collection time to
record the "old" probabilities, and at replay time to recompute "new"
probabilities under updated parameters for the PPO ratio. Sharing this
module is what makes that guarantee structural rather than a matter of
remembering to keep two implementations in sync.

Replay never samples: it reuses the stored ``stop_selected`` and
``sampled_query`` flags, the stored ``selected_action_index``, and
(critically) the exact stored Plan Maker confidence vector -- never
recomputing or re-fetching any of them.

**Hierarchical decision** (see :mod:`marla.learning.recurrent_policy`). The
complete decision's log-probability is::

    FINISH:    log p_t = log sigma(s_t)
    CONTINUE:  log p_t = log sigma(-s_t) + log pi^q(q_t|z_t)
                         + (1-q_t) log pi^0(a_t) + q_t log pi^PM(a_t)

with ``s_t`` the stop logit (log-sigmoid, numerically stable at any
logit), the query term present only for the assisted variant, and
``pi^0``/``pi^PM`` softmaxes over the ORDINARY actions only. FINISH involves
no query decision and no consultation. With no ordinary action legal,
FINISH is forced (``log p_t = 0``). PPO uses this one joint
log-probability, so the stop and action decisions share one ratio and one
advantage.

**PPO-ratio consistency of subnet routing (subnet-scoped consultation,
design audit)**: for a CONTINUE decision, the action part of the joint
log-probability is::

    log pi^q(q_t|z_t) + (1-q_t) log pi^0(a_t) + q_t log pi^PM(a_t)

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
import torch.nn.functional as F
from torch import Tensor

from marla.learning.query_gate import compute_base_entropy, compute_top_two_margin
from marla.learning.recurrent_policy import PolicyStepOutput, RecurrentPolicy

_EPSILON = 1e-12


def compute_query_probability(policy: RecurrentPolicy, step_output: PolicyStepOutput, kappa: float) -> Tensor:
    """p_t^q, before sampling (spec section 14) -- over the ordinary actions
    only: the gate is consulted only after a CONTINUE decision."""
    device = step_output.z.device
    entropy = compute_base_entropy(step_output.base_probs)
    margin = compute_top_two_margin(step_output.base_logits)
    num_actions_t = torch.tensor(float(step_output.num_ordinary_actions), device=device)
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
    SUBSET of the ordinary candidates (that subnet's actions; never FINISH),
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
    # log pi(stop decision actually taken): log sigma(s) for FINISH,
    # log sigma(-s) for CONTINUE, 0 when FINISH was forced.
    stop_log_prob: Tensor
    # Both None for a FINISH decision (no query, no ordinary action).
    query_log_prob: Tensor | None
    action_log_prob: Tensor | None


def compute_joint_log_probability(
    stop_logit: Tensor,
    num_ordinary_actions: int,
    stop_selected: bool,
    final_logits: Tensor | None = None,
    selected_action_index: int | None = None,
    query_probability: Tensor | None = None,
    sampled_query: bool = False,
) -> JointLogProbability:
    """Log-probability of one complete hierarchical decision (module docstring).

    For CONTINUE, ``final_logits`` (``(K,)`` ordinary-action logits: base, or
    advised when queried and accepted) and ``selected_action_index`` (< K) are
    required; ``query_probability`` is given exactly for the assisted
    variant. The single ``final_logits`` softmax already *is* pi^0 when not
    queried (or when advice was rejected, since beta=0) and pi^PM when queried
    and accepted, so one formula covers both branches.
    """
    if num_ordinary_actions == 0:
        if not stop_selected:
            raise ValueError("CONTINUE is impossible with no ordinary action")
        zero = stop_logit.new_zeros(())
        return JointLogProbability(joint_log_prob=zero, stop_log_prob=zero, query_log_prob=None, action_log_prob=None)
    if stop_selected:
        stop_log_prob = F.logsigmoid(stop_logit)
        return JointLogProbability(
            joint_log_prob=stop_log_prob, stop_log_prob=stop_log_prob, query_log_prob=None, action_log_prob=None
        )

    if final_logits is None or selected_action_index is None:
        raise ValueError("final_logits and selected_action_index are required for a CONTINUE decision")
    if not 0 <= selected_action_index < num_ordinary_actions:
        raise ValueError(f"selected_action_index={selected_action_index} is not an ordinary action")
    stop_log_prob = F.logsigmoid(-stop_logit)
    action_log_prob = F.log_softmax(final_logits, dim=-1)[selected_action_index]
    joint = stop_log_prob + action_log_prob
    query_log_prob = None
    if query_probability is not None:
        query_prob_used = query_probability if sampled_query else (1 - query_probability)
        query_log_prob = torch.log(query_prob_used.clamp_min(_EPSILON))
        joint = joint + query_log_prob
    return JointLogProbability(
        joint_log_prob=joint, stop_log_prob=stop_log_prob, query_log_prob=query_log_prob,
        action_log_prob=action_log_prob,
    )


def bernoulli_entropy_from_logit(logit: Tensor) -> Tensor:
    """H(Bernoulli(sigma(logit))), stable for any finite logit."""
    p = torch.sigmoid(logit)
    return -(p * F.logsigmoid(logit) + (1 - p) * F.logsigmoid(-logit))


@dataclass
class HierarchicalEntropy:
    # H_stop + (1 - p_stop) * H(pi_action): entropy of the stop/action part of
    # the decision (the query decision is reported separately).
    action_entropy: Tensor
    stop_entropy: Tensor
    conditional_action_entropy: Tensor  # H(pi_action), not continuation-weighted
    # (1 - p_stop) * H(pi^q); None without a query gate.
    query_entropy: Tensor | None


def compute_hierarchical_entropy(
    stop_logit: Tensor, final_logits: Tensor, query_probability: Tensor | None = None
) -> HierarchicalEntropy:
    """Entropy of the hierarchical distribution at one decision state.

    ``final_logits`` is the conditional ordinary-action distribution of the
    decision's own branch (base when not queried). The conditional terms are
    continuation-weighted by ``1 - p_stop``, the probability that the
    ordinary-action (and query) decision is reached at all. With no ordinary
    action, FINISH is forced and every term is 0.
    """
    zero = stop_logit.new_zeros(())
    if final_logits.numel() == 0:
        return HierarchicalEntropy(zero, zero, zero, zero if query_probability is not None else None)
    stop_entropy = bernoulli_entropy_from_logit(stop_logit)
    continue_probability = torch.sigmoid(-stop_logit)
    log_probs = F.log_softmax(final_logits, dim=-1)
    conditional = -(log_probs.exp() * log_probs).sum()
    query_entropy = None
    if query_probability is not None:
        q = query_probability.clamp(1e-6, 1 - 1e-6)
        query_entropy = continue_probability * -(q * torch.log(q) + (1 - q) * torch.log(1 - q))
    return HierarchicalEntropy(
        action_entropy=stop_entropy + continue_probability * conditional,
        stop_entropy=stop_entropy,
        conditional_action_entropy=conditional,
        query_entropy=query_entropy,
    )


def hierarchical_action_probabilities(stop_logit: Tensor, ordinary_logits: Tensor) -> Tensor:
    """The complete distribution over ``legal_actions`` (``(K+1,)``, FINISH
    last): ``(1 - p_stop) * softmax(ordinary_logits)`` followed by
    ``p_stop`` -- or ``[1.0]`` when no ordinary action exists. Sums to 1."""
    if ordinary_logits.numel() == 0:
        return stop_logit.new_ones(1)
    p_stop = torch.sigmoid(stop_logit)
    return torch.cat([(1 - p_stop) * torch.softmax(ordinary_logits, dim=-1), p_stop.reshape(1)])
