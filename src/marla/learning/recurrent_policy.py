"""Ties the graph encoder, action encoder, GRU, base scorer, critic,
completion predictor, and stop head together into a single live
rollout-collection step.

Hierarchical action distribution (FINISH is NOT an ordinary candidate)::

    p_hat_t   = sigmoid(f_completion(z_t))              auxiliary, supervised
    p_stop_t  = sigmoid(f_stop([z_t, sg(p_hat_t)]))     PPO-trained
    pi(FINISH | h_t) = p_stop_t
    pi(a | h_t)      = (1 - p_stop_t) * pi_action(a | h_t)    a ordinary

``pi_action`` is a softmax over the ordinary (non-FINISH) legal actions
only, so FINISH's probability does not depend on how many ordinary
candidates exist. ``sg`` is stop-gradient: PPO's stop-head gradient reaches
the shared recurrent representation ``z_t`` directly but never the
completion head, whose only training signal is its own BCE loss -- policy
gradients cannot trade calibration for return. The simulator's completion
label is a training target only; it never enters ``x_t``, the action
descriptors, or the recurrent state.

Optional evidence/history action context (``policy.action_evidence``,
``policy.action_history``; :mod:`marla.environment.action_context`) enters
``pi_action`` only, through the scorer's ``W_x`` term:
``l_t(a) = f(z_t, e(a), e(h), c_t(h, a), m_t(h, a))``. It never reaches
``z_t`` (the GRU still receives the unchanged embedding of the previous
action), the critic, the completion head or the stop head, so ``p_stop`` and
the factorization above are unchanged.

This module handles exactly one environment instance at a time (MARLA does
not vectorize environments -- see spec section 25 non-goals), so all
"batch" dimensions here are size 1. Fixed-size, padded multi-sequence
batching for PPO minibatch replay is built on the same components in
Milestone 4's ``learning/ppo.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.utils import softmax as scatter_softmax

from marla.config.models import PolicyConfig
from marla.environment.action_context import EVIDENCE_FEATURE_DIM, HISTORY_FEATURE_DIM
from marla.environment.actions import ActionDescriptor, count_ordinary_actions
from marla.environment.graph import NODE_FEATURE_DIM, GraphObservation
from marla.environment.visible_facts import visible_progress_dim
from marla.learning.action_encoder import ActionEncoder
from marla.learning.advice import (
    AdvisedOutput,
    AdviceScale,
    TrustHead,
    clip_and_logit,
    compute_advice_summary,
    compute_agreement_features,
    normalize_advice,
    summary_to_tensor,
)
from marla.learning.base_scorer import BaseActionScorer
from marla.learning.critic import Critic
from marla.learning.graph_encoder import GraphEncoder
from marla.learning.query_gate import QueryGate
from marla.learning.recurrent_core import RecurrentCore


@dataclass
class RecurrentState:
    """The GRU hidden state plus everything needed to build the next ``x_t``."""

    z: Tensor  # (hidden_size,)
    previous_action_embedding: Tensor  # (action_hidden_size,)
    previous_reward: float = 0.0
    previous_query: float = 0.0


@dataclass
class PolicyStepOutput:
    z: Tensor  # (hidden_size,) -- becomes the next state's z
    # Ordinary (non-FINISH) actions only: index i is legal_actions[i] (FINISH
    # is always last, see marla.environment.actions.count_ordinary_actions).
    base_logits: Tensor  # (K,), K = N - 1
    base_probs: Tensor  # (K,), conditional pi_action; sums to 1 when K > 0
    action_embeddings: Tensor  # (N, action_hidden_size) -- FINISH's row included
    value: Tensor  # scalar tensor
    stop_logit: Tensor  # scalar tensor, logit of p_stop
    # Scalar tensor, logit of the completion prediction; None when the
    # completion predictor is disabled.
    completion_logit: Tensor | None

    @property
    def num_ordinary_actions(self) -> int:
        return int(self.base_logits.shape[0])


class RecurrentPolicy(nn.Module):
    """The trainable RL Orchestrator policy (spec section 11).

    Only this module's parameters are optimized by PPO -- the Plan Maker is
    frozen and lives entirely outside this class.
    """

    def __init__(
        self,
        policy_config: PolicyConfig,
        node_feature_dim: int = NODE_FEATURE_DIM,
        consultation_enabled: bool = False,
    ):
        super().__init__()
        ge_cfg = policy_config.graph_encoder
        ae_cfg = policy_config.action_encoder
        rc_cfg = policy_config.recurrent
        self.consultation_enabled = consultation_enabled
        # Architecture-ablation flags (spec: v2/v3-target/v3-full) -- read
        # once at construction time and exposed here so rollout collection
        # (which has no other access to policy_config) can assemble the
        # matching-width visible-progress tensor and decide whether
        # to_pyg_data should include the subnet-scan graph feature,
        # without duplicating this decision anywhere else.
        self.visible_target_progress_enabled = rc_cfg.visible_target_progress
        self.visible_subnet_exploration_enabled = rc_cfg.visible_subnet_exploration
        # Ordinary-action context ablation (A: neither, B: evidence, C:
        # history, D: both) -- see select_action_context.
        self.action_evidence_enabled = policy_config.action_evidence.enabled
        self.action_history_enabled = policy_config.action_history.enabled
        self.action_context_dim = (EVIDENCE_FEATURE_DIM if self.action_evidence_enabled else 0) + (
            HISTORY_FEATURE_DIM if self.action_history_enabled else 0
        )

        self.graph_encoder = GraphEncoder(node_feature_dim, ge_cfg.hidden_size, ge_cfg.layers)
        self.action_encoder = ActionEncoder(
            node_embedding_size=ge_cfg.hidden_size,
            action_type_embedding_size=ae_cfg.action_type_embedding_size,
            hidden_size=ae_cfg.hidden_size,
        )
        self.recurrent_core = RecurrentCore(
            graph_embedding_size=ge_cfg.hidden_size,
            action_embedding_size=ae_cfg.hidden_size,
            hidden_size=rc_cfg.hidden_size,
            visible_progress_dim=visible_progress_dim(
                self.visible_target_progress_enabled, self.visible_subnet_exploration_enabled
            ),
        )
        self.base_scorer = BaseActionScorer(
            recurrent_hidden_size=rc_cfg.hidden_size,
            action_hidden_size=ae_cfg.hidden_size,
            scorer_hidden_size=rc_cfg.hidden_size,
            action_context_size=self.action_context_dim,
        )
        self.critic = Critic(rc_cfg.hidden_size)

        # Completion predictor + stop head (shared by baseline and assisted).
        # Training-time settings (loss coefficient, class weighting) are kept
        # here so PPO can read them without extra plumbing.
        self.completion_predictor_config = policy_config.completion_predictor
        self.completion_predictor_enabled = policy_config.completion_predictor.enabled
        hidden = rc_cfg.hidden_size
        if self.completion_predictor_enabled:
            self.completion_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.stop_head = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # Start with a low stop probability so early episodes explore instead
        # of ending at once (p_stop = 0.5 everywhere would make the expected
        # episode length ~2 decisions); PPO moves it from there.
        initial = policy_config.stop_head.initial_stop_probability
        with torch.no_grad():
            self.stop_head[-1].bias.fill_(math.log(initial / (1.0 - initial)))

        # Assisted-variant-only heads (spec section 11): baseline and
        # assisted otherwise share every parameter above unchanged, so the
        # additional parameter count is exactly these three modules.
        if consultation_enabled:
            self.query_gate = QueryGate(recurrent_hidden_size=rc_cfg.hidden_size)
            self.trust_head = TrustHead(recurrent_hidden_size=rc_cfg.hidden_size)
            self.advice_scale = AdviceScale()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initial_recurrent_state(self) -> RecurrentState:
        """The recurrent state at the start of a new episode.

        Must be used at every episode boundary -- hidden state is never
        propagated across episodes (spec section 17).
        """
        device = self.device
        z = self.recurrent_core.initial_hidden_state(1, device).squeeze(0)
        previous_action_embedding = self.recurrent_core.initial_previous_action_embedding(
            1, device
        ).squeeze(0)
        return RecurrentState(z=z, previous_action_embedding=previous_action_embedding)

    def select_action_context(self, evidence: Tensor | None, history: Tensor | None) -> Tensor | None:
        """The scorer's per-action context for this policy's ablation: the
        enabled blocks of the stored evidence/history matrices, concatenated
        (evidence first), or ``None`` when neither is enabled. Used by
        rollout collection and PPO replay alike, so both feed the scorer the
        identical tensor for a stored transition."""
        parts = []
        for enabled, block, name in (
            (self.action_evidence_enabled, evidence, "evidence"),
            (self.action_history_enabled, history, "history"),
        ):
            if not enabled:
                continue
            if block is None:
                raise ValueError(f"policy.action_{name} is enabled but no {name} features were provided")
            parts.append(block)
        if not parts:
            return None
        return torch.cat(parts, dim=-1)

    def step(
        self,
        graph_observation: GraphObservation,
        legal_actions: list[ActionDescriptor],
        recurrent_state: RecurrentState,
        compatibility_matrix: Tensor,
        visible_progress: Tensor,
        action_context: Tensor | None = None,
    ) -> PolicyStepOutput:
        """``compatibility_matrix``: ``(len(legal_actions), COMPATIBILITY_FEATURE_DIM)``,
        the fixed-width action/target compatibility features (spec section
        5) in the same order as ``legal_actions`` -- computed once by the
        caller (:mod:`marla.environment.action_compatibility`) and passed
        in explicitly, never recomputed here, so rollout collection and
        PPO replay are structurally forced to use the identical tensor for
        a given stored transition (spec section 11).

        ``visible_progress``: ``(VISIBLE_PROGRESS_DIM,)``, the compact
        visible-only global objective-progress summary (see
        :func:`marla.environment.visible_facts.compute_visible_progress`)
        -- likewise computed once by the caller and passed in explicitly,
        never recomputed here, for the same replay-consistency reason.

        ``action_context``: ``(len(legal_actions), action_context_dim)`` from
        :meth:`select_action_context`; required iff ``action_context_dim > 0``.
        Only its ordinary-action rows are used, and only by the scorer.
        """
        num_ordinary = count_ordinary_actions(legal_actions)
        if (action_context is None) != (self.action_context_dim == 0):
            raise ValueError(
                f"this policy expects {'no ' if self.action_context_dim == 0 else ''}action context "
                f"(action_context_dim={self.action_context_dim})"
            )

        device = self.device
        data = graph_observation.data.to(device)

        node_embeddings, graph_embedding = self.graph_encoder(data)
        action_embeddings = self.action_encoder.encode_descriptors(
            legal_actions, node_embeddings, graph_observation.node_key_to_index, device, compatibility_matrix
        )

        previous_reward = torch.tensor(
            [[recurrent_state.previous_reward]], dtype=torch.float32, device=device
        )
        previous_query = torch.tensor(
            [[recurrent_state.previous_query]], dtype=torch.float32, device=device
        )
        z = self.recurrent_core(
            graph_embedding,
            visible_progress.unsqueeze(0).to(device),
            recurrent_state.previous_action_embedding.unsqueeze(0).to(device),
            previous_reward,
            previous_query,
            recurrent_state.z.unsqueeze(0).to(device),
        ).squeeze(0)

        ordinary_embeddings = action_embeddings[:num_ordinary]
        if num_ordinary > 0:
            action_to_sample = torch.zeros(num_ordinary, dtype=torch.long, device=device)
            ordinary_context = action_context[:num_ordinary].to(device) if action_context is not None else None
            base_logits = self.base_scorer(z.unsqueeze(0), ordinary_embeddings, action_to_sample, ordinary_context)
            base_probs = scatter_softmax(base_logits, action_to_sample)
        else:
            base_logits = z.new_zeros(0)
            base_probs = z.new_zeros(0)
        value = self.critic(z.unsqueeze(0)).squeeze(0)
        completion_logit, stop_logit = self.stop_logits(z)

        return PolicyStepOutput(
            z=z,
            base_logits=base_logits,
            base_probs=base_probs,
            action_embeddings=action_embeddings,
            value=value,
            stop_logit=stop_logit,
            completion_logit=completion_logit,
        )

    def stop_logits(self, z: Tensor) -> tuple[Tensor | None, Tensor]:
        """``(completion_logit, stop_logit)`` for one recurrent state ``z``.

        The stop head sees ``z`` and the completion probability through a
        stop-gradient (see the module docstring); with the predictor disabled
        it sees a constant 0 in that slot.
        """
        if self.completion_predictor_enabled:
            completion_logit = self.completion_head(z).squeeze(-1)
            completion_feature = torch.sigmoid(completion_logit).detach().unsqueeze(-1)
        else:
            completion_logit = None
            completion_feature = z.new_zeros(1)
        stop_logit = self.stop_head(torch.cat([z, completion_feature], dim=-1)).squeeze(-1)
        return completion_logit, stop_logit

    def apply_advice(
        self, z: Tensor, base_logits: Tensor, confidence_local: Tensor, consulted_indices: Tensor
    ) -> AdvisedOutput:
        """Sparse, subnet-scoped advice residual (subnet-scoped consultation spec, section 10).

        ``confidence_local`` covers ONLY the consulted actions (length K),
        in the same order as ``consulted_indices`` (a LongTensor of K
        positions into the GLOBAL ``base_logits`` this scope was built
        from -- reconstructed by stable action ID, never raw vector
        position, exactly as the dense predecessor of this method already
        required). Log-odds conversion, z-score normalization, and the
        TrustHead's advice-summary/agreement features are computed OVER
        THE CONSULTED SUBSET ONLY -- never re-normalized against the
        global action count, which is the whole point: an unrelated
        action from another subnet must never change this normalization
        (spec section 10/18).

        The resulting per-action residual is SPARSE: constructed as zeros
        the full width of ``base_logits``, with ``normalized_local``
        scattered in at exactly ``consulted_indices`` and left at 0.0
        everywhere else -- so ``final_logits[i] == base_logits[i]`` EXACTLY
        for every non-consulted action ``i`` (spec section 10's central
        correctness rule; no unnecessary numerical transformation of
        unconsulted logits at all). No global re-normalization ever
        happens after the scatter.

        When every non-FINISH candidate belongs to one subnet,
        ``consulted_indices`` covers the full action range and this
        reduces exactly to the previous dense computation (single-subnet
        equivalence, spec section 17).
        """
        if not self.consultation_enabled:
            raise RuntimeError("apply_advice() called on a policy built with consultation_enabled=False")

        log_odds_local = clip_and_logit(confidence_local)
        normalized_local = normalize_advice(log_odds_local)
        summary = summary_to_tensor(compute_advice_summary(log_odds_local))
        base_logits_local = base_logits[consulted_indices]
        agreement = compute_agreement_features(base_logits_local, log_odds_local)

        beta = self.trust_head(z, summary, agreement)
        alpha = self.advice_scale()

        sparse_residual = base_logits.new_zeros(base_logits.shape)
        sparse_residual[consulted_indices] = normalized_local
        final_logits = base_logits + beta * alpha * sparse_residual

        return AdvisedOutput(final_logits=final_logits, beta=beta, alpha=alpha, normalized_advice=sparse_residual)

    def advance_recurrent_state(
        self,
        step_output: PolicyStepOutput,
        selected_action_index: int,
        training_reward: float,
        query: bool,
    ) -> RecurrentState:
        """Build the next step's :class:`RecurrentState` after an action is selected."""
        return RecurrentState(
            z=step_output.z.detach(),
            previous_action_embedding=step_output.action_embeddings[selected_action_index].detach(),
            previous_reward=training_reward,
            previous_query=float(query),
        )
