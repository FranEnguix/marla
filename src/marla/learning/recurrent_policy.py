"""Ties the graph encoder, action encoder, GRU, base scorer, and critic
together into a single live rollout-collection step.

This module handles exactly one environment instance at a time (MARLA does
not vectorize environments -- see spec section 25 non-goals), so all
"batch" dimensions here are size 1. Fixed-size, padded multi-sequence
batching for PPO minibatch replay is built on the same components in
Milestone 4's ``learning/ppo.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.utils import softmax as scatter_softmax

from marla.config.models import PolicyConfig
from marla.environment.actions import ActionDescriptor
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
    base_logits: Tensor  # (N,)
    base_probs: Tensor  # (N,), sums to 1 over legal actions
    action_embeddings: Tensor  # (N, action_hidden_size)
    value: Tensor  # scalar tensor


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
        )
        self.critic = Critic(rc_cfg.hidden_size)

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

    def step(
        self,
        graph_observation: GraphObservation,
        legal_actions: list[ActionDescriptor],
        recurrent_state: RecurrentState,
        compatibility_matrix: Tensor,
        visible_progress: Tensor,
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
        """
        if not legal_actions:
            raise ValueError("legal_actions must include at least FINISH")

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

        action_to_sample = torch.zeros(len(legal_actions), dtype=torch.long, device=device)
        base_logits = self.base_scorer(z.unsqueeze(0), action_embeddings, action_to_sample)
        base_probs = scatter_softmax(base_logits, action_to_sample)
        value = self.critic(z.unsqueeze(0)).squeeze(0)

        return PolicyStepOutput(
            z=z,
            base_logits=base_logits,
            base_probs=base_probs,
            action_embeddings=action_embeddings,
            value=value,
        )

    def apply_advice(self, z: Tensor, base_logits: Tensor, confidence: Tensor) -> AdvisedOutput:
        """Accepted-advice residual adjustment (spec section 15).

        ``confidence`` must already be in the *same order* as ``base_logits``
        (i.e. the legal-action order), reconstructed by stable action ID --
        never by raw vector position.
        """
        if not self.consultation_enabled:
            raise RuntimeError("apply_advice() called on a policy built with consultation_enabled=False")

        log_odds = clip_and_logit(confidence)
        normalized_advice = normalize_advice(log_odds)
        summary = summary_to_tensor(compute_advice_summary(log_odds))
        agreement = compute_agreement_features(base_logits, log_odds)

        beta = self.trust_head(z, summary, agreement)
        alpha = self.advice_scale()
        final_logits = base_logits + beta * alpha * normalized_advice

        return AdvisedOutput(final_logits=final_logits, beta=beta, alpha=alpha, normalized_advice=normalized_advice)

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
