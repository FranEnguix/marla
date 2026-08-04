"""Recurrent PPO optimization (spec sections 16-17).

Sequences are chunked per-episode (never spanning an episode boundary) and
replayed by re-running :meth:`RecurrentPolicy.step` forward through the
chunk with gradients enabled, seeded by each chunk's *stored* initial hidden
state and previous-action embedding (a "stored state" truncated-BPTT
scheme: the chunk boundary is a fixed input, not backpropagated through).
Because ``NASimEmu`` observations are already fixed in the buffer, this
recomputes only the *encoding* of each transition under the current
parameters -- it never touches the environment or the Plan Maker again. For
the assisted variant, the exact stored Plan Maker confidence vector is
reused verbatim; the Plan Maker itself is never re-invoked during replay.

Sequences are batched and shuffled as whole units (never individual
transitions), and each chunk stays within one episode by construction, so
there is no cross-episode hidden-state leakage to mask out. Because MARLA
does not vectorize environments, chunks are replayed one at a time rather
than as a single padded tensor -- correctness first, matching the project's
"NASimEmu simulation correctness is the priority" scoping for v0.1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor

from marla.config.models import PPOConfig
from marla.environment.graph import GraphObservation
from marla.learning.decision import compute_final_decision, compute_joint_log_probability, compute_query_probability
from marla.learning.recurrent_policy import RecurrentPolicy, RecurrentState
from marla.learning.rollout import StepRecord


@dataclass
class SequenceChunk:
    records: list[StepRecord]
    advantages: list[float]
    returns: list[float]


def build_sequence_chunks(
    records: list[StepRecord],
    advantages: list[float],
    returns: list[float],
    sequence_length: int,
) -> list[SequenceChunk]:
    """Group consecutive same-episode records into chunks of at most ``sequence_length``.

    A chunk never spans two episodes: whenever ``episode_id`` changes, a new
    chunk starts, which is exactly what prevents hidden-state leakage across
    episode boundaries during replay.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")

    chunks: list[SequenceChunk] = []
    start = 0
    n = len(records)
    while start < n:
        episode_id = records[start].episode_id
        end = start + 1
        while end < n and records[end].episode_id == episode_id and (end - start) < sequence_length:
            end += 1
        chunks.append(
            SequenceChunk(
                records=records[start:end],
                advantages=advantages[start:end],
                returns=returns[start:end],
            )
        )
        start = end
    return chunks


def _bernoulli_entropy(probability: Tensor) -> Tensor:
    p = probability.clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


@dataclass
class ReplayOutputs:
    new_joint_log_probs: Tensor
    new_values: Tensor
    action_entropies: Tensor
    query_entropies: Tensor | None  # None for a policy built with consultation_enabled=False


def _replay_chunk(
    policy: RecurrentPolicy, chunk: SequenceChunk, device: torch.device, consultation_cost: float
) -> ReplayOutputs:
    """Re-encodes a chunk under current parameters; gradients flow through it."""
    first = chunk.records[0]
    z = first.initial_gru_hidden_state.to(device)
    previous_action_embedding = first.previous_action_embedding.to(device)
    previous_reward = first.previous_training_reward
    previous_query = float(first.previous_query)

    new_joint_log_probs: list[Tensor] = []
    new_values: list[Tensor] = []
    action_entropies: list[Tensor] = []
    query_entropies: list[Tensor] | None = [] if policy.consultation_enabled else None

    for record in chunk.records:
        graph_obs = GraphObservation(data=record.graph_data.to(device), node_key_to_index=record.node_key_to_index)
        rstate = RecurrentState(
            z=z,
            previous_action_embedding=previous_action_embedding,
            previous_reward=previous_reward,
            previous_query=previous_query,
        )
        out = policy.step(graph_obs, record.legal_action_descriptors, rstate)
        new_values.append(out.value)

        if policy.consultation_enabled:
            query_probability = compute_query_probability(
                policy, out, len(record.legal_action_descriptors), consultation_cost
            )
            query_entropies.append(_bernoulli_entropy(query_probability))

            plan_maker_confidence = None
            if record.sampled_query and record.plan_maker_validation_status == "accepted":
                assert record.plan_maker_scores_in_action_order is not None
                plan_maker_confidence = torch.tensor(
                    record.plan_maker_scores_in_action_order, dtype=torch.float32, device=device
                )

            final_decision = compute_final_decision(policy, out, record.sampled_query, plan_maker_confidence)
            joint = compute_joint_log_probability(
                query_probability, record.sampled_query, final_decision.final_logits, record.selected_action_index
            )
            new_joint_log_probs.append(joint.joint_log_prob)

            final_probs = torch.softmax(final_decision.final_logits, dim=-1)
            final_log_probs = torch.log(final_probs.clamp_min(1e-12))
            action_entropies.append(-(final_probs * final_log_probs).sum())
        else:
            log_probs_all = torch.log(out.base_probs.clamp_min(1e-12))
            new_joint_log_probs.append(log_probs_all[record.selected_action_index])
            action_entropies.append(-(out.base_probs * log_probs_all).sum())

        z = out.z
        previous_action_embedding = out.action_embeddings[record.selected_action_index]
        previous_reward = record.training_reward
        previous_query = float(record.sampled_query)

    return ReplayOutputs(
        new_joint_log_probs=torch.stack(new_joint_log_probs),
        new_values=torch.stack(new_values),
        action_entropies=torch.stack(action_entropies),
        query_entropies=torch.stack(query_entropies) if query_entropies else None,
    )


def ppo_update(
    policy: RecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    chunks: list[SequenceChunk],
    ppo_config: PPOConfig,
    device: torch.device,
    consultation_cost: float = 0.0,
) -> dict[str, float]:
    """One gradient step over a minibatch of sequence chunks."""
    old_log_probs_parts, advantages_parts, returns_parts, old_values_parts = [], [], [], []
    new_log_probs_parts, new_values_parts, action_entropy_parts = [], [], []
    query_entropy_parts: list[Tensor] = []

    for chunk in chunks:
        replay = _replay_chunk(policy, chunk, device, consultation_cost)
        old_log_probs_parts.append(
            torch.tensor([r.old_joint_log_probability for r in chunk.records], device=device)
        )
        advantages_parts.append(torch.tensor(chunk.advantages, dtype=torch.float32, device=device))
        returns_parts.append(torch.tensor(chunk.returns, dtype=torch.float32, device=device))
        old_values_parts.append(
            torch.tensor([r.critic_value for r in chunk.records], dtype=torch.float32, device=device)
        )
        new_log_probs_parts.append(replay.new_joint_log_probs)
        new_values_parts.append(replay.new_values)
        action_entropy_parts.append(replay.action_entropies)
        if replay.query_entropies is not None:
            query_entropy_parts.append(replay.query_entropies)

    old_log_probs = torch.cat(old_log_probs_parts)
    advantages = torch.cat(advantages_parts)
    returns = torch.cat(returns_parts)
    old_values = torch.cat(old_values_parts)
    new_log_probs = torch.cat(new_log_probs_parts)
    new_values = torch.cat(new_values_parts)
    action_entropies = torch.cat(action_entropy_parts)

    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    ratio = torch.exp(new_log_probs - old_log_probs)
    surrogate_1 = ratio * advantages
    surrogate_2 = torch.clamp(ratio, 1 - ppo_config.clip_epsilon, 1 + ppo_config.clip_epsilon) * advantages
    actor_loss = -torch.min(surrogate_1, surrogate_2).mean()

    value_pred_clipped = old_values + torch.clamp(
        new_values - old_values, -ppo_config.clip_epsilon, ppo_config.clip_epsilon
    )
    value_loss_unclipped = (new_values - returns).pow(2)
    value_loss_clipped = (value_pred_clipped - returns).pow(2)
    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

    action_entropy = action_entropies.mean()
    query_entropy = torch.cat(query_entropy_parts).mean() if query_entropy_parts else torch.zeros((), device=device)

    loss = (
        actor_loss
        + ppo_config.value_coefficient * value_loss
        - ppo_config.action_entropy_coefficient * action_entropy
        - ppo_config.query_entropy_coefficient * query_entropy
    )

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo_config.max_grad_norm)
    optimizer.step()

    with torch.no_grad():
        approximate_kl = (old_log_probs - new_log_probs).mean().item()
        clip_fraction = (torch.abs(ratio - 1.0) > ppo_config.clip_epsilon).float().mean().item()
        return_var = returns.var(unbiased=False)
        explained_variance = (
            1.0 - (returns - new_values).var(unbiased=False) / (return_var + 1e-8)
        ).item()

    return {
        "policy_loss": actor_loss.item(),
        "value_loss": value_loss.item(),
        "query_entropy": query_entropy.item(),
        "action_entropy": action_entropy.item(),
        "approximate_kl": approximate_kl,
        "clip_fraction": clip_fraction,
        "explained_variance": explained_variance,
        "gradient_norm": float(grad_norm),
    }


def optimize(
    policy: RecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    records: list[StepRecord],
    advantages: list[float],
    returns: list[float],
    ppo_config: PPOConfig,
    sequence_length: int,
    minibatch_sequences: int,
    device: torch.device,
    rng: random.Random,
    consultation_cost: float = 0.0,
) -> list[dict[str, float]]:
    """Runs ``ppo_config.epochs`` passes of shuffled-minibatch updates over the rollout."""
    chunks = build_sequence_chunks(records, advantages, returns, sequence_length)
    metrics: list[dict[str, float]] = []

    for _epoch in range(ppo_config.epochs):
        order = list(range(len(chunks)))
        rng.shuffle(order)
        for start in range(0, len(order), minibatch_sequences):
            batch_indices = order[start : start + minibatch_sequences]
            minibatch = [chunks[i] for i in batch_indices]
            metrics.append(ppo_update(policy, optimizer, minibatch, ppo_config, device, consultation_cost))

    return metrics
