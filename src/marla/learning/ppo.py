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


class TrainingDivergedError(Exception):
    """Raised when a critical training quantity (advantage/return/log-prob/
    ratio/loss/grad-norm/network parameter) becomes non-finite (NaN or Inf)
    during a PPO update -- fails loudly and immediately (research-readiness
    audit: a run should never silently produce further steps of corrupted
    data after diverging). Checked once per PPO minibatch update, over
    already-computed tensors (a cheap ``torch.isfinite(...).all()``, not a
    per-element or per-timestep scan), so this does not materially slow
    training.
    """


def _assert_finite(name: str, value: Tensor | float) -> None:
    tensor = value if isinstance(value, Tensor) else torch.tensor(float(value))
    if not torch.isfinite(tensor).all():
        raise TrainingDivergedError(
            f"Non-finite value(s) detected in {name!r} during a PPO update -- training has diverged."
        )


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
        # Exactly the tensor stored at collection time (spec section 11) --
        # never recomputed from current simulator state, which by replay
        # time is both stale and inaccessible without calling the
        # environment again (forbidden during PPO epochs).
        out = policy.step(
            graph_obs, record.legal_action_descriptors, rstate, record.compatibility_features.to(device),
            record.visible_progress.to(device),
        )
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


def _replay_chunk_z_only(policy: RecurrentPolicy, chunk: SequenceChunk, device: torch.device) -> Tensor:
    """Recomputes just the recurrent hidden state ``z`` for every record in
    a chunk, under the CURRENT shared backbone (``graph_encoder``/
    ``action_encoder``/``recurrent_core``) -- critic-only refinement's
    only need. Never calls ``base_scorer`` (the actor head) at all, since
    its output isn't needed here -- a deliberate, cheap side benefit of
    isolating the critic-only path, not merely an optimization.

    Callers MUST wrap this in ``torch.no_grad()`` (see
    :func:`critic_refinement_update`) -- that, combined with freezing
    every non-critic parameter's ``requires_grad``, is what guarantees
    the shared backbone never receives a gradient during refinement
    (spec sections 5-6): the returned ``z`` then has no ``grad_fn`` at
    all, so even a caller that forgot to freeze parameters could not
    accidentally backpropagate into this function's own computation.
    """
    first = chunk.records[0]
    z = first.initial_gru_hidden_state.to(device)
    previous_action_embedding = first.previous_action_embedding.to(device)
    previous_reward = first.previous_training_reward
    previous_query = float(first.previous_query)

    zs: list[Tensor] = []
    for record in chunk.records:
        data = record.graph_data.to(device)
        node_embeddings, graph_embedding = policy.graph_encoder(data)
        action_embeddings = policy.action_encoder.encode_descriptors(
            record.legal_action_descriptors, node_embeddings, record.node_key_to_index, device,
            record.compatibility_features.to(device),
        )
        previous_reward_t = torch.tensor([[previous_reward]], dtype=torch.float32, device=device)
        previous_query_t = torch.tensor([[previous_query]], dtype=torch.float32, device=device)
        z = policy.recurrent_core(
            graph_embedding, record.visible_progress.unsqueeze(0).to(device),
            previous_action_embedding.unsqueeze(0).to(device), previous_reward_t, previous_query_t,
            z.unsqueeze(0).to(device),
        ).squeeze(0)
        zs.append(z)

        previous_action_embedding = action_embeddings[record.selected_action_index]
        previous_reward = record.training_reward
        previous_query = float(record.sampled_query)

    return torch.stack(zs)


def critic_refinement_update(
    policy: RecurrentPolicy, optimizer: torch.optim.Optimizer, chunks: list[SequenceChunk],
    max_grad_norm: float, device: torch.device,
) -> dict[str, float]:
    """One value-only gradient step over one minibatch of chunks (spec:
    critic-only refinement -- "does better fitting of the value head
    alone prevent the miscalibration?"). Freezes every parameter except
    ``policy.critic``'s own two tensors (``value_head.weight``/``bias``),
    recomputes ``z`` via a no-grad forward pass through the frozen shared
    backbone (:func:`_replay_chunk_z_only`), then takes a gradient step
    on the critic head alone using the SAME GAE return targets the
    normal actor+critic PPO update already computed for this rollout
    (never Monte Carlo/oracle/future data -- this function is never
    passed anything but ``chunk.returns``, sourced from the same
    ``compute_gae`` call ``optimize()`` uses; spec sections 9/40).

    Uses the RAW (unweighted) value loss: ``value_coefficient`` exists
    only to balance the joint actor+critic PPO objective and has no
    defined meaning for a critic-only update (spec section 10) -- using
    it here would just rescale the effective learning rate for no
    principled reason. This is also deliberately simpler than the joint
    PPO update's own value loss (see ``optimize()``'s ``value_loss_clipped``/
    ``value_pred_clipped``): PPO's value-clipping term exists to keep the
    critic from moving too far in a single joint update relative to the
    *actor's* trust region, a concept with no defined analogue for a
    critic-only pass with no actor update alongside it -- so refinement
    uses the plain ``0.5 * (new_values - returns).pow(2).mean()`` MSE
    loss, never the clipped-max variant.

    Isolation is enforced twice, independently (spec section 5's "strict/
    numerically justified tolerances"): (1) every non-critic parameter's
    ``requires_grad`` is set ``False`` for the duration of this call
    (restored in a ``finally`` block, even on an exception), and (2) the
    backbone forward pass itself runs under ``torch.no_grad()``, so its
    output tensor carries no ``grad_fn`` regardless of (1). Either
    mechanism alone would already prevent the shared backbone from
    changing; both together make it structurally, not just practically,
    impossible for this function to update anything but the critic head.
    """
    critic_params = list(policy.critic.parameters())
    non_critic_params = [p for name, p in policy.named_parameters() if not name.startswith("critic.")]
    original_requires_grad = [p.requires_grad for p in non_critic_params]
    for p in non_critic_params:
        p.requires_grad_(False)

    try:
        with torch.no_grad():
            z_parts = [_replay_chunk_z_only(policy, chunk, device) for chunk in chunks]
        returns_parts = [torch.tensor(chunk.returns, dtype=torch.float32, device=device) for chunk in chunks]
        z_all = torch.cat(z_parts)
        returns = torch.cat(returns_parts)

        with torch.no_grad():
            value_loss_before = 0.5 * (policy.critic(z_all) - returns).pow(2).mean().item()

        new_values = policy.critic(z_all)
        value_loss = 0.5 * (new_values - returns).pow(2).mean()
        _assert_finite("critic_refinement returns", returns)
        _assert_finite("critic_refinement value_loss", value_loss)

        optimizer.zero_grad()
        value_loss.backward()
        grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(critic_params, max_grad_norm)
        _assert_finite("critic_refinement grad_norm_before_clip", grad_norm_before_clip)
        grad_norm_after_clip = min(float(grad_norm_before_clip), max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            value_loss_after = 0.5 * (policy.critic(z_all) - returns).pow(2).mean().item()
    finally:
        for p, orig in zip(non_critic_params, original_requires_grad):
            p.requires_grad_(orig)

    return {
        "critic_loss_before_refinement": value_loss_before,
        "critic_loss_after_refinement": value_loss_after,
        "critic_refinement_grad_norm_before_clip": float(grad_norm_before_clip),
        "critic_refinement_grad_norm_after_clip": grad_norm_after_clip,
    }


def critic_refinement(
    policy: RecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    records: list[StepRecord],
    advantages: list[float],
    returns: list[float],
    sequence_length: int,
    minibatch_sequences: int,
    critic_refinement_epochs: int,
    max_grad_norm: float,
    device: torch.device,
    rng: random.Random,
) -> list[dict[str, float]]:
    """Runs ``critic_refinement_epochs`` extra value-only passes over this
    rollout's data, AFTER the normal actor+critic PPO update
    (``optimize()``) has already completed (spec section 4's exact
    ordering). ``critic_refinement_epochs=0`` (the default) makes this a
    no-op returning an empty list -- byte-identical to not calling it at
    all, which is what makes ``critic_refinement_epochs=0`` an exact
    behavioral no-op for every existing/default config (spec section 72).

    Rebuilds chunks from the SAME ``records``/``advantages``/``returns``/
    ``sequence_length`` the actor update already used (cheap regrouping,
    not a second GAE computation) -- reuses the exact existing return
    targets, never recomputes them. Shuffles minibatches from the SAME
    ``rng`` instance ``optimize()`` used (continuing its stream, not a
    separate one), matching the actor loop's own minibatching pattern
    (spec section 20: this is what makes "critic-only optimizer steps
    per rollout" a well-defined, comparable quantity).
    """
    if critic_refinement_epochs <= 0:
        return []

    chunks = build_sequence_chunks(records, advantages, returns, sequence_length)
    metrics: list[dict[str, float]] = []

    for epoch in range(1, critic_refinement_epochs + 1):
        order = list(range(len(chunks)))
        rng.shuffle(order)
        for minibatch_index, start in enumerate(range(0, len(order), minibatch_sequences), start=1):
            batch_indices = order[start : start + minibatch_sequences]
            minibatch = [chunks[i] for i in batch_indices]
            update_metrics = critic_refinement_update(policy, optimizer, minibatch, max_grad_norm, device)
            update_metrics["critic_refinement_epoch"] = epoch
            update_metrics["critic_refinement_minibatch"] = minibatch_index
            metrics.append(update_metrics)

    return metrics


def _tensor_stats(prefix: str, x: Tensor, suffix: str = "") -> dict[str, float]:
    """Compact mean/std/min/max summary of a 1-D tensor, empty-safe (never
    called on a genuinely empty tensor by this module, but defensive
    anyway) -- used for the collapse-autopsy diagnostics (spec sections
    18/22/25), never a per-sample dump. Key shape is ``<prefix>_<stat><suffix>``
    (``suffix`` lets advantage stats match spec section 18's exact naming,
    e.g. ``advantage_mean_raw``).
    """
    if x.numel() == 0:
        return {f"{prefix}_mean{suffix}": None, f"{prefix}_std{suffix}": None, f"{prefix}_min{suffix}": None, f"{prefix}_max{suffix}": None}
    return {
        f"{prefix}_mean{suffix}": x.mean().item(),
        f"{prefix}_std{suffix}": x.std(unbiased=False).item() if x.numel() > 1 else 0.0,
        f"{prefix}_min{suffix}": x.min().item(),
        f"{prefix}_max{suffix}": x.max().item(),
    }


def _finish_advantage_diagnostics(
    all_records: list[StepRecord], raw_advantages: Tensor
) -> dict[str, float | int | None]:
    """FINISH-specific raw-advantage diagnostics (spec sections 20-21) for
    one PPO minibatch -- computed on the RAW (pre-normalization) advantage,
    since that is the quantity whose scale is actually comparable to a
    reward/return; the z-score below then expresses a FINISH outlier in
    units of this same minibatch's own raw-advantage spread. Diagnostic
    only -- never fed back into the loss, never changes FINISH reward.
    """
    finish_mask = [r.selected_action_type == "finish" for r in all_records]
    finish_count = sum(finish_mask)
    successful_mask = [r.selected_action_type == "finish" and r.objective_satisfied_before_action for r in all_records]
    premature_mask = [r.selected_action_type == "finish" and not r.objective_satisfied_before_action for r in all_records]

    values = raw_advantages.tolist()
    finish_values = [v for v, m in zip(values, finish_mask) if m]
    non_finish_values = [v for v, m in zip(values, finish_mask) if not m]
    successful_values = [v for v, m in zip(values, successful_mask) if m]
    premature_values = [v for v, m in zip(values, premature_mask) if m]

    overall_mean = raw_advantages.mean().item() if raw_advantages.numel() else 0.0
    overall_std = raw_advantages.std(unbiased=False).item() if raw_advantages.numel() > 1 else 0.0
    finish_advantage_z_max = (
        (max(finish_values) - overall_mean) / (overall_std + 1e-8) if finish_values else None
    )

    return {
        "finish_selected_count": finish_count,
        "successful_finish_count": sum(successful_mask),
        "premature_finish_count": sum(premature_mask),
        "mean_raw_advantage_finish": (sum(finish_values) / len(finish_values)) if finish_values else None,
        "max_raw_advantage_finish": max(finish_values) if finish_values else None,
        "min_raw_advantage_finish": min(finish_values) if finish_values else None,
        "mean_raw_advantage_non_finish": (sum(non_finish_values) / len(non_finish_values)) if non_finish_values else None,
        "max_raw_advantage_non_finish": max(non_finish_values) if non_finish_values else None,
        "mean_raw_advantage_successful_finish": (sum(successful_values) / len(successful_values)) if successful_values else None,
        "mean_raw_advantage_premature_finish": (sum(premature_values) / len(premature_values)) if premature_values else None,
        "finish_advantage_z_max": finish_advantage_z_max,
    }


def _state_n_advantage_diagnostics(
    all_records: list[StepRecord], raw_advantages: Tensor, normalized_advantages: Tensor
) -> dict[str, float | int | None]:
    """State-N-conditioned advantage diagnostics for one PPO minibatch
    (no-visible-target-FINISH collapse investigation, spec sections 26-28)
    -- ``State N`` (no sensitive target currently visible,
    :func:`visible_target_state`) is a decision-time, policy-observable
    category, independent of whether FINISH was selected. Both RAW and
    NORMALIZED advantage are reported for State-N FINISH selections
    specifically, to test whether per-minibatch normalization *amplifies*
    an otherwise-modest raw signal (spec section 27) -- normalization
    itself is not changed anywhere in this module.

    Statistics over *selected* actions only (spec section 15/28): MARLA's
    PPO never has a counterfactual "what would the advantage of FINISH
    have been had a different action been chosen" value, so no such
    quantity is computed or implied here.
    """
    is_state_n = [visible_target_state(r) == "N" for r in all_records]
    is_finish = [r.selected_action_type == "finish" for r in all_records]
    state_n_finish_mask = [n and f for n, f in zip(is_state_n, is_finish)]

    raw_values = raw_advantages.tolist()
    normalized_values = normalized_advantages.tolist()
    state_n_finish_raw = [v for v, m in zip(raw_values, state_n_finish_mask) if m]
    state_n_finish_normalized = [v for v, m in zip(normalized_values, state_n_finish_mask) if m]

    return {
        "real_sample_count": len(all_records),
        "state_N_count": sum(is_state_n),
        "state_N_finish_count": len(state_n_finish_raw),
        "mean_raw_advantage_state_N_finish": _mean(state_n_finish_raw),
        "max_raw_advantage_state_N_finish": max(state_n_finish_raw) if state_n_finish_raw else None,
        "mean_normalized_advantage_state_N_finish": _mean(state_n_finish_normalized),
        "max_normalized_advantage_state_N_finish": max(state_n_finish_normalized) if state_n_finish_normalized else None,
    }


def ppo_update(
    policy: RecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    chunks: list[SequenceChunk],
    ppo_config: PPOConfig,
    device: torch.device,
    consultation_cost: float = 0.0,
) -> dict[str, float]:
    """One gradient step over a minibatch of sequence chunks.

    Returns the original PPO health metrics plus a compact set of
    collapse-autopsy diagnostics (spec sections 15-25): PPO ratio stats,
    raw-vs-normalized advantage stats, FINISH-specific raw-advantage
    stats, return stats, and both the pre-clip and post-clip gradient
    norm. None of these change the optimization itself -- every new field
    is a read-only summary of tensors this function already computes.
    """
    old_log_probs_parts, advantages_parts, returns_parts, old_values_parts = [], [], [], []
    new_log_probs_parts, new_values_parts, action_entropy_parts = [], [], []
    query_entropy_parts: list[Tensor] = []
    all_records: list[StepRecord] = []

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
        all_records.extend(chunk.records)

    old_log_probs = torch.cat(old_log_probs_parts)
    raw_advantages = torch.cat(advantages_parts)
    returns = torch.cat(returns_parts)
    old_values = torch.cat(old_values_parts)
    new_log_probs = torch.cat(new_log_probs_parts)
    new_values = torch.cat(new_values_parts)
    action_entropies = torch.cat(action_entropy_parts)

    # Advantage normalization audit (spec section 19): this happens HERE,
    # per PPO minibatch (i.e. over exactly the `minibatch_sequences` chunks
    # -- typically a few hundred timesteps, not the whole rollout), on the
    # concatenation of every timestep from every chunk in this minibatch.
    # There is no padding/masking step anywhere in this module: chunks are
    # ragged (0 < len <= sequence_length) and replayed one real timestep at
    # a time via a Python loop (_replay_chunk), never packed into a fixed-
    # width padded tensor -- so a padded position influencing this mean/std
    # is structurally impossible, not merely avoided by a mask (see
    # tests/test_collapse_autopsy.py for a regression test of this).
    # Fail fast (spec: NaN/Inf safety) before any further computation
    # builds on top of already-corrupted values -- a rollout with a
    # non-finite return/advantage/log-prob means GAE, the environment
    # reward, or a prior update already diverged; continuing would just
    # spread the corruption through this update's loss and gradients too.
    _assert_finite("raw_advantages", raw_advantages)
    _assert_finite("returns", returns)
    _assert_finite("old_log_probs", old_log_probs)
    _assert_finite("new_log_probs", new_log_probs)
    _assert_finite("new_values", new_values)

    if raw_advantages.numel() > 1:
        normalized_advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std(unbiased=False) + 1e-8)
    else:
        normalized_advantages = raw_advantages
    advantages = normalized_advantages

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
    _assert_finite("loss", loss)

    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_ returns the TOTAL gradient norm BEFORE
    # any scaling is applied (computed first, then grads are scaled
    # in-place by max_norm/total_norm only if total_norm > max_norm) --
    # spec section 17's "grad_norm_before_clip". The post-clip norm is not
    # separately measured (that would need a second full pass over every
    # parameter's .grad) -- it follows exactly from clip_grad_norm_'s own
    # definition: post-clip norm == min(pre-clip norm, max_grad_norm),
    # since a total_norm at or below max_norm is left untouched, and one
    # above it is rescaled to land exactly at max_norm.
    grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo_config.max_grad_norm)
    _assert_finite("grad_norm_before_clip", grad_norm_before_clip)
    grad_norm_after_clip = min(float(grad_norm_before_clip), ppo_config.max_grad_norm)
    optimizer.step()

    # One cheap isfinite() per parameter tensor (not per-element inspection
    # beyond what isfinite already does internally) -- catches a diverged
    # optimizer step (e.g. an extreme LR combined with an extreme gradient)
    # immediately, rather than after N further updates of silently
    # NaN-poisoned weights.
    for name, param in policy.named_parameters():
        _assert_finite(f"policy parameter {name!r}", param.detach())

    with torch.no_grad():
        approximate_kl = (old_log_probs - new_log_probs).mean().item()
        clip_fraction = (torch.abs(ratio - 1.0) > ppo_config.clip_epsilon).float().mean().item()
        return_var = returns.var(unbiased=False)
        explained_variance = (
            1.0 - (returns - new_values).var(unbiased=False) / (return_var + 1e-8)
        ).item()
        fraction_ratio_below_clip = (ratio < (1 - ppo_config.clip_epsilon)).float().mean().item()
        fraction_ratio_above_clip = (ratio > (1 + ppo_config.clip_epsilon)).float().mean().item()

    metrics: dict[str, float | int | None] = {
        "policy_loss": actor_loss.item(),
        "value_loss": value_loss.item(),
        "query_entropy": query_entropy.item(),
        "action_entropy": action_entropy.item(),
        "approximate_kl": approximate_kl,
        "clip_fraction": clip_fraction,
        "explained_variance": explained_variance,
        "gradient_norm": float(grad_norm_before_clip),
        "grad_norm_before_clip": float(grad_norm_before_clip),
        "grad_norm_after_clip": grad_norm_after_clip,
        "fraction_ratio_below_clip": fraction_ratio_below_clip,
        "fraction_ratio_above_clip": fraction_ratio_above_clip,
    }
    metrics.update(_tensor_stats("ratio", ratio.detach()))
    metrics.update(_tensor_stats("advantage", raw_advantages, suffix="_raw"))
    metrics.update(_tensor_stats("advantage", normalized_advantages, suffix="_normalized"))
    metrics["advantage_abs_max_raw"] = raw_advantages.abs().max().item() if raw_advantages.numel() else None
    metrics.update(_tensor_stats("return", returns))
    metrics.update(_finish_advantage_diagnostics(all_records, raw_advantages))
    metrics.update(_state_n_advantage_diagnostics(all_records, raw_advantages, normalized_advantages))
    return metrics


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

    for epoch in range(1, ppo_config.epochs + 1):
        order = list(range(len(chunks)))
        rng.shuffle(order)
        for minibatch_index, start in enumerate(range(0, len(order), minibatch_sequences), start=1):
            batch_indices = order[start : start + minibatch_sequences]
            minibatch = [chunks[i] for i in batch_indices]
            update_metrics = ppo_update(policy, optimizer, minibatch, ppo_config, device, consultation_cost)
            update_metrics["epoch"] = epoch
            update_metrics["minibatch"] = minibatch_index
            metrics.append(update_metrics)

    return metrics


def _group_records_by_episode(records: list[StepRecord]) -> list[list[StepRecord]]:
    """Groups consecutive same-episode records -- like :func:`build_sequence_chunks`
    but WITHOUT a ``sequence_length`` truncation. :func:`compute_policy_probe`
    below never backpropagates (no TBPTT memory constraint), so it can
    replay each episode in one unbroken pass for maximum fidelity to what
    the current policy actually predicts along the real, stored trajectory.
    """
    groups: list[list[StepRecord]] = []
    start = 0
    n = len(records)
    while start < n:
        episode_id = records[start].episode_id
        end = start + 1
        while end < n and records[end].episode_id == episode_id:
            end += 1
        groups.append(records[start:end])
        start = end
    return groups


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def visible_target_state(record: StepRecord) -> str:
    """Decision-time visible-state category (the no-visible-target-FINISH
    collapse investigation, spec sections 9-10) -- uses ONLY
    ``has_visible_sensitive_target_before_action``/
    ``all_visible_sensitive_targets_rooted_before_action``, both
    decision-time, visible-only facts (never ``objective_satisfied_before_action``,
    which is hidden evaluator truth and may legitimately disagree with
    these -- see that field's own docstring on ``StepRecord``):

    - ``"N"``: no sensitive target currently visible.
    - ``"I"``: >=1 visible sensitive target, not all yet ROOT.
    - ``"C"``: >=1 visible sensitive target, all currently visible ones ROOT.
    """
    if not record.has_visible_sensitive_target_before_action:
        return "N"
    return "C" if record.all_visible_sensitive_targets_rooted_before_action else "I"


def compute_policy_probe(
    policy: RecurrentPolicy, records: list[StepRecord], device: torch.device
) -> dict[str, float | None]:
    """A read-only, no-grad forward pass of the CURRENT policy over exactly
    this rollout's already-collected, stored observations (spec sections
    11-14 of the seq32-vs-64 PPO collapse investigation) -- never calls the
    environment, never reconstructs anything from live simulator state.
    Called once immediately before ``optimize()`` and once immediately
    after (same ``records``, byte-identical), so the two calls' outputs
    are directly comparable: any difference is attributable entirely to
    the PPO update that ran in between, not to a different trajectory.

    Groups records by true episode boundaries, not ``sequence_length``-
    limited chunks (see :func:`_group_records_by_episode`) -- no gradient
    is taken here, so there is no memory reason to truncate.

    Also splits FINISH probability/max-action-probability/entropy/value by
    :func:`visible_target_state` (spec sections 9/23 of the no-visible-
    target-FINISH investigation) -- ``State N/I/C`` are policy-observable
    (decision-time visible facts), a different axis from the
    ``objective_reached``/``not_reached`` split above (hidden evaluator
    truth).
    """
    finish_probs: list[float] = []
    finish_probs_reached: list[float] = []
    finish_probs_not_reached: list[float] = []
    max_probs: list[float] = []
    max_abs_logits: list[float] = []
    values: list[float] = []
    entropies: list[float] = []
    by_state: dict[str, dict[str, list[float]]] = {
        s: {"finish_prob": [], "max_prob": [], "entropy": [], "value": [], "abs_logit": []} for s in ("N", "I", "C")
    }

    with torch.no_grad():
        for group in _group_records_by_episode(records):
            first = group[0]
            z = first.initial_gru_hidden_state.to(device)
            previous_action_embedding = first.previous_action_embedding.to(device)
            previous_reward = first.previous_training_reward
            previous_query = float(first.previous_query)

            for record in group:
                graph_obs = GraphObservation(
                    data=record.graph_data.to(device), node_key_to_index=record.node_key_to_index
                )
                rstate = RecurrentState(
                    z=z, previous_action_embedding=previous_action_embedding,
                    previous_reward=previous_reward, previous_query=previous_query,
                )
                out = policy.step(
                    graph_obs, record.legal_action_descriptors, rstate,
                    record.compatibility_features.to(device), record.visible_progress.to(device),
                )

                finish_index = next(
                    (i for i, d in enumerate(record.legal_action_descriptors) if d.is_finish), None
                )
                if finish_index is not None:
                    finish_p = out.base_probs[finish_index].item()
                    finish_probs.append(finish_p)
                    if record.objective_satisfied_before_action:
                        finish_probs_reached.append(finish_p)
                    else:
                        finish_probs_not_reached.append(finish_p)

                max_prob = out.base_probs.max().item()
                abs_logit = out.base_logits.abs().max().item()
                value = out.value.item()
                log_probs = torch.log(out.base_probs.clamp_min(1e-12))
                entropy = -(out.base_probs * log_probs).sum().item()

                max_probs.append(max_prob)
                max_abs_logits.append(abs_logit)
                values.append(value)
                entropies.append(entropy)

                state = visible_target_state(record)
                bucket = by_state[state]
                bucket["max_prob"].append(max_prob)
                bucket["entropy"].append(entropy)
                bucket["value"].append(value)
                bucket["abs_logit"].append(abs_logit)
                if finish_index is not None:
                    bucket["finish_prob"].append(out.base_probs[finish_index].item())

                z = out.z
                previous_action_embedding = out.action_embeddings[record.selected_action_index]
                previous_reward = record.training_reward
                previous_query = float(record.sampled_query)

    result: dict[str, float | None] = {
        "action_entropy": _mean(entropies),
        "mean_finish_probability": _mean(finish_probs),
        "max_finish_probability": max(finish_probs) if finish_probs else None,
        "mean_finish_probability_objective_reached": _mean(finish_probs_reached),
        "mean_finish_probability_objective_not_reached": _mean(finish_probs_not_reached),
        "mean_max_action_probability": _mean(max_probs),
        "max_max_action_probability": max(max_probs) if max_probs else None,
        "mean_abs_logit": _mean(max_abs_logits),
        "max_abs_logit": max(max_abs_logits) if max_abs_logits else None,
        "value_prediction_mean": _mean(values),
        "value_prediction_std": _std(values),
        "value_prediction_min": min(values) if values else None,
        "value_prediction_max": max(values) if values else None,
    }
    for state, bucket in by_state.items():
        result[f"mean_finish_probability_state_{state}"] = _mean(bucket["finish_prob"])
        result[f"finish_probability_count_state_{state}"] = len(bucket["finish_prob"])
        result[f"mean_max_action_probability_state_{state}"] = _mean(bucket["max_prob"])
        result[f"mean_entropy_state_{state}"] = _mean(bucket["entropy"])
        result[f"mean_value_state_{state}"] = _mean(bucket["value"])
        result[f"mean_abs_logit_state_{state}"] = _mean(bucket["abs_logit"])
        result[f"decision_count_state_{state}"] = len(bucket["max_prob"])
    return result
