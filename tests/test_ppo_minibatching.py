"""v0.11.0 recurrent-minibatch partition: minibatches are sized by REAL
transition count, so optimizer steps per update do not scale with episode
fragmentation (see marla.learning.ppo.partition_minibatches).
"""

import copy
import random
from pathlib import Path

import pytest
import torch

import marla.learning.ppo as ppo
from marla.config.loader import load_config, parse_config
from marla.environment.action_compatibility import COMPATIBILITY_FEATURE_DIM
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.environment.visible_facts import VISIBLE_PROGRESS_DIM
from marla.learning.gae import compute_gae
from marla.learning.ppo import build_sequence_chunks, optimize, partition_minibatches
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import RolloutCollector, StepRecord

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


def _fake_record(episode_id: int, step: int = 0) -> StepRecord:
    return StepRecord(
        run_id="r", episode_id=episode_id, environment_step=step, observation_id=f"o-{episode_id}-{step}",
        graph_data=None, node_key_to_index={}, legal_action_descriptors=[],
        compatibility_features=torch.zeros((0, COMPATIBILITY_FEATURE_DIM)),
        visible_progress=torch.zeros(VISIBLE_PROGRESS_DIM),
        initial_gru_hidden_state=torch.zeros(2), previous_action_embedding=torch.zeros(2),
        previous_training_reward=0.0, previous_query=False,
        base_logits=torch.zeros(1), final_logits=torch.zeros(1), selected_action_index=0,
        old_action_log_probability=0.0, old_joint_log_probability=0.0, critic_value=0.0,
        nasimemu_reward=0.0, consultation_cost=0.0, training_reward=0.0,
        terminated=False, truncated=False,
    )


def _records(per_env_episode_lengths: list[list[int]]) -> list[StepRecord]:
    """Concatenated per-env streams with disjoint episode ids, as the trainer builds them."""
    records = []
    for env, lengths in enumerate(per_env_episode_lengths):
        for k, length in enumerate(lengths):
            records += [_fake_record(env * 100_000 + k + 1, s) for s in range(length)]
    return records


def _window(lengths, total):
    out, used = [], 0
    for length in lengths:
        take = min(length, total - used)
        out.append(take)
        used += take
        if used == total:
            return out
    raise AssertionError("episode-length stream exhausted")


def _rollout(kind: str, steps_per_env: int = 512, num_envs: int = 4, seed: int = 0) -> list[StepRecord]:
    rng = random.Random(seed)

    def lengths():
        while True:
            if kind == "long":
                yield 100
            elif kind == "fragmented":
                yield rng.randint(1, 5)
            elif kind == "one_step":
                yield 1
            else:
                yield rng.randint(1, 150)

    return _records([_window(lengths(), steps_per_env) for _ in range(num_envs)])


class _Epochs:
    def __init__(self, epochs):
        self.epochs = epochs


def _recorded_minibatches(records, epochs=6, sequence_length=64, minibatch_sequences=16, seed=0):
    """Runs the real optimize() with ppo_update replaced by a recorder."""
    seen = []

    def recorder(policy, optimizer, chunks, ppo_config, device, consultation_cost=0.0):
        seen.append(chunks)
        return {}

    original = ppo.ppo_update
    ppo.ppo_update = recorder
    try:
        metrics = optimize(None, None, records, [0.0] * len(records), [0.0] * len(records), _Epochs(epochs),
                           sequence_length, minibatch_sequences, None, random.Random(seed))
    finally:
        ppo.ppo_update = original
    return seen, metrics


# --- A. fragmentation does not change optimizer-step count ------------------


def test_optimizer_steps_do_not_depend_on_episode_fragmentation():
    counts = {}
    real_per_step = {}
    for kind in ("long", "fragmented", "one_step", "mixed"):
        records = _rollout(kind)
        assert len(records) == 2048
        seen, metrics = _recorded_minibatches(records)
        counts[kind] = len(seen)
        real_per_step[kind] = [sum(len(c.records) for c in mb) for mb in seen]
        assert [m["minibatch"] for m in metrics[: len(seen) // 6]] == list(range(1, len(seen) // 6 + 1))
    # 2048 real transitions / (16 * 64) capacity = 2 minibatches/epoch x 6 epochs, for every fragmentation level.
    assert set(counts.values()) == {12}, counts
    for kind, sizes in real_per_step.items():
        assert all(1024 - 64 <= s <= 1024 + 64 for s in sizes), (kind, sizes)


def test_v0_10_fixed_chunk_count_would_have_exploded_for_the_same_rollouts():
    """Documents the pathology the partition removes (v0.10.x: 16 chunks per minibatch)."""
    for kind, expected in (("long", 18), ("one_step", 768)):
        chunks = build_sequence_chunks(_rollout(kind), [0.0] * 2048, [0.0] * 2048, 64)
        assert 6 * -(-len(chunks) // 16) == expected


# --- B/C. every real transition exactly once per epoch; nothing dropped/duplicated


@pytest.mark.parametrize("kind", ["long", "fragmented", "one_step", "mixed"])
@pytest.mark.parametrize("sequence_length,minibatch_sequences", [(64, 16), (8, 3), (4, 1), (32, 64)])
def test_every_real_transition_used_exactly_once_per_epoch(kind, sequence_length, minibatch_sequences):
    records = _rollout(kind, steps_per_env=97, num_envs=3, seed=sequence_length)
    epochs = 3
    seen, _ = _recorded_minibatches(records, epochs, sequence_length, minibatch_sequences, seed=7)
    all_ids = sorted(id(r) for r in records)
    per_epoch = len(seen) // epochs
    assert per_epoch * epochs == len(seen)
    for e in range(epochs):
        used = [id(r) for mb in seen[e * per_epoch:(e + 1) * per_epoch] for c in mb for r in c.records]
        assert len(used) == len(set(used)), "a transition was duplicated within an epoch"
        assert sorted(used) == all_ids, "a transition was dropped (or foreign one added) within an epoch"
        assert all(mb for mb in seen[e * per_epoch:(e + 1) * per_epoch]), "empty minibatch"


def test_partition_minibatches_properties():
    rng = random.Random(3)
    for trial in range(200):
        sequence_length = rng.randint(1, 16)
        minibatch_sequences = rng.randint(1, 8)
        lengths = [rng.randint(1, sequence_length) for _ in range(rng.randint(1, 120))]
        chunks = [ppo.SequenceChunk(records=[None] * n, advantages=[0.0] * n, returns=[0.0] * n) for n in lengths]
        order = list(range(len(chunks)))
        rng.shuffle(order)
        groups = partition_minibatches(chunks, order, minibatch_sequences, sequence_length)
        capacity = minibatch_sequences * sequence_length
        total = sum(lengths)
        assert [i for g in groups for i in g] == order, "groups must be contiguous, order-preserving, complete"
        assert all(groups)
        assert len(groups) <= -(-total // capacity)
        for g in groups:  # balanced to within one chunk of total/K
            size = sum(lengths[i] for i in g)
            assert size <= total / -(-total // capacity) + sequence_length
    assert partition_minibatches([], [], 16, 64) == []
    with pytest.raises(ValueError):
        partition_minibatches([], [], 0, 64)


# --- F. identical to v0.10.x where the two partitionings coincide ------------


def _v0_10_slices(order, minibatch_sequences):
    return [order[s:s + minibatch_sequences] for s in range(0, len(order), minibatch_sequences)]


def test_partition_equals_v0_10_slices_for_full_length_chunks():
    for num_minibatches in (1, 2, 3, 5):
        chunks = [ppo.SequenceChunk([None] * 64, [0.0] * 64, [0.0] * 64) for _ in range(16 * num_minibatches)]
        for seed in range(5):
            order = list(range(len(chunks)))
            random.Random(seed).shuffle(order)
            assert partition_minibatches(chunks, order, 16, 64) == _v0_10_slices(order, 16)


# --- real-rollout tests (D, E, F numeric) -----------------------------------


def _tiny_config(minibatch_sequences=2, sequence_length=4, steps_per_env=24, epochs=2):
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    data["policy"]["ppo"]["steps_per_env"] = steps_per_env
    data["policy"]["ppo"]["epochs"] = epochs
    data["policy"]["ppo"]["minibatch_sequences"] = minibatch_sequences
    data["policy"]["recurrent"]["sequence_length"] = sequence_length
    return parse_config(data)


async def _collect(config, base_seed=1):
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy)
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO, max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    collector = RolloutCollector(adapter, policy, run_id="t", base_seed=base_seed)
    records, _ = await collector.collect(config.policy.ppo.steps_per_env)
    advantages, returns = compute_gae(
        [r.training_reward for r in records], [r.critic_value for r in records],
        [r.terminated for r in records], [r.truncated for r in records], [r.bootstrap_value for r in records],
        config.policy.ppo.gamma, config.policy.ppo.gae_lambda,
    )
    return policy, records, advantages, returns


@pytest.mark.asyncio
async def test_padding_contributes_nothing_losses_are_means_over_real_transitions_only():
    """D: MARLA never pads (chunks are ragged and replayed step by step), so policy
    loss, value loss, entropy, KL and clip fraction must equal plain means over
    exactly the minibatch's real transitions -- recomputed here independently."""
    config = _tiny_config(sequence_length=16, steps_per_env=40)
    policy, records, advantages, returns = await _collect(config)
    chunks = build_sequence_chunks(records, advantages, returns, config.policy.recurrent.sequence_length)
    ragged = sorted(chunks, key=lambda c: len(c.records))
    minibatch = [ragged[0], ragged[-1]]
    assert len(minibatch[0].records) < len(minibatch[1].records), "fixture must contain ragged chunks"
    real = sum(len(c.records) for c in minibatch)
    ppo_cfg = config.policy.ppo

    reference = copy.deepcopy(policy)
    with torch.no_grad():
        parts = [ppo._replay_chunk(reference, c, torch.device("cpu"), 0.0) for c in minibatch]
    new_lp = torch.cat([p.new_joint_log_probs for p in parts])
    new_v = torch.cat([p.new_values for p in parts])
    ent = torch.cat([p.action_entropies for p in parts])
    old_lp = torch.tensor([r.old_joint_log_probability for c in minibatch for r in c.records])
    old_v = torch.tensor([r.critic_value for c in minibatch for r in c.records])
    adv = torch.tensor([a for c in minibatch for a in c.advantages])
    ret = torch.tensor([x for c in minibatch for x in c.returns])
    assert new_lp.numel() == real == adv.numel()
    norm = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8) if real > 1 else adv
    ratio = torch.exp(new_lp - old_lp)
    eps = ppo_cfg.clip_epsilon
    expected = {
        "policy_loss": -torch.min(ratio * norm, ratio.clamp(1 - eps, 1 + eps) * norm).mean().item(),
        "value_loss": 0.5 * torch.max((new_v - ret) ** 2,
                                      (old_v + (new_v - old_v).clamp(-eps, eps) - ret) ** 2).mean().item(),
        "action_entropy": ent.mean().item(),
        "approximate_kl": (old_lp - new_lp).mean().item(),
        "clip_fraction": ((ratio - 1).abs() > eps).float().mean().item(),
    }
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_cfg.optimizer.learning_rate)
    metrics = ppo.ppo_update(policy, optimizer, minibatch, ppo_cfg, torch.device("cpu"))
    assert metrics["real_sample_count"] == real
    for key, value in expected.items():
        assert metrics[key] == pytest.approx(value, rel=1e-5, abs=1e-7), key


@pytest.mark.asyncio
async def test_hidden_state_initialization_and_chunk_integrity_are_preserved():
    """E: minibatches contain whole chunks exactly as build_sequence_chunks made them;
    an episode-start chunk begins from the policy's initial state, and a
    mid-episode chunk begins from the collection-time hidden state that the
    previous chunk's replay (unchanged policy) reproduces."""
    config = _tiny_config(steps_per_env=40)
    policy, records, advantages, returns = await _collect(config)
    sequence_length = config.policy.recurrent.sequence_length
    chunks = build_sequence_chunks(records, advantages, returns, sequence_length)
    initial = policy.initial_recurrent_state()
    checked_start = checked_continuation = 0

    for prev, chunk in zip([None] + chunks[:-1], chunks):
        first = chunk.records[0]
        assert len({r.episode_id for r in chunk.records}) == 1
        if first.environment_step == 0:
            assert torch.equal(first.initial_gru_hidden_state, initial.z)
            assert torch.equal(first.previous_action_embedding, initial.previous_action_embedding)
            checked_start += 1
        elif prev is not None and prev.records[-1].episode_id == first.episode_id:
            with torch.no_grad():
                z_prev = ppo._replay_chunk_z_only(policy, prev, torch.device("cpu"))[-1]
            assert torch.allclose(z_prev, first.initial_gru_hidden_state, atol=1e-6)
            checked_continuation += 1
    assert checked_start >= 1 and checked_continuation >= 1, (checked_start, checked_continuation)

    seen, _ = _recorded_minibatches(records, 2, sequence_length, config.policy.ppo.minibatch_sequences)
    reference = {tuple(id(r) for r in c.records) for c in chunks}
    for mb in seen:
        for c in mb:
            assert tuple(id(r) for r in c.records) in reference, "a chunk was split or merged"


@pytest.mark.asyncio
async def test_optimize_is_numerically_identical_to_v0_10_when_partitions_coincide():
    """F: all chunks full-length and a chunk count that is a multiple of
    minibatch_sequences -> the new partition equals v0.10.x's slices, and the
    whole optimize() (params, metrics, rng stream) is bit-identical."""
    config = _tiny_config(minibatch_sequences=2, sequence_length=4, steps_per_env=16, epochs=3)
    policy, records, advantages, returns = await _collect(config)
    # One stream id for all 16 records -> four full 4-step chunks (inputs are identical for both paths).
    records = [copy.copy(r) for r in records]
    for r in records:
        r.episode_id = 1
    chunks = build_sequence_chunks(records, advantages, returns, 4)
    assert [len(c.records) for c in chunks] == [4, 4, 4, 4]
    ppo_cfg = config.policy.ppo

    def v0_10_optimize(pol, opt, rng):
        out = []
        for epoch in range(1, ppo_cfg.epochs + 1):
            order = list(range(len(chunks)))
            rng.shuffle(order)
            for k, start in enumerate(range(0, len(order), 2), start=1):
                m = ppo.ppo_update(pol, opt, [chunks[i] for i in order[start:start + 2]], ppo_cfg, torch.device("cpu"))
                m["epoch"], m["minibatch"] = epoch, k
                out.append(m)
        return out

    old_policy, new_policy = copy.deepcopy(policy), copy.deepcopy(policy)
    old_opt = torch.optim.Adam(old_policy.parameters(), lr=ppo_cfg.optimizer.learning_rate)
    new_opt = torch.optim.Adam(new_policy.parameters(), lr=ppo_cfg.optimizer.learning_rate)
    old_rng, new_rng = random.Random(11), random.Random(11)
    old_metrics = v0_10_optimize(old_policy, old_opt, old_rng)
    new_metrics = optimize(new_policy, new_opt, records, advantages, returns, ppo_cfg, 4, 2,
                           torch.device("cpu"), new_rng)
    assert len(old_metrics) == len(new_metrics) == 6
    for k, v in old_policy.state_dict().items():
        assert torch.equal(v, new_policy.state_dict()[k]), k
    for a, b in zip(old_metrics, new_metrics):
        assert a == b
    assert old_rng.random() == new_rng.random(), "rng stream must advance identically"
