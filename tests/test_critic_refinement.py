"""Tests for critic-only refinement (spec: State-N critic-calibration
collapse investigation, sections 71-72): the extra value-head-only PPO
passes ``learning.ppo.critic_refinement`` runs AFTER the normal actor+
critic PPO update, with every non-critic parameter frozen.

Covers: critic/shared parameter partition and freeze semantics, actor
logits/probabilities/FINISH probability unchanged by refinement, shared
parameter hashes unchanged, critic-head parameters changed, optimizer
state reuse (Adam moments for non-critic params untouched),
gradient-clipping semantics, ``critic_refinement_epochs=0`` exact
baseline-equivalence, config validation, and structural "no MC/oracle
data" guards. Deliberately does NOT duplicate the full MC calibration
study here -- that lives in research/diagnostics/value_calibration/ and
is reused, not re-implemented, by this phase's evaluation scripts.
"""

import copy
import csv
import inspect
import random
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from marla.config.loader import load_config, parse_config
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.learning.gae import compute_gae
from marla.learning.ppo import (
    build_sequence_chunks,
    compute_policy_probe,
    critic_refinement,
    critic_refinement_update,
    optimize,
)
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import RolloutCollector

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


def _tiny_config(critic_refinement_epochs: int = 0):
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    data["policy"]["ppo"]["steps_per_env"] = 16
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["ppo"]["critic_refinement_epochs"] = critic_refinement_epochs
    data["policy"]["recurrent"]["sequence_length"] = 4
    return parse_config(data)


async def _collect_records(config, seed: int = 1, steps: int | None = None):
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=config.policy.ppo.optimizer.learning_rate, eps=config.policy.ppo.optimizer.eps
    )
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    collector = RolloutCollector(adapter, policy, run_id="test-run", base_seed=seed)
    records, _summaries = await collector.collect(steps or config.policy.ppo.steps_per_env)

    rewards = [r.training_reward for r in records]
    values = [r.critic_value for r in records]
    terminated = [r.terminated for r in records]
    truncated = [r.truncated for r in records]
    bootstrap_values = [r.bootstrap_value for r in records]
    advantages, returns = compute_gae(
        rewards, values, terminated, truncated, bootstrap_values,
        config.policy.ppo.gamma, config.policy.ppo.gae_lambda,
    )
    return policy, optimizer, records, advantages, returns


def _named_non_critic_params(policy):
    return [(name, p) for name, p in policy.named_parameters() if not name.startswith("critic.")]


def _named_critic_params(policy):
    return [(name, p) for name, p in policy.named_parameters() if name.startswith("critic.")]


@pytest.mark.asyncio
async def test_critic_refinement_epochs_zero_is_exact_noop():
    """Spec section 72: ``critic_refinement_epochs=0`` must be byte-identical
    to not calling ``critic_refinement`` at all -- no extra optimizer step,
    no RNG consumption, no changed parameters.
    """
    config = _tiny_config(critic_refinement_epochs=0)
    policy, optimizer, records, advantages, returns = await _collect_records(config)

    before = copy.deepcopy(policy.state_dict())
    optimizer_state_before = copy.deepcopy(optimizer.state_dict())
    rng = random.Random(0)
    rng_state_before = rng.getstate()

    metrics = critic_refinement(
        policy=policy, optimizer=optimizer, records=records, advantages=advantages, returns=returns,
        sequence_length=config.policy.recurrent.sequence_length,
        minibatch_sequences=config.policy.ppo.minibatch_sequences,
        critic_refinement_epochs=config.policy.ppo.critic_refinement_epochs,
        max_grad_norm=config.policy.ppo.max_grad_norm, device=torch.device("cpu"), rng=rng,
    )

    assert metrics == []
    after = policy.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "epochs=0 must not change any parameter"
    assert optimizer.state_dict() == optimizer_state_before or not optimizer.state_dict()["state"], (
        "epochs=0 must not take any optimizer step"
    )
    assert rng.getstate() == rng_state_before, "epochs=0 must not consume any RNG draw"


@pytest.mark.asyncio
async def test_critic_refinement_changes_only_critic_head_parameters():
    """Spec sections 3/5/6: after refinement, every GraphEncoder/
    ActionEncoder/RecurrentCore/BaseActionScorer (actor) parameter must be
    byte-identical; only ``critic.*`` parameters may change.
    """
    config = _tiny_config(critic_refinement_epochs=2)
    policy, optimizer, records, advantages, returns = await _collect_records(config)

    before = copy.deepcopy(policy.state_dict())

    metrics = critic_refinement(
        policy=policy, optimizer=optimizer, records=records, advantages=advantages, returns=returns,
        sequence_length=config.policy.recurrent.sequence_length,
        minibatch_sequences=config.policy.ppo.minibatch_sequences,
        critic_refinement_epochs=config.policy.ppo.critic_refinement_epochs,
        max_grad_norm=config.policy.ppo.max_grad_norm, device=torch.device("cpu"), rng=random.Random(0),
    )
    assert len(metrics) > 0

    after = policy.state_dict()
    for name, tensor in before.items():
        if name.startswith("critic."):
            continue
        assert torch.equal(tensor, after[name]), f"non-critic parameter {name} changed during critic refinement"

    critic_changed = any(not torch.equal(before[name], after[name]) for name in before if name.startswith("critic."))
    assert critic_changed, "critic-head parameters should change during refinement"

    for m in metrics:
        for key, value in m.items():
            if value is None:
                continue
            assert torch.isfinite(torch.tensor(float(value))), f"{key} is not finite: {value}"


@pytest.mark.asyncio
async def test_critic_refinement_preserves_actor_logits_and_finish_probability():
    """Spec section 5's deterministic isolation probe: same fixed rollout
    records, actor entropy/FINISH probability/max-action-probability/
    abs-logit statistics must be EXACTLY identical before vs. after
    refinement (computed via the same no-grad forward pass used by the
    collapse-diagnostics probe); only the value-prediction statistics may
    change.
    """
    config = _tiny_config(critic_refinement_epochs=2)
    policy, optimizer, records, advantages, returns = await _collect_records(config)

    probe_before = compute_policy_probe(policy, records, torch.device("cpu"))

    critic_refinement(
        policy=policy, optimizer=optimizer, records=records, advantages=advantages, returns=returns,
        sequence_length=config.policy.recurrent.sequence_length,
        minibatch_sequences=config.policy.ppo.minibatch_sequences,
        critic_refinement_epochs=config.policy.ppo.critic_refinement_epochs,
        max_grad_norm=config.policy.ppo.max_grad_norm, device=torch.device("cpu"), rng=random.Random(0),
    )

    probe_after = compute_policy_probe(policy, records, torch.device("cpu"))

    actor_only_fields = [
        "action_entropy", "mean_finish_probability", "max_finish_probability",
        "mean_finish_probability_objective_reached", "mean_finish_probability_objective_not_reached",
        "mean_max_action_probability", "max_max_action_probability",
        "mean_abs_logit", "max_abs_logit",
        "mean_finish_probability_state_N", "mean_finish_probability_state_I", "mean_finish_probability_state_C",
        "mean_max_action_probability_state_N", "mean_max_action_probability_state_I",
        "mean_max_action_probability_state_C",
        "mean_entropy_state_N", "mean_entropy_state_I", "mean_entropy_state_C",
        "mean_abs_logit_state_N", "mean_abs_logit_state_I", "mean_abs_logit_state_C",
    ]
    for field in actor_only_fields:
        b, a = probe_before[field], probe_after[field]
        if b is None:
            assert a is None, field
            continue
        assert a == pytest.approx(b, abs=0.0), f"{field} changed by critic-only refinement: {b} -> {a}"


@pytest.mark.asyncio
async def test_critic_refinement_reuses_optimizer_state_for_critic_only():
    """Spec sections 11/20: reusing the SAME Adam optimizer instance must
    touch only the critic head's own moment buffers -- every non-critic
    parameter's Adam state (if any existed from a prior ordinary PPO
    update) must be untouched, since ``optimizer.step()`` skips any
    parameter whose ``.grad is None`` (guaranteed by freezing
    ``requires_grad`` during refinement).
    """
    config = _tiny_config(critic_refinement_epochs=2)
    policy, optimizer, records, advantages, returns = await _collect_records(config)

    # Run one ordinary PPO update first so non-critic params have real Adam
    # moment buffers to check for non-interference (a fresh optimizer has
    # no state for any parameter yet, which would make this check vacuous).
    optimize(
        policy, optimizer, records, advantages, returns, config.policy.ppo,
        config.policy.recurrent.sequence_length, config.policy.ppo.minibatch_sequences,
        torch.device("cpu"), random.Random(0),
    )
    non_critic_params = [p for _, p in _named_non_critic_params(policy)]
    non_critic_state_before = {id(p): copy.deepcopy(optimizer.state.get(p)) for p in non_critic_params}

    critic_refinement(
        policy=policy, optimizer=optimizer, records=records, advantages=advantages, returns=returns,
        sequence_length=config.policy.recurrent.sequence_length,
        minibatch_sequences=config.policy.ppo.minibatch_sequences,
        critic_refinement_epochs=config.policy.ppo.critic_refinement_epochs,
        max_grad_norm=config.policy.ppo.max_grad_norm, device=torch.device("cpu"), rng=random.Random(0),
    )

    for p in non_critic_params:
        state_after = optimizer.state.get(p)
        state_before = non_critic_state_before[id(p)]
        if state_before is None:
            assert state_after is None or not state_after, "refinement must not create Adam state for a non-critic param"
            continue
        for key in state_before:
            if isinstance(state_before[key], torch.Tensor):
                assert torch.equal(state_before[key], state_after[key]), f"non-critic optimizer state '{key}' changed"
            else:
                assert state_before[key] == state_after[key]

    critic_params = [p for _, p in _named_critic_params(policy)]
    assert all(optimizer.state.get(p) for p in critic_params), "critic head should have Adam state after refinement"


@pytest.mark.asyncio
async def test_critic_refinement_does_not_change_scheduled_learning_rate():
    """Spec section 12: extra critic-only optimizer steps must not perturb
    ``optimizer.param_groups[0]['lr']`` -- it is set once per rollout by
    the linear-schedule code, and critic refinement must inherit exactly
    that value without altering it (LR schedule stays driven by
    environment-step progress, never by optimizer-step count).
    """
    config = _tiny_config(critic_refinement_epochs=2)
    policy, optimizer, records, advantages, returns = await _collect_records(config)
    optimizer.param_groups[0]["lr"] = 1.2345e-4  # arbitrary scheduled value for this rollout

    lr_before = optimizer.param_groups[0]["lr"]
    critic_refinement(
        policy=policy, optimizer=optimizer, records=records, advantages=advantages, returns=returns,
        sequence_length=config.policy.recurrent.sequence_length,
        minibatch_sequences=config.policy.ppo.minibatch_sequences,
        critic_refinement_epochs=config.policy.ppo.critic_refinement_epochs,
        max_grad_norm=config.policy.ppo.max_grad_norm, device=torch.device("cpu"), rng=random.Random(0),
    )
    assert optimizer.param_groups[0]["lr"] == lr_before


@pytest.mark.asyncio
async def test_critic_refinement_gradient_clip_norm_matches_definition():
    """Spec section 13: ``grad_norm_after_clip`` must equal
    ``min(grad_norm_before_clip, max_grad_norm)`` -- the same relationship
    ``torch.nn.utils.clip_grad_norm_`` itself defines, and the clipping
    threshold must be the existing ``max_grad_norm``, unchanged.
    """
    config = _tiny_config(critic_refinement_epochs=1)
    policy, optimizer, records, advantages, returns = await _collect_records(config)
    chunks = build_sequence_chunks(records, advantages, returns, config.policy.recurrent.sequence_length)

    result = critic_refinement_update(
        policy, optimizer, chunks[: config.policy.ppo.minibatch_sequences],
        config.policy.ppo.max_grad_norm, torch.device("cpu"),
    )
    assert result["critic_refinement_grad_norm_after_clip"] == pytest.approx(
        min(result["critic_refinement_grad_norm_before_clip"], config.policy.ppo.max_grad_norm)
    )
    assert result["critic_refinement_grad_norm_after_clip"] <= config.policy.ppo.max_grad_norm + 1e-6


def test_critic_refinement_epochs_config_validation():
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    assert data["policy"]["ppo"]["critic_refinement_epochs"] == 0, "production default must remain 0"

    data["policy"]["ppo"]["critic_refinement_epochs"] = 2
    parse_config(data)  # must not raise

    data["policy"]["ppo"]["critic_refinement_epochs"] = -1
    with pytest.raises(Exception):
        parse_config(data)


def test_critic_refinement_signature_has_no_mc_or_oracle_access():
    """Structural guard (spec section 9): critic-only refinement must never
    be able to receive Monte Carlo calibration values, oracle/Plan-Maker
    returns, or any future-trajectory information -- only this rollout's
    own already-computed ``records``/``advantages``/``returns``.
    """
    forbidden_substrings = ("monte_carlo", "oracle", "mc_return", "plan_maker_return", "future")
    for fn in (critic_refinement, critic_refinement_update):
        params = set(inspect.signature(fn).parameters.keys())
        for name in params:
            lowered = name.lower()
            assert not any(bad in lowered for bad in forbidden_substrings), (
                f"{fn.__name__} accepts a suspicious parameter '{name}'"
            )


@pytest.mark.asyncio
async def test_critic_refinement_optimizer_step_count_is_epochs_times_minibatches():
    """Spec section 20: critic-head optimizer steps per rollout must equal
    exactly ``critic_refinement_epochs * minibatches_per_epoch`` -- never
    silently folded into or confused with the actor's own epoch count.
    """
    config = _tiny_config(critic_refinement_epochs=3)
    policy, optimizer, records, advantages, returns = await _collect_records(config)
    chunks = build_sequence_chunks(records, advantages, returns, config.policy.recurrent.sequence_length)
    expected_minibatches_per_epoch = (
        len(chunks) + config.policy.ppo.minibatch_sequences - 1
    ) // config.policy.ppo.minibatch_sequences

    metrics = critic_refinement(
        policy=policy, optimizer=optimizer, records=records, advantages=advantages, returns=returns,
        sequence_length=config.policy.recurrent.sequence_length,
        minibatch_sequences=config.policy.ppo.minibatch_sequences,
        critic_refinement_epochs=config.policy.ppo.critic_refinement_epochs,
        max_grad_norm=config.policy.ppo.max_grad_norm, device=torch.device("cpu"), rng=random.Random(0),
    )
    assert len(metrics) == 3 * expected_minibatches_per_epoch
    assert {m["critic_refinement_epoch"] for m in metrics} == {1, 2, 3}


@pytest.mark.asyncio
async def test_critic_refinement_rows_are_persisted_to_updates_csv(tmp_path):
    """Regression test for a real bug caught during this phase's own
    integration check: ``run_training_loop``'s incremental (real,
    ``run_dir``-backed) persistence path flushes each rollout's PPO
    update rows via a LOCAL ``update_metrics`` list -- critic-refinement
    rows were being computed and correctly appended to
    ``result.update_metrics`` (the in-memory, whole-run accumulator) but
    were never added to that local list, so ``updates.csv`` on disk
    silently never contained a single critic-refinement row for any real
    training run (only the in-process ``TrainingResult`` had them). Spec
    section 25 requires these to actually be persisted -- this test runs
    a real tiny training loop through the full incremental-writer path
    (exactly what a genuine ``marla run`` uses) and reads ``updates.csv``
    back from disk, not from the in-memory result.
    """
    from marla.learning.trainer import build_policy_and_optimizer, run_training_loop
    from marla.metrics.writer import initialize_run_directory
    from marla.runtime.device import resolve_device

    config = _tiny_config(critic_refinement_epochs=2)
    resolved_device = resolve_device(config.device)
    run_dir = tmp_path / "run"
    initialize_run_directory(run_dir, config, resolved_device, datetime.now(timezone.utc), None)

    torch.manual_seed(0)
    policy, optimizer = build_policy_and_optimizer(config, resolved_device.torch_device)
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )

    await run_training_loop(
        policy=policy, optimizer=optimizer, adapter=adapter, run_id="test-run",
        ppo_config=config.policy.ppo, sequence_length=config.policy.recurrent.sequence_length,
        num_rollouts=2, device=resolved_device.torch_device, seed=1,
        config=config, run_dir=run_dir, collapse_diagnostics=True,
    )

    with open(run_dir / "updates.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    refinement_rows = [r for r in rows if r.get("critic_refinement_epoch")]
    assert refinement_rows, "critic-refinement update rows must actually reach updates.csv on disk"

    # Every "update" id assigned across a rollout (actor rows + refinement
    # rows together) must be a contiguous run -- no gaps, no duplicates --
    # proving the numbering scheme and the persistence path agree with
    # each other now that both are fixed.
    update_ids = sorted(int(r["update"]) for r in rows)
    assert update_ids == list(range(update_ids[0], update_ids[0] + len(update_ids))), (
        f"update ids must be contiguous, got gaps: {update_ids}"
    )

    for r in refinement_rows:
        for key in (
            "critic_loss_before_refinement", "critic_loss_after_refinement",
            "critic_refinement_grad_norm_before_clip", "critic_refinement_grad_norm_after_clip",
        ):
            assert r[key] not in ("", None), f"{key} missing from a persisted refinement row"
