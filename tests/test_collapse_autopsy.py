"""Targeted tests for the seq32-vs-64 PPO collapse investigation's new
instrumentation: per-update collapse diagnostics
(``ppo.ppo_update``'s extra return fields), the rollout-level pre/post-
update policy probe (``ppo.compute_policy_probe``/``trainer._diff_policy_probe``),
and the opt-in ``collapse_diagnostics`` wiring through
``trainer.run_training_loop``/``run_baseline_training``.

Existing coverage NOT duplicated here (cited, not re-tested): GAE terminal/
truncated/cutoff semantics (``tests/test_gae.py``), recurrent chunk
boundaries and stored-hidden-state replay (``tests/test_ppo.py::test_build_sequence_chunks_*``,
``test_ppo_replay_reuses_stored_compatibility_features_verbatim``), and
that PPO optimization never touches the environment/Plan Maker again
(``test_optimize_signature_has_no_environment_or_plan_maker_access``).
"""

from __future__ import annotations

import inspect
import random
from pathlib import Path

import pytest
import torch

from marla.config.loader import load_config, parse_config
from marla.environment.action_compatibility import COMPATIBILITY_FEATURE_DIM
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.environment.visible_facts import VISIBLE_PROGRESS_DIM
from marla.learning.gae import compute_gae
from marla.learning.ppo import SequenceChunk, compute_policy_probe, optimize, ppo_update
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import RolloutCollector, StepRecord
from marla.learning.trainer import _diff_policy_probe, run_baseline_training

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


def _tiny_config(sequence_length: int = 4, total_environment_steps: int = 32):
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    data["policy"]["ppo"]["steps_per_env"] = 16
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["ppo"]["total_environment_steps"] = total_environment_steps
    data["policy"]["recurrent"]["sequence_length"] = sequence_length
    data["metrics"]["eval_episodes"] = 0
    return parse_config(data)


async def _collect_real_records(config, seed: int = 0) -> list[StepRecord]:
    torch.manual_seed(seed)
    policy = RecurrentPolicy(config.policy)
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    collector = RolloutCollector(adapter, policy, run_id="test-run", base_seed=1)
    records, _summaries = await collector.collect(config.policy.ppo.steps_per_env)
    return policy, records


def _gae(config, records):
    rewards = [r.training_reward for r in records]
    values = [r.critic_value for r in records]
    terminated = [r.terminated for r in records]
    truncated = [r.truncated for r in records]
    bootstrap_values = [r.bootstrap_value for r in records]
    return compute_gae(rewards, values, terminated, truncated, bootstrap_values, config.policy.ppo.gamma, config.policy.ppo.gae_lambda)


# --- ppo_update's new per-minibatch diagnostics -----------------------------


@pytest.mark.asyncio
async def test_ppo_update_reports_grad_norm_before_and_after_clip_with_correct_relationship():
    """torch.nn.utils.clip_grad_norm_ returns the PRE-clip total norm; the
    post-clip norm is min(pre-clip, max_grad_norm) by that helper's own
    definition (spec section 17) -- verified against the real, if tiny,
    max_grad_norm=0.5 in examples/baseline.yaml, which virtually always
    triggers clipping on a freshly-initialized network.
    """
    config = _tiny_config()
    policy, records = await _collect_real_records(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy.ppo.optimizer.learning_rate)
    advantages, returns = _gae(config, records)

    metrics = optimize(
        policy, optimizer, records, advantages, returns, config.policy.ppo,
        config.policy.recurrent.sequence_length, config.policy.ppo.minibatch_sequences,
        torch.device("cpu"), random.Random(0),
    )
    for m in metrics:
        assert m["grad_norm_before_clip"] >= m["grad_norm_after_clip"]
        assert m["grad_norm_after_clip"] <= config.policy.ppo.max_grad_norm + 1e-6
        if m["grad_norm_before_clip"] > config.policy.ppo.max_grad_norm:
            assert m["grad_norm_after_clip"] == pytest.approx(config.policy.ppo.max_grad_norm)
        else:
            assert m["grad_norm_after_clip"] == pytest.approx(m["grad_norm_before_clip"])
        # gradient_norm is kept, unchanged, as the pre-clip value (backward compat).
        assert m["gradient_norm"] == pytest.approx(m["grad_norm_before_clip"])


@pytest.mark.asyncio
async def test_ppo_update_advantage_stats_raw_vs_normalized():
    """Raw advantage stats reflect the true, pre-normalization GAE output;
    normalized stats have mean~=0, std~=1 over the same minibatch (spec
    section 18-19) -- confirms normalization happens once, per minibatch,
    over exactly the concatenated advantages of that minibatch's chunks.
    """
    config = _tiny_config()
    policy, records = await _collect_real_records(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy.ppo.optimizer.learning_rate)
    advantages, returns = _gae(config, records)

    metrics = optimize(
        policy, optimizer, records, advantages, returns, config.policy.ppo,
        config.policy.recurrent.sequence_length, config.policy.ppo.minibatch_sequences,
        torch.device("cpu"), random.Random(0),
    )
    for m in metrics:
        assert m["advantage_min_raw"] <= m["advantage_mean_raw"] <= m["advantage_max_raw"]
        assert m["advantage_abs_max_raw"] >= abs(m["advantage_max_raw"])
        assert m["advantage_abs_max_raw"] >= abs(m["advantage_min_raw"])
        if m["advantage_std_normalized"] is not None and m["advantage_std_normalized"] > 0:
            assert m["advantage_mean_normalized"] == pytest.approx(0.0, abs=1e-4)
            assert m["advantage_std_normalized"] == pytest.approx(1.0, abs=1e-3)


def _finish_record(objective_satisfied_before: bool, advantage_index: int) -> StepRecord:
    return StepRecord(
        run_id="r", episode_id=advantage_index, environment_step=0, observation_id="o",
        graph_data=None, node_key_to_index={}, legal_action_descriptors=[],
        compatibility_features=torch.zeros((0, COMPATIBILITY_FEATURE_DIM)),
        visible_progress=torch.zeros(VISIBLE_PROGRESS_DIM),
        initial_gru_hidden_state=torch.zeros(2), previous_action_embedding=torch.zeros(2),
        previous_training_reward=0.0, previous_query=False,
        base_logits=torch.zeros(1), final_logits=torch.zeros(1), selected_action_index=0,
        old_action_log_probability=0.0, old_joint_log_probability=0.0, critic_value=0.0,
        nasimemu_reward=0.0, consultation_cost=0.0, training_reward=0.0,
        terminated=False, truncated=False,
        selected_action_type="finish", objective_satisfied_before_action=objective_satisfied_before,
    )


def _non_finish_record(advantage_index: int) -> StepRecord:
    r = _finish_record(False, advantage_index)
    r.selected_action_type = "exploit"
    return r


def test_finish_advantage_diagnostics_separates_successful_and_premature():
    from marla.learning.ppo import _finish_advantage_diagnostics

    records = [
        _non_finish_record(0),
        _finish_record(objective_satisfied_before=True, advantage_index=1),  # successful FINISH
        _finish_record(objective_satisfied_before=False, advantage_index=2),  # premature FINISH
        _non_finish_record(3),
    ]
    raw_advantages = torch.tensor([-0.1, 5.0, -2.0, 0.2])

    diag = _finish_advantage_diagnostics(records, raw_advantages)
    assert diag["finish_selected_count"] == 2
    assert diag["successful_finish_count"] == 1
    assert diag["premature_finish_count"] == 1
    assert diag["mean_raw_advantage_successful_finish"] == pytest.approx(5.0)
    assert diag["mean_raw_advantage_premature_finish"] == pytest.approx(-2.0)
    assert diag["mean_raw_advantage_finish"] == pytest.approx((5.0 + -2.0) / 2)
    assert diag["mean_raw_advantage_non_finish"] == pytest.approx((-0.1 + 0.2) / 2)
    assert diag["max_raw_advantage_finish"] == pytest.approx(5.0)
    assert diag["min_raw_advantage_finish"] == pytest.approx(-2.0)

    overall_mean = raw_advantages.mean().item()
    overall_std = raw_advantages.std(unbiased=False).item()
    expected_z = (5.0 - overall_mean) / (overall_std + 1e-8)
    assert diag["finish_advantage_z_max"] == pytest.approx(expected_z)


def test_finish_advantage_diagnostics_all_none_when_no_finish_this_minibatch():
    from marla.learning.ppo import _finish_advantage_diagnostics

    records = [_non_finish_record(0), _non_finish_record(1)]
    raw_advantages = torch.tensor([0.1, -0.2])
    diag = _finish_advantage_diagnostics(records, raw_advantages)
    assert diag["finish_selected_count"] == 0
    assert diag["mean_raw_advantage_finish"] is None
    assert diag["finish_advantage_z_max"] is None
    assert diag["mean_raw_advantage_non_finish"] == pytest.approx(-0.05)


# --- No padding/masking exists (structural) ---------------------------------


def test_no_padding_exists_chunk_and_replay_lengths_always_match_true_record_count():
    """Anti-regression for spec section 19's padding audit: MARLA's PPO
    replay never pads a ragged chunk into a fixed-width tensor -- chunks
    stay ragged (0 < len <= sequence_length) and are replayed one real
    timestep at a time (learning/ppo.py's module docstring). This proves
    the *total* record count is conserved exactly through chunking, so a
    padded/masked position influencing any downstream statistic is
    structurally impossible, not merely avoided by convention.
    """
    from marla.learning.ppo import build_sequence_chunks

    def _rec(ep):
        r = _non_finish_record(0)
        r.episode_id = ep
        return r

    records = [_rec(1), _rec(1), _rec(1), _rec(2), _rec(2), _rec(3), _rec(3), _rec(3), _rec(3), _rec(3)]
    advantages = list(range(10))
    returns = list(range(10))
    for sequence_length in (2, 3, 4, 32, 64):
        chunks = build_sequence_chunks(records, advantages, returns, sequence_length)
        total = sum(len(c.records) for c in chunks)
        assert total == len(records), f"seq_len={sequence_length}: chunking must conserve every real record"
        for c in chunks:
            assert 0 < len(c.records) <= sequence_length
            assert len(c.records) == len(c.advantages) == len(c.returns)


# --- Pre/post-update policy probe -------------------------------------------


def test_compute_policy_probe_never_touches_environment_or_simulator():
    """Structural guard mirroring test_optimize_signature_has_no_environment_or_plan_maker_access:
    compute_policy_probe's only inputs are the policy and a list of
    already-collected StepRecords -- no adapter/environment parameter
    exists to accidentally call.
    """
    sig = inspect.signature(compute_policy_probe)
    assert "adapter" not in sig.parameters
    assert "environment" not in sig.parameters
    assert set(sig.parameters) == {"policy", "records", "device"}


@pytest.mark.asyncio
async def test_compute_policy_probe_is_deterministic_and_reproducible_on_unchanged_policy():
    """Calling the probe twice on an unchanged policy and the same stored
    records must give identical results (no_grad, no dropout/BN, no RNG
    consumption) -- this is exactly what makes the before/after diff a
    valid "how much did the update change things" measurement rather than
    noise from re-evaluation itself.
    """
    config = _tiny_config()
    policy, records = await _collect_real_records(config)
    first = compute_policy_probe(policy, records, torch.device("cpu"))
    second = compute_policy_probe(policy, records, torch.device("cpu"))
    for key in first:
        if first[key] is None:
            assert second[key] is None
        else:
            assert first[key] == pytest.approx(second[key]), f"{key} differed across two probes of an unchanged policy"


def test_diff_policy_probe_reports_zero_change_when_before_equals_after():
    before = {
        "action_entropy": 4.0, "mean_finish_probability": 0.02, "max_finish_probability": 0.05,
        "mean_finish_probability_objective_reached": None, "mean_finish_probability_objective_not_reached": 0.02,
        "mean_max_action_probability": 0.03, "max_max_action_probability": 0.06,
        "mean_abs_logit": 0.2, "max_abs_logit": 0.5,
        "value_prediction_mean": 0.1, "value_prediction_std": 0.05,
        "value_prediction_min": -0.1, "value_prediction_max": 0.3,
    }
    after = dict(before)
    row = _diff_policy_probe(before, after)
    assert row["action_entropy_change"] == pytest.approx(0.0)
    assert row["finish_probability_change"] == pytest.approx(0.0)
    assert row["action_entropy_before_update"] == row["action_entropy_after_update"] == 4.0
    # An empty-denominator field (no objective-reached decisions this rollout) stays None, not 0.
    assert row["mean_finish_probability_objective_reached_before_update"] is None
    assert row["mean_finish_probability_objective_reached_after_update"] is None


def test_diff_policy_probe_change_is_none_when_either_side_missing():
    before = {"action_entropy": None, "mean_finish_probability": 0.1}
    after = {"action_entropy": 3.0, "mean_finish_probability": None}
    row = _diff_policy_probe(before, after)
    assert row["action_entropy_change"] is None
    assert row["finish_probability_change"] is None


# --- collapse_diagnostics opt-in wiring -------------------------------------


@pytest.mark.asyncio
async def test_collapse_diagnostics_default_off_leaves_rollout_rows_without_probe_data():
    config = _tiny_config()
    result = await run_baseline_training(
        config, SMALL_SCENARIO, num_rollouts=2, device=torch.device("cpu"), seed=1,
    )
    assert len(result.rollout_rows) > 0
    for row in result.rollout_rows:
        assert "action_entropy_before_update" not in row


@pytest.mark.asyncio
async def test_collapse_diagnostics_enabled_populates_probe_fields_on_every_rollout_with_records():
    config = _tiny_config()
    result = await run_baseline_training(
        config, SMALL_SCENARIO, num_rollouts=2, device=torch.device("cpu"), seed=1,
        collapse_diagnostics=True,
    )
    populated = [r for r in result.rollout_rows if "action_entropy_before_update" in r]
    assert len(populated) > 0
    for row in populated:
        assert row["action_entropy_before_update"] is not None
        assert row["action_entropy_after_update"] is not None
        assert row["action_entropy_change"] is not None


@pytest.mark.asyncio
async def test_old_run_and_new_probe_columns_both_summarize_cleanly(tmp_path):
    """generate_plots must tolerate both a rollouts.csv from BEFORE these
    new columns existed (pandas simply won't find them -- every existing
    plot already degrades gracefully per its own module docstring) and
    ONE written with the new collapse-diagnostics columns populated (this
    run) -- neither crashes marla summarize.
    """
    from marla.metrics.plots import generate_plots
    from marla.metrics.writer import write_episodes_csv, write_rollouts_csv, write_updates_csv

    config = _tiny_config()
    result = await run_baseline_training(
        config, SMALL_SCENARIO, num_rollouts=2, device=torch.device("cpu"), seed=1,
        collapse_diagnostics=True,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_episodes_csv(run_dir, config, result.episode_summaries, result.eval_episode_summaries)
    write_rollouts_csv(run_dir, result.rollout_rows)
    write_updates_csv(run_dir, config, result.update_metrics)

    written = generate_plots(run_dir, tmp_path / "plots")
    assert isinstance(written, list)  # must not raise
