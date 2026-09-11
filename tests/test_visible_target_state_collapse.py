"""Targeted tests for the no-visible-target-FINISH collapse investigation's
new instrumentation: :func:`marla.learning.ppo.visible_target_state`'s
State N/I/C classification, the state-conditioned pre/post-update policy
probe, and the state-N-conditioned per-minibatch advantage diagnostics.

Behavioral/statistical analysis of the actual collapse (why PPO learns
``P(FINISH | State N) >> P(FINISH | State I/C)``) lives in
``research/diagnostics/ppo_learnability/README.md`` and its analysis
scripts, not here -- these tests are about the instrumentation being
correct, not about interpreting any particular run's numbers.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from marla.config.loader import load_config, parse_config
from marla.environment.action_compatibility import COMPATIBILITY_FEATURE_DIM
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.environment.visible_facts import VISIBLE_PROGRESS_DIM
from marla.learning.gae import compute_gae
from marla.learning.ppo import (
    _finish_advantage_diagnostics,
    _state_n_advantage_diagnostics,
    compute_policy_probe,
    optimize,
    visible_target_state,
)
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import RolloutCollector, StepRecord
from marla.learning.trainer import _diff_policy_probe

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


def _record(
    has_visible_sensitive_target: bool, all_rooted: bool | None, action_type: str = "exploit",
    objective_satisfied_before: bool = False,
) -> StepRecord:
    return StepRecord(
        run_id="r", episode_id=0, environment_step=0, observation_id="o",
        graph_data=None, node_key_to_index={}, legal_action_descriptors=[],
        compatibility_features=torch.zeros((0, COMPATIBILITY_FEATURE_DIM)),
        visible_progress=torch.zeros(VISIBLE_PROGRESS_DIM),
        initial_gru_hidden_state=torch.zeros(2), previous_action_embedding=torch.zeros(2),
        previous_training_reward=0.0, previous_query=False,
        base_logits=torch.zeros(1), final_logits=torch.zeros(1), selected_action_index=0,
        old_action_log_probability=0.0, old_joint_log_probability=0.0, critic_value=0.0,
        nasimemu_reward=0.0, consultation_cost=0.0, training_reward=0.0,
        terminated=False, truncated=False,
        selected_action_type=action_type,
        has_visible_sensitive_target_before_action=has_visible_sensitive_target,
        all_visible_sensitive_targets_rooted_before_action=all_rooted,
        objective_satisfied_before_action=objective_satisfied_before,
    )


# --- visible_target_state classification -------------------------------


def test_state_n_is_no_visible_sensitive_target():
    r = _record(has_visible_sensitive_target=False, all_rooted=None)
    assert visible_target_state(r) == "N"


def test_state_i_is_visible_and_incomplete():
    r = _record(has_visible_sensitive_target=True, all_rooted=False)
    assert visible_target_state(r) == "I"


def test_state_c_is_visible_and_all_rooted():
    r = _record(has_visible_sensitive_target=True, all_rooted=True)
    assert visible_target_state(r) == "C"


def test_state_uses_visible_facts_never_true_objective():
    """A record can be objective_satisfied_before_action=True (hidden
    evaluator truth) while visibly incomplete (State I) -- e.g. a hidden
    hostile host outside what's currently discovered. The classifier must
    ignore objective_satisfied_before_action entirely (spec section 10).
    """
    r = _record(has_visible_sensitive_target=True, all_rooted=False, objective_satisfied_before=True)
    assert visible_target_state(r) == "I"


# --- state-N advantage diagnostics (per minibatch) --------------------------


def test_state_n_advantage_diagnostics_separates_finish_from_nonfinish():
    records = [
        _record(False, None, action_type="finish"),  # State N, FINISH
        _record(False, None, action_type="exploit"),  # State N, non-FINISH
        _record(True, False, action_type="finish"),  # State I, FINISH -- not counted as State N
        _record(True, True, action_type="exploit"),  # State C, non-FINISH
    ]
    raw = torch.tensor([2.0, -0.5, 5.0, 0.1])
    normalized = torch.tensor([1.0, -0.2, 3.0, 0.05])

    diag = _state_n_advantage_diagnostics(records, raw, normalized)
    assert diag["real_sample_count"] == 4
    assert diag["state_N_count"] == 2
    assert diag["state_N_finish_count"] == 1
    assert diag["mean_raw_advantage_state_N_finish"] == pytest.approx(2.0)
    assert diag["max_raw_advantage_state_N_finish"] == pytest.approx(2.0)
    assert diag["mean_normalized_advantage_state_N_finish"] == pytest.approx(1.0)


def test_state_n_advantage_diagnostics_none_when_no_state_n_finish_this_minibatch():
    records = [_record(True, False, action_type="finish"), _record(False, None, action_type="exploit")]
    raw = torch.tensor([1.0, -1.0])
    normalized = torch.tensor([1.0, -1.0])
    diag = _state_n_advantage_diagnostics(records, raw, normalized)
    assert diag["state_N_finish_count"] == 0
    assert diag["mean_raw_advantage_state_N_finish"] is None
    assert diag["mean_normalized_advantage_state_N_finish"] is None


def test_state_n_diagnostics_never_reports_a_counterfactual_advantage():
    """Only actually-selected actions ever contribute -- a State-N step
    where a non-FINISH action was selected contributes nothing to the
    State-N-FINISH statistics, even though FINISH was presumably legal
    there too (spec section 15/28's "no counterfactual" limitation).
    """
    records = [_record(False, None, action_type="exploit")] * 5
    raw = torch.tensor([10.0] * 5)
    normalized = torch.tensor([10.0] * 5)
    diag = _state_n_advantage_diagnostics(records, raw, normalized)
    assert diag["state_N_count"] == 5
    assert diag["state_N_finish_count"] == 0
    assert diag["mean_raw_advantage_state_N_finish"] is None


# --- state-conditioned pre/post-update probe --------------------------------


def _tiny_config():
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    data["policy"]["ppo"]["steps_per_env"] = 16
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["metrics"]["eval_episodes"] = 0
    return parse_config(data)


@pytest.mark.asyncio
async def test_compute_policy_probe_state_counts_partition_all_records():
    config = _tiny_config()
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy)
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO, max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    collector = RolloutCollector(adapter, policy, run_id="test-run", base_seed=1)
    records, _ = await collector.collect(config.policy.ppo.steps_per_env)

    probe = compute_policy_probe(policy, records, torch.device("cpu"))
    total = probe["decision_count_state_N"] + probe["decision_count_state_I"] + probe["decision_count_state_C"]
    assert total == len(records)
    # Every probe field this task's spec asks for must exist, even if
    # some states have zero decisions this rollout (small tiny_config
    # scenario may never visit State C, say) -- the *key* must still be
    # present with value None, never silently missing.
    for state in ("N", "I", "C"):
        assert f"mean_finish_probability_state_{state}" in probe
        assert f"decision_count_state_{state}" in probe


@pytest.mark.asyncio
async def test_diff_policy_probe_includes_state_conditioned_change_fields():
    config = _tiny_config()
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy)
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO, max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    collector = RolloutCollector(adapter, policy, run_id="test-run", base_seed=1)
    records, _ = await collector.collect(config.policy.ppo.steps_per_env)

    before = compute_policy_probe(policy, records, torch.device("cpu"))
    advantages, returns = (
        lambda r: compute_gae(
            [x.training_reward for x in r], [x.critic_value for x in r], [x.terminated for x in r],
            [x.truncated for x in r], [x.bootstrap_value for x in r], config.policy.ppo.gamma, config.policy.ppo.gae_lambda,
        )
    )(records)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy.ppo.optimizer.learning_rate)
    optimize(
        policy, optimizer, records, advantages, returns, config.policy.ppo,
        config.policy.recurrent.sequence_length, config.policy.ppo.minibatch_sequences,
        torch.device("cpu"), random.Random(0),
    )
    after = compute_policy_probe(policy, records, torch.device("cpu"))

    row = _diff_policy_probe(before, after)
    for state in ("N", "I", "C"):
        assert f"mean_finish_probability_state_{state}_before_update" in row
        assert f"mean_finish_probability_state_{state}_after_update" in row
        assert f"finish_probability_state_{state}_change" in row
