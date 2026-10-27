"""Default target-dependent FINISH reward: +50 * N on a successful FINISH (every
sensitive target at ROOT), -100 * M on a premature FINISH (M targets still lacking
ROOT), applied only on the FINISH transition and identical for baseline and
assisted agents. Legacy explicit configurations keep their exact behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from nasimemu.nasim.envs.utils import AccessLevel

from marla.config.loader import config_hash, load_config, parse_config
from marla.config.models import ObjectiveConfig
from marla.environment.finish import (
    DEFAULT_COMPLETION_REWARD_PER_TARGET,
    DEFAULT_PREMATURE_FINISH_PENALTY,
    DEFAULT_PREMATURE_FINISH_PENALTY_PER_REMAINING_TARGET,
    compute_finish_reward,
)
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.evaluation.overrides import EvaluationOverrides
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import ConsultationResult, RolloutCollector

REPO_ROOT = Path(__file__).resolve().parent.parent
UNI_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/uni.v2.yaml").resolve())
# uni.v2 reset seeds yielding a given number of sensitive targets (verified in test_seed_target_counts).
SEED_FOR_TARGETS = {0: 22, 1: 17, 2: 4, 3: 2, 5: 6}


def default_adapter(**kw):
    return NasimEmuAdapter(scenario=UNI_SCENARIO, max_episode_steps=50, **kw)


def legacy_adapter():
    return NasimEmuAdapter(scenario=UNI_SCENARIO, max_episode_steps=50, completion_reward=1.0,
                           premature_finish_penalty=-1.0, premature_finish_penalty_per_remaining_target=0.0)


def set_access(adapter, addresses, level):
    state = adapter._env.env.current_state
    for addr in addresses:
        state.get_host(addr).access = level


def finish(adapter, state):
    return adapter.step(next(a for a in adapter.legal_actions(state) if a.is_finish))


# --- compute_finish_reward -------------------------------------------------------------------------------


def test_default_constants():
    assert DEFAULT_COMPLETION_REWARD_PER_TARGET == 50.0
    assert DEFAULT_PREMATURE_FINISH_PENALTY == 0.0
    assert DEFAULT_PREMATURE_FINISH_PENALTY_PER_REMAINING_TARGET == -100.0


def _default_reward(satisfied, total, remaining):
    return compute_finish_reward(
        satisfied, None, DEFAULT_PREMATURE_FINISH_PENALTY, DEFAULT_PREMATURE_FINISH_PENALTY_PER_REMAINING_TARGET,
        remaining, completion_reward_per_target=DEFAULT_COMPLETION_REWARD_PER_TARGET, total_sensitive_targets=total)


@pytest.mark.parametrize("total,expected", [(1, 50.0), (2, 100.0), (3, 150.0), (5, 250.0)])
def test_successful_finish_scales_with_total_targets(total, expected):
    assert _default_reward(True, total, 0) == expected


@pytest.mark.parametrize("total,remaining,expected", [(3, 3, -300.0), (3, 2, -200.0), (3, 1, -100.0), (5, 2, -200.0)])
def test_premature_finish_is_minus_100_per_remaining_target(total, remaining, expected):
    assert _default_reward(False, total, remaining) == expected


def test_explicit_fixed_completion_reward_takes_precedence_over_per_target():
    assert compute_finish_reward(True, 1.0, 0.0, -100.0, 0, completion_reward_per_target=50.0, total_sensitive_targets=3) == 1.0


# --- ObjectiveConfig ---------------------------------------------------------------------------------------


def test_objective_defaults_when_fields_omitted():
    o = ObjectiveConfig(type="capture_target", description="x")
    assert o.completion_reward is None
    assert o.completion_reward_per_target == 50.0
    assert o.premature_finish_penalty == 0.0
    assert o.premature_finish_penalty_per_remaining_target == -100.0


def test_explicit_legacy_overrides_are_kept():
    o = ObjectiveConfig(type="capture_target", description="x", completion_reward=1.0,
                        premature_finish_penalty=-1.0, premature_finish_penalty_per_remaining_target=0.0)
    assert (o.completion_reward, o.premature_finish_penalty, o.premature_finish_penalty_per_remaining_target) == (1.0, -1.0, 0.0)


def test_explicit_per_target_override_is_kept():
    o = ObjectiveConfig(type="capture_target", description="x", completion_reward_per_target=10.0,
                        premature_finish_penalty_per_remaining_target=-5.0)
    assert (o.completion_reward, o.completion_reward_per_target, o.premature_finish_penalty_per_remaining_target) == (None, 10.0, -5.0)


def test_explicit_fixed_completion_reward_wins_when_both_are_present():
    o = ObjectiveConfig(type="capture_target", description="x", completion_reward=1.0, completion_reward_per_target=50.0)
    assert o.completion_reward == 1.0  # precedence: fixed reward; per-target value unused


def test_legacy_config_survives_model_dump_round_trip():
    """model_dump() writes completion_reward_per_target explicitly; re-parsing must still work and keep semantics."""
    data = load_config(REPO_ROOT / "examples" / "baseline.yaml").model_dump()
    data["objective"].update(completion_reward=1.0, premature_finish_penalty=-1.0,
                             premature_finish_penalty_per_remaining_target=0.0)
    first = parse_config(data)
    again = parse_config(first.model_dump())
    assert again.objective == first.objective
    assert config_hash(again) == config_hash(first)


def test_legacy_explicit_config_hash_ignores_unused_per_target_field():
    """With an explicit fixed completion_reward the per-target value is unused, so it is not hashed:
    a legacy config keeps the hash it had before the field existed."""
    data = load_config(REPO_ROOT / "examples" / "baseline.yaml").model_dump()
    data["objective"].update(completion_reward=1.0, premature_finish_penalty=-1.0,
                             premature_finish_penalty_per_remaining_target=0.0)
    data["objective"].pop("completion_reward_per_target")
    explicit = parse_config(data)
    other = explicit.model_copy(update={"objective": explicit.objective.model_copy(update={"completion_reward_per_target": 7.0})})
    assert config_hash(explicit) == config_hash(other)
    # ...whereas with the per-target default in effect, the value is part of the hash.
    default = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    changed = default.model_copy(update={"objective": default.objective.model_copy(update={"completion_reward_per_target": 7.0})})
    assert config_hash(default) != config_hash(changed)


def test_example_configs_use_new_defaults():
    for name in ("baseline.yaml", "assisted.yaml"):
        o = load_config(REPO_ROOT / "examples" / name).objective
        assert (o.completion_reward, o.completion_reward_per_target, o.premature_finish_penalty,
                o.premature_finish_penalty_per_remaining_target) == (None, 50.0, 0.0, -100.0)


# --- NasimEmuAdapter (real simulator) ------------------------------------------------------------------------


def test_seed_target_counts():
    adapter = default_adapter()
    for n, seed in SEED_FOR_TARGETS.items():
        adapter.reset(seed=seed)
        assert adapter.sensitive_target_status()[0] == n


def test_adapter_defaults_match_config_defaults():
    adapter = default_adapter()
    assert adapter.completion_reward is None
    assert adapter.completion_reward_per_target == 50.0
    assert adapter.premature_finish_penalty == 0.0
    assert adapter.premature_finish_penalty_per_remaining_target == -100.0


@pytest.mark.parametrize("n,expected", [(1, 50.0), (2, 100.0), (3, 150.0), (5, 250.0)])
def test_successful_finish_reward_and_termination(n, expected):
    adapter = default_adapter()
    state = adapter.reset(seed=SEED_FOR_TARGETS[n])
    set_access(adapter, adapter._env.env.network.sensitive_addresses, AccessLevel.ROOT)
    result = finish(adapter, state)
    assert result.info["objective_satisfied"] is True
    assert result.nasimemu_reward == expected
    assert result.terminated is True and result.truncated is False and result.state is None


@pytest.mark.parametrize("rooted,expected", [(0, -300.0), (1, -200.0), (2, -100.0)])
def test_premature_finish_reward_per_remaining_target(rooted, expected):
    adapter = default_adapter()
    state = adapter.reset(seed=SEED_FOR_TARGETS[3])
    set_access(adapter, adapter._env.env.network.sensitive_addresses[:rooted], AccessLevel.ROOT)
    result = finish(adapter, state)
    assert result.info["sensitive_targets_remaining"] == 3 - rooted
    assert result.nasimemu_reward == expected
    assert result.terminated is True


def test_user_access_does_not_complete_a_target():
    adapter = default_adapter()
    state = adapter.reset(seed=SEED_FOR_TARGETS[3])
    addresses = adapter._env.env.network.sensitive_addresses
    set_access(adapter, addresses[:2], AccessLevel.ROOT)
    set_access(adapter, addresses[2:], AccessLevel.USER)
    result = finish(adapter, state)
    assert result.info["objective_satisfied"] is False
    assert result.nasimemu_reward == -100.0


def test_non_sensitive_hosts_are_excluded():
    adapter = default_adapter()
    state = adapter.reset(seed=SEED_FOR_TARGETS[3])
    network = adapter._env.env.network
    ordinary = [a for a in network.address_space if a not in network.sensitive_addresses]
    assert ordinary
    set_access(adapter, ordinary, AccessLevel.ROOT)
    result = finish(adapter, state)
    assert result.info["sensitive_targets_total"] == 3
    assert result.nasimemu_reward == -300.0


def test_zero_target_scenario_finish_gives_zero():
    adapter = default_adapter()
    state = adapter.reset(seed=SEED_FOR_TARGETS[0])
    result = finish(adapter, state)
    assert result.info["objective_satisfied"] is True
    assert result.nasimemu_reward == 0.0


def test_completion_bonus_requires_explicit_finish_and_non_finish_rewards_unchanged():
    """With every target already at ROOT, a non-FINISH action earns only NASimEmu's own reward
    (identical to the legacy-configured adapter) and does not end the episode."""
    new, old = default_adapter(), legacy_adapter()
    s_new, s_old = new.reset(seed=SEED_FOR_TARGETS[3]), old.reset(seed=SEED_FOR_TARGETS[3])
    for a in (new, old):
        set_access(a, a._env.env.network.sensitive_addresses, AccessLevel.ROOT)
    assert new.objective_satisfied()
    act_new = [a for a in new.legal_actions(s_new) if not a.is_finish]
    act_old = [a for a in old.legal_actions(s_old) if not a.is_finish]
    assert [a.action_id for a in act_new] == [a.action_id for a in act_old]
    for a_new, a_old in list(zip(act_new, act_old))[:8]:
        r_new, r_old = new.step(a_new), old.step(a_old)
        assert r_new.nasimemu_reward == r_old.nasimemu_reward
        assert r_new.nasimemu_reward < 50.0 and r_new.terminated is False


def test_legacy_explicit_adapter_unchanged():
    adapter = legacy_adapter()
    state = adapter.reset(seed=SEED_FOR_TARGETS[3])
    assert finish(adapter, state).nasimemu_reward == -1.0
    state = adapter.reset(seed=SEED_FOR_TARGETS[3])
    set_access(adapter, adapter._env.env.network.sensitive_addresses, AccessLevel.ROOT)
    assert finish(adapter, state).nasimemu_reward == 1.0


# --- reward flow through rollout collection (baseline and assisted) --------------------------------------


async def _collect_with_forced_finish(consultation_enabled, root_all_first):
    config = load_config(REPO_ROOT / "examples" / ("assisted.yaml" if consultation_enabled else "baseline.yaml"))
    o = config.objective
    adapter = NasimEmuAdapter(scenario=UNI_SCENARIO, max_episode_steps=50, completion_reward=o.completion_reward,
                              premature_finish_penalty=o.premature_finish_penalty,
                              premature_finish_penalty_per_remaining_target=o.premature_finish_penalty_per_remaining_target,
                              completion_reward_per_target=o.completion_reward_per_target)
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy, consultation_enabled=consultation_enabled)

    async def consult_fn(legal_actions, episode_id, step, oid, observation, subnet, n_global):
        return ConsultationResult(status="accepted", scores={a.action_id: 0.5 for a in legal_actions}, request_id="r")

    kw = dict(consultation_enabled=True, consultation_cost=0.1, consult_fn=consult_fn,
              overrides=EvaluationOverrides(query_mode="always")) if consultation_enabled else {}
    collector = RolloutCollector(adapter, policy, run_id="finish-reward", base_seed=SEED_FOR_TARGETS[3], **kw)
    original = collector._decide

    async def forced(legal_actions, step_out, state):
        decision = await original(legal_actions, step_out, state)
        if root_all_first:
            set_access(adapter, adapter._env.env.network.sensitive_addresses, AccessLevel.ROOT)
        decision["action_index"] = next(i for i, a in enumerate(legal_actions) if a.is_finish)
        return decision

    collector._decide = forced
    return await collector.collect(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("consultation_enabled", [False, True])
@pytest.mark.parametrize("root_all_first,expected", [(False, -300.0), (True, 150.0)])
async def test_finish_reward_flows_into_records_and_returns(consultation_enabled, root_all_first, expected):
    records, summaries = await _collect_with_forced_finish(consultation_enabled, root_all_first)
    assert len(records) == 1 and len(summaries) == 1
    rec, summ = records[0], summaries[0]
    cost = 0.1 if consultation_enabled else 0.0
    assert rec.nasimemu_reward == expected
    assert rec.training_reward == pytest.approx(expected - cost)  # consultation cost unchanged, applied once
    assert summ.nasimemu_return == expected
    assert summ.training_return == pytest.approx(expected - cost)
    assert summ.finish_reason == "finish"
    assert summ.successful_finish is root_all_first


# --- FINISH-behaviour metrics: successful vs premature vs timeout (deterministic, forced decisions) -----------


async def _three_outcome_episodes(consultation_enabled):
    """Episode 1: root every target then FINISH; episode 2: FINISH immediately; episode 3: never FINISH
    (time-limit truncation after 3 steps)."""
    config = load_config(REPO_ROOT / "examples" / ("assisted.yaml" if consultation_enabled else "baseline.yaml"))
    adapter = NasimEmuAdapter(scenario=UNI_SCENARIO, max_episode_steps=3)
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy, consultation_enabled=consultation_enabled)

    async def consult_fn(legal_actions, episode_id, step, oid, observation, subnet, n_global):
        return ConsultationResult(status="accepted", scores={a.action_id: 0.5 for a in legal_actions}, request_id="r")

    kw = dict(consultation_enabled=True, consultation_cost=0.1, consult_fn=consult_fn,
              overrides=EvaluationOverrides(query_mode="always")) if consultation_enabled else {}
    collector = RolloutCollector(adapter, policy, run_id="finish-metrics", base_seed=SEED_FOR_TARGETS[3], **kw)
    original = collector._decide

    async def forced(legal_actions, step_out, state):
        decision = await original(legal_actions, step_out, state)
        episode = collector._episode_id
        if episode == 1:
            set_access(adapter, adapter._env.env.network.sensitive_addresses, AccessLevel.ROOT)
        finish_index = next(i for i, a in enumerate(legal_actions) if a.is_finish)
        other_index = next(i for i, a in enumerate(legal_actions) if not a.is_finish)
        decision["action_index"] = finish_index if episode in (1, 2) else other_index
        return decision

    collector._decide = forced
    _records, summaries = await collector.collect(5)
    return config, summaries


@pytest.mark.asyncio
@pytest.mark.parametrize("consultation_enabled", [False, True])
async def test_metrics_distinguish_successful_premature_and_timeout(tmp_path, consultation_enabled):
    from collections import Counter

    from marla.metrics.accumulators import ConsultationStats
    from marla.metrics.writer import build_episode_rows, write_summary_json

    config, summaries = await _three_outcome_episodes(consultation_enabled)
    assert [s.finish_reason for s in summaries] == ["finish", "finish", "truncated"]
    rows = build_episode_rows(config, summaries)
    n1, n2 = summaries[0].sensitive_targets_total, summaries[1].sensitive_targets_total
    assert [r["finish_reward"] for r in rows] == [50.0 * n1, -100.0 * n2, 0.0]
    for r, s in zip(rows, summaries):
        assert r["native_return"] == pytest.approx(r["nasimemu_return"] - r["finish_reward"])
        assert r["benchmark_return"] == r["nasimemu_return"]
        assert r["training_return"] == pytest.approx(r["nasimemu_return"] - s.consultation_cost)
    assert rows[0]["native_return"] == 0.0 and rows[1]["native_return"] == 0.0  # FINISH-only episodes
    assert rows[0]["target_completion_fraction"] == 1.0 and rows[1]["target_completion_fraction"] == 0.0
    assert [r["successful_finish"] for r in rows] == [True, False, False]

    write_summary_json(tmp_path, summaries, ConsultationStats(), Counter(), 5, 1.0, {})
    summary = __import__("json").loads((tmp_path / "summary.json").read_text())
    for key in ("successful_finish_rate", "premature_finish_rate", "timeout_rate"):
        assert summary[key] == pytest.approx(1 / 3), key
    assert summary["mean_finish_reward"] == pytest.approx((50.0 * n1 - 100.0 * n2) / 3)
    assert summary["mean_native_return"] == pytest.approx(sum(r["native_return"] for r in rows) / 3)
    assert summary["mean_benchmark_return"] == pytest.approx(summary["mean_native_return"] + summary["mean_finish_reward"])
    assert summary["mean_training_return"] == pytest.approx(sum(s.training_return for s in summaries) / 3)
    assert summary["mean_environment_steps"] == pytest.approx(5 / 3)
    assert summary["mean_target_completion_fraction"] == pytest.approx(sum(r["target_completion_fraction"] for r in rows) / 3)
