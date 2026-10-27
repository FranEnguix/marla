"""FINISH reward: +N on a successful FINISH (every one of N sensitive targets at ROOT),
-M on a premature FINISH (M targets still lacking ROOT), paid only on the FINISH
transition. Non-FINISH actions keep NASimEmu's own reward; zero-target instances never
produce a (vacuous) successful FINISH.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from nasimemu.env import NASimEmuEnv
from nasimemu.nasim.envs.utils import AccessLevel
from pydantic import ValidationError

from marla.config.loader import load_config
from marla.config.models import ObjectiveConfig
from marla.environment.actions import resolve_action_target
from marla.environment.finish import (
    DEFAULT_PREMATURE_FINISH_REWARD_PER_REMAINING_TARGET,
    DEFAULT_SUCCESSFUL_FINISH_REWARD_PER_TARGET,
    compute_finish_reward,
)
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.evaluation.overrides import EvaluationOverrides
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import ConsultationResult, RolloutCollector

REPO_ROOT = Path(__file__).resolve().parent.parent
UNI_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/uni.v2.yaml").resolve())
VACUOUS_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())
# uni.v2 reset seeds yielding a given number of sensitive targets (verified in test_seed_target_counts).
SEED_FOR_TARGETS = {1: 17, 2: 4, 3: 2, 5: 0}
# A uni.v2 seed whose raw NASimEmu realization has no sensitive target at all.
ZERO_TARGET_SEED = 22


def adapter(**kw):
    return NasimEmuAdapter(scenario=UNI_SCENARIO, max_episode_steps=50, **kw)


def set_access(adapter, addresses, level):
    state = adapter._env.env.current_state
    for addr in addresses:
        state.get_host(addr).access = level


def finish(adapter, state):
    return adapter.step(next(a for a in adapter.legal_actions(state) if a.is_finish))


# --- compute_finish_reward -------------------------------------------------------------------------------


def test_default_coefficients():
    assert DEFAULT_SUCCESSFUL_FINISH_REWARD_PER_TARGET == 1.0
    assert DEFAULT_PREMATURE_FINISH_REWARD_PER_REMAINING_TARGET == -1.0


@pytest.mark.parametrize(
    "total,rooted,expected",
    [(1, 0, -1.0), (1, 1, 1.0), (3, 0, -3.0), (3, 1, -2.0), (3, 2, -1.0), (3, 3, 3.0), (5, 5, 5.0)],
)
def test_reward_table(total, rooted, expected):
    reward = compute_finish_reward(total, total - rooted)
    assert reward == expected and isinstance(reward, float)


def test_custom_coefficients():
    assert compute_finish_reward(4, 0, 2.5, -0.5) == 10.0
    assert compute_finish_reward(4, 3, 2.5, -0.5) == -1.5


@pytest.mark.parametrize("total,remaining", [(0, 0), (-1, 0), (2, 3), (2, -1)])
def test_invalid_target_counts_are_rejected(total, remaining):
    with pytest.raises(ValueError):
        compute_finish_reward(total, remaining)


# --- ObjectiveConfig -------------------------------------------------------------------------------------


def test_objective_defaults():
    o = ObjectiveConfig(type="capture_target", description="x")
    assert (o.successful_finish_reward_per_target, o.premature_finish_reward_per_remaining_target) == (1.0, -1.0)
    assert o.effective_finish_reward() == {
        "version": 2, "successful_finish_reward_per_target": 1.0, "premature_finish_reward_per_remaining_target": -1.0,
    }


@pytest.mark.parametrize("field,value", [
    ("successful_finish_reward_per_target", -1.0),
    ("premature_finish_reward_per_remaining_target", 1.0),
    ("completion_reward", 1.0),  # obsolete fixed-bonus parameters are no longer accepted
    ("premature_finish_penalty", -1.0),
])
def test_invalid_or_obsolete_reward_fields_are_rejected(field, value):
    with pytest.raises(ValidationError):
        ObjectiveConfig(type="capture_target", description="x", **{field: value})


def test_example_configs_use_defaults():
    for name in ("baseline.yaml", "assisted.yaml"):
        o = load_config(REPO_ROOT / "examples" / name).objective
        assert (o.successful_finish_reward_per_target, o.premature_finish_reward_per_remaining_target) == (1.0, -1.0)


# --- NasimEmuAdapter (real simulator) --------------------------------------------------------------------


def test_seed_target_counts():
    a = adapter()
    for n, seed in SEED_FOR_TARGETS.items():
        a.reset(seed=seed)
        assert a.sensitive_target_status()[0] == n


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_successful_finish_gives_plus_n_and_terminates(n):
    a = adapter()
    state = a.reset(seed=SEED_FOR_TARGETS[n])
    set_access(a, a._env.env.network.sensitive_addresses, AccessLevel.ROOT)
    result = finish(a, state)
    assert result.info["objective_satisfied"] is True
    assert result.nasimemu_reward == float(n)
    assert result.terminated is True and result.truncated is False and result.state is None


@pytest.mark.parametrize("rooted,expected", [(0, -3.0), (1, -2.0), (2, -1.0)])
def test_premature_finish_gives_minus_m(rooted, expected):
    a = adapter()
    state = a.reset(seed=SEED_FOR_TARGETS[3])
    set_access(a, a._env.env.network.sensitive_addresses[:rooted], AccessLevel.ROOT)
    result = finish(a, state)
    assert result.info["sensitive_targets_remaining"] == 3 - rooted
    assert result.nasimemu_reward == expected
    assert result.terminated is True


def test_user_access_does_not_count():
    a = adapter()
    state = a.reset(seed=SEED_FOR_TARGETS[3])
    addresses = a._env.env.network.sensitive_addresses
    set_access(a, addresses[:2], AccessLevel.ROOT)
    set_access(a, addresses[2:], AccessLevel.USER)
    result = finish(a, state)
    assert result.info["objective_satisfied"] is False
    assert result.nasimemu_reward == -1.0


def test_non_sensitive_hosts_are_ignored():
    a = adapter()
    state = a.reset(seed=SEED_FOR_TARGETS[3])
    network = a._env.env.network
    ordinary = [h for h in network.address_space if h not in network.sensitive_addresses]
    assert ordinary
    set_access(a, ordinary, AccessLevel.ROOT)
    result = finish(a, state)
    assert result.info["sensitive_targets_total"] == 3
    assert result.nasimemu_reward == -3.0


def test_non_finish_rewards_are_nasimemus_own_and_no_bonus_when_last_target_rooted():
    """With every target already at ROOT, non-FINISH actions earn exactly what an identically
    seeded raw NASimEmuEnv returns for the same actions, never a completion bonus, and never
    end the episode -- independent of the FINISH coefficients."""
    seed = SEED_FOR_TARGETS[3]
    marla, scaled = adapter(), adapter(successful_finish_reward_per_target=10.0)
    s1, s2 = marla.reset(seed=seed), scaled.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    raw = NASimEmuEnv(scenario_name=UNI_SCENARIO, emulate=False, step_limit=None, random_init=False,
                      observation_format="matrix", fully_obs=False, augment_with_action=False)
    random.seed(seed)
    np.random.seed(seed)
    raw.reset()
    for env in (marla, scaled):
        set_access(env, env._env.env.network.sensitive_addresses, AccessLevel.ROOT)
    for h in raw.env.network.sensitive_addresses:
        raw.env.current_state.get_host(h).access = AccessLevel.ROOT
    assert marla.objective_satisfied() and scaled.objective_satisfied()

    actions = [a for a in marla.legal_actions(s1) if not a.is_finish][:8]
    assert [a.action_id for a in actions] == [a.action_id for a in scaled.legal_actions(s2) if not a.is_finish][:8]
    for action in actions:
        r1, r2 = marla.step(action), scaled.step(action)
        _obs, raw_reward, _done, _info = raw.step(resolve_action_target(raw, action.action_id))
        assert r1.nasimemu_reward == r2.nasimemu_reward == pytest.approx(float(raw_reward))
        assert r1.terminated is False and r1.nasimemu_reward < 1.0


def test_zero_target_realization_is_regenerated_deterministically():
    random.seed(ZERO_TARGET_SEED)
    np.random.seed(ZERO_TARGET_SEED)
    raw = NASimEmuEnv(scenario_name=UNI_SCENARIO, emulate=False, step_limit=None, random_init=False,
                      observation_format="matrix", fully_obs=False, augment_with_action=False)
    random.seed(ZERO_TARGET_SEED)
    np.random.seed(ZERO_TARGET_SEED)
    raw.reset()
    assert raw.env.network.sensitive_addresses == []  # the raw instance would make FINISH vacuously "successful"

    first, second = adapter(), adapter()
    s1, s2 = first.reset(seed=ZERO_TARGET_SEED), second.reset(seed=ZERO_TARGET_SEED)
    total, remaining = first.sensitive_target_status()
    assert total >= 1 and remaining == total and not first.objective_satisfied()
    assert second.sensitive_target_status() == (total, remaining)
    assert s1.host_addresses == s2.host_addresses
    assert finish(first, s1).nasimemu_reward == -float(total)


def test_vacuous_scenario_is_rejected():
    with pytest.raises(ValueError, match="no sensitive target"):
        NasimEmuAdapter(scenario=VACUOUS_SCENARIO, max_episode_steps=10).reset(seed=0)


# --- reward flow through rollout collection (baseline and assisted) --------------------------------------


def _collector(config_name, adapter, consult_calls, base_seed, **extra):
    config = load_config(REPO_ROOT / "examples" / config_name)
    assisted = config.consultation.mode == "learned"
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy, consultation_enabled=assisted)

    async def consult_fn(actions, episode_id, step, oid, observation, subnet, n_global):
        consult_calls.append([a.action_id for a in actions])
        return ConsultationResult(status="accepted", scores={a.action_id: 0.5 for a in actions}, request_id="r")

    kw = dict(consultation_enabled=True, consultation_cost=0.1, consult_fn=consult_fn,
              overrides=EvaluationOverrides(query_mode="always")) if assisted else {}
    return config, RolloutCollector(adapter, policy, run_id="finish-reward", base_seed=base_seed, **kw, **extra)


def _force_stop(collector, decide):
    """Replace the stop draw by ``decide(episode_id) -> bool`` (the stop log-probability is still the
    policy's own for that decision)."""
    original = collector._sample_stop

    def forced(step_out):
        _stop, probability = original(step_out)
        return decide(collector._episode_id), probability

    collector._sample_stop = forced


@pytest.mark.asyncio
@pytest.mark.parametrize("config_name", ["baseline.yaml", "assisted.yaml"])
@pytest.mark.parametrize("root_all_first,expected", [(False, -3.0), (True, 3.0)])
async def test_finish_reward_flows_into_records_and_skips_consultation(config_name, root_all_first, expected):
    a = adapter()
    calls: list = []
    _config, collector = _collector(config_name, a, calls, SEED_FOR_TARGETS[3])
    _force_stop(collector, lambda episode: True)
    if root_all_first:
        collector._start_new_episode()
        set_access(a, a._env.env.network.sensitive_addresses, AccessLevel.ROOT)
    records, summaries = await collector.collect(1)
    rec, summ = records[0], summaries[0]
    assert rec.stop_selected and rec.selected_action_type == "finish"
    assert calls == [] and rec.sampled_query is False and rec.consultation_cost == 0.0  # FINISH never consults
    assert rec.nasimemu_reward == expected and rec.training_reward == expected
    assert summ.finish_reward == expected and summ.successful_finish is root_all_first


async def _three_outcome_episodes(config_name):
    """Episode 1: root every target, then FINISH; episode 2: FINISH at once; episode 3: CONTINUE until
    the 3-step time limit."""
    a = NasimEmuAdapter(scenario=UNI_SCENARIO, max_episode_steps=3)
    calls: list = []
    config, collector = _collector(config_name, a, calls, SEED_FOR_TARGETS[3])
    _force_stop(collector, lambda episode: episode in (1, 2))
    collector._start_new_episode()
    set_access(a, a._env.env.network.sensitive_addresses, AccessLevel.ROOT)
    _records, summaries = await collector.collect(5)
    return config, summaries, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("config_name", ["baseline.yaml", "assisted.yaml"])
async def test_metrics_distinguish_successful_premature_and_timeout(tmp_path, config_name):
    from marla.metrics.accumulators import ConsultationStats
    from marla.metrics.writer import build_episode_rows, write_summary_json

    config, summaries, calls = await _three_outcome_episodes(config_name)
    assert [s.finish_reason for s in summaries] == ["finish", "finish", "truncated"]
    if config.consultation.mode == "learned":
        assert len(calls) == 3  # consulted only on the timeout episode's three CONTINUE decisions
    rows = build_episode_rows(config, summaries)
    n1, n2 = summaries[0].sensitive_targets_total, summaries[1].sensitive_targets_total
    assert [r["finish_reward"] for r in rows] == [float(n1), -float(n2), 0.0]
    for r, s in zip(rows, summaries):
        assert r["native_return"] == pytest.approx(r["nasimemu_return"] - r["finish_reward"])
        assert r["training_return"] == pytest.approx(r["nasimemu_return"] - s.consultation_cost)
    assert [r["successful_finish"] for r in rows] == [True, False, False]

    write_summary_json(tmp_path, summaries, ConsultationStats(), Counter(), 5, 1.0, {})
    summary = json.loads((tmp_path / "summary.json").read_text())
    for key in ("successful_finish_rate", "premature_finish_rate", "timeout_rate"):
        assert summary[key] == pytest.approx(1 / 3), key
    assert summary["mean_finish_reward"] == pytest.approx((n1 - n2) / 3)
    assert summary["mean_target_completion_fraction"] == pytest.approx(
        sum(r["target_completion_fraction"] for r in rows) / 3
    )
