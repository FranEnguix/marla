"""Checkpoint-evaluation policy modes (v0.11.0): GREEDY (default, unchanged)
and STOCHASTIC_POLICY (sample the learned compound policy from explicit,
reproducible evaluation random streams -- see marla.evaluation.sampling).
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import marla.evaluation.checkpoint_eval as checkpoint_eval
import marla.learning.rollout as rollout_module
from marla.environment.consultation_scope import select_consulted_subnet
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.evaluation.overrides import ALWAYS_QUERY, BETA_ONE, BETA_ZERO, NO_QUERY, PLAN_MAKER_ONLY
from marla.evaluation.sampling import (
    ACTION_STREAM,
    EVALUATION_RNG_SCHEME,
    QUERY_STREAM,
    EvaluationPolicyMode,
    EvaluationSampler,
    evaluation_stream_seed,
)
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import RolloutCollector
from tests.support import eval_trajectory as et

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())
GOLDEN = REPO_ROOT / "tests" / "data" / "greedy_eval_golden_v0_10_1.json"
AGENT_SEED = 42  # experiment.seed of examples/baseline.yaml and examples/assisted.yaml


# --------------------------------------------------------------------------- RNG scheme


def _reference_seed(agent_seed, eval_seed, purpose):
    key = f"marla-stochastic-eval|{agent_seed}|{eval_seed}|{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def _reference_categorical(u, probs):
    p = probs.detach().double().cpu().numpy()
    return min(int(np.searchsorted(np.cumsum(p), u * p.sum(), side="right")), len(p) - 1)


def test_stream_seed_derivation_is_the_documented_sha256_scheme():
    assert evaluation_stream_seed(959, 7101, "action") == _reference_seed(959, 7101, "action")
    seeds = {evaluation_stream_seed(a, e, p) for a in (1, 2) for e in (10, 11) for p in (QUERY_STREAM, ACTION_STREAM)}
    assert len(seeds) == 8, "agent seed, eval seed and purpose must each separate the streams"
    assert "sha256" in EVALUATION_RNG_SCHEME and "pcg64" in EVALUATION_RNG_SCHEME


def test_sampler_draws_follow_the_documented_streams_and_inverse_cdf():
    probs = torch.softmax(torch.tensor([0.3, -1.0, 2.0, 0.0, 0.5]), dim=0)
    sampler = EvaluationSampler(agent_seed=7)
    sampler.start_episode(123)
    actions = [sampler.sample_action(probs) for _ in range(50)]
    queries = [sampler.sample_query(torch.tensor(0.4)) for _ in range(50)]
    ua = np.random.default_rng(_reference_seed(7, 123, "action")).random(50)
    uq = np.random.default_rng(_reference_seed(7, 123, "query")).random(50)
    assert actions == [_reference_categorical(u, probs) for u in ua]
    assert queries == [bool(u < 0.4) for u in uq]
    assert sampler.draws == {QUERY_STREAM: 50, ACTION_STREAM: 50}


def test_query_and_action_streams_are_independent():
    probs = torch.full((6,), 1 / 6)
    with_queries, without_queries = EvaluationSampler(3), EvaluationSampler(3)
    with_queries.start_episode(5)
    without_queries.start_episode(5)
    a, b = [], []
    for _ in range(40):
        with_queries.sample_query(torch.tensor(0.5))  # interleaved query draws
        a.append(with_queries.sample_action(probs))
        b.append(without_queries.sample_action(probs))
    assert a == b


def test_each_episode_restarts_its_streams_from_its_own_seed():
    probs = torch.full((10,), 0.1)
    s = EvaluationSampler(3)
    s.start_episode(1)
    first = [s.sample_action(probs) for _ in range(20)]
    s.start_episode(2)
    s.sample_action(probs)
    s.start_episode(1)
    assert [s.sample_action(probs) for _ in range(20)] == first
    with pytest.raises(RuntimeError):
        EvaluationSampler(3).sample_action(probs)


def test_collector_rejects_a_sampler_combined_with_greedy_selection():
    from marla.config.loader import load_config

    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    adapter = NasimEmuAdapter(scenario=SMALL_SCENARIO, max_episode_steps=5, completion_reward=1.0,
                              premature_finish_penalty=-1.0)
    with pytest.raises(ValueError):
        RolloutCollector(adapter, RecurrentPolicy(config.policy), "x", base_seed=1, deterministic=True,
                         evaluation_sampler=EvaluationSampler(1))


# --------------------------------------------------------------------------- GREEDY compatibility


@pytest.mark.asyncio
@pytest.mark.parametrize("mode_kwargs", [{}, {"policy_mode": EvaluationPolicyMode.GREEDY}, {"policy_mode": "GREEDY"}])
async def test_greedy_reproduces_v0_10_1_trajectories_for_every_condition(tmp_path, mode_kwargs):
    """Golden recorded by tests/support/eval_trajectory.py under MARLA v0.10.1
    (same code, same fixed-seed checkpoint): PPO_ONLY and MARLA_FULL under
    NORMAL/NO_QUERY/ALWAYS_QUERY/BETA_ZERO/BETA_ONE/PLAN_MAKER_ONLY."""
    golden = json.loads(GOLDEN.read_text())
    assert golden["recorded_with_marla_version"] == "0.10.1"
    got = await et.run_greedy_conditions(tmp_path, REPO_ROOT, SMALL_SCENARIO, **mode_kwargs)
    assert set(got) == set(golden["trajectories"])
    for condition, expected in golden["trajectories"].items():
        assert got[condition]["episodes"] == expected["episodes"], condition
        assert len(got[condition]["steps"]) == len(expected["steps"]), condition
        for g, e in zip(got[condition]["steps"], expected["steps"]):
            assert g[:-1] == e[:-1], (condition, g, e)
            assert g[-1] == pytest.approx(e[-1], abs=2e-3), (condition, g, e)
    # the fixture really exercises advice: greedy NORMAL consults, BETA_ONE != BETA_ZERO
    assert any(s[3] for s in golden["trajectories"]["MARLA_FULL/NORMAL"]["steps"])
    assert golden["trajectories"]["MARLA_FULL/BETA_ONE"] != golden["trajectories"]["MARLA_FULL/BETA_ZERO"]


@pytest.mark.asyncio
async def test_greedy_metadata_records_mode_and_no_rng(tmp_path):
    run_dir = et.make_run_dir(tmp_path, REPO_ROOT, SMALL_SCENARIO, consultation=False)
    result = await checkpoint_eval.evaluate_checkpoint(run_dir, seed_start=11, num_episodes=2, device="cpu")
    meta = result.evaluation_metadata()
    assert meta["evaluation_policy_mode"] == "GREEDY"
    assert meta["evaluation_seed_start"] == 11 and meta["evaluation_seeds"] == [11, 12]
    assert meta["rng_scheme"] is None and meta["rng_draws"] is None
    assert all(s.is_eval for s in result.summaries)


@pytest.mark.asyncio
async def test_invalid_policy_mode_is_rejected(tmp_path):
    run_dir = et.make_run_dir(tmp_path, REPO_ROOT, SMALL_SCENARIO, consultation=False)
    with pytest.raises(ValueError):
        await checkpoint_eval.evaluate_checkpoint(run_dir, seed_start=1, num_episodes=1, device="cpu",
                                                  policy_mode="SAMPLE")


# --------------------------------------------------------------------------- PPO_ONLY stochastic


async def _stochastic(run_dir, **kwargs):
    return await checkpoint_eval.evaluate_checkpoint(
        run_dir, seed_start=et.SEED_START, num_episodes=3, device="cpu",
        policy_mode=EvaluationPolicyMode.STOCHASTIC_POLICY, **kwargs)


def _key(result):
    return [(r.episode_id, r.environment_step, r.selected_action_index, r.sampled_query, r.nasimemu_reward,
             r.terminated, r.truncated, r.objective_satisfied_before_action) for r in result.records]


def _episodes(result):
    """{episode_id: [records in order]}"""
    out = {}
    for r in result.records:
        out.setdefault(r.episode_id, []).append(r)
    return out


def _uniforms(eval_seed, purpose, n):
    return list(np.random.default_rng(_reference_seed(AGENT_SEED, eval_seed, purpose)).random(n))


@pytest.mark.asyncio
async def test_ppo_only_stochastic_samples_pi_from_the_action_stream_reproducibly(tmp_path):
    run_dir = et.make_run_dir(tmp_path, REPO_ROOT, SMALL_SCENARIO, consultation=False)
    torch_state = torch.get_rng_state()
    a = await _stochastic(run_dir)
    assert torch.equal(torch_state, torch.get_rng_state()), "evaluation must not consume torch's global RNG"
    b = await _stochastic(run_dir)
    assert _key(a) == _key(b), "same checkpoint + eval seeds + mode must give the same trajectory"

    meta = a.evaluation_metadata()
    assert meta["evaluation_policy_mode"] == "STOCHASTIC_POLICY"
    assert meta["rng_scheme"] == EVALUATION_RNG_SCHEME and meta["rng_agent_seed"] == AGENT_SEED
    assert meta["evaluation_seeds"] == [et.SEED_START, et.SEED_START + 2]
    assert meta["rng_draws"][QUERY_STREAM] == 0, "PPO_ONLY must not consume the query stream"
    assert meta["rng_draws"][ACTION_STREAM] >= len(a.records)
    assert all(s.is_eval for s in a.summaries)

    by_seed = {s.episode_id: s.seed for s in a.summaries}
    for episode_id, records in _episodes(a).items():
        us = _uniforms(by_seed[episode_id], "action", len(records))
        for u, r in zip(us, records):
            probs = torch.distributions.Categorical(probs=torch.softmax(r.base_logits, dim=-1)).probs
            assert r.selected_action_index == _reference_categorical(u, probs)
            assert torch.equal(r.final_logits, r.base_logits)
            assert not r.sampled_query
    distinct = {r.selected_action_index for r in a.records}
    assert len(distinct) > 1, "an untrained policy sampled over 90 decisions should not be constant"


@pytest.mark.asyncio
async def test_logging_and_global_rng_use_cannot_change_a_stochastic_trajectory(tmp_path, monkeypatch):
    run_dir = et.make_run_dir(tmp_path, REPO_ROOT, SMALL_SCENARIO, consultation=False)
    reference = _key(await _stochastic(run_dir))

    original_info = rollout_module.logger.info

    def noisy_info(*args, **kwargs):  # a logging call that also consumes every global RNG
        torch.rand(3)
        random.random()
        np.random.random()
        return original_info(*args, **kwargs)

    monkeypatch.setattr(rollout_module.logger, "info", noisy_info)
    torch.manual_seed(12345)
    assert _key(await _stochastic(run_dir)) == reference


# --------------------------------------------------------------------------- MARLA_FULL stochastic


@pytest.fixture
def scripted_consultant(monkeypatch):
    consultant = et.ScriptedConsultant()
    monkeypatch.setattr(checkpoint_eval, "_build_consult_fn", lambda *a, **k: consultant)
    return consultant


def _assert_final_action_sampled(records, eval_seed):
    action_records = list(records)
    us = _uniforms(eval_seed, "action", len(action_records))
    for u, r in zip(us, action_records):
        probs = torch.distributions.Categorical(probs=torch.softmax(r.final_logits, dim=-1)).probs
        assert r.selected_action_index == _reference_categorical(u, probs)


@pytest.mark.asyncio
async def test_marla_full_stochastic_structural_path(tmp_path, scripted_consultant):
    torch.manual_seed(0)
    policy = RecurrentPolicy(
        checkpoint_eval.load_run_config(et.make_run_dir(tmp_path, REPO_ROOT, SMALL_SCENARIO, consultation=True)).policy,
        consultation_enabled=True,
    ).eval()  # plain untrained gate: p_q ~= 0.45 -> a genuine mix of sampled queries
    run_dir = tmp_path / "run_full"
    result = await _stochastic(run_dir, policy_override=policy)
    by_seed = {s.episode_id: s.seed for s in result.summaries}

    queried = [r for r in result.records if r.sampled_query]
    assert 0 < len(queried) < len(result.records), "expected a mix of sampled queries"
    # consults only when q_t is true -- one call per queried decision, nothing else
    kept_calls = [c for c in scripted_consultant.calls if c["episode_id"] in by_seed]
    assert [(c["episode_id"], c["step"]) for c in kept_calls] == [(r.episode_id, r.environment_step) for r in queried]

    for episode_id, records in _episodes(result).items():
        uq = _uniforms(by_seed[episode_id], "query", len(records))
        for u, r in zip(uq, records):  # q_t ~ Bernoulli(p_t^q), one query draw per decision
            assert r.sampled_query == (u < r.old_query_probability)
        _assert_final_action_sampled(records, by_seed[episode_id])  # a_t ~ softmax(final logits)
        assert records[0].previous_query is False
        for prev, cur in zip(records, records[1:]):  # recurrent previous_query = the SAMPLED q_t
            assert cur.previous_query == prev.sampled_query

    for call, r in zip(kept_calls, queried):  # normal subnet-scoped advice path
        legal = r.legal_action_descriptors
        assert r.consulted_subnet == select_consulted_subnet(legal, r.base_logits)
        assert call["consulted_subnet"] == r.consulted_subnet
        assert call["action_ids"] == [legal[i].action_id for i in r.consulted_action_indices]
        assert r.plan_maker_scores_in_action_order == pytest.approx([et.scripted_score(a) for a in call["action_ids"]])
    assert any(not torch.equal(r.final_logits, r.base_logits) for r in queried), "accepted advice must reach pi"
    assert all(torch.equal(r.final_logits, r.base_logits) for r in result.records if not r.sampled_query)

    again = await _stochastic(run_dir, policy_override=policy)
    assert _key(again) == _key(result)


@pytest.mark.asyncio
async def test_override_precedence_under_stochastic_policy(tmp_path, scripted_consultant):
    run_dir = et.make_run_dir(tmp_path, REPO_ROOT, SMALL_SCENARIO, consultation=True)

    no_query = await _stochastic(run_dir, overrides=NO_QUERY)
    assert not any(r.sampled_query for r in no_query.records) and not scripted_consultant.calls
    assert no_query.rng_draws[QUERY_STREAM] == 0 and no_query.rng_draws[ACTION_STREAM] > 0
    for e, records in _episodes(no_query).items():
        _assert_final_action_sampled(records, {s.episode_id: s.seed for s in no_query.summaries}[e])

    always = await _stochastic(run_dir, overrides=ALWAYS_QUERY)
    assert all(r.sampled_query for r in always.records)
    assert always.rng_draws[QUERY_STREAM] == 0, "a forced query consumes no query draw"
    for e, records in _episodes(always).items():
        _assert_final_action_sampled(records, {s.episode_id: s.seed for s in always.summaries}[e])

    for overrides, beta in ((BETA_ZERO, 0.0), (BETA_ONE, 1.0)):
        result = await _stochastic(run_dir, overrides=overrides)
        assert result.rng_draws[QUERY_STREAM] > 0, "BETA_* leaves the learned (sampled) query gate in place"
        advised = [r for r in result.records if r.normalized_advice is not None]
        for r in advised:
            assert r.beta == beta
            expected = r.base_logits + beta * r.alpha * torch.tensor(r.normalized_advice)
            assert torch.allclose(r.final_logits, expected, atol=1e-5)
        for e, records in _episodes(result).items():
            _assert_final_action_sampled(records, {s.episode_id: s.seed for s in result.summaries}[e])

    pm_only = await _stochastic(run_dir, overrides=PLAN_MAKER_ONLY)
    assert pm_only.rng_draws == {QUERY_STREAM: 0, ACTION_STREAM: 0}, "PLAN_MAKER_ONLY never samples the policy"
    for r in pm_only.records:
        assert r.sampled_query
        local = int(np.argmax(r.plan_maker_scores_in_action_order))
        assert r.selected_action_index == r.consulted_action_indices[local]
