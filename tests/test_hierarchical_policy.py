"""Completion predictor + PPO-trained stop head (hierarchical FINISH/CONTINUE policy).

pi(FINISH | h) = p_stop, pi(a | h) = (1 - p_stop) * pi_action(a) for ordinary a, with
pi_action a softmax over the ordinary actions only. The completion head is supervised
by decision-time simulator labels (training only) and feeds the stop head through a
stop-gradient.
"""

from __future__ import annotations

import asyncio
import copy
import math
from pathlib import Path

import pytest
import torch
import yaml

from marla.config.loader import parse_config
from marla.environment.action_compatibility import compute_compatibility_matrix
from marla.environment.actions import count_ordinary_actions, finish_descriptor
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.environment.visible_facts import (
    assemble_visible_progress,
    extract_visible_host_facts,
    extract_visible_network_exploration,
)
from marla.evaluation.overrides import EvaluationOverrides
from marla.evaluation.sampling import EvaluationSampler
from marla.learning.checkpoint import PolicyRepresentationMismatchError, load_checkpoint, save_checkpoint
from marla.learning.decision import (
    compute_final_decision,
    compute_hierarchical_entropy,
    compute_joint_log_probability,
    hierarchical_action_probabilities,
)
from marla.learning.gae import compute_gae
from marla.learning.ppo import (
    SequenceChunk,
    _replay_chunk,
    completion_positive_class_weight,
    completion_prediction_loss,
    ppo_update,
)
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import ConsultationResult, RolloutCollector, run_evaluation_episodes
from marla.learning.trainer import run_baseline_training

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "tests/data/scenarios/sm_entry_dmz_one_target.v2.yaml").resolve())
UNI_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/uni.v2.yaml").resolve())
# One host, one target; exploit e_win_root grants ROOT deterministically.
MICRO_A_SCENARIO = str((REPO_ROOT / "src/marla/scenarios/diagnostic/micro_a_direct_root.v2.yaml").resolve())


def small_config(name="baseline.yaml", max_episode_steps=12, **policy_overrides):
    data = yaml.safe_load((REPO_ROOT / "examples" / name).read_text())
    data["environment"].update(scenario=SMALL_SCENARIO, max_episode_steps=max_episode_steps)
    data["policy"]["graph_encoder"]["hidden_size"] = 16
    data["policy"]["action_encoder"].update(hidden_size=16, action_type_embedding_size=8)
    data["policy"]["recurrent"].update(hidden_size=16, sequence_length=8)
    data["policy"]["ppo"].update(total_environment_steps=64, steps_per_env=32, epochs=1, minibatch_sequences=2)
    for key, value in policy_overrides.items():
        data["policy"][key] = value
    return parse_config(data)


def make_adapter(config, scenario=SMALL_SCENARIO):
    return NasimEmuAdapter(scenario=scenario, max_episode_steps=config.environment.max_episode_steps)


def policy_inputs(policy, adapter, state, legal=None):
    legal = legal if legal is not None else adapter.legal_actions(state)
    facts = extract_visible_host_facts(state)
    progress = assemble_visible_progress(
        facts, extract_visible_network_exploration(state),
        policy.visible_target_progress_enabled, policy.visible_subnet_exploration_enabled,
    )
    graph = adapter.to_pyg_data(state, include_subnet_scan_feature=policy.visible_subnet_exploration_enabled)
    return graph, legal, compute_compatibility_matrix(legal, facts), progress


def step_on(policy, adapter, state, rstate):
    graph, legal, compat, progress = policy_inputs(policy, adapter, state)
    return policy.step(graph, legal, rstate, compat, progress)


def set_stop_probability(policy, p):
    """Make p_stop == p for every state (final stop-head layer: zero weights, logit(p) bias)."""
    with torch.no_grad():
        policy.stop_head[-1].weight.zero_()
        policy.stop_head[-1].bias.fill_(math.log(p / (1 - p)))


def scores_for(actions):
    return {a.action_id: 0.1 + 0.8 * ((sum(map(ord, a.action_id)) % 7) / 6) for a in actions}


def build_collector(config, policy, adapter, seed=0, consult_calls=None, **kw):
    if not policy.consultation_enabled:
        return RolloutCollector(adapter, policy, run_id="hier", base_seed=seed, **kw)

    async def consult_fn(actions, episode_id, step, oid, observation, subnet, n_global):
        if consult_calls is not None:
            consult_calls.append([a.action_id for a in actions])
        return ConsultationResult(status="accepted", scores=scores_for(actions), request_id=f"r{episode_id}-{step}")

    return RolloutCollector(adapter, policy, run_id="hier", base_seed=seed, consultation_enabled=True,
                            consultation_cost=0.1, consult_fn=consult_fn, **kw)


def collect(config, steps, assisted=False, seed=0, stop_probability=None, consult_calls=None, **kw):
    torch.manual_seed(seed)
    policy = RecurrentPolicy(config.policy, consultation_enabled=assisted)
    if stop_probability is not None:
        set_stop_probability(policy, stop_probability)
    adapter = make_adapter(config)
    collector = build_collector(config, policy, adapter, seed=seed, consult_calls=consult_calls, **kw)
    records, summaries = asyncio.run(collector.collect(steps))
    return policy, records, summaries


def chunk_of(records, advantages=None):
    advantages = advantages if advantages is not None else [0.0] * len(records)
    return SequenceChunk(records=records, advantages=list(advantages), returns=[0.0] * len(records))


# --- completion predictor ---------------------------------------------------------------------------------


def test_heads_output_scalar_logits_and_probabilities():
    config = small_config()
    policy = RecurrentPolicy(config.policy)
    adapter = make_adapter(config)
    state = adapter.reset(seed=0)
    out = step_on(policy, adapter, state, policy.initial_recurrent_state())
    assert out.completion_logit.shape == () and out.stop_logit.shape == ()
    assert 0.0 < float(torch.sigmoid(out.completion_logit)) < 1.0
    assert out.base_logits.shape == (count_ordinary_actions(adapter.legal_actions(state)),)
    assert out.action_embeddings.shape[0] == len(adapter.legal_actions(state))  # FINISH's row kept
    assert float(torch.sigmoid(out.stop_logit)) == pytest.approx(0.05, abs=0.05)  # initial_stop_probability


def test_completion_labels_are_the_pre_action_state():
    """Rooting the only target with step t's action leaves step t's label 0; step t+1's label is 1."""
    config = small_config()
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy)
    set_stop_probability(policy, 1e-6)
    adapter = make_adapter(config, scenario=MICRO_A_SCENARIO)
    collector = RolloutCollector(adapter, policy, run_id="labels", base_seed=0)
    plan = iter(["service-scan:host-1-0", "exploit:host-1-0:e_win_root", "os-scan:host-1-0", "os-scan:host-1-0"])

    def scripted(probs):
        wanted = next(plan)
        return next(i for i, a in enumerate(adapter.legal_actions(collector._state)) if a.action_id == wanted)

    collector._sample_ordinary_action = scripted
    records, _ = asyncio.run(collector.collect(4))
    assert [r.objective_satisfied_before_action for r in records] == [False, False, True, True]
    assert records[1].objective_satisfied is True  # post-action state of step t is already complete
    assert all(r.completion_probability is not None for r in records)


def test_completion_loss_gradients_reach_completion_head_and_shared_layers_only():
    config = small_config()
    policy = RecurrentPolicy(config.policy)
    adapter = make_adapter(config)
    out = step_on(policy, adapter, adapter.reset(seed=0), policy.initial_recurrent_state())
    torch.nn.functional.binary_cross_entropy_with_logits(out.completion_logit, torch.tensor(1.0)).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.completion_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.recurrent_core.parameters())
    assert all(p.grad is None for p in policy.stop_head.parameters())
    assert all(p.grad is None for p in policy.base_scorer.parameters())


def test_stop_gradients_reach_shared_layers_but_not_the_completion_head():
    config = small_config()
    policy = RecurrentPolicy(config.policy)
    adapter = make_adapter(config)
    out = step_on(policy, adapter, adapter.reset(seed=0), policy.initial_recurrent_state())
    out.stop_logit.backward()
    assert all(p.grad is None for p in policy.completion_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.stop_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.recurrent_core.parameters())


def test_policy_outputs_do_not_depend_on_simulator_truth():
    """Flipping the hidden simulator state to "complete" (the completion label) without a new
    observation changes nothing the policy computes: labels never enter its inputs."""
    config = small_config()
    policy = RecurrentPolicy(config.policy)
    adapter = make_adapter(config)
    state = adapter.reset(seed=0)
    rstate = policy.initial_recurrent_state()
    with torch.no_grad():
        before = step_on(policy, adapter, state, rstate)
        assert not adapter.objective_satisfied()
        for addr in adapter._env.env.network.sensitive_addresses:
            adapter._env.env.current_state.get_host(addr).access = 3
        assert adapter.objective_satisfied()
        after = step_on(policy, adapter, state, rstate)
    for name in ("z", "base_logits", "stop_logit", "completion_logit", "value"):
        assert torch.equal(getattr(before, name), getattr(after, name)), name


def test_completion_loss_ignores_padded_entries():
    logits = torch.tensor([2.0, -1.0, 0.5, 50.0, -50.0])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    mask = torch.tensor([True, True, True, False, False])
    masked = completion_prediction_loss(logits, labels, mask, positive_class_weight=3.0)
    valid_only = completion_prediction_loss(logits[:3], labels[:3], mask[:3], positive_class_weight=3.0)
    assert masked == pytest.approx(float(valid_only))
    # weighted mean: the loss of a perfectly balanced, unweighted batch is the plain mean BCE
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logits[:3], labels[:3])
    assert completion_prediction_loss(logits[:3], labels[:3], mask[:3], 1.0) == pytest.approx(float(plain))


def test_positive_class_weighting():
    cfg = small_config().policy.completion_predictor
    mask = torch.ones(10, dtype=torch.bool)
    assert completion_positive_class_weight(torch.tensor([1.0] * 2 + [0.0] * 8), mask, cfg) == 4.0
    assert completion_positive_class_weight(torch.tensor([1.0] + [0.0] * 9), mask, cfg) == 9.0
    assert completion_positive_class_weight(torch.tensor([0.0] * 10), mask, cfg) == 1.0  # no positives
    assert completion_positive_class_weight(torch.tensor([1.0] * 8 + [0.0] * 2), mask, cfg) == 1.0  # never < 1
    capped = cfg.model_copy(update={"max_positive_class_weight": 3.0})
    assert completion_positive_class_weight(torch.tensor([1.0] + [0.0] * 9), mask, capped) == 3.0
    fixed = cfg.model_copy(update={"positive_class_weight": 2.5})
    assert completion_positive_class_weight(torch.tensor([0.0] * 10), mask, fixed) == 2.5


def test_synthetic_supervised_dataset_teaches_the_completion_head():
    """A strongly supervised, linearly separable completion dataset in z-space is learned by the
    completion head alone (labels: 'complete' iff z0 + z1 > 0, ~20% positives)."""
    torch.manual_seed(0)
    config = small_config()
    policy = RecurrentPolicy(config.policy)
    z = torch.randn(512, 16)
    labels = (z[:, 0] + z[:, 1] > 1.2).float()
    assert 0.05 < labels.mean() < 0.4
    mask = torch.ones_like(labels, dtype=torch.bool)
    optimizer = torch.optim.Adam(policy.completion_head.parameters(), lr=1e-2)
    weight = completion_positive_class_weight(labels, mask, config.policy.completion_predictor)
    for _ in range(300):
        logits = policy.completion_head(z).squeeze(-1)
        loss = completion_prediction_loss(logits, labels, mask, weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probabilities = torch.stack([torch.sigmoid(policy.stop_logits(zi)[0]) for zi in z])
    predicted = probabilities >= 0.5
    accuracy = (predicted == labels.bool()).float().mean()
    recall = (predicted & labels.bool()).sum() / labels.sum()
    assert accuracy > 0.95 and recall > 0.9


# --- stop head / hierarchical distribution ----------------------------------------------------------------


def test_finish_probability_is_independent_of_the_ordinary_candidate_count():
    config = small_config()
    policy = RecurrentPolicy(config.policy)
    adapter = make_adapter(config, scenario=UNI_SCENARIO)
    state = adapter.reset(seed=2)
    graph, legal, compat, progress = policy_inputs(policy, adapter, state)
    rstate = policy.initial_recurrent_state()
    assert len(legal) > 10
    stop_probs, sizes = [], []
    with torch.no_grad():
        for keep in (1, 4, len(legal) - 1):
            subset = legal[:keep] + [legal[-1]]
            sub_compat = torch.cat([compat[:keep], compat[-1:]])
            out = policy.step(graph, subset, rstate, sub_compat, progress)
            stop_probs.append(float(torch.sigmoid(out.stop_logit)))
            sizes.append(out.num_ordinary_actions)
    assert sizes == [1, 4, len(legal) - 1]
    assert stop_probs[0] == stop_probs[1] == stop_probs[2]


@pytest.mark.parametrize("stop_logit", [-60.0, -3.0, 0.0, 2.0, 60.0])
def test_complete_distribution_sums_to_one_and_entropy_matches_definition(stop_logit):
    torch.manual_seed(0)
    logits = torch.randn(7)
    s = torch.tensor(stop_logit)
    probs = hierarchical_action_probabilities(s, logits)
    assert probs.shape == (8,) and torch.isfinite(probs).all()
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(probs[-1]) == pytest.approx(float(torch.sigmoid(s)))
    entropy = compute_hierarchical_entropy(s, logits)
    brute = -(probs * torch.log(probs.clamp_min(1e-30))).sum()
    assert float(entropy.action_entropy) == pytest.approx(float(brute), abs=1e-5)
    assert torch.isfinite(entropy.stop_entropy) and torch.isfinite(entropy.conditional_action_entropy)


@pytest.mark.parametrize("stop_logit", [-80.0, -1.5, 0.7, 80.0])
def test_joint_log_probabilities_match_the_hierarchical_policy(stop_logit):
    torch.manual_seed(1)
    logits = torch.randn(5)
    s = torch.tensor(stop_logit)
    probs = hierarchical_action_probabilities(s, logits).double()
    finish = compute_joint_log_probability(s, 5, stop_selected=True)
    assert torch.isfinite(finish.joint_log_prob)
    assert float(finish.joint_log_prob) == pytest.approx(float(torch.nn.functional.logsigmoid(s)), abs=1e-5)
    for a in range(5):
        cont = compute_joint_log_probability(s, 5, stop_selected=False, final_logits=logits, selected_action_index=a)
        assert torch.isfinite(cont.joint_log_prob)
        expected = float(torch.nn.functional.logsigmoid(-s) + torch.log_softmax(logits, -1)[a])
        assert float(cont.joint_log_prob) == pytest.approx(expected, abs=1e-5)
        if abs(stop_logit) < 50:
            assert math.exp(float(cont.joint_log_prob)) == pytest.approx(float(probs[a]), rel=1e-4)
    q = torch.tensor(0.3)
    queried = compute_joint_log_probability(s, 5, False, logits, 2, query_probability=q, sampled_query=True)
    unqueried = compute_joint_log_probability(s, 5, False, logits, 2, query_probability=q, sampled_query=False)
    base = compute_joint_log_probability(s, 5, False, logits, 2)
    assert float(queried.joint_log_prob) == pytest.approx(float(base.joint_log_prob) + math.log(0.3), abs=1e-5)
    assert float(unqueried.joint_log_prob) == pytest.approx(float(base.joint_log_prob) + math.log(0.7), abs=1e-5)


def test_invalid_continue_decisions_are_rejected():
    s = torch.tensor(0.0)
    with pytest.raises(ValueError):
        compute_joint_log_probability(s, 0, stop_selected=False)
    with pytest.raises(ValueError):
        compute_joint_log_probability(s, 3, stop_selected=False, final_logits=torch.zeros(3), selected_action_index=3)


def test_no_ordinary_action_forces_finish_without_invalid_probabilities():
    config = small_config()
    policy = RecurrentPolicy(config.policy, consultation_enabled=True)
    adapter = make_adapter(config)
    state = adapter.reset(seed=0)
    graph, _legal, compat, progress = policy_inputs(policy, adapter, state)
    only_finish = [finish_descriptor()]
    out = policy.step(graph, only_finish, policy.initial_recurrent_state(), compat[-1:], progress)
    assert out.num_ordinary_actions == 0 and out.base_logits.numel() == 0
    joint = compute_joint_log_probability(out.stop_logit, 0, stop_selected=True)
    assert float(joint.joint_log_prob) == 0.0
    assert float(compute_hierarchical_entropy(out.stop_logit, out.base_logits).action_entropy) == 0.0
    assert hierarchical_action_probabilities(out.stop_logit, out.base_logits).tolist() == [1.0]
    collector = build_collector(config, policy, adapter)
    assert collector._sample_stop(out) == (True, 1.0)
    decision = asyncio.run(collector._decide(only_finish, out, state))
    assert decision["stop_selected"] and decision["joint_log_prob"] == 0.0 and decision["sampled_query"] is False


def test_legal_actions_must_end_with_a_single_finish():
    config = small_config()
    policy = RecurrentPolicy(config.policy)
    adapter = make_adapter(config)
    graph, legal, compat, progress = policy_inputs(policy, adapter, adapter.reset(seed=0))
    with pytest.raises(ValueError):
        policy.step(graph, [legal[-1], *legal[:-1]], policy.initial_recurrent_state(), compat, progress)
    with pytest.raises(ValueError):
        policy.step(graph, legal[:-1], policy.initial_recurrent_state(), compat[:-1], progress)


def test_stop_sampling_follows_the_bernoulli_and_deterministic_mode_uses_its_mode():
    config = small_config()
    for p in (0.2, 0.7):
        policy, records, _ = collect(config, 400, seed=3, stop_probability=p)
        # every decision with ordinary actions shares p_stop = p; FINISH frequency ~= p
        assert all(r.old_stop_probability == pytest.approx(p) for r in records)
        frequency = sum(r.stop_selected for r in records) / len(records)
        assert frequency == pytest.approx(p, abs=0.06)
        for r in records:
            expected = math.log(p) if r.stop_selected else math.log(1 - p)
            assert r.old_stop_log_probability == pytest.approx(expected, abs=1e-5)
    for p, expect_finish in ((0.49, False), (0.5, True)):
        _policy, records, _ = collect(config, 3, seed=0, stop_probability=p, deterministic=True)
        assert records[0].stop_selected is expect_finish


def test_stochastic_evaluation_stop_draws_are_reproducible():
    config = small_config()

    def run():
        sampler = EvaluationSampler(agent_seed=7)
        _policy, records, _ = collect(config, 40, seed=1, stop_probability=0.3, evaluation_sampler=sampler)
        return [(r.stop_selected, r.selected_action_index) for r in records], dict(sampler.draws)

    first, draws = run()
    second, _ = run()
    assert first == second
    assert draws["stop"] == 40 and draws["action"] == sum(1 for s, _ in first if not s)


# --- PPO --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("assisted", [False, True])
def test_replay_at_theta_old_reproduces_collected_log_probabilities(assisted):
    config = small_config("assisted.yaml" if assisted else "baseline.yaml")
    policy, records, _ = collect(config, 48, assisted=assisted, seed=4, stop_probability=0.15)
    assert any(r.stop_selected for r in records) and any(not r.stop_selected for r in records)
    if assisted:
        assert any(r.sampled_query for r in records) and any(not r.sampled_query for r in records if not r.stop_selected)
    episodes: dict[int, list] = {}
    for r in records:
        episodes.setdefault(r.episode_id, []).append(r)
    with torch.no_grad():
        for episode_records in episodes.values():
            replay = _replay_chunk(policy, chunk_of(episode_records), torch.device("cpu"), 0.1)
            old = torch.tensor([r.old_joint_log_probability for r in episode_records])
            assert torch.allclose(replay.new_joint_log_probs, old, atol=1e-5)
            assert torch.allclose(
                replay.stop_probabilities, torch.tensor([r.old_stop_probability for r in episode_records]), atol=1e-6
            )


@pytest.mark.parametrize("assisted", [False, True])
def test_ppo_update_trains_stop_head_and_completion_head(assisted):
    config = small_config("assisted.yaml" if assisted else "baseline.yaml")
    policy, records, _ = collect(config, 48, assisted=assisted, seed=5, stop_probability=0.2)
    rewards = [r.training_reward for r in records]
    advantages, returns = compute_gae(rewards, [r.critic_value for r in records], [r.terminated for r in records],
                                      [r.truncated for r in records], [r.bootstrap_value for r in records], 0.99, 0.95)
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    chunks = [SequenceChunk(records=records, advantages=advantages, returns=returns)]
    metrics = ppo_update(policy, optimizer, chunks, config.policy.ppo, torch.device("cpu"), consultation_cost=0.1)
    changed = {n for n, p in policy.named_parameters() if not torch.equal(p.detach(), before[n])}
    assert any(n.startswith("stop_head.") for n in changed)
    assert any(n.startswith("completion_head.") for n in changed)
    for key in ("completion_prediction_loss", "stop_probability", "stop_entropy", "conditional_action_entropy",
                "completion_prediction_accuracy", "completion_label_positive_count", "completion_positive_class_weight"):
        assert metrics[key] is not None, key
    assert math.isfinite(metrics["completion_prediction_loss"])
    assert metrics["action_entropy"] >= metrics["stop_entropy"] > 0


def test_completion_predictor_can_be_disabled():
    config = small_config(completion_predictor={"enabled": False})
    policy, records, _ = collect(config, 24, seed=6, stop_probability=0.2)
    assert not hasattr(policy, "completion_head")
    assert all(r.completion_probability is None for r in records)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    metrics = ppo_update(policy, optimizer, [chunk_of(records, [1.0] * len(records))], config.policy.ppo,
                         torch.device("cpu"))
    assert metrics["completion_prediction_loss"] is None and metrics["stop_probability"] is not None


# --- assisted mode ----------------------------------------------------------------------------------------


def test_continue_consults_without_finish_and_charges_cost_once():
    config = small_config("assisted.yaml")
    calls: list = []
    _policy, records, summaries = collect(config, 30, assisted=True, seed=7, stop_probability=0.2,
                                          consult_calls=calls, overrides=EvaluationOverrides(query_mode="always"))
    continue_records = [r for r in records if not r.stop_selected]
    finish_records = [r for r in records if r.stop_selected]
    assert finish_records and continue_records
    assert len(calls) == len(continue_records)  # one consultation per CONTINUE, none for FINISH
    assert all("finish" not in ids for ids in calls)
    for r in continue_records:
        assert r.sampled_query and r.consultation_cost == pytest.approx(0.1)
        assert r.training_reward == pytest.approx(r.nasimemu_reward - 0.1)
        assert all(0 <= i < len(r.base_logits) for i in r.consulted_action_indices)
        assert r.global_candidate_action_count == len(r.legal_action_descriptors) - 1
    for r in finish_records:
        assert not r.sampled_query and r.consultation_cost == 0.0 and r.old_query_probability is None
        assert r.training_reward == r.nasimemu_reward
    for s in summaries:
        episode = [r for r in records if r.episode_id == s.episode_id]
        assert s.consultation_count == sum(r.sampled_query for r in episode)
        assert s.consultation_cost == pytest.approx(0.1 * s.consultation_count)


def test_advice_changes_only_the_conditional_action_distribution():
    config = small_config("assisted.yaml")
    torch.manual_seed(8)
    policy = RecurrentPolicy(config.policy, consultation_enabled=True)
    with torch.no_grad():
        policy.advice_scale.alpha_bar.fill_(2.0)
    adapter = make_adapter(config, scenario=UNI_SCENARIO)
    state = adapter.reset(seed=2)
    graph, legal, compat, progress = policy_inputs(policy, adapter, state)
    out = policy.step(graph, legal, policy.initial_recurrent_state(), compat, progress)
    indices = torch.arange(0, min(6, out.num_ordinary_actions))
    confidence = torch.linspace(0.05, 0.95, len(indices))
    advised = compute_final_decision(policy, out, True, confidence, indices)
    assert not torch.allclose(advised.final_logits, out.base_logits)
    base_probs = hierarchical_action_probabilities(out.stop_logit, out.base_logits)
    final_probs = hierarchical_action_probabilities(out.stop_logit, advised.final_logits)
    assert float(base_probs[-1]) == float(final_probs[-1])  # p_stop untouched
    assert float(final_probs[:-1].sum()) == pytest.approx(float(1 - torch.sigmoid(out.stop_logit)), abs=1e-6)
    untouched = torch.arange(len(indices), out.num_ordinary_actions)
    assert torch.equal(advised.final_logits[untouched], out.base_logits[untouched])  # sparse residual


# --- integration ------------------------------------------------------------------------------------------


def test_end_to_end_multi_env_training_checkpoint_and_evaluation(tmp_path):
    config = small_config()
    data = config.model_dump(mode="json")
    data["policy"]["ppo"].update(num_envs=2, steps_per_env=24, total_environment_steps=96, epochs=2)
    config = parse_config(data)
    result = asyncio.run(run_baseline_training(config, SMALL_SCENARIO, num_rollouts=2))
    assert len(result.update_metrics) >= 2
    for metric in result.update_metrics:
        assert math.isfinite(metric["policy_loss"]) and math.isfinite(metric["completion_prediction_loss"])
    assert result.update_metrics[0]["approximate_kl"] == pytest.approx(0.0, abs=1e-6)
    row = result.rollout_rows[-1]
    for key in ("stop_probability", "completion_probability", "completion_prediction_brier",
                "completion_label_negative_count"):
        assert row[key] is not None, key
    assert {r["env_index"] for r in result.decision_rows} == {0, 1}
    for d in result.decision_rows:
        assert d["base_finish_probability"] == pytest.approx(d["stop_probability"])
        assert d["final_finish_probability"] == pytest.approx(d["stop_probability"])
        assert d["finish_selected"] == d["stop_selected"]

    path = tmp_path / "checkpoint.pt"
    optimizer = torch.optim.Adam(result.policy.parameters(), lr=1e-3)
    save_checkpoint(path, result.policy, optimizer, update_count=2, environment_steps=96, config_hash="h",
                    finish_reward_semantics=config.objective.effective_finish_reward())
    restored = RecurrentPolicy(config.policy)
    load_checkpoint(path, restored)
    for (name, p), (_n, q) in zip(result.policy.state_dict().items(), restored.state_dict().items()):
        assert torch.equal(p, q), name

    adapter = make_adapter(config)
    greedy_a = asyncio.run(run_evaluation_episodes(result.policy, adapter, "eval", 3, seed_start=100))
    greedy_b = asyncio.run(run_evaluation_episodes(restored, make_adapter(config), "eval", 3, seed_start=100))
    assert [(s.finish_reason, s.nasimemu_return) for s in greedy_a] == [(s.finish_reason, s.nasimemu_return) for s in greedy_b]


def test_incompatible_checkpoints_fail_clearly(tmp_path):
    config = small_config()
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, policy, optimizer, update_count=0, environment_steps=0, config_hash="h")

    old = torch.load(path, weights_only=False)
    old["policy_representation_version"] = 4  # flat-softmax FINISH architecture
    torch.save(old, tmp_path / "old.pt")
    with pytest.raises(PolicyRepresentationMismatchError, match="policy_representation_version"):
        load_checkpoint(tmp_path / "old.pt", RecurrentPolicy(config.policy))

    disabled = small_config(completion_predictor={"enabled": False})
    with pytest.raises(PolicyRepresentationMismatchError, match="completion_predictor_enabled"):
        load_checkpoint(path, RecurrentPolicy(disabled.policy))

    legacy_weights = copy.deepcopy(old)
    legacy_weights["policy_representation_version"] = 5
    for key in [k for k in legacy_weights["policy_state_dict"] if k.startswith(("stop_head.", "completion_head."))]:
        del legacy_weights["policy_state_dict"][key]
    torch.save(legacy_weights, tmp_path / "missing_heads.pt")
    with pytest.raises(RuntimeError, match="stop_head"):
        load_checkpoint(tmp_path / "missing_heads.pt", RecurrentPolicy(config.policy))
