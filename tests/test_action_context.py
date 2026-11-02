"""Evidence-aware action features and episode-local host-action memory
(marla.environment.action_context), and their use by the ordinary-action
scorer only (policy.action_evidence / policy.action_history).

Lettered tests follow the task's checklist: A matching service, B unknown
service, C cross-host independence, D repeated stochastic failure, E state
changes, F PPO replay, G no privileged information, H unchanged stopping
architecture, I assisted mode.
"""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import pytest
import torch
from nasimemu.nasim.envs.utils import AccessLevel

from marla.config.loader import config_hash
from marla.environment.action_context import (
    EVIDENCE_FEATURE_DIM,
    HISTORY_FEATURE_DIM,
    EpisodeActionMemory,
    FailureCategory,
    ObservedActionOutcome,
    observed_action_outcome,
)
from marla.environment.actions import ActionDescriptor, finish_descriptor
from marla.environment.nasimemu_adapter import EnvironmentState, NasimEmuAdapter
from marla.environment.visible_facts import VisibleHostFacts, extract_visible_host_facts
from marla.learning.checkpoint import PolicyRepresentationMismatchError, load_checkpoint, save_checkpoint
from marla.learning.decision import hierarchical_action_probabilities
from marla.learning.gae import compute_gae
from marla.learning.ppo import _replay_chunk, ppo_update
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.metrics.writer import build_decision_rows, build_rollout_row
from tests.test_hierarchical_policy import (
    SMALL_SCENARIO,
    build_collector,
    chunk_of,
    collect,
    make_adapter,
    policy_inputs,
    small_config,
)

DRUPAL = "80_linux_drupal"
FTP = "21_linux_proftpd"
STATE = EnvironmentState(raw_observation=None, host_rows=None, host_addresses=[])

# Evidence row indices (see action_context's index table).
SERVICE_INFO, REQUIRED_SERVICE_NOT_FOUND, CONFIRMED_FRACTION, UNKNOWN_FRACTION, CONTRADICTED = 0, 3, 5, 6, 7
SCAN_INFO_AVAILABLE, SCAN_UNRESOLVED = 8, 9
# History row indices.
ATTEMPTED, ATTEMPTS, UNPRODUCTIVE, LAST_SUCCEEDED, LAST_PRODUCTIVE, RECENCY, CHANGED, NEW_INFO = range(8)
STOCHASTIC_FAILURE, UNSPECIFIED_FAILURE = 10, 11


def config_for(evidence: bool, history: bool, name="baseline.yaml"):
    return small_config(name, action_evidence={"enabled": evidence}, action_history={"enabled": history})


def facts(key, services=(), os=(), access=AccessLevel.NONE, compromised=False):
    return VisibleHostFacts(
        target_key=key, reachable=True, compromised=compromised, access=access,
        known_os=frozenset(os), known_services=frozenset(services), known_processes=frozenset(),
    )


def exploit(key, name="e_drupal", service=DRUPAL):
    return ActionDescriptor(f"exploit:{key}:{name}", "exploit", key, {"service": service, "os": "linux"})


def scan(key, kind="service-scan", action_type="service_scan"):
    return ActionDescriptor(f"{kind}:{key}", action_type, key, {})


FAILED = ObservedActionOutcome(succeeded=False, productive=False, failure_category=FailureCategory.UNSPECIFIED)


def collect_with_trained_context(config, steps, seed, assisted=False):
    """Collect with W_x set to random non-zero values (as after training), so
    the context genuinely shapes the logits being replayed."""
    torch.manual_seed(seed)
    policy = RecurrentPolicy(config.policy, consultation_enabled=assisted)
    with torch.no_grad():
        policy.base_scorer.w_x.normal_(std=1.0)
    collector = build_collector(config, policy, make_adapter(config), seed=seed)
    records, summaries = asyncio.run(collector.collect(steps))
    return policy, records, summaries


def by_episode(records):
    episodes: dict[int, list] = {}
    for record in records:
        episodes.setdefault(record.episode_id, []).append(record)
    return list(episodes.values())


# --- A / B: compatibility evidence ------------------------------------------------------------------------


def test_a_matching_service_is_marked_confirmed():
    memory = EpisodeActionMemory()
    host = facts("host-1-0", services=[DRUPAL], os=["linux"])
    actions = [exploit("host-1-0"), exploit("host-1-0", "e_proftpd", FTP), finish_descriptor()]
    evidence = memory.evidence_matrix(actions, {"host-1-0": host}, STATE)

    assert evidence.shape == (3, EVIDENCE_FEATURE_DIM)
    drupal, proftpd, finish = evidence
    assert drupal[SERVICE_INFO] == 1.0 and drupal[CONFIRMED_FRACTION] == 1.0 and drupal[CONTRADICTED] == 0.0
    # Services were revealed and FTP is not among them: a genuine negative observation.
    assert proftpd[REQUIRED_SERVICE_NOT_FOUND] == 1.0 and proftpd[CONTRADICTED] == 1.0
    assert torch.count_nonzero(finish) == 0
    assert memory.selected_action_context(actions[0], {"host-1-0": host}, drupal).evidence_status == "confirmed"


def test_b_unscanned_service_is_unknown_not_absent():
    memory = EpisodeActionMemory()
    host = facts("host-1-1")  # nothing observed yet
    action = exploit("host-1-1")
    row = memory.evidence_matrix([action, finish_descriptor()], {"host-1-1": host}, STATE)[0]

    assert row[SERVICE_INFO] == 0.0
    assert row[REQUIRED_SERVICE_NOT_FOUND] == 0.0 and row[CONTRADICTED] == 0.0
    assert row[UNKNOWN_FRACTION] == 1.0 and row[CONFIRMED_FRACTION] == 0.0
    context = memory.selected_action_context(action, {"host-1-1": host}, row)
    assert context.evidence_status == "unknown" and context.required_service_status == "unknown"


def test_scan_features_describe_what_a_scan_would_still_reveal():
    memory = EpisodeActionMemory()
    actions = [scan("host-1-0"), exploit("host-1-0"), exploit("host-1-0", "e_proftpd", FTP), finish_descriptor()]
    before = memory.evidence_matrix(actions, {"host-1-0": facts("host-1-0")}, STATE)[0]
    assert before[SCAN_INFO_AVAILABLE] == 0.0 and before[SCAN_UNRESOLVED] == 1.0

    after = memory.evidence_matrix(actions, {"host-1-0": facts("host-1-0", services=[FTP])}, STATE)[0]
    assert after[SCAN_INFO_AVAILABLE] == 1.0 and after[SCAN_UNRESOLVED] == 0.0


def test_a_successful_scan_counts_as_information_even_when_it_finds_nothing_required():
    memory = EpisodeActionMemory()
    host = facts("host-1-0")
    memory.record(scan("host-1-0", "process-scan", "process_scan"), 0, {"host-1-0": host},
                  ObservedActionOutcome(True, False, FailureCategory.NONE))
    assert memory.information_available("host-1-0", host) == (False, False, True)


# --- C / D / E: host-action history ------------------------------------------------------------------------


def test_c_history_is_specific_to_the_host_action_pair():
    memory = EpisodeActionMemory()
    host_a, host_b = facts("host-1-0"), facts("host-1-1", services=[DRUPAL], os=["linux"])
    by_target = {"host-1-0": host_a, "host-1-1": host_b}
    memory.record(exploit("host-1-0"), 0, by_target, FAILED)

    actions = [exploit("host-1-0"), exploit("host-1-1"), exploit("host-1-0", "e_proftpd", FTP), finish_descriptor()]
    history = memory.history_matrix(actions, by_target, step=1)
    assert history.shape == (4, HISTORY_FEATURE_DIM)
    assert history[0, ATTEMPTED] == 1.0 and history[0, UNSPECIFIED_FAILURE] == 1.0
    assert torch.count_nonzero(history[1]) == 0  # host B's e_drupal is untouched
    assert torch.count_nonzero(history[2]) == 0  # host A's other exploits too
    assert torch.count_nonzero(history[3]) == 0

    # Host B's Drupal evidence is unaffected, and the cross-host retry is only flagged as a metric.
    evidence = memory.evidence_matrix(actions, by_target, STATE)
    assert evidence[1, CONFIRMED_FRACTION] == 1.0
    assert memory.selected_action_context(actions[1], by_target, evidence[1]).failed_on_other_host is True


def test_d_a_stochastic_failure_is_recorded_but_never_disables_the_action():
    memory = EpisodeActionMemory()
    host = facts("host-1-0", services=[DRUPAL], os=["linux"])
    stochastic = ObservedActionOutcome(False, False, FailureCategory.STOCHASTIC)
    for step in range(3):
        memory.record(exploit("host-1-0"), step, {"host-1-0": host}, stochastic)
    row = memory.history_matrix([exploit("host-1-0")], {"host-1-0": host}, step=3)[0]
    assert row[STOCHASTIC_FAILURE] == 1.0 and row[UNSPECIFIED_FAILURE] == 0.0
    assert row[ATTEMPTS] == 0.75 and row[UNPRODUCTIVE] == 0.75 and row[RECENCY] == 1.0
    # The evidence for the exploit still says its service is confirmed.
    assert memory.evidence_matrix([exploit("host-1-0")], {"host-1-0": host}, STATE)[0, CONFIRMED_FRACTION] == 1.0


def test_d_policy_keeps_positive_probability_for_repeatedly_failed_actions():
    config = config_for(True, True)
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy)
    with torch.no_grad():
        policy.base_scorer.w_x.normal_(std=3.0)  # any learned weighting, however strong
    adapter = NasimEmuAdapter(SMALL_SCENARIO, config.environment.max_episode_steps)
    state = adapter.reset(seed=1)
    graph, legal, compat, progress = policy_inputs(policy, adapter, state)
    memory = EpisodeActionMemory()
    by_target = extract_visible_host_facts(state)
    target = next(a for a in legal if a.action_type == "exploit")
    for step in range(4):
        memory.record(target, step, by_target, ObservedActionOutcome(False, False, FailureCategory.STOCHASTIC))
    context = policy.select_action_context(
        memory.evidence_matrix(legal, by_target, state), memory.history_matrix(legal, by_target, 4)
    )
    out = policy.step(graph, legal, policy.initial_recurrent_state(), compat, progress, action_context=context)
    assert out.base_probs.shape[0] == len(legal) - 1  # no candidate removed
    assert torch.isclose(out.base_probs.sum(), torch.tensor(1.0))
    assert out.base_probs[legal.index(target)] > 0


def test_e_history_reports_new_evidence_after_a_failed_exploit():
    """Real simulator: e_drupal fails against a target without Drupal, then a
    service scan reveals the target's services."""
    adapter = NasimEmuAdapter(SMALL_SCENARIO, max_episode_steps=20)
    memory = EpisodeActionMemory()
    for seed in range(200):
        state = adapter.reset(seed=seed)
        legal = adapter.legal_actions(state)
        drupal = next(a for a in legal if a.action_id.endswith(":e_drupal"))
        before = extract_visible_host_facts(state)
        transition = adapter.step(drupal)
        after = extract_visible_host_facts(transition.state)
        outcome = observed_action_outcome(state, transition, before, after)
        if not outcome.succeeded:
            break
    else:
        pytest.fail("no realization without Drupal found")
    assert outcome == ObservedActionOutcome(False, False, FailureCategory.UNSPECIFIED)
    memory.record(drupal, 0, after, outcome)

    unchanged = memory.history_matrix([drupal], after, step=1)[0]
    assert unchanged[CHANGED] == 0.0 and unchanged[NEW_INFO] == 0.0
    context = memory.selected_action_context(drupal, after, memory.evidence_matrix([drupal], after, transition.state)[0])
    assert context.unproductive_repeat is True

    service_scan = next(
        a for a in adapter.legal_actions(transition.state)
        if a.action_type == "service_scan" and a.target_key == drupal.target_key
    )
    scanned = adapter.step(service_scan)
    scanned_facts = extract_visible_host_facts(scanned.state)
    scan_outcome = observed_action_outcome(transition.state, scanned, after, scanned_facts)
    assert scan_outcome.succeeded and scan_outcome.productive
    memory.record(service_scan, 1, scanned_facts, scan_outcome)

    changed = memory.history_matrix([drupal], scanned_facts, step=2)[0]
    assert changed[CHANGED] == 1.0 and changed[NEW_INFO] == 1.0 and changed[RECENCY] == 0.5
    evidence = memory.evidence_matrix([drupal], scanned_facts, scanned.state)[0]
    assert evidence[SERVICE_INFO] == 1.0 and evidence[REQUIRED_SERVICE_NOT_FOUND] == 1.0
    context = memory.selected_action_context(drupal, scanned_facts, evidence)
    assert context.unproductive_repeat is False and context.evidence_status == "contradicted"


def test_memory_resets_at_every_episode_start():
    config = config_for(False, True)
    _, records, summaries = collect(config, 40, seed=2)
    assert len(summaries) >= 2
    for record in records:
        if record.environment_step == 0:
            assert torch.count_nonzero(record.history_features) == 0


# --- F: PPO replay ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("evidence,history", [(True, False), (False, True), (True, True)])
def test_f_replay_at_theta_old_reproduces_collected_log_probabilities(evidence, history):
    policy, records, _ = collect_with_trained_context(config_for(evidence, history), 48, seed=4)
    assert any(torch.count_nonzero(r.history_features if history else r.evidence_features) for r in records)
    with torch.no_grad():
        for episode_records in by_episode(records):
            replay = _replay_chunk(policy, chunk_of(episode_records), torch.device("cpu"), 0.0)
            old = torch.tensor([r.old_joint_log_probability for r in episode_records])
            assert torch.allclose(replay.new_joint_log_probs, old, atol=1e-5)


def test_f_stored_features_are_reconstructible_from_the_observable_trajectory():
    """Re-simulating each recorded episode's actions from its seed with a fresh
    memory reproduces every stored evidence/history matrix exactly: the
    features depend only on what was observable up to each decision."""
    config = config_for(True, True)
    _, records, summaries = collect(config, 60, seed=6)
    seeds = {s.episode_id: s.seed for s in summaries}
    adapter = NasimEmuAdapter(SMALL_SCENARIO, config.environment.max_episode_steps)
    checked = 0
    for episode_id, seed in seeds.items():
        episode = [r for r in records if r.episode_id == episode_id]
        state, memory = adapter.reset(seed=seed), EpisodeActionMemory()
        for record in episode:
            legal = adapter.legal_actions(state)
            by_target = extract_visible_host_facts(state)
            assert torch.equal(memory.evidence_matrix(legal, by_target, state), record.evidence_features)
            assert torch.equal(memory.history_matrix(legal, by_target, record.environment_step), record.history_features)
            selected = legal[record.selected_action_index]
            transition = adapter.step(selected)
            checked += 1
            if transition.state is None:
                break
            after = extract_visible_host_facts(transition.state)
            memory.record(selected, record.environment_step, after,
                          observed_action_outcome(state, transition, by_target, after))
            state = transition.state
    assert checked == sum(1 for r in records if r.episode_id in seeds)


def test_f_replay_never_touches_the_live_memory(monkeypatch):
    config = config_for(True, True)
    policy, records, _ = collect(config, 32, seed=7)

    def forbidden(*args, **kwargs):
        raise AssertionError("PPO replay must reuse stored features")

    monkeypatch.setattr(EpisodeActionMemory, "evidence_matrix", forbidden)
    monkeypatch.setattr(EpisodeActionMemory, "history_matrix", forbidden)
    episode = [r for r in records if r.episode_id == records[0].episode_id]
    _replay_chunk(policy, chunk_of(episode), torch.device("cpu"), 0.0)


def test_ppo_update_trains_the_context_weights():
    config = config_for(True, True)
    policy, records, _ = collect(config, 48, seed=5)
    advantages, returns = compute_gae(
        [r.training_reward for r in records], [r.critic_value for r in records], [r.terminated for r in records],
        [r.truncated for r in records], [r.bootstrap_value for r in records], 0.99, 0.95,
    )
    chunks, start = [], 0
    for episode_records in by_episode(records):
        end = start + len(episode_records)
        chunk = chunk_of(episode_records, advantages[start:end])
        chunk.returns = list(returns[start:end])
        chunks.append(chunk)
        start = end
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    ppo_update(policy, optimizer, chunks, config.policy.ppo, torch.device("cpu"))
    assert torch.count_nonzero(policy.base_scorer.w_x) > 0


# --- G: no privileged information --------------------------------------------------------------------------


def test_g_features_depend_only_on_the_visible_observation():
    """Two realizations whose targets run different hidden services (neither
    with Drupal) look identical before and after a failed e_drupal -- and so
    do their evidence/history features."""
    adapter = NasimEmuAdapter(SMALL_SCENARIO, max_episode_steps=20)

    def hidden_services(seed):
        adapter.reset(seed=seed)
        network = adapter._env.env.network  # test-only peek at simulator truth
        target = network.sensitive_addresses[0]
        return frozenset(s for s, on in adapter._env.env.current_state.get_host(target).services.items() if on)

    candidates = {}
    for seed in range(300):
        services = hidden_services(seed)
        if DRUPAL not in services:
            candidates.setdefault(services, seed)
        if len(candidates) >= 2:
            break
    seed_a, seed_b = list(candidates.values())[:2]

    def run(seed):
        memory = EpisodeActionMemory()
        state = adapter.reset(seed=seed)
        legal = adapter.legal_actions(state)
        by_target = extract_visible_host_facts(state)
        first = (memory.evidence_matrix(legal, by_target, state), memory.history_matrix(legal, by_target, 0))
        drupal = next(a for a in legal if a.action_id.endswith(":e_drupal"))
        transition = adapter.step(drupal)
        after = extract_visible_host_facts(transition.state)
        memory.record(drupal, 0, after, observed_action_outcome(state, transition, by_target, after))
        legal = adapter.legal_actions(transition.state)
        second = (memory.evidence_matrix(legal, after, transition.state), memory.history_matrix(legal, after, 1))
        return state.host_rows.copy(), transition.state.host_rows.copy(), first, second

    rows_a0, rows_a1, first_a, second_a = run(seed_a)
    rows_b0, rows_b1, first_b, second_b = run(seed_b)
    assert (rows_a0 == rows_b0).all() and (rows_a1 == rows_b1).all()  # same visible observation
    for x, y in zip(first_a + second_a, first_b + second_b):
        assert torch.equal(x, y)


# --- H: unchanged stopping architecture --------------------------------------------------------------------


def test_h_context_changes_only_the_ordinary_action_scores():
    config = config_for(True, True)
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy)
    with torch.no_grad():
        policy.base_scorer.w_x.normal_(std=1.0)
    adapter = NasimEmuAdapter(SMALL_SCENARIO, config.environment.max_episode_steps)
    graph, legal, compat, progress = policy_inputs(policy, adapter, adapter.reset(seed=3))
    rstate = policy.initial_recurrent_state()
    zeros = torch.zeros(len(legal), policy.action_context_dim)
    ones = torch.ones(len(legal), policy.action_context_dim)
    a = policy.step(graph, legal, rstate, compat, progress, action_context=zeros)
    b = policy.step(graph, legal, rstate, compat, progress, action_context=ones)
    for name in ("z", "value", "stop_logit", "completion_logit", "action_embeddings"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert not torch.equal(a.base_logits, b.base_logits)
    probs = hierarchical_action_probabilities(b.stop_logit, b.base_logits)
    p_stop = torch.sigmoid(b.stop_logit)
    assert torch.isclose(probs[-1], p_stop)
    assert torch.allclose(probs[:-1], (1 - p_stop) * torch.softmax(b.base_logits, dim=-1))


def test_h_all_conditions_share_the_original_initialization():
    states = {}
    for evidence, history in ((False, False), (True, False), (False, True), (True, True)):
        torch.manual_seed(11)
        states[(evidence, history)] = RecurrentPolicy(config_for(evidence, history).policy).state_dict()
    baseline = states[(False, False)]
    assert "base_scorer.w_x" not in baseline
    for key, state in states.items():
        assert all(torch.equal(baseline[name], state[name]) for name in baseline), key
        if key != (False, False):
            assert torch.count_nonzero(state["base_scorer.w_x"]) == 0
    assert states[(True, False)]["base_scorer.w_x"].shape[1] == EVIDENCE_FEATURE_DIM
    assert states[(False, True)]["base_scorer.w_x"].shape[1] == HISTORY_FEATURE_DIM
    assert states[(True, True)]["base_scorer.w_x"].shape[1] == EVIDENCE_FEATURE_DIM + HISTORY_FEATURE_DIM


def test_h_disabled_context_is_rejected_and_required_context_enforced():
    adapter = NasimEmuAdapter(SMALL_SCENARIO, 12)
    state = adapter.reset(seed=0)
    for evidence, history in ((False, False), (True, True)):
        policy = RecurrentPolicy(config_for(evidence, history).policy)
        graph, legal, compat, progress = policy_inputs(policy, adapter, state)
        wrong = None if policy.action_context_dim else torch.zeros(len(legal), 3)
        with pytest.raises(ValueError):
            policy.step(graph, legal, policy.initial_recurrent_state(), compat, progress, action_context=wrong)


# --- I: assisted mode --------------------------------------------------------------------------------------


def test_i_assisted_mode_applies_advice_on_top_of_the_evidence_aware_scorer():
    config = config_for(True, True, "assisted.yaml")
    policy, records, _ = collect_with_trained_context(config, 48, seed=4, assisted=True)
    queried = [r for r in records if r.sampled_query]
    assert queried and any(not r.sampled_query for r in records if not r.stop_selected)
    assert all(not r.sampled_query and r.plan_maker_scores_in_action_order is None for r in records if r.stop_selected)
    for record in queried:
        residual = torch.tensor(record.normalized_advice)
        assert torch.allclose(record.final_logits, record.base_logits + record.beta * record.alpha * residual, atol=1e-5)
    with torch.no_grad():
        for episode_records in by_episode(records):
            replay = _replay_chunk(policy, chunk_of(episode_records), torch.device("cpu"), 0.1)
            old = torch.tensor([r.old_joint_log_probability for r in episode_records])
            assert torch.allclose(replay.new_joint_log_probs, old, atol=1e-5)
    # Consultation cost is charged once per query, never on FINISH.
    assert sum(r.consultation_cost for r in records) == pytest.approx(0.1 * len(queried))


# --- configuration, checkpoints, metrics -------------------------------------------------------------------


def test_disabled_blocks_keep_the_config_hash():
    base = small_config()
    explicit = config_for(False, False)
    assert config_hash(base) == config_hash(explicit)
    assert config_hash(config_for(True, False)) != config_hash(base)
    assert config_hash(config_for(False, True)) != config_hash(config_for(True, False))


def test_checkpoint_rejects_a_mismatched_context_configuration(tmp_path: Path):
    def saved(evidence, history):
        policy = RecurrentPolicy(config_for(evidence, history).policy)
        path = tmp_path / f"{evidence}_{history}.pt"
        save_checkpoint(path, policy, torch.optim.Adam(policy.parameters()), 0, 0, "hash")
        return path

    both = saved(True, True)
    load_checkpoint(both, RecurrentPolicy(config_for(True, True).policy))
    with pytest.raises(PolicyRepresentationMismatchError, match="action_evidence_enabled"):
        load_checkpoint(both, RecurrentPolicy(config_for(False, False).policy))

    # A checkpoint written before the flags existed has neither input.
    legacy = saved(False, False)
    data = torch.load(legacy, weights_only=False)
    del data["action_evidence_enabled"], data["action_history_enabled"]
    torch.save(data, legacy)
    assert load_checkpoint(legacy, RecurrentPolicy(config_for(False, False).policy)).action_evidence_enabled is False
    with pytest.raises(PolicyRepresentationMismatchError):
        load_checkpoint(legacy, RecurrentPolicy(config_for(False, True).policy))


def test_decision_rows_and_rollout_aggregates_describe_the_selected_action():
    config = config_for(False, False)  # metrics are recorded whatever the policy receives
    _, records, _ = collect(config, 60, seed=8)
    rows = build_decision_rows(records, [0.0] * len(records), [0.0] * len(records))
    for record, row in zip(records, rows):
        if record.stop_selected:
            assert row["selected_action_succeeded"] is None and row["selected_action_prior_attempts"] is None
            continue
        assert row["selected_action_succeeded"] == record.observed_outcome.succeeded
        assert row["selected_action_prior_attempts"] == record.selected_action_context.prior_attempts
        assert row["selected_action_failure_category"] in {c.value for c in FailureCategory}
        if row["selected_action_type"] == "exploit":
            assert row["selected_action_required_service_status"] in {"confirmed", "unknown", "contradicted"}
    rollout = build_rollout_row("r", 1, len(records), records, 1, [0.0] * len(records), [0.0] * len(records),
                                0.0, 0.0, None, decision_rows=rows)
    exploits = [r for r in rows if r["selected_action_type"] == "exploit"]
    assert rollout["exploit_rate"] == pytest.approx(len(exploits) / len(rows))
    if exploits:
        assert rollout["exploit_success_rate"] == pytest.approx(
            sum(r["selected_action_succeeded"] for r in exploits) / len(exploits)
        )
    assert rollout["unproductive_repeat_rate"] == pytest.approx(
        sum(r["selected_action_unproductive_repeat"] is True for r in rows) / len(rows)
    )


def test_baseline_collection_is_unchanged_by_the_feature_computation():
    """Condition A (both disabled) behaves exactly like the original policy:
    the same seed gives the same actions and log-probabilities whether or
    not anything reads the stored evidence/history."""
    config = config_for(False, False)
    _, first, _ = collect(config, 40, seed=9)
    _, second, _ = collect(copy.deepcopy(config), 40, seed=9)
    assert [r.selected_action_index for r in first] == [r.selected_action_index for r in second]
    assert [r.old_joint_log_probability for r in first] == [r.old_joint_log_probability for r in second]
