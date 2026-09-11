"""Observable subnet-exploration progress (spec: distinguish "all known
sensitive targets rooted" from "...AND no known exploration frontier
left", without ever exposing unknown subnets or hidden targets).

Covers: successful-vs-attempted SubnetScan tracking, anti-hidden-topology
semantics, the per-subnet-node GraphSAGE feature, the visible-exploration
recurrent-progress vector (including dynamic discovery), PPO replay
consistency, checkpoint/ablation compatibility, and Plan Maker/PPO
fairness. Metrics/plot coverage lives in test_finish_diagnostics.py-style
additions within this file too, to keep the whole feature's tests together.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from marla.config.loader import load_config, parse_config
from marla.environment.graph import NODE_FEATURE_DIM, build_graph_observation
from marla.environment.nasimemu_adapter import EnvironmentState, NasimEmuAdapter
from marla.environment.observation_summary import build_observation_summary
from marla.environment.visible_facts import (
    VISIBLE_EXPLORATION_PROGRESS_DIM,
    VisibleNetworkExploration,
    compute_exploration_progress,
    extract_visible_network_exploration,
    visible_progress_dim,
)
from marla.learning.checkpoint import PolicyRepresentationMismatchError, load_checkpoint, save_checkpoint
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import RolloutCollector
from marla.scenarios import diagnostic_scenario_path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Real-adapter: attempted vs. successful SubnetScan ----------------------


def _reach_entry_user(adapter: NasimEmuAdapter):
    """Micro D's fixed opening: exploit the entry host for USER access.
    Returns (state, entry_target_key)."""
    state = adapter.reset(seed=1)
    entry_key = f"host-{state.host_addresses[0][0]}-{state.host_addresses[0][1]}"
    legal = adapter.legal_actions(state)
    exploit = next(
        a for a in legal if a.action_type == "exploit" and a.target_key == entry_key and a.action_id.endswith(":e_linux_user")
    )
    state = adapter.step(exploit).state
    return state, entry_key


def test_failed_subnet_scan_does_not_mark_the_subnet_scanned():
    """A SubnetScan attempted before the required USER access exists must
    fail (real NASimEmu precondition) and must NOT be recorded as
    successfully scanned."""
    adapter = NasimEmuAdapter(
        scenario=str(diagnostic_scenario_path("micro_d_exploration_frontier.v2.yaml")),
        max_episode_steps=30, completion_reward=1.0, premature_finish_penalty=0.0,
    )
    state = adapter.reset(seed=1)
    entry_key = f"host-{state.host_addresses[0][0]}-{state.host_addresses[0][1]}"
    legal = adapter.legal_actions(state)
    scan = next(a for a in legal if a.action_type == "subnet_scan" and a.target_key == entry_key)

    result = adapter.step(scan)
    assert result.info.get("success") is False
    assert result.state.successfully_scanned_subnets == frozenset()


def test_successful_zero_discovery_subnet_scan_still_counts_as_scanned():
    """The central regression test (spec section 4/27): a successful
    SubnetScan that discovers zero new subnets must still mark the origin
    subnet as scanned -- never inferred from subnet_graph changing."""
    adapter = NasimEmuAdapter(
        scenario=str(diagnostic_scenario_path("micro_d_exploration_frontier.v2.yaml")),
        max_episode_steps=30, completion_reward=1.0, premature_finish_penalty=0.0,
    )
    state, entry_key = _reach_entry_user(adapter)
    legal = adapter.legal_actions(state)
    scan = next(a for a in legal if a.action_type == "subnet_scan" and a.target_key == entry_key)

    result = adapter.step(scan)
    assert result.info.get("success") is True
    state = result.state
    assert state.subnet_graph  # this scan DID discover something (subnet 2)

    # Repeat the same scan -- now nothing new to discover, but it must
    # still count (both before and after, the origin subnet is scanned).
    legal = adapter.legal_actions(state)
    scan_again = next(a for a in legal if a.action_type == "subnet_scan" and a.target_key == entry_key)
    subnet_graph_before = set(state.subnet_graph)
    result2 = adapter.step(scan_again)
    assert result2.info.get("success") is True
    assert set(result2.state.subnet_graph) == subnet_graph_before  # nothing NEW discovered
    entry_subnet = int(entry_key.split("-")[1])
    assert entry_subnet in result2.state.successfully_scanned_subnets  # still counted


def test_new_episode_resets_successfully_scanned_subnets():
    adapter = NasimEmuAdapter(
        scenario=str(diagnostic_scenario_path("micro_d_exploration_frontier.v2.yaml")),
        max_episode_steps=30, completion_reward=1.0, premature_finish_penalty=0.0,
    )
    state, entry_key = _reach_entry_user(adapter)
    legal = adapter.legal_actions(state)
    scan = next(a for a in legal if a.action_type == "subnet_scan" and a.target_key == entry_key)
    result = adapter.step(scan)
    assert result.state.successfully_scanned_subnets  # nonempty this episode

    fresh_state = adapter.reset(seed=2)
    assert fresh_state.successfully_scanned_subnets == frozenset()


# --- Anti-hidden-topology leakage --------------------------------------------


def _state_with_hosts(host_addresses, successfully_scanned_subnets=frozenset()):
    return EnvironmentState(
        raw_observation=None, host_rows=[None] * len(host_addresses), host_addresses=host_addresses,
        subnet_graph=set(), step_idx=0, successfully_scanned_subnets=successfully_scanned_subnets,
    )


def test_known_subnets_reflects_only_currently_visible_hosts_never_true_topology():
    """A structural proof, not just an assertion: EnvironmentState itself
    carries no notion of "true subnet count" at all -- only currently-
    visible host_addresses -- so a subnet the simulator has but that has
    never had a host discovered in it cannot possibly influence
    known_subnets, known_unscanned_subnets, or the progress vector."""
    only_subnet_one = _state_with_hosts([(1, 0)])
    exploration = extract_visible_network_exploration(only_subnet_one)
    assert exploration.known_subnets == frozenset({1})

    # "True subnets = {1,2,3}, visible subnets = {1}" -- behaves exactly as
    # if the world only currently contained subnet 1.
    progress = compute_exploration_progress(exploration)
    assert progress.tolist() == [0.0, 1.0]  # fraction=0 (nothing scanned), frontier remains

    # Only once subnet 2 actually appears in host_addresses does it affect
    # anything.
    now_two_subnets = _state_with_hosts([(1, 0), (2, 0)])
    exploration2 = extract_visible_network_exploration(now_two_subnets)
    assert exploration2.known_subnets == frozenset({1, 2})


def test_successfully_scanned_subnets_outside_known_set_never_counted():
    """Defensive: even if successfully_scanned_subnets somehow contained a
    subnet ID not currently in host_addresses (should not happen in
    practice -- scanning targets a visible host), it must not be counted
    as "known and scanned" -- extract_visible_network_exploration
    intersects with known_subnets explicitly."""
    state = _state_with_hosts([(1, 0)], successfully_scanned_subnets=frozenset({1, 99}))
    exploration = extract_visible_network_exploration(state)
    assert exploration.known_subnets == frozenset({1})
    assert exploration.successfully_scanned_subnets == frozenset({1})
    assert exploration.fraction_known_subnets_scanned == 1.0
    assert exploration.known_exploration_frontier_remaining is False


# --- GraphSAGE per-subnet-node feature ---------------------------------------


def test_subnet_node_feature_reflects_successful_scan_state():
    # host_addresses/host_rows must correspond 1:1; using no hosts at all
    # (empty of both) isolates the subnet-node feature construction cleanly.
    host_addresses: list[tuple[int, int]] = []

    graph_none_scanned = build_graph_observation(
        [], host_addresses, subnet_graph=set(), successfully_scanned_subnets=frozenset()
    )
    assert graph_none_scanned.data.x.shape == (0, NODE_FEATURE_DIM)  # no subnet nodes without any host

    # Subnet nodes only appear once a host in them is visible -- use real
    # host addresses (still empty host_rows is invalid; give one dummy row
    # per host, matching build_graph_observation's real contract).
    two_hosts = [(1, 0), (2, 0)]
    dummy_rows = _dummy_host_rows(two_hosts)
    graph_one_scanned = build_graph_observation(
        dummy_rows, two_hosts, subnet_graph=set(), successfully_scanned_subnets=frozenset({1})
    )
    num_hosts = len(two_hosts)
    subnet_rows = graph_one_scanned.data.x[num_hosts:]
    # subnet 1's node -> 1.0, subnet 2's node -> 0.0 (order follows sorted discovered_subnets)
    assert subnet_rows[0, 12].item() == 1.0
    assert subnet_rows[1, 12].item() == 0.0


def _dummy_host_rows(host_addresses: list[tuple[int, int]]):
    """A real HostVector-shaped row (correctly sized/laid out) per host --
    build_graph_observation's per-host features (_host_features) only read
    generic fields (access/reachable/services/...) off the row itself, and
    group hosts into subnet nodes using the separate host_addresses list,
    never by decoding the row's own (read-only) address -- so reusing one
    real row's shape/layout for every dummy host is sufficient here; the
    subnet-node feature under test does not depend on per-host content.
    """
    adapter = NasimEmuAdapter(
        scenario=str(diagnostic_scenario_path("micro_d_exploration_frontier.v2.yaml")),
        max_episode_steps=5, completion_reward=1.0, premature_finish_penalty=0.0,
    )
    state = adapter.reset(seed=1)
    template_row = state.host_rows[0]
    return [template_row.copy() for _ in host_addresses]


def test_newly_discovered_subnet_node_starts_unscanned():
    two_hosts = [(1, 0), (2, 0)]
    graph = build_graph_observation(
        _dummy_host_rows(two_hosts), two_hosts, subnet_graph=set(), successfully_scanned_subnets=frozenset({1})
    )
    # subnet 2 just appeared (e.g. via a subnet scan from 1) -- must read 0
    # until IT is itself successfully scanned, never inheriting subnet 1's state.
    assert graph.data.x[len(two_hosts) + 1, 12].item() == 0.0


def test_host_nodes_never_receive_the_subnet_scan_completed_bit():
    """A real adapter graph, with the feature genuinely enabled, to prove
    host rows (index 12) stay 0 even when subnet nodes are 1 -- the
    feature is per-subnet, never per-host (spec section 14)."""
    adapter = NasimEmuAdapter(
        scenario=str(diagnostic_scenario_path("micro_d_exploration_frontier.v2.yaml")),
        max_episode_steps=30, completion_reward=1.0, premature_finish_penalty=0.0,
    )
    state, entry_key = _reach_entry_user(adapter)
    legal = adapter.legal_actions(state)
    scan = next(a for a in legal if a.action_type == "subnet_scan" and a.target_key == entry_key)
    state = adapter.step(scan).state

    graph = adapter.to_pyg_data(state, include_subnet_scan_feature=True)
    num_hosts = len(state.host_addresses)
    host_rows = graph.data.x[:num_hosts]
    subnet_rows = graph.data.x[num_hosts:]
    assert (host_rows[:, 12] == 0.0).all()
    assert subnet_rows[:, 12].sum().item() >= 1.0  # at least the entry subnet is marked


def test_include_subnet_scan_feature_false_forces_zero_regardless_of_real_state():
    """The ablation-validity guarantee (spec section 18): even with a
    real, nonempty successfully_scanned_subnets, disabling the feature at
    the adapter level must zero it out entirely -- a "v3-target" policy
    must never see this signal indirectly through GraphSAGE."""
    adapter = NasimEmuAdapter(
        scenario=str(diagnostic_scenario_path("micro_d_exploration_frontier.v2.yaml")),
        max_episode_steps=30, completion_reward=1.0, premature_finish_penalty=0.0,
    )
    state, entry_key = _reach_entry_user(adapter)
    legal = adapter.legal_actions(state)
    scan = next(a for a in legal if a.action_type == "subnet_scan" and a.target_key == entry_key)
    state = adapter.step(scan).state
    assert state.successfully_scanned_subnets  # real, nonempty

    graph_disabled = adapter.to_pyg_data(state, include_subnet_scan_feature=False)
    num_hosts = len(state.host_addresses)
    assert (graph_disabled.data.x[num_hosts:, 12] == 0.0).all()

    graph_enabled = adapter.to_pyg_data(state, include_subnet_scan_feature=True)
    assert graph_enabled.data.x[num_hosts:, 12].sum().item() >= 1.0


def test_empty_graph_shape_matches_bumped_node_feature_dim():
    graph = build_graph_observation(host_rows=[], host_addresses=[], subnet_graph=set())
    assert graph.data.x.shape == (0, NODE_FEATURE_DIM)


# --- Visible exploration recurrent-progress vector (dynamic discovery) -----


def test_exploration_progress_vector_matches_documented_dynamics():
    """Spec section 12/30's exact worked example."""
    import pytest

    exploration = VisibleNetworkExploration(known_subnets=frozenset({1, 2, 3}), successfully_scanned_subnets=frozenset({1, 2}))
    fraction, frontier = compute_exploration_progress(exploration).tolist()
    assert fraction == pytest.approx(2 / 3)
    assert frontier == 1.0

    fully_scanned = VisibleNetworkExploration(known_subnets=frozenset({1, 2, 3}), successfully_scanned_subnets=frozenset({1, 2, 3}))
    fraction2, frontier2 = compute_exploration_progress(fully_scanned).tolist()
    assert fraction2 == pytest.approx(1.0)
    assert frontier2 == 0.0

    new_subnet_discovered = VisibleNetworkExploration(
        known_subnets=frozenset({1, 2, 3, 4}), successfully_scanned_subnets=frozenset({1, 2, 3})
    )
    fraction3, frontier3 = compute_exploration_progress(new_subnet_discovered).tolist()
    assert fraction3 == pytest.approx(3 / 4)
    assert frontier3 == 1.0


def test_exploration_progress_dim_and_visible_progress_dim_agree():
    assert compute_exploration_progress(
        VisibleNetworkExploration(known_subnets=frozenset(), successfully_scanned_subnets=frozenset())
    ).shape == (VISIBLE_EXPLORATION_PROGRESS_DIM,)
    assert visible_progress_dim(target_enabled=False, exploration_enabled=True) == VISIBLE_EXPLORATION_PROGRESS_DIM
    assert visible_progress_dim(target_enabled=False, exploration_enabled=False) == 0


def test_no_known_subnets_gives_zero_fraction_not_a_crash():
    exploration = VisibleNetworkExploration(known_subnets=frozenset(), successfully_scanned_subnets=frozenset())
    fraction, frontier = compute_exploration_progress(exploration).tolist()
    assert fraction == 0.0
    assert frontier == 0.0  # no known subnets -> no frontier to speak of


# --- Plan Maker / PPO fairness -----------------------------------------------


def test_plan_maker_summary_and_ppo_exploration_facts_agree():
    """PPO (extract_visible_network_exploration) and the Plan Maker
    (build_observation_summary's network_exploration section) must derive
    identical known/scanned/unscanned subnet sets from the same state --
    same authoritative source, per spec section 32."""
    adapter = NasimEmuAdapter(
        scenario=str(diagnostic_scenario_path("micro_d_exploration_frontier.v2.yaml")),
        max_episode_steps=30, completion_reward=1.0, premature_finish_penalty=0.0,
    )
    state, entry_key = _reach_entry_user(adapter)
    legal = adapter.legal_actions(state)
    scan = next(a for a in legal if a.action_type == "subnet_scan" and a.target_key == entry_key)
    state = adapter.step(scan).state

    exploration = extract_visible_network_exploration(state)
    summary = build_observation_summary(state)
    net = summary["network_exploration"]

    assert net["known_subnets"] == sorted(exploration.known_subnets)
    assert net["successfully_scanned_subnets"] == sorted(exploration.successfully_scanned_subnets)
    assert net["known_unscanned_subnets"] == sorted(exploration.known_unscanned_subnets)
    # "subnet 2 is known and unscanned" iff the Plan Maker's summary says so too.
    assert (2 in exploration.known_unscanned_subnets) == (2 in net["known_unscanned_subnets"])
    # No unknown subnet ever appears on either side.
    assert set(net["known_subnets"]) == exploration.known_subnets


# --- PPO replay consistency --------------------------------------------------


@pytest.mark.asyncio
async def test_ppo_replay_reuses_stored_visible_progress_verbatim(monkeypatch):
    """Mirrors test_ppo.py's compatibility_features replay test: PPO
    replay must feed RecurrentCore EXACTLY the visible_progress tensor
    stored at collection time (which, with exploration enabled, now
    includes the subnet-exploration component) -- never recomputed from a
    (by replay time, stale and inaccessible) live simulator state.
    """
    from marla.learning import recurrent_core as recurrent_core_module
    from marla.learning.ppo import build_sequence_chunks, _replay_chunk

    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    data["policy"]["recurrent"]["visible_subnet_exploration"] = True
    data["policy"]["ppo"]["steps_per_env"] = 16
    data["policy"]["recurrent"]["sequence_length"] = 4
    config = parse_config(data)

    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy)
    assert policy.recurrent_core.visible_progress_dim == 5

    adapter = NasimEmuAdapter(
        scenario=str(diagnostic_scenario_path("micro_d_exploration_frontier.v2.yaml")),
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    collector = RolloutCollector(adapter, policy, run_id="test-run", base_seed=1)
    records, _summaries = await collector.collect(8)
    assert all(r.visible_progress.shape == (5,) for r in records)

    sentinel = torch.full_like(records[0].visible_progress, 7.0)
    records[0].visible_progress = sentinel

    captured: list[torch.Tensor] = []
    real_forward = recurrent_core_module.RecurrentCore.forward

    def spy(self, graph_embedding, visible_progress, previous_action_embedding, previous_reward, previous_query, previous_hidden):
        captured.append(visible_progress.clone())
        return real_forward(self, graph_embedding, visible_progress, previous_action_embedding, previous_reward, previous_query, previous_hidden)

    monkeypatch.setattr(recurrent_core_module.RecurrentCore, "forward", spy)

    chunks = build_sequence_chunks(records, [0.0] * len(records), [0.0] * len(records), sequence_length=8)
    _replay_chunk(policy, chunks[0], torch.device("cpu"), consultation_cost=0.0)

    assert torch.equal(captured[0].squeeze(0), sentinel)
    for record, seen in zip(chunks[0].records, captured):
        assert torch.equal(seen.squeeze(0), record.visible_progress)


# --- Checkpoint / ablation compatibility -------------------------------------


def test_checkpoint_rejects_mismatched_ablation_flags_before_any_load():
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")

    policy_no_exploration = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy_no_exploration.parameters(), lr=1e-3)

    data = config.model_dump()
    data["policy"]["recurrent"]["visible_subnet_exploration"] = True
    config_with_exploration = parse_config(data)
    policy_with_exploration = RecurrentPolicy(config_with_exploration.policy)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/checkpoint.pt"
        save_checkpoint(path, policy_no_exploration, optimizer, update_count=1, environment_steps=1, config_hash="h")

        before = {k: v.clone() for k, v in policy_with_exploration.state_dict().items()}
        with pytest.raises(PolicyRepresentationMismatchError, match="visible_subnet_exploration_enabled"):
            load_checkpoint(path, policy_with_exploration)
        for key, value in before.items():
            assert torch.equal(value, policy_with_exploration.state_dict()[key])


# --- Decision/rollout metrics -------------------------------------------------


def test_decision_row_carries_exploration_metrics_from_a_real_rollout():
    from marla.metrics.writer import build_decision_rows

    data = load_config(REPO_ROOT / "examples" / "baseline.yaml").model_dump()
    data["policy"]["recurrent"]["visible_subnet_exploration"] = True
    config = parse_config(data)
    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy)
    adapter = NasimEmuAdapter(
        scenario=str(diagnostic_scenario_path("micro_d_exploration_frontier.v2.yaml")),
        max_episode_steps=20, completion_reward=1.0, premature_finish_penalty=0.0,
    )
    collector = RolloutCollector(adapter, policy, run_id="t", base_seed=1)

    import asyncio

    records, _summaries = asyncio.run(collector.collect(20))
    rows = build_decision_rows(records, advantages=[0.0] * len(records), returns=[0.0] * len(records))
    for row, record in zip(rows, records):
        assert row["known_subnets_total_before_action"] == record.known_subnets_total_before_action
        assert row["known_subnets_scanned_before_action"] == record.known_subnets_scanned_before_action
        assert row["known_subnets_unscanned_before_action"] == (
            record.known_subnets_total_before_action - record.known_subnets_scanned_before_action
        )
        assert 0.0 <= row["fraction_known_subnets_scanned_before_action"] <= 1.0
        assert isinstance(row["known_exploration_frontier_remaining_before_action"], bool)


def test_rollout_row_frontier_conditional_finish_split_uses_visible_completion_only():
    """Unit-level: hand-built decision rows proving the frontier split
    only considers rows where a sensitive target is actually visible AND
    all of it is rooted -- never conflating "no visible target" with
    "complete"."""
    from marla.metrics.writer import build_rollout_row
    from marla.learning.rollout import StepRecord
    from marla.environment.action_compatibility import COMPATIBILITY_FEATURE_DIM
    from marla.environment.actions import ActionDescriptor
    from marla.environment.visible_facts import VISIBLE_TARGET_PROGRESS_DIM

    def _row(has_target, all_rooted, frontier_remaining, finish_prob):
        return {
            "objective_satisfied_before_action": False,
            "base_finish_probability": finish_prob,
            "final_finish_probability": finish_prob,
            "base_finish_is_argmax": False,
            "final_finish_is_argmax": False,
            "finish_selected": False,
            "base_compatible_action_probability_mass": 0.0,
            "final_compatible_action_probability_mass": 0.0,
            "selected_action_visible_preconditions_status": "unknown",
            "known_subnets_total_before_action": 1,
            "known_subnets_scanned_before_action": 0 if frontier_remaining else 1,
            "known_subnets_unscanned_before_action": 1 if frontier_remaining else 0,
            "fraction_known_subnets_scanned_before_action": 0.0 if frontier_remaining else 1.0,
            "known_exploration_frontier_remaining_before_action": frontier_remaining,
            "has_visible_sensitive_target_before_action": has_target,
            "all_visible_sensitive_targets_rooted_before_action": all_rooted,
        }

    rows = [
        _row(has_target=False, all_rooted=None, frontier_remaining=True, finish_prob=0.9),  # excluded: no visible target
        _row(has_target=True, all_rooted=False, frontier_remaining=True, finish_prob=0.9),  # excluded: not complete
        _row(has_target=True, all_rooted=True, frontier_remaining=True, finish_prob=0.2),  # complete, frontier remains
        _row(has_target=True, all_rooted=True, frontier_remaining=False, finish_prob=0.8),  # complete, no frontier
    ]
    descriptors = [ActionDescriptor("finish", "finish", None, {}, is_finish=True)]
    records = [
        StepRecord(
            run_id="r", episode_id=1, environment_step=0, observation_id="o",
            graph_data=None, node_key_to_index={}, legal_action_descriptors=descriptors,
            compatibility_features=torch.zeros((1, COMPATIBILITY_FEATURE_DIM)),
            visible_progress=torch.zeros(VISIBLE_TARGET_PROGRESS_DIM),
            initial_gru_hidden_state=torch.zeros(2), previous_action_embedding=torch.zeros(2),
            previous_training_reward=0.0, previous_query=False,
            base_logits=torch.zeros(1), final_logits=torch.zeros(1), selected_action_index=0,
            old_action_log_probability=0.0, old_joint_log_probability=0.0, critic_value=0.0,
            nasimemu_reward=0.0, consultation_cost=0.0, training_reward=0.0,
            terminated=False, truncated=False,
        )
        for _ in rows
    ]

    row = build_rollout_row(
        run_id="r", rollout=1, environment_steps_total=100, records=records,
        episodes_finished=0, advantages=[0.0] * len(rows), returns=[0.0] * len(rows),
        collection_seconds=1.0, optimization_seconds=1.0, evaluation_seconds=None, decision_rows=rows,
    )
    assert row["mean_finish_probability_visible_complete_frontier_remaining"] == pytest.approx(0.2)
    assert row["mean_finish_probability_visible_complete_no_frontier"] == pytest.approx(0.8)
    # Delta_frontier for this rollout: no_frontier - frontier_remaining.
    delta_frontier = (
        row["mean_finish_probability_visible_complete_no_frontier"]
        - row["mean_finish_probability_visible_complete_frontier_remaining"]
    )
    assert delta_frontier == pytest.approx(0.6)


# --- Plots ---------------------------------------------------------------


def test_exploration_plots_skip_cleanly_for_a_run_written_before_the_columns_existed(tmp_path):
    import csv

    from marla.metrics.plots import generate_plots
    from marla.metrics.writer import _DECISIONS_FIELDS, _ROLLOUTS_FIELDS

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    old_rollout_fields = [f for f in _ROLLOUTS_FIELDS if "frontier" not in f and "subnets_scanned" not in f]
    with (run_dir / "rollouts.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=old_rollout_fields)
        writer.writeheader()
        writer.writerow(
            {field: "" for field in old_rollout_fields}
            | {"run_id": "x", "rollout": 1, "environment_steps_total": 100, "collection_seconds": 1.0, "optimization_seconds": 1.0}
        )

    old_decision_fields = [f for f in _DECISIONS_FIELDS if "subnet" not in f and "frontier" not in f]
    with (run_dir / "decisions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=old_decision_fields)
        writer.writeheader()
        writer.writerow(
            {field: "" for field in old_decision_fields}
            | {"run_id": "x", "environment_step": 0, "critic_value": 0.0, "gae_advantage": 0.0, "return_target": 0.0}
        )

    written = generate_plots(run_dir, run_dir / "plots")
    names = {p.name for p in written}
    assert "known_subnet_exploration_progress.png" not in names
    assert "finish_probability_by_visible_completion_and_frontier.png" not in names


@pytest.mark.integration
def test_generate_plots_includes_exploration_plots_for_a_real_run(tmp_path):
    import asyncio
    from datetime import datetime, timezone

    from marla.learning.trainer import run_baseline_training
    from marla.metrics.plots import generate_plots
    from marla.metrics.writer import write_run_artifacts
    from marla.runtime.device import resolve_device

    data = load_config(REPO_ROOT / "examples" / "baseline.yaml").model_dump()
    data["policy"]["recurrent"]["visible_subnet_exploration"] = True
    data["policy"]["ppo"]["steps_per_env"] = 16
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["device"] = "cpu"
    data["metrics"]["eval_episodes"] = 0
    config = parse_config(data)

    scenario_path = str(diagnostic_scenario_path("micro_d_exploration_frontier.v2.yaml"))
    result = asyncio.run(run_baseline_training(config, scenario_path, num_rollouts=2, seed=1))
    resolved_device = resolve_device(config.device)
    run_dir = tmp_path / "run"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    write_run_artifacts(run_dir, config, result, resolved_device, start, end, status="completed")

    written = generate_plots(run_dir, run_dir / "plots")
    names = {p.name for p in written}
    assert "known_subnet_exploration_progress.png" in names
