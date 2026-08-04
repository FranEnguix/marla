from pathlib import Path

import pytest
import torch

from marla.environment.graph import NODE_FEATURE_DIM, build_graph_observation
from marla.environment.nasimemu_adapter import NasimEmuAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


@pytest.fixture
def small_adapter():
    return NasimEmuAdapter(
        scenario=SMALL_SCENARIO, max_episode_steps=20, completion_reward=1.0, premature_finish_penalty=-1.0
    )


def test_graph_shapes_match_host_and_subnet_counts(small_adapter):
    state = small_adapter.reset(seed=1)
    graph = build_graph_observation(state.host_rows, state.host_addresses, state.subnet_graph)

    num_hosts = len(state.host_addresses)
    num_subnets = len({addr[0] for addr in state.host_addresses})

    assert graph.data.x.shape == (num_hosts + num_subnets, NODE_FEATURE_DIM)
    assert graph.data.x.dtype == torch.float32
    assert graph.data.edge_index.shape[0] == 2
    assert torch.isfinite(graph.data.x).all()


def test_node_key_to_index_matches_host_rows(small_adapter):
    state = small_adapter.reset(seed=1)
    graph = build_graph_observation(state.host_rows, state.host_addresses, state.subnet_graph)

    for i, address in enumerate(state.host_addresses):
        key = f"host-{address[0]}-{address[1]}"
        assert graph.node_key_to_index[key] == i
        assert graph.data.x[i, 0].item() == 0.0  # host node marker


def test_subnet_nodes_marked_and_connected_to_their_hosts(small_adapter):
    state = small_adapter.reset(seed=1)
    graph = build_graph_observation(state.host_rows, state.host_addresses, state.subnet_graph)

    num_hosts = len(state.host_addresses)
    subnet_rows = graph.data.x[num_hosts:]
    assert (subnet_rows[:, 0] == 1.0).all()
    assert (subnet_rows[:, 1:] == 0.0).all()

    edges = set(map(tuple, graph.data.edge_index.T.tolist()))
    for host_index in range(num_hosts):
        # every host must have at least one edge to *some* subnet node
        assert any(src == host_index and dst >= num_hosts for src, dst in edges)
        assert any(dst == host_index and src >= num_hosts for src, dst in edges)


def test_host_access_level_is_one_hot(small_adapter):
    state = small_adapter.reset(seed=1)
    graph = build_graph_observation(state.host_rows, state.host_addresses, state.subnet_graph)

    num_hosts = len(state.host_addresses)
    access_columns = graph.data.x[:num_hosts, 1:4]
    assert torch.allclose(access_columns.sum(dim=1), torch.ones(num_hosts))


def test_empty_graph_does_not_crash():
    graph = build_graph_observation(host_rows=[], host_addresses=[], subnet_graph=set())
    assert graph.data.x.shape == (0, NODE_FEATURE_DIM)
    assert graph.data.edge_index.shape == (2, 0)
    assert graph.node_key_to_index == {}
