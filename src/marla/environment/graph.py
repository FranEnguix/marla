"""Convert a visible NASimEmu observation into a Torch Geometric graph.

Only fields derivable from the partially-observable observation are used
(spec section 12): node type, access level, reachability, known
service/process fractions, a coarse "has any services/processes been
identified" scan-status proxy, the objective-target flag (a value-bearing
host, matching the ``capture_target`` objective = all sensitive hosts
compromised), and normalized value fields. No hidden simulator state (e.g.
undiscovered services, true exploit success probabilities) is encoded.

The feature width is fixed (:data:`NODE_FEATURE_DIM`) and independent of a
given scenario's number of distinct services/processes/OSes, so the same
graph encoder works across scenarios with different vocabularies.

A subnet node's last feature (index 12, ``subnet_scan_completed``) is 1
iff a ``SubnetScan`` originating from that subnet has *succeeded* at least
once this episode -- never merely attempted, and never encoding the
subnet's own ID (see :func:`_subnet_features`,
``marla.environment.nasimemu_adapter.EnvironmentState.successfully_scanned_subnets``).
Host nodes never receive this bit. It is threaded through
``build_graph_observation``'s ``successfully_scanned_subnets`` parameter,
which the caller (:class:`marla.environment.nasimemu_adapter.NasimEmuAdapter`)
zeroes out entirely when ``policy.recurrent.visible_subnet_exploration`` is
disabled, so the ablation flag controls this feature identically to the
recurrent-policy vector -- see :doc:`/configuration`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel
from torch_geometric.data import Data

from marla.environment.actions import host_target_key

NODE_FEATURE_DIM = 13
_VALUE_SCALE = 10.0


@dataclass(frozen=True)
class GraphObservation:
    """A single-episode-step graph observation plus the target lookup table."""

    data: Data
    node_key_to_index: dict[str, int]


def _host_features(row: np.ndarray) -> np.ndarray:
    host = HostVector(row)
    features = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)

    features[0] = 0.0  # node_type: host
    access = int(host.access)
    features[1] = float(access == AccessLevel.NONE)
    features[2] = float(access == AccessLevel.USER)
    features[3] = float(access == AccessLevel.ROOT)
    features[4] = float(bool(host.reachable))

    services = host.services
    processes = host.processes
    known_services = sum(services.values())
    known_processes = sum(processes.values())

    features[5] = known_services / max(len(services), 1)
    features[6] = known_processes / max(len(processes), 1)
    features[7] = float(known_services > 0)
    features[8] = float(known_processes > 0)
    features[9] = float(host.value > 0)
    features[10] = float(host.value) / _VALUE_SCALE
    features[11] = float(host.discovery_value) / _VALUE_SCALE

    return features


def _subnet_features(subnet_id: int, successfully_scanned_subnets: frozenset[int]) -> np.ndarray:
    features = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
    features[0] = 1.0  # node_type: subnet
    # Observable subnet-exploration progress (spec): 1 iff a SubnetScan
    # from a host in this subnet has *succeeded* at least once this
    # episode (never merely attempted -- see NasimEmuAdapter.step()'s
    # tracking of successfully_scanned_subnets, the sole source of truth
    # here). Never the subnet ID itself -- only this one bit.
    features[12] = float(subnet_id in successfully_scanned_subnets)
    return features


def build_graph_observation(
    host_rows: np.ndarray,
    host_addresses: list[tuple[int, int]],
    subnet_graph: set[tuple[int, int]],
    successfully_scanned_subnets: frozenset[int] = frozenset(),
) -> GraphObservation:
    """Build a graph from visible host rows (``raw_observation[:-1]``).

    ``host_addresses[i]`` must correspond to ``host_rows[i]``.
    ``subnet_graph`` is the set of ``(from_subnet, to_subnet)`` edges
    discovered so far via subnet scans (tracked by the adapter across an
    episode -- NASimEmu itself does not persist this).
    ``successfully_scanned_subnets``: which of the subnets that appear as
    nodes here (i.e. currently known, per ``host_addresses``) have had a
    successful ``SubnetScan`` -- defaults to empty, which makes every
    subnet node's ``subnet_scan_completed`` feature read 0 (the safe,
    inert default for any caller not thinking about this ablation; see
    ``NasimEmuAdapter.to_pyg_data``'s ``include_subnet_scan_feature`` for
    where a real caller decides this explicitly).
    """
    num_hosts = len(host_addresses)

    host_features = (
        np.stack([_host_features(row) for row in host_rows], axis=0)
        if num_hosts
        else np.zeros((0, NODE_FEATURE_DIM), dtype=np.float32)
    )

    discovered_subnets = sorted({addr[0] for addr in host_addresses})
    subnet_features = (
        np.stack(
            [_subnet_features(subnet_id, successfully_scanned_subnets) for subnet_id in discovered_subnets],
            axis=0,
        )
        if discovered_subnets
        else np.zeros((0, NODE_FEATURE_DIM), dtype=np.float32)
    )

    node_features = np.concatenate([host_features, subnet_features], axis=0)

    node_key_to_index = {host_target_key(*addr): i for i, addr in enumerate(host_addresses)}
    subnet_node_index = {subnet_id: num_hosts + i for i, subnet_id in enumerate(discovered_subnets)}

    edges: list[tuple[int, int]] = []
    for i, addr in enumerate(host_addresses):
        subnet_node = subnet_node_index[addr[0]]
        edges.append((i, subnet_node))
        edges.append((subnet_node, i))

    for src_subnet, dst_subnet in subnet_graph:
        if src_subnet in subnet_node_index and dst_subnet in subnet_node_index:
            src, dst = subnet_node_index[src_subnet], subnet_node_index[dst_subnet]
            edges.append((src, dst))
            edges.append((dst, src))

    edge_index = (
        torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros((2, 0), dtype=torch.long)
    )
    x = torch.tensor(node_features, dtype=torch.float32)

    return GraphObservation(data=Data(x=x, edge_index=edge_index), node_key_to_index=node_key_to_index)
