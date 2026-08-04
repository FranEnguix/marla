"""Torch Geometric GraphSAGE encoder (spec section 12).

Produces per-node embeddings (used as target-host embeddings for the action
encoder) and a global mean-pooled graph embedding (fed into the GRU).
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.hidden_dim = hidden_dim
        self.convs = nn.ModuleList(
            [SAGEConv(input_dim, hidden_dim)]
            + [SAGEConv(hidden_dim, hidden_dim) for _ in range(layers - 1)]
        )

    def forward(self, data: Batch) -> tuple[Tensor, Tensor]:
        """Returns ``(node_embeddings, graph_embeddings)``.

        ``data.batch`` is required when ``data`` holds more than one graph
        (a :class:`~torch_geometric.data.Batch`); for a single graph it may
        be omitted (``global_mean_pool`` treats all nodes as one graph).
        """
        h = data.x
        for conv in self.convs:
            h = F.relu(conv(h, data.edge_index))
        batch_index = getattr(data, "batch", None)
        g = global_mean_pool(h, batch_index)
        return h, g
