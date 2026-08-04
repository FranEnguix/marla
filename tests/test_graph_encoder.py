import torch
from torch_geometric.data import Data

from marla.learning.graph_encoder import GraphEncoder


def _small_graph(num_nodes=4, input_dim=6):
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def test_output_shapes_single_layer():
    data = _small_graph(num_nodes=5, input_dim=6)
    encoder = GraphEncoder(input_dim=6, hidden_dim=8, layers=1)
    node_emb, graph_emb = encoder(data)
    assert node_emb.shape == (5, 8)
    assert graph_emb.shape == (1, 8)
    assert torch.isfinite(node_emb).all()
    assert torch.isfinite(graph_emb).all()


def test_output_shapes_multi_layer():
    data = _small_graph(num_nodes=7, input_dim=6)
    encoder = GraphEncoder(input_dim=6, hidden_dim=16, layers=3)
    node_emb, graph_emb = encoder(data)
    assert node_emb.shape == (7, 16)
    assert graph_emb.shape == (1, 16)


def test_rejects_zero_layers():
    import pytest

    with pytest.raises(ValueError):
        GraphEncoder(input_dim=6, hidden_dim=8, layers=0)


def test_graph_embedding_is_mean_of_node_embeddings_when_single_graph():
    data = _small_graph(num_nodes=4, input_dim=6)
    encoder = GraphEncoder(input_dim=6, hidden_dim=8, layers=1)
    node_emb, graph_emb = encoder(data)
    assert torch.allclose(graph_emb.squeeze(0), node_emb.mean(dim=0), atol=1e-5)
