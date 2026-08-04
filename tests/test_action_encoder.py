import torch

from marla.environment.actions import ActionDescriptor
from marla.learning.action_encoder import ActionEncoder, action_type_id, parameter_features


def test_action_type_id_covers_all_known_types():
    for t in ["service_scan", "os_scan", "subnet_scan", "process_scan", "exploit", "privilege_escalation", "finish"]:
        assert isinstance(action_type_id(t), int)


def test_action_type_id_rejects_unknown():
    import pytest

    with pytest.raises(ValueError):
        action_type_id("not-a-real-type")


def test_parameter_features_shape_and_values():
    assert parameter_features({}).tolist() == [0.0, 0.0]
    assert parameter_features({"service": "http"}).tolist() == [1.0, 0.0]
    assert parameter_features({"os": "linux"}).tolist() == [0.0, 1.0]
    assert parameter_features({"process": "cron", "os": "linux"}).tolist() == [1.0, 1.0]


def test_encode_descriptors_shape_and_finiteness():
    encoder = ActionEncoder(node_embedding_size=8, action_type_embedding_size=4, hidden_size=16)
    node_embeddings = torch.randn(3, 8)
    node_key_to_index = {"host-1-0": 0, "host-1-1": 1, "host-2-0": 2}
    descriptors = [
        ActionDescriptor("service-scan:host-1-0", "service_scan", "host-1-0", {}),
        ActionDescriptor("exploit:host-1-1:e1", "exploit", "host-1-1", {"service": "http"}),
        ActionDescriptor("finish", "finish", None, {}, is_finish=True),
    ]
    out = encoder.encode_descriptors(descriptors, node_embeddings, node_key_to_index, device=torch.device("cpu"))
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()


def test_finish_uses_learned_no_target_embedding_not_any_host_embedding():
    encoder = ActionEncoder(node_embedding_size=8, action_type_embedding_size=4, hidden_size=16)
    node_embeddings = torch.randn(2, 8)
    node_key_to_index = {"host-1-0": 0, "host-1-1": 1}

    finish_only = [ActionDescriptor("finish", "finish", None, {}, is_finish=True)]
    out_a = encoder.encode_descriptors(finish_only, node_embeddings, node_key_to_index, torch.device("cpu"))

    # Change the (unused, since target_key=None) node embeddings entirely;
    # FINISH's encoding must be unaffected because it never reads them.
    out_b = encoder.encode_descriptors(
        finish_only, torch.randn(2, 8) * 100, node_key_to_index, torch.device("cpu")
    )
    assert torch.allclose(out_a, out_b)


def test_same_target_gives_same_embedding_for_same_action_type():
    encoder = ActionEncoder(node_embedding_size=8, action_type_embedding_size=4, hidden_size=16)
    node_embeddings = torch.randn(1, 8)
    node_key_to_index = {"host-1-0": 0}
    descriptors = [
        ActionDescriptor("service-scan:host-1-0", "service_scan", "host-1-0", {}),
        ActionDescriptor("service-scan:host-1-0", "service_scan", "host-1-0", {}),
    ]
    out = encoder.encode_descriptors(descriptors, node_embeddings, node_key_to_index, torch.device("cpu"))
    assert torch.allclose(out[0], out[1])
