"""ActionEncoder tests, including the aliasing regression this module's
compatibility features (spec section 5) exist to fix.

**The bug being fixed** (documented here rather than by keeping a second,
dead copy of the old encoder around): before this change, ``ActionEncoder``
only received ``parameter_features()`` -- ``[has_service_or_process,
has_os]`` -- as action-specific input, with no relation at all to what had
actually been observed on the target. Two exploits (or two privescs)
targeting the same host, sharing the same broad ``action_type`` and the
same *shape* of parameters (both have a service, both have an OS), were
therefore **structurally forced to produce byte-identical embeddings** --
``e_elasticsearch`` was indistinguishable from ``e_wp_ninja`` on the same
Windows host, and a Windows-only privesc was indistinguishable from a
Linux-only one, no matter what the agent had observed. ``test_old_aliasing_bug_reproduced_with_zero_compatibility_features``
below reproduces exactly that forced equality using today's encoder with
all-zero compatibility features (i.e. the informational content the old
encoder had) -- proving the aliasing was a structural property of the
inputs, not an artifact of untrained weights -- and the tests after it
prove real (non-zero, fact-derived) compatibility features break that
equality.
"""

import torch

from marla.environment.action_compatibility import COMPATIBILITY_FEATURE_DIM, compute_compatibility_matrix
from marla.environment.actions import ActionDescriptor
from marla.environment.visible_facts import VisibleHostFacts
from marla.learning.action_encoder import ActionEncoder, action_type_id, parameter_features
from nasimemu.nasim.envs.utils import AccessLevel

TARGET = "host-2-0"


def _zeros(n: int) -> torch.Tensor:
    return torch.zeros((n, COMPATIBILITY_FEATURE_DIM), dtype=torch.float32)


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
    out = encoder.encode_descriptors(
        descriptors, node_embeddings, node_key_to_index, device=torch.device("cpu"),
        compatibility_matrix=_zeros(len(descriptors)),
    )
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()


def test_encode_descriptors_rejects_wrong_compatibility_shape():
    import pytest

    encoder = ActionEncoder(node_embedding_size=8, action_type_embedding_size=4, hidden_size=16)
    descriptors = [ActionDescriptor("finish", "finish", None, {}, is_finish=True)]
    with pytest.raises(ValueError):
        encoder.encode_descriptors(
            descriptors, torch.randn(1, 8), {}, torch.device("cpu"), compatibility_matrix=_zeros(2)
        )


def test_finish_uses_learned_no_target_embedding_not_any_host_embedding():
    encoder = ActionEncoder(node_embedding_size=8, action_type_embedding_size=4, hidden_size=16)
    node_embeddings = torch.randn(2, 8)
    node_key_to_index = {"host-1-0": 0, "host-1-1": 1}

    finish_only = [ActionDescriptor("finish", "finish", None, {}, is_finish=True)]
    out_a = encoder.encode_descriptors(
        finish_only, node_embeddings, node_key_to_index, torch.device("cpu"), _zeros(1)
    )

    # Change the (unused, since target_key=None) node embeddings entirely;
    # FINISH's encoding must be unaffected because it never reads them.
    out_b = encoder.encode_descriptors(
        finish_only, torch.randn(2, 8) * 100, node_key_to_index, torch.device("cpu"), _zeros(1)
    )
    assert torch.allclose(out_a, out_b)


def test_same_target_gives_same_embedding_for_same_action_type_and_compatibility():
    encoder = ActionEncoder(node_embedding_size=8, action_type_embedding_size=4, hidden_size=16)
    node_embeddings = torch.randn(1, 8)
    node_key_to_index = {"host-1-0": 0}
    descriptors = [
        ActionDescriptor("service-scan:host-1-0", "service_scan", "host-1-0", {}),
        ActionDescriptor("service-scan:host-1-0", "service_scan", "host-1-0", {}),
    ]
    out = encoder.encode_descriptors(
        descriptors, node_embeddings, node_key_to_index, torch.device("cpu"), _zeros(2)
    )
    assert torch.allclose(out[0], out[1])


# --- The aliasing bug: reproduced, then shown fixed ------------------------


def _windows_host_facts(known_services=frozenset(), known_os=frozenset({"windows"})) -> dict[str, VisibleHostFacts]:
    return {
        TARGET: VisibleHostFacts(
            target_key=TARGET, reachable=True, compromised=False, access=AccessLevel.NONE,
            known_os=known_os, known_services=known_services, known_processes=frozenset(),
        )
    }


def _exploit_pair():
    e_elasticsearch = ActionDescriptor(
        f"exploit:{TARGET}:e_elasticsearch", "exploit", TARGET,
        {"service": "9200_windows_elasticsearch", "os": "windows"},
    )
    e_wp_ninja = ActionDescriptor(
        f"exploit:{TARGET}:e_wp_ninja", "exploit", TARGET,
        {"service": "80_windows_wp_ninja", "os": "windows"},
    )
    return e_elasticsearch, e_wp_ninja


def test_old_aliasing_bug_reproduced_with_zero_compatibility_features():
    """With all-zero compatibility features (the old encoder's exact
    informational content -- see module docstring), two exploits sharing
    the same action_type/parameter-shape on the same target are
    STRUCTURALLY forced to identical embeddings. This is the bug; it is
    reproduced here as a permanent regression marker, not merely asserted
    once and discarded.
    """
    encoder = ActionEncoder(node_embedding_size=8, action_type_embedding_size=4, hidden_size=16)
    node_embeddings = torch.randn(1, 8)
    node_key_to_index = {TARGET: 0}
    e_elasticsearch, e_wp_ninja = _exploit_pair()

    out = encoder.encode_descriptors(
        [e_elasticsearch, e_wp_ninja], node_embeddings, node_key_to_index, torch.device("cpu"), _zeros(2)
    )
    assert torch.allclose(out[0], out[1])  # forced-equal: this IS the bug


def test_formerly_aliased_exploits_now_differ_once_facts_are_observed():
    """Same target, same encoder, same two exploits -- but now with REAL
    compatibility features derived from an observed Windows host whose
    elasticsearch service has been confirmed present. The embeddings must
    now differ: this is the fix.
    """
    encoder = ActionEncoder(node_embedding_size=8, action_type_embedding_size=4, hidden_size=16)
    node_embeddings = torch.randn(1, 8)
    node_key_to_index = {TARGET: 0}
    e_elasticsearch, e_wp_ninja = _exploit_pair()

    facts = _windows_host_facts(known_services=frozenset({"9200_windows_elasticsearch"}))
    compatibility = compute_compatibility_matrix([e_elasticsearch, e_wp_ninja], facts)
    assert not torch.allclose(compatibility[0], compatibility[1])  # precondition: features differ

    out = encoder.encode_descriptors(
        [e_elasticsearch, e_wp_ninja], node_embeddings, node_key_to_index, torch.device("cpu"), compatibility
    )
    assert not torch.allclose(out[0], out[1])


def test_windows_and_linux_privescs_now_differ_once_os_is_observed():
    windows_privesc = ActionDescriptor(
        f"privilege-escalation:{TARGET}:marla_repair_windows_root_privesc", "privilege_escalation", TARGET,
        {"process": None, "os": "windows"},
    )
    linux_privesc = ActionDescriptor(
        f"privilege-escalation:{TARGET}:pe_kernel", "privilege_escalation", TARGET,
        {"process": None, "os": "linux"},
    )
    # Both are "privilege_escalation, has_process=False, has_os=True" --
    # identical under the old parameter-presence-only features.
    assert torch.equal(parameter_features(windows_privesc.parameters), parameter_features(linux_privesc.parameters))

    facts = {
        TARGET: VisibleHostFacts(
            target_key=TARGET, reachable=True, compromised=True, access=AccessLevel.USER,
            known_os=frozenset({"windows"}), known_services=frozenset(), known_processes=frozenset(),
        )
    }
    compatibility = compute_compatibility_matrix([windows_privesc, linux_privesc], facts)
    assert not torch.allclose(compatibility[0], compatibility[1])  # os_known_present vs. os_known_incompatible

    encoder = ActionEncoder(node_embedding_size=8, action_type_embedding_size=4, hidden_size=16)
    node_embeddings = torch.randn(1, 8)
    out = encoder.encode_descriptors(
        [windows_privesc, linux_privesc], node_embeddings, {TARGET: 0}, torch.device("cpu"), compatibility
    )
    assert not torch.allclose(out[0], out[1])


def test_actions_remain_equivalent_when_genuinely_no_evidence_separates_them():
    """Not overcorrecting (spec section 9): if nothing has been observed
    yet, two exploits with identical type/parameter-shape SHOULD still be
    representationally equivalent -- the model must not hallucinate a
    distinction before any evidence exists.
    """
    e_elasticsearch, e_wp_ninja = _exploit_pair()
    facts = _windows_host_facts(known_services=frozenset())  # nothing scanned yet
    compatibility = compute_compatibility_matrix([e_elasticsearch, e_wp_ninja], facts)
    assert torch.allclose(compatibility[0], compatibility[1])  # both UNKNOWN, identical features

    encoder = ActionEncoder(node_embedding_size=8, action_type_embedding_size=4, hidden_size=16)
    node_embeddings = torch.randn(1, 8)
    out = encoder.encode_descriptors(
        [e_elasticsearch, e_wp_ninja], node_embeddings, {TARGET: 0}, torch.device("cpu"), compatibility
    )
    assert torch.allclose(out[0], out[1])
