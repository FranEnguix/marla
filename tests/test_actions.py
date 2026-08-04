import pytest

from marla.environment.actions import (
    FINISH_ACTION_ID,
    build_legal_actions,
    finish_descriptor,
    host_target_key,
    parse_host_target_key,
    resolve_action_target,
)


class FakeNasimEnv:
    """Minimal stand-in for NASimEmuEnv's action-list surface."""

    def __init__(self, exploit_list, privesc_list):
        self.exploit_list = exploit_list
        self.privesc_list = privesc_list


def test_host_target_key_round_trip():
    assert host_target_key(1, 2) == "host-1-2"
    assert parse_host_target_key("host-1-2") == (1, 2)


def test_parse_host_target_key_rejects_malformed():
    with pytest.raises(ValueError):
        parse_host_target_key("not-a-host-key")


def test_finish_descriptor_is_stable_and_has_no_target():
    descriptor = finish_descriptor()
    assert descriptor.action_id == FINISH_ACTION_ID == "finish"
    assert descriptor.is_finish is True
    assert descriptor.target_key is None


def test_build_legal_actions_covers_all_hosts_and_action_kinds():
    env = FakeNasimEnv(
        exploit_list=[("exploit-e1", {"service": "http"}), ("exploit-e2", {"service": "ssh"})],
        privesc_list=[("privesc-p1", {"process": "cron"})],
    )
    host_addresses = [(1, 0), (1, 1)]

    descriptors = build_legal_actions(env, host_addresses)
    ids = [d.action_id for d in descriptors]

    # 4 scans + 2 exploits + 1 privesc per host, plus one finish.
    assert len(descriptors) == 2 * (4 + 2 + 1) + 1
    assert ids[-1] == "finish"

    assert "service-scan:host-1-0" in ids
    assert "os-scan:host-1-1" in ids
    assert "subnet-scan:host-1-0" in ids
    assert "process-scan:host-1-1" in ids
    assert "exploit:host-1-0:exploit-e1" in ids
    assert "exploit:host-1-1:exploit-e2" in ids
    assert "privilege-escalation:host-1-0:privesc-p1" in ids

    exploit_descriptor = next(d for d in descriptors if d.action_id == "exploit:host-1-0:exploit-e1")
    assert exploit_descriptor.parameters == {"service": "http", "os": None}
    assert exploit_descriptor.action_type == "exploit"
    assert exploit_descriptor.target_key == "host-1-0"


def test_action_ids_are_unique():
    env = FakeNasimEnv(
        exploit_list=[("e1", {}), ("e2", {})],
        privesc_list=[("p1", {})],
    )
    descriptors = build_legal_actions(env, [(1, 0), (1, 1), (2, 0)])
    ids = [d.action_id for d in descriptors]
    assert len(ids) == len(set(ids))


def test_resolve_action_target_round_trips_with_build_legal_actions():
    env = FakeNasimEnv(
        exploit_list=[("e1", {}), ("e2", {})],
        privesc_list=[("p1", {})],
    )
    host_addresses = [(1, 0), (1, 1)]
    descriptors = build_legal_actions(env, host_addresses)

    seen_indices = set()
    for descriptor in descriptors:
        if descriptor.is_finish:
            assert resolve_action_target(env, descriptor.action_id) is None
            continue
        target, index = resolve_action_target(env, descriptor.action_id)
        assert target == parse_host_target_key(descriptor.target_key)
        assert 0 <= index < 4 + len(env.exploit_list) + len(env.privesc_list)
        seen_indices.add((target, index))

    # every (target, action_list_index) combination should be distinct
    assert len(seen_indices) == len(descriptors) - 1  # minus finish


def test_resolve_action_target_rejects_unknown_exploit():
    env = FakeNasimEnv(exploit_list=[("e1", {})], privesc_list=[])
    with pytest.raises(ValueError):
        resolve_action_target(env, "exploit:host-1-0:not-a-real-exploit")


def test_resolve_action_target_rejects_malformed_id():
    env = FakeNasimEnv(exploit_list=[], privesc_list=[])
    with pytest.raises(ValueError):
        resolve_action_target(env, "not-a-valid-action-id")
