from pathlib import Path

from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.environment.observation_summary import build_observation_summary

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


def test_observation_summary_shape_and_values():
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO, max_episode_steps=20, completion_reward=1.0, premature_finish_penalty=-1.0
    )
    state = adapter.reset(seed=1)
    summary = build_observation_summary(state)

    assert "hosts" in summary
    assert len(summary["hosts"]) == len(state.host_addresses)
    assert isinstance(summary["sensitive_hosts_total"], int)
    assert isinstance(summary["sensitive_hosts_with_root_access"], int)
    assert summary["sensitive_hosts_with_root_access"] <= summary["sensitive_hosts_total"]

    for host in summary["hosts"]:
        assert host["access"] in ("none", "user", "root")
        assert isinstance(host["reachable"], bool)
        assert isinstance(host["compromised"], bool)
        assert isinstance(host["known_os"], list)
        assert isinstance(host["known_services"], list)
        assert isinstance(host["known_processes"], list)
        assert isinstance(host["is_sensitive_target"], bool)
        assert host["target"].startswith("host-")


def test_observation_summary_counts_sensitive_hosts_with_root_access():
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO, max_episode_steps=20, completion_reward=1.0, premature_finish_penalty=-1.0
    )
    state = adapter.reset(seed=1)
    summary = build_observation_summary(state)

    expected_total = sum(1 for h in summary["hosts"] if h["is_sensitive_target"])
    expected_captured = sum(1 for h in summary["hosts"] if h["is_sensitive_target"] and h["access"] == "root")
    assert summary["sensitive_hosts_total"] == expected_total
    assert summary["sensitive_hosts_with_root_access"] == expected_captured


def test_observation_summary_is_json_serializable():
    import json

    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO, max_episode_steps=20, completion_reward=1.0, premature_finish_penalty=-1.0
    )
    state = adapter.reset(seed=1)
    summary = build_observation_summary(state)
    json.dumps(summary)  # must not raise
