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

    for host in summary["hosts"]:
        assert host["access"] in ("none", "user", "root")
        assert isinstance(host["reachable"], bool)
        assert isinstance(host["known_services"], int)
        assert isinstance(host["known_processes"], int)
        assert isinstance(host["is_objective_target"], bool)
        assert host["target"].startswith("host-")


def test_observation_summary_is_json_serializable():
    import json

    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO, max_episode_steps=20, completion_reward=1.0, premature_finish_penalty=-1.0
    )
    state = adapter.reset(seed=1)
    summary = build_observation_summary(state)
    json.dumps(summary)  # must not raise
