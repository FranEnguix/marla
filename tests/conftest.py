import copy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


@pytest.fixture
def baseline_config_path() -> Path:
    return EXAMPLES_DIR / "baseline.yaml"


@pytest.fixture
def assisted_config_path() -> Path:
    return EXAMPLES_DIR / "assisted.yaml"


def write_yaml(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def minimal_config_dict(scenario_path: str) -> dict:
    """A minimal, valid baseline config dict, deep-copyable for mutation in tests."""
    return copy.deepcopy(
        {
            "schema_version": "1.0",
            "experiment": {"name": "unit-test", "seed": 1},
            "execution": {"mode": "local"},
            "device": "cpu",
            "xmpp": {"server": "localhost"},
            "environment": {
                "mode": "simulation",
                "scenario": scenario_path,
                "max_episode_steps": 50,
            },
            "objective": {"type": "capture_target", "description": "test objective"},
            "policy": {
                "algorithm": "recurrent_ppo",
                "graph_encoder": {"type": "graphsage", "hidden_size": 32, "layers": 2},
                "action_encoder": {"hidden_size": 32, "action_type_embedding_size": 8},
                "recurrent": {"hidden_size": 32, "sequence_length": 8},
                "ppo": {
                    "total_environment_steps": 1000,
                    "rollout_steps": 64,
                    "epochs": 2,
                    "minibatch_sequences": 2,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_epsilon": 0.2,
                    "value_coefficient": 0.5,
                    "query_entropy_coefficient": 0.01,
                    "action_entropy_coefficient": 0.01,
                    "max_grad_norm": 0.5,
                    "learning_rate": 0.0003,
                },
            },
            "consultation": {"mode": "disabled"},
            "rl_orchestrator": {"alias": "rl_orchestrator", "jid": "rl-orchestrator@localhost"},
        }
    )
