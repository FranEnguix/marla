from pathlib import Path

import torch

from marla.config.loader import load_config
from marla.learning.checkpoint import load_checkpoint, save_checkpoint
from marla.learning.recurrent_policy import RecurrentPolicy

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_save_and_reload_round_trips_parameters(tmp_path):
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # perturb parameters so a naive "load did nothing" bug would be caught
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    before = {k: v.clone() for k, v in policy.state_dict().items()}

    checkpoint_path = tmp_path / "checkpoints" / "update-1.pt"
    save_checkpoint(checkpoint_path, policy, optimizer, update_count=1, environment_steps=128, config_hash="abc123")

    fresh_policy = RecurrentPolicy(config.policy)  # different random init
    fresh_optimizer = torch.optim.Adam(fresh_policy.parameters(), lr=1e-3)
    metadata = load_checkpoint(checkpoint_path, fresh_policy, fresh_optimizer)

    for key, value in before.items():
        assert torch.equal(value, fresh_policy.state_dict()[key])

    assert metadata.update_count == 1
    assert metadata.environment_steps == 128
    assert metadata.config_hash == "abc123"


def test_load_without_optimizer_still_restores_policy(tmp_path):
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "update-1.pt"
    save_checkpoint(checkpoint_path, policy, optimizer, update_count=5, environment_steps=64, config_hash="xyz")

    fresh_policy = RecurrentPolicy(config.policy)
    metadata = load_checkpoint(checkpoint_path, fresh_policy, optimizer=None)
    assert metadata.update_count == 5
    for key, value in policy.state_dict().items():
        assert torch.equal(value, fresh_policy.state_dict()[key])
