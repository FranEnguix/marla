"""Checkpoint save/load for the recurrent PPO policy + optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from marla.learning.recurrent_policy import RecurrentPolicy


@dataclass(frozen=True)
class CheckpointMetadata:
    update_count: int
    environment_steps: int
    config_hash: str


def save_checkpoint(
    path: str | Path,
    policy: RecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    update_count: int,
    environment_steps: int,
    config_hash: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update_count": update_count,
            "environment_steps": environment_steps,
            "config_hash": config_hash,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    policy: RecurrentPolicy,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointMetadata:
    data = torch.load(Path(path), map_location=map_location, weights_only=False)
    policy.load_state_dict(data["policy_state_dict"])
    if optimizer is not None and data.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(data["optimizer_state_dict"])
    return CheckpointMetadata(
        update_count=data["update_count"],
        environment_steps=data["environment_steps"],
        config_hash=data["config_hash"],
    )
