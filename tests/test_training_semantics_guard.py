"""PPO training-semantics checkpoint guard (v0.11.0).

v0.11.0 changed how a rollout becomes optimizer steps (minibatches of ~fixed
REAL-transition count, ``marla.learning.ppo.partition_minibatches``). Weight
shapes did not change, so a v0.10.x checkpoint loads into a v0.11 policy:

A. v0.11 checkpoint  -> v0.11 training resume: allowed
B. v0.10.x checkpoint -> v0.11 evaluation:      allowed
C. v0.10.x checkpoint -> v0.11 training resume: refused, clearly
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from marla.config.loader import load_config
from marla.evaluation.checkpoint_eval import evaluate_checkpoint
from marla.learning.checkpoint import (
    CheckpointResumeError,
    TrainingSemanticsMismatchError,
    load_checkpoint,
    peek_checkpoint_metadata,
    save_checkpoint,
)
from marla.learning.lr_scheduler import build_scheduler
from marla.learning.ppo import TRAINING_SEMANTICS_VERSION
from marla.learning.recurrent_policy import RecurrentPolicy
from tests.test_checkpoint_eval import make_fake_run_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


def _save_resumable(path: Path, training_semantics_version="current"):
    """A full, resumable checkpoint (scheduler state included). ``None``
    removes the field, exactly like a checkpoint written by v0.10.x."""
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, config.policy.ppo)
    save_checkpoint(
        path, policy, optimizer, update_count=2, environment_steps=64, config_hash="h", next_episode_seed=9,
        rng_state=torch.get_rng_state(), scheduler=scheduler,
        scheduler_config=config.policy.ppo.optimizer.scheduler.model_dump(),
    )
    if training_semantics_version != "current":
        data = torch.load(path, weights_only=False)
        if training_semantics_version is None:
            del data["training_semantics_version"]
        else:
            data["training_semantics_version"] = training_semantics_version
        torch.save(data, path)
    return config


def test_current_version_is_v0_11_real_transition_batching():
    assert TRAINING_SEMANTICS_VERSION == 2


def test_checkpoint_records_training_semantics_version(tmp_path):
    _save_resumable(tmp_path / "c.pt")
    assert torch.load(tmp_path / "c.pt", weights_only=False)["training_semantics_version"] == TRAINING_SEMANTICS_VERSION
    assert peek_checkpoint_metadata(tmp_path / "c.pt").training_semantics_version == TRAINING_SEMANTICS_VERSION


def test_a_current_checkpoint_resumes(tmp_path):
    config = _save_resumable(tmp_path / "c.pt")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    metadata = load_checkpoint(tmp_path / "c.pt", policy, optimizer, restore_rng_state=True)
    assert metadata.training_semantics_version == TRAINING_SEMANTICS_VERSION
    assert metadata.rng_state_restored is True


@pytest.mark.parametrize("saved", [None, 1, TRAINING_SEMANTICS_VERSION + 1])
def test_a_different_semantics_checkpoint_is_refused_for_resume(tmp_path, saved):
    config = _save_resumable(tmp_path / "legacy.pt", training_semantics_version=saved)
    policy = RecurrentPolicy(config.policy)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    rng_before = torch.get_rng_state()
    with pytest.raises(TrainingSemanticsMismatchError) as exc:
        load_checkpoint(tmp_path / "legacy.pt", policy, torch.optim.Adam(policy.parameters()), restore_rng_state=True)
    assert isinstance(exc.value, CheckpointResumeError)
    message = str(exc.value)
    assert f"training_semantics_version={saved!r}" in message and "evaluation" in message
    # Refused before anything was loaded: weights and global RNG untouched.
    assert all(torch.equal(v, policy.state_dict()[k]) for k, v in before.items())
    assert torch.equal(rng_before, torch.get_rng_state())


def test_a_legacy_checkpoint_still_loads_for_evaluation_with_identical_weights(tmp_path):
    config = _save_resumable(tmp_path / "legacy.pt", training_semantics_version=None)
    saved = torch.load(tmp_path / "legacy.pt", weights_only=False)["policy_state_dict"]
    policy = RecurrentPolicy(config.policy)
    metadata = load_checkpoint(tmp_path / "legacy.pt", policy, restore_rng_state=False)
    assert metadata.training_semantics_version is None
    assert saved.keys() == policy.state_dict().keys()
    assert all(torch.equal(v, policy.state_dict()[k]) for k, v in saved.items())


@pytest.mark.asyncio
async def test_evaluate_checkpoint_accepts_a_legacy_checkpoint(tmp_path):
    run_dir = make_fake_run_dir(tmp_path)
    data = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    del data["training_semantics_version"]
    torch.save(data, run_dir / "checkpoint.pt")
    for mode in ("GREEDY", "STOCHASTIC_POLICY"):
        result = await evaluate_checkpoint(run_dir, seed_start=1, num_episodes=1, device="cpu", policy_mode=mode)
        assert len(result.summaries) == 1


@pytest.mark.integration
def test_marla_run_resume_refuses_a_legacy_checkpoint_before_creating_a_run_directory(tmp_path):
    _save_resumable(tmp_path / "legacy.pt", training_semantics_version=None)
    data = yaml.safe_load((REPO_ROOT / "examples" / "baseline.yaml").read_text())
    run_id = "semantics-guard-legacy-resume"
    data["experiment"]["run_id"] = run_id
    data["environment"]["scenario"] = SMALL_SCENARIO
    data["policy"]["ppo"]["total_environment_steps"] = 128
    config_path = tmp_path / "resume.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    run_dir = REPO_ROOT / "runs" / "ppo_baseline" / run_id
    assert not run_dir.exists()

    result = subprocess.run(
        [sys.executable, "-m", "marla", "run", "--resume", str(tmp_path / "legacy.pt"), str(config_path)],
        cwd=REPO_ROOT, env={**os.environ, "MARLA_RL_ORCHESTRATOR_PASSWORD": "testpass"},
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "training_semantics_version=None" in result.stdout + result.stderr
    assert not run_dir.exists()
