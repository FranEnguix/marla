"""Checkpoint resume (research/aamas2027's staged-training design): a
second `marla run --resume` invocation continues training rather than
restarting -- policy/optimizer state, environment-step/update counters,
and episode-seed progression all carry forward.

Two SEPARATE subprocesses (see test_local_runtime.py's module docstring:
SPADE's Container is a process-wide singleton, so two run_local() calls in
one interpreter reuse a closed event loop and fail) -- this also matches
real usage exactly: two real `marla run` invocations, not a mocked resume.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from marla.learning.checkpoint import load_checkpoint, peek_checkpoint_metadata, save_checkpoint
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.config.loader import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
# sm_entry_dmz_two_subnets.v2.yaml is not universally solvable (see
# marla.scenario.solvability's analysis -- a real, separate finding this
# purely-mechanical resume test is not about) and would now fail `marla
# run`'s preflight check; sm_entry_dmz_one_subnet.v2.yaml has a vacuous
# objective (0 sensitive-host probability) and is trivially solvable,
# matching the sibling subprocess tests in test_local_runtime.py /
# test_local_runtime_assisted.py / test_distributed.py.
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


def _write_tiny_baseline_config(tmp_path: Path, run_id: str, total_environment_steps: int) -> Path:
    data = yaml.safe_load((REPO_ROOT / "examples" / "baseline.yaml").read_text())
    data["experiment"]["run_id"] = run_id
    data["environment"]["scenario"] = SMALL_SCENARIO
    data["environment"]["max_episode_steps"] = 5
    data["policy"]["ppo"]["total_environment_steps"] = total_environment_steps
    data["policy"]["ppo"]["rollout_steps"] = 12
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["metrics"]["eval_episodes"] = 0  # keep this test focused on the resume mechanics, not eval

    config_path = tmp_path / f"{run_id}.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def _run_marla(args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "MARLA_RL_ORCHESTRATOR_PASSWORD": "testpass", **(env_extra or {})}
    return subprocess.run(
        [sys.executable, "-m", "marla", *args], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=90
    )


# --- unit-level: checkpoint.py's new fields --------------------------------


def test_save_checkpoint_round_trips_resume_fields(tmp_path):
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint.pt"

    torch.manual_seed(42)
    rng_snapshot = torch.get_rng_state()
    save_checkpoint(
        checkpoint_path, policy, optimizer, update_count=5, environment_steps=512, config_hash="abc",
        next_episode_seed=137, rng_state=rng_snapshot,
    )

    fresh_policy = RecurrentPolicy(config.policy)
    metadata = load_checkpoint(checkpoint_path, fresh_policy, restore_rng_state=False)
    assert metadata.next_episode_seed == 137
    assert metadata.rng_state_restored is False  # not requested


def test_load_checkpoint_restores_rng_state_only_when_requested(tmp_path):
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint.pt"

    torch.manual_seed(42)
    saved_state = torch.get_rng_state()
    save_checkpoint(
        checkpoint_path, policy, optimizer, update_count=1, environment_steps=1, config_hash="x",
        next_episode_seed=1, rng_state=saved_state,
    )

    torch.manual_seed(999)  # perturb the global RNG state to something else
    metadata = load_checkpoint(checkpoint_path, policy, restore_rng_state=True)
    assert metadata.rng_state_restored is True
    assert torch.equal(torch.get_rng_state(), saved_state)


def test_peek_checkpoint_metadata_does_not_need_a_policy(tmp_path):
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path, policy, optimizer, update_count=7, environment_steps=999, config_hash="y",
        next_episode_seed=42, rng_state=None,
    )
    metadata = peek_checkpoint_metadata(checkpoint_path)
    assert metadata.update_count == 7
    assert metadata.environment_steps == 999
    assert metadata.next_episode_seed == 42


def test_load_checkpoint_without_resume_fields_still_works(tmp_path):
    """A checkpoint saved before resume support existed (no next_episode_seed/
    rng_state keys at all) must still load -- backward compatibility."""
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "old_style.pt"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update_count": 3,
            "environment_steps": 256,
            "config_hash": "old",
        },
        checkpoint_path,
    )
    metadata = load_checkpoint(checkpoint_path, policy)
    assert metadata.next_episode_seed is None
    assert metadata.environment_steps == 256


# --- end-to-end: two real `marla run` processes ---------------------------


@pytest.mark.integration
def test_resume_continues_rather_than_restarts(tmp_path):
    stage_a_config = _write_tiny_baseline_config(tmp_path, run_id="resume-stage-a", total_environment_steps=12)
    result_a = _run_marla(["run", str(stage_a_config)])
    assert result_a.returncode == 0, result_a.stdout + result_a.stderr
    assert "12 environment steps" in result_a.stdout

    checkpoint_a = REPO_ROOT / "runs" / "ppo_baseline" / "resume-stage-a" / "checkpoint.pt"
    metadata_a = peek_checkpoint_metadata(checkpoint_a)
    assert metadata_a.environment_steps == 12
    assert metadata_a.next_episode_seed is not None

    stage_b_config = _write_tiny_baseline_config(tmp_path, run_id="resume-stage-b", total_environment_steps=24)
    result_b = _run_marla(["run", "--resume", str(checkpoint_a), str(stage_b_config)])
    assert result_b.returncode == 0, result_b.stdout + result_b.stderr
    assert "already done" in result_b.stdout
    assert "24 environment steps" in result_b.stdout  # cumulative, not just the new 12

    run_dir_b = REPO_ROOT / "runs" / "ppo_baseline" / "resume-stage-b"
    with (run_dir_b / "episodes.csv").open() as fh:
        episode_seeds = [int(r["seed"]) for r in csv.DictReader(fh)]
    # Every seed in stage B's own episodes.csv must be >= stage A's ending
    # seed -- i.e. continuing the sequence, never repeating stage A's seeds.
    assert min(episode_seeds) >= metadata_a.next_episode_seed

    metadata_b = peek_checkpoint_metadata(run_dir_b / "checkpoint.pt")
    assert metadata_b.environment_steps == 24
    assert metadata_b.update_count > metadata_a.update_count  # continued numbering, not reset

    # Clean up the shared runs/ppo_baseline/ directory this test writes into
    # (matches other tests in this suite using real example configs).
    import shutil

    shutil.rmtree(REPO_ROOT / "runs" / "ppo_baseline" / "resume-stage-a", ignore_errors=True)
    shutil.rmtree(run_dir_b, ignore_errors=True)


@pytest.mark.integration
def test_resume_rejects_a_checkpoint_already_past_the_target(tmp_path):
    stage_a_config = _write_tiny_baseline_config(tmp_path, run_id="resume-overshoot-a", total_environment_steps=24)
    result_a = _run_marla(["run", str(stage_a_config)])
    assert result_a.returncode == 0, result_a.stdout + result_a.stderr

    checkpoint_a = REPO_ROOT / "runs" / "ppo_baseline" / "resume-overshoot-a" / "checkpoint.pt"
    stage_b_config = _write_tiny_baseline_config(tmp_path, run_id="resume-overshoot-b", total_environment_steps=12)
    result_b = _run_marla(["run", "--resume", str(checkpoint_a), str(stage_b_config)])
    assert result_b.returncode == 1
    assert "already has" in result_b.stdout + result_b.stderr

    import shutil

    shutil.rmtree(REPO_ROOT / "runs" / "ppo_baseline" / "resume-overshoot-a", ignore_errors=True)
