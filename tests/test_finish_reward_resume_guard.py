"""Checkpoint-resume guard for FINISH-reward semantics: a resume must never silently continue under a
different reward objective; evaluation loads are never affected."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from marla.config.loader import load_config
from marla.config.models import ObjectiveConfig
from marla.learning.checkpoint import FinishRewardMismatchError, load_checkpoint, peek_checkpoint_metadata, save_checkpoint
from marla.learning.lr_scheduler import build_scheduler
from marla.learning.recurrent_policy import RecurrentPolicy

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "tests/data/scenarios/sm_entry_dmz_one_target.v2.yaml").resolve())

DEFAULT = ObjectiveConfig(type="capture_target", description="x").effective_finish_reward()
SCALED = ObjectiveConfig(type="capture_target", description="x", successful_finish_reward_per_target=2.0,
                         premature_finish_reward_per_remaining_target=-0.5).effective_finish_reward()


def _save(tmp_path, semantics, name="ckpt.pt"):
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, config.policy.ppo)
    path = tmp_path / name
    save_checkpoint(path, policy, optimizer, update_count=1, environment_steps=10, config_hash="h",
                    next_episode_seed=5, rng_state=torch.get_rng_state(), scheduler=scheduler,
                    scheduler_config={"type": "linear"}, finish_reward_semantics=semantics)
    return path, config


def _load(path, config, **kw):
    policy = RecurrentPolicy(config.policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    return load_checkpoint(path, policy, optimizer, **kw)


def test_semantics_distinguish_coefficients():
    assert DEFAULT != SCALED
    assert DEFAULT == ObjectiveConfig(type="capture_target", description="y").effective_finish_reward()


def test_new_checkpoints_record_semantics(tmp_path):
    path, _ = _save(tmp_path, DEFAULT)
    assert peek_checkpoint_metadata(path).finish_reward_semantics == DEFAULT


def test_resume_with_matching_semantics(tmp_path):
    path, config = _save(tmp_path, DEFAULT)
    assert _load(path, config, restore_rng_state=True, expected_finish_reward=DEFAULT).finish_reward_semantics == DEFAULT


@pytest.mark.parametrize("saved", [SCALED, None])
def test_resume_with_different_or_missing_semantics_is_refused(tmp_path, saved):
    path, config = _save(tmp_path, saved)
    with pytest.raises(FinishRewardMismatchError, match="different reward objective"):
        _load(path, config, restore_rng_state=True, expected_finish_reward=DEFAULT)


def test_evaluation_load_is_unaffected(tmp_path):
    path, config = _save(tmp_path, SCALED)
    _load(path, config)
    _load(path, config, expected_finish_reward=DEFAULT)  # not a resume -> no check


# --- CLI integration (real `marla run --resume` subprocesses) -------------------------------------------------


def _write_config(tmp_path, run_id, total_steps, **objective):
    data = yaml.safe_load((REPO_ROOT / "examples" / "baseline.yaml").read_text())
    data["experiment"]["run_id"] = run_id
    data["environment"]["scenario"] = SMALL_SCENARIO
    data["environment"]["max_episode_steps"] = 5
    data["policy"]["ppo"].update(total_environment_steps=total_steps, steps_per_env=12, epochs=1, minibatch_sequences=2)
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["metrics"]["eval_episodes"] = 0
    data["metrics"]["output_directory"] = str(tmp_path / "runs")
    data["objective"].update(objective)
    path = tmp_path / f"{run_id}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _marla(args):
    env = {**os.environ, "MARLA_RL_ORCHESTRATOR_PASSWORD": "testpass"}
    return subprocess.run([sys.executable, "-m", "marla", *args], cwd=REPO_ROOT, env=env, capture_output=True,
                          text=True, timeout=120)


def test_cli_resume_guard_end_to_end(tmp_path):
    first = _marla(["run", str(_write_config(tmp_path, "guard-a", 12))])
    assert first.returncode == 0, first.stdout + first.stderr
    ckpt = next((tmp_path / "runs").rglob("checkpoint.pt"))
    assert peek_checkpoint_metadata(ckpt).finish_reward_semantics == DEFAULT

    ok = _marla(["run", str(_write_config(tmp_path, "guard-b", 24, successful_finish_reward_per_target=1.0)),
                 "--resume", str(ckpt)])
    assert ok.returncode == 0, ok.stdout + ok.stderr

    bad = _marla(["run", str(_write_config(tmp_path, "guard-c", 24, premature_finish_reward_per_remaining_target=-2.0)),
                  "--resume", str(ckpt)])
    assert bad.returncode != 0 and "different reward objective" in bad.stdout + bad.stderr
    assert not any((tmp_path / "runs").rglob("guard-c"))
