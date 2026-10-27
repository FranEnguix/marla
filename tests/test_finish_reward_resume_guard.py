"""Checkpoint-resume guard for FINISH-reward semantics: a resume must never silently continue under a
different reward objective; legacy checkpoints (no recorded semantics) need explicit acknowledgement;
evaluation loads are never affected."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from marla.config.loader import load_config
from marla.config.models import ObjectiveConfig
from marla.learning.checkpoint import (
    FinishRewardMismatchError,
    FinishRewardUnverifiedError,
    load_checkpoint,
    peek_checkpoint_metadata,
    save_checkpoint,
)
from marla.learning.lr_scheduler import build_scheduler
from marla.learning.recurrent_policy import RecurrentPolicy

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())

DEFAULT = ObjectiveConfig(type="capture_target", description="x").effective_finish_reward()
LEGACY = ObjectiveConfig(type="capture_target", description="x", completion_reward=1.0,
                         premature_finish_penalty=-1.0, premature_finish_penalty_per_remaining_target=0.0
                         ).effective_finish_reward()


def _obj(**kw):
    return ObjectiveConfig(type="capture_target", description="x", **kw).effective_finish_reward()


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


# --- effective semantics -----------------------------------------------------------------------------------


def test_effective_semantics_ignore_irrelevant_differences():
    assert _obj() == _obj(completion_reward_per_target=50.0) == DEFAULT
    # fixed completion_reward makes the per-target value irrelevant
    assert _obj(completion_reward=1.0, completion_reward_per_target=50.0) == _obj(completion_reward=1.0,
                                                                                 completion_reward_per_target=7.0)
    assert DEFAULT["successful_finish"] == {"mode": "per_target", "value": 50.0}
    assert DEFAULT["premature_finish"] == {"fixed": 0.0, "per_remaining_target": -100.0}


def test_effective_semantics_detect_real_differences():
    assert _obj(completion_reward=50.0) != DEFAULT  # fixed 50 is not 50 per target
    assert _obj(premature_finish_penalty=-1.0) != DEFAULT
    assert _obj(premature_finish_penalty_per_remaining_target=-20.0) != DEFAULT
    assert LEGACY != DEFAULT


# --- load_checkpoint -----------------------------------------------------------------------------------------


def test_new_checkpoints_record_semantics(tmp_path):
    path, _ = _save(tmp_path, DEFAULT)
    assert peek_checkpoint_metadata(path).finish_reward_semantics == DEFAULT


def test_resume_with_matching_semantics(tmp_path):
    path, config = _save(tmp_path, DEFAULT)
    meta = _load(path, config, restore_rng_state=True, expected_finish_reward=_obj(completion_reward_per_target=50.0))
    assert meta.finish_reward_semantics == DEFAULT


def test_resume_with_mismatched_semantics_is_refused(tmp_path):
    path, config = _save(tmp_path, LEGACY)
    with pytest.raises(FinishRewardMismatchError, match="different reward objective"):
        _load(path, config, restore_rng_state=True, expected_finish_reward=DEFAULT)


def test_resume_with_explicit_override_matching_checkpoint(tmp_path):
    path, config = _save(tmp_path, LEGACY)
    _load(path, config, restore_rng_state=True, expected_finish_reward=LEGACY)


def test_legacy_checkpoint_resume_requires_acknowledgement(tmp_path, caplog):
    path, config = _save(tmp_path, None)
    with pytest.raises(FinishRewardUnverifiedError, match="--accept-unverified-finish-reward"):
        _load(path, config, restore_rng_state=True, expected_finish_reward=DEFAULT)
    with caplog.at_level(logging.WARNING, logger="marla.learning.checkpoint"):
        meta = _load(path, config, restore_rng_state=True, expected_finish_reward=DEFAULT,
                     allow_unverified_finish_reward=True)
    assert meta.finish_reward_semantics is None
    assert "explicitly accepted" in caplog.text


def test_acknowledgement_never_overrides_a_recorded_mismatch(tmp_path):
    path, config = _save(tmp_path, LEGACY)
    with pytest.raises(FinishRewardMismatchError):
        _load(path, config, restore_rng_state=True, expected_finish_reward=DEFAULT, allow_unverified_finish_reward=True)


def test_evaluation_load_of_legacy_or_mismatched_checkpoint_is_unaffected(tmp_path):
    for semantics in (None, LEGACY):
        path, config = _save(tmp_path, semantics, name=f"{semantics is None}.pt")
        _load(path, config)  # plain evaluation load: no reward check
        _load(path, config, expected_finish_reward=DEFAULT)  # not a resume -> still no check


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

    # same effective reward -> resumes
    ok = _marla(["run", str(_write_config(tmp_path, "guard-b", 24, completion_reward_per_target=50.0)), "--resume", str(ckpt)])
    assert ok.returncode == 0, ok.stdout + ok.stderr

    # different reward -> refused before any run directory is created
    bad = _marla(["run", str(_write_config(tmp_path, "guard-c", 24, premature_finish_penalty=-1.0)), "--resume", str(ckpt)])
    assert bad.returncode != 0 and "different reward objective" in bad.stdout + bad.stderr
    assert not any((tmp_path / "runs").rglob("guard-c"))

    # legacy checkpoint (no recorded semantics) -> needs explicit acknowledgement
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    data.pop("finish_reward_semantics")
    legacy = tmp_path / "legacy.pt"
    torch.save(data, legacy)
    refused = _marla(["run", str(_write_config(tmp_path, "guard-d", 24)), "--resume", str(legacy)])
    assert refused.returncode != 0 and "--accept-unverified-finish-reward" in refused.stdout + refused.stderr
    accepted = _marla(["run", str(_write_config(tmp_path, "guard-e", 24)), "--resume", str(legacy),
                       "--accept-unverified-finish-reward"])
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
