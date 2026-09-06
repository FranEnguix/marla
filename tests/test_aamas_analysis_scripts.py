"""research/aamas2027's analysis scripts against the *current* marla
metrics schema.

These scripts live outside the ``marla`` package (under ``research/``) and
are not imported by anything in ``src/``, so a metrics-schema change (e.g.
moving a column from ``updates.csv`` to ``rollouts.csv``) can silently
break them without any of marla's own tests noticing -- exactly what
happened here: ``analyze.py``'s ``query_rate_over_training``/
``beta_over_training`` read ``updates.csv``'s ``actual_query_rate``/
``mean_beta`` columns, which moved to ``rollouts.csv`` (rollout-level
aggregates are no longer repeated onto every PPO minibatch's updates.csv
row); and ``evaluate_checkpoint.py`` called ``marla.metrics.writer``'s
``_decision_row`` with its old two-argument ``(config, record)`` signature,
which is now ``(record, advantage, return_target)`` -- silently wrong
argument types/count, not even a graceful skip.

These tests load the real scripts by file path (they are standalone,
outside any Python package) and run them against a real, tiny, currently-
produced run directory -- not a hand-built fixture -- so a future schema
change that these scripts don't handle will fail loudly here instead of
only producing an empty aggregate CSV during a real paper run.
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch

from marla.config.loader import load_config, parse_config
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.learning.rollout import ConsultationResult
from marla.learning.trainer import build_policy_and_optimizer, run_training_loop
from marla.metrics.writer import write_run_artifacts
from marla.runtime.device import resolve_device

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())
ANALYZE_PY = REPO_ROOT / "research" / "aamas2027" / "analyze.py"
EVALUATE_CHECKPOINT_PY = REPO_ROOT / "research" / "aamas2027" / "scripts" / "evaluate_checkpoint.py"


def _load_module(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_assisted_run_dir(tmp_path) -> Path:
    """A real, tiny, currently-produced assisted run (forced-accepting
    consult_fn, no SPADE needed) -- exercises the actual current
    rollouts.csv/updates.csv/decisions.csv schema, not a hand-written stub
    of it.
    """
    config = load_config(REPO_ROOT / "examples" / "assisted.yaml")
    data = config.model_dump()
    data["environment"]["scenario"] = SMALL_SCENARIO
    data["environment"]["max_episode_steps"] = 20
    data["policy"]["ppo"]["rollout_steps"] = 16
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["device"] = "cpu"
    data["experiment"]["run_id"] = "aamas-script-check"
    config = parse_config(data)

    device = torch.device("cpu")
    policy, optimizer = build_policy_and_optimizer(config, device, consultation_enabled=True)
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO, max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )

    async def consult_fn(legal_actions, episode_id, step, source_observation_id, observation):
        scores = {a.action_id: 1.0 / (i + 1) for i, a in enumerate(legal_actions)}
        return ConsultationResult(status="accepted", scores=scores, request_id=f"req-{episode_id}-{step}", latency_ms=42.0)

    result = asyncio.run(
        run_training_loop(
            policy, optimizer, adapter, "aamas-script-check", config.policy.ppo,
            config.policy.recurrent.sequence_length, num_rollouts=3, device=device, seed=1,
            consultation_enabled=True, consultation_cost=config.consultation.cost, consult_fn=consult_fn,
        )
    )

    run_dir = tmp_path / "runs" / "aamas2027_marla_full" / "marla-full-seed-999"
    now = datetime.now(timezone.utc)
    write_run_artifacts(run_dir, config, result, resolve_device("cpu"), now, now, status="completed")
    return run_dir


def test_analyze_query_rate_and_beta_over_training_read_rollouts_csv(tiny_assisted_run_dir, tmp_path, monkeypatch):
    analyze = _load_module(ANALYZE_PY, "aamas_analyze_under_test")
    # REPO_ROOT / run_info["run_dir"] must resolve to tiny_assisted_run_dir;
    # monkeypatching REPO_ROOT to the filesystem root lets an absolute
    # run_dir path pass through `REPO_ROOT / run_info["run_dir"]` unchanged
    # (Path.__truediv__ with an absolute right-hand side discards the left).
    monkeypatch.setattr(analyze, "REPO_ROOT", Path("/"))
    manifest = {"conditions": {"MARLA_FULL": {"runs": {999: {"run_dir": str(tiny_assisted_run_dir)}}}}}
    out_dir = tmp_path / "aggregate"
    out_dir.mkdir()

    analyze.query_rate_over_training(manifest, out_dir)
    analyze.beta_over_training(manifest, out_dir)

    with (out_dir / "query_rate_over_training.csv").open() as fh:
        query_rows = list(csv.DictReader(fh))
    with (out_dir / "beta_over_training.csv").open() as fh:
        beta_rows = list(csv.DictReader(fh))

    # One row per rollout (3 rollouts) -- not silently empty, which is
    # exactly what the pre-fix code produced (it read a column that no
    # longer exists on updates.csv, via `.get(...)`, which returns None
    # for every row and skips it without ever raising).
    assert len(query_rows) == 3
    assert len(beta_rows) == 3
    for row in query_rows:
        assert 0.0 <= float(row["actual_query_rate"]) <= 1.0
    for row in beta_rows:
        float(row["mean_beta"])  # must parse as a real number, not empty


@pytest.fixture
def tiny_baseline_run_dir(tmp_path) -> Path:
    """A real, tiny, currently-produced PPO_ONLY-shaped (baseline) run --
    no Plan Maker/consultation involved, so evaluate_checkpoint.py's
    PPO_ONLY condition (below) never needs to load a real language model.
    """
    from marla.learning.trainer import run_baseline_training

    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    # Absolute, matching how the real AAMAS configs do this (see e.g.
    # research/aamas2027/configs/ppo_only_seed101.yaml's own comment): the
    # checkpoint-evaluation harness reloads run_dir/config.yaml from
    # *inside* the run directory, so a relative scenario path would
    # resolve against the wrong base directory at that point.
    data["environment"]["scenario"] = SMALL_SCENARIO
    data["policy"]["ppo"]["rollout_steps"] = 12
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["device"] = "cpu"
    data["experiment"]["run_id"] = "aamas-script-check-ppo-only"
    config = parse_config(data)

    result = asyncio.run(run_baseline_training(config, SMALL_SCENARIO, num_rollouts=1, seed=1))
    run_dir = tmp_path / "runs" / "aamas2027_ppo_only" / "ppo-only-seed-999"
    now = datetime.now(timezone.utc)
    write_run_artifacts(run_dir, config, result, resolve_device("cpu"), now, now, status="completed")
    return run_dir


def test_evaluate_checkpoint_main_writes_decisions_csv_end_to_end(tiny_baseline_run_dir, tmp_path, monkeypatch):
    """Regression test: exercises evaluate_checkpoint.py's *actual* main()
    call site (not a reimplementation of it) end to end against a real
    checkpoint -- the old ``_decision_row(None, record)`` call (matching
    marla.metrics.writer._decision_row's former ``(config, record)``
    signature, now ``(record, advantage, return_target)``) would raise
    inside this exact call, not just mismatch types silently.
    """
    module = _load_module(EVALUATE_CHECKPOINT_PY, "aamas_evaluate_checkpoint_under_test")
    out_dir = tmp_path / "eval_out"

    monkeypatch.setattr(
        sys, "argv",
        [
            "evaluate_checkpoint.py",
            "--run-dir", str(tiny_baseline_run_dir),
            "--condition", "PPO_ONLY",
            "--seed-start", "5001",
            "--num-episodes", "2",
            "--out-dir", str(out_dir),
            "--training-seed", "999",
            "--device", "cpu",
        ],
    )
    module.main()

    with (out_dir / "decisions.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "evaluate_checkpoint.py wrote zero decision rows"
    for row in rows:
        assert row["condition"] == "PPO_ONLY"
        assert row["selected_action_id"]
        # PPO_ONLY makes no gradient update during evaluation -- gae_advantage/
        # return_target are meaningless here and must be left empty, not raise.
        assert row["gae_advantage"] == ""
        assert row["return_target"] == ""
    with (out_dir / "episodes.csv").open() as fh:
        episode_rows = list(csv.DictReader(fh))
    assert len(episode_rows) >= 1
