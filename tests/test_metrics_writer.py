"""Metrics writer (spec section 21): a real tiny baseline training run's
TrainingResult, written to disk and checked for structure/content."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from marla.config.loader import load_config
from marla.learning.trainer import run_baseline_training
from marla.metrics.writer import write_run_artifacts
from marla.runtime.device import resolve_device

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


def _tiny_config():
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    data["policy"]["ppo"]["rollout_steps"] = 12
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["device"] = "cpu"
    from marla.config.loader import parse_config

    return parse_config(data)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture
def written_run_dir(tmp_path):
    import asyncio

    config = _tiny_config()
    result = asyncio.run(run_baseline_training(config, SMALL_SCENARIO, num_rollouts=2, seed=1))
    resolved_device = resolve_device(config.device)
    run_dir = tmp_path / "run"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    write_run_artifacts(run_dir, config, result, resolved_device, start, end, status="completed")
    return run_dir, config, result


def test_write_run_artifacts_creates_every_expected_file(written_run_dir):
    run_dir, _config, _result = written_run_dir
    for name in ("config.yaml", "metadata.json", "episodes.csv", "decisions.csv", "updates.csv", "summary.json"):
        assert (run_dir / name).is_file(), name


def test_metadata_json_has_required_fields(written_run_dir):
    run_dir, config, _result = written_run_dir
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == config.experiment.run_id
    assert metadata["variant"] == "baseline"
    assert metadata["execution_mode"] == "local"
    assert metadata["device_resolved"] == "cpu"
    assert metadata["status"] == "completed"
    assert metadata["config_hash"]
    assert "torch" in metadata["dependency_versions"]
    assert metadata["gatekeeper"] is None
    assert metadata["plan_maker"] is None


def test_episodes_csv_has_one_row_per_episode_with_expected_columns(written_run_dir):
    run_dir, _config, result = written_run_dir
    rows = _read_csv(run_dir / "episodes.csv")
    assert len(rows) == len(result.episode_summaries)
    assert len(rows) >= 1
    row = rows[0]
    assert row["variant"] == "baseline"
    assert row["consultation_count"] == "0"  # baseline never consults
    assert float(row["benchmark_return"]) == float(row["nasimemu_return"])


def test_decisions_csv_has_one_row_per_step_with_derived_fields(written_run_dir):
    run_dir, _config, result = written_run_dir
    rows = _read_csv(run_dir / "decisions.csv")
    assert len(rows) == result.environment_steps
    row = rows[0]
    assert row["base_top_action_id"]
    assert row["selected_action_id"]
    assert row["selected_action_base_rank"] == "1" or int(row["selected_action_base_rank"]) >= 1
    # Not populated in this release (see writer.py's module docstring).
    assert row["schema_revision_count"] == ""
    assert row["action_success"] == ""
    assert row["artifact_path"] == ""


def test_decisions_csv_omitted_when_record_decisions_is_false(tmp_path):
    import asyncio

    from marla.config.loader import parse_config

    config = _tiny_config()
    data = config.model_dump()
    data["metrics"]["record_decisions"] = False
    config = parse_config(data)

    result = asyncio.run(run_baseline_training(config, SMALL_SCENARIO, num_rollouts=1, seed=1))
    resolved_device = resolve_device(config.device)
    run_dir = tmp_path / "run"
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    write_run_artifacts(run_dir, config, result, resolved_device, now, now, status="completed")

    assert not (run_dir / "decisions.csv").exists()
    assert (run_dir / "episodes.csv").exists()


def test_updates_csv_has_run_id_and_expected_columns(written_run_dir):
    run_dir, config, result = written_run_dir
    rows = _read_csv(run_dir / "updates.csv")
    assert len(rows) == len(result.update_metrics)
    assert len(rows) >= 1
    for row in rows:
        assert row["run_id"] == config.experiment.run_id
        float(row["policy_loss"])  # must parse as a number
        assert row["mean_beta"] == ""  # baseline never queries


def test_summary_json_has_required_aggregate_fields(written_run_dir):
    run_dir, _config, result = written_run_dir
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["episode_count"] == len(result.episode_summaries)
    assert summary["total_training_environment_steps"] == result.environment_steps
    assert 0.0 <= summary["goal_success_rate"] <= 1.0
    assert summary["mean_plan_maker_latency_ms"] is None  # baseline never consults
    assert summary["schema_rejection_rate"] is None


def test_write_run_artifacts_handles_none_result(tmp_path):
    """A construction failure before training happened still gets a metadata.json."""
    config = _tiny_config()
    resolved_device = resolve_device(config.device)
    run_dir = tmp_path / "run"
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    write_run_artifacts(run_dir, config, None, resolved_device, now, now, status="failed")

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert _read_csv(run_dir / "episodes.csv") == []


def test_config_yaml_is_a_valid_redacted_dump(written_run_dir):
    run_dir, config, _result = written_run_dir
    dumped = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert dumped["experiment"]["run_id"] == config.experiment.run_id
    assert dumped["rl_orchestrator"]["password_env"] == config.rl_orchestrator.password_env


@pytest.fixture
def written_assisted_run_dir(tmp_path):
    """A real assisted TrainingResult (fake consult_fn, no SPADE needed), written to disk."""
    import asyncio

    from marla.config.loader import parse_config
    from marla.environment.nasimemu_adapter import NasimEmuAdapter
    from marla.learning.recurrent_policy import RecurrentPolicy
    from marla.learning.rollout import ConsultationResult
    from marla.learning.trainer import build_policy_and_optimizer, run_training_loop

    config = load_config(REPO_ROOT / "examples" / "assisted.yaml")
    data = config.model_dump()
    data["environment"]["scenario"] = SMALL_SCENARIO
    data["environment"]["max_episode_steps"] = 50
    data["policy"]["ppo"]["rollout_steps"] = 12
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["device"] = "cpu"
    config = parse_config(data)

    device = __import__("torch").device("cpu")
    policy, optimizer = build_policy_and_optimizer(config, device, consultation_enabled=True)
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )

    async def consult_fn(legal_actions, episode_id, step, source_observation_id, observation):
        scores = {a.action_id: 1.0 / (i + 1) for i, a in enumerate(legal_actions)}
        return ConsultationResult(status="accepted", scores=scores, request_id=f"req-{episode_id}-{step}", latency_ms=42.0)

    result = asyncio.run(
        run_training_loop(
            policy, optimizer, adapter, config.experiment.run_id or config.experiment.name,
            config.policy.ppo, config.policy.recurrent.sequence_length, num_rollouts=2,
            device=device, seed=1, consultation_enabled=True, consultation_cost=config.consultation.cost,
            consult_fn=consult_fn,
        )
    )

    resolved_device = resolve_device(config.device)
    run_dir = tmp_path / "run"
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    write_run_artifacts(run_dir, config, result, resolved_device, now, now, status="completed")
    return run_dir, config, result


def test_decisions_csv_populates_plan_maker_fields_for_assisted_runs(written_assisted_run_dir):
    run_dir, _config, result = written_assisted_run_dir
    rows = _read_csv(run_dir / "decisions.csv")
    assert len(rows) == result.environment_steps
    queried_rows = [r for r in rows if r["queried"] == "True"]
    assert queried_rows, "expected at least one queried decision with a forced consult_fn"
    for row in queried_rows:
        assert row["response_status"] == "accepted"
        assert row["response_latency_ms"] == "42.0"
        assert row["plan_maker_top_action_id"]
        assert row["beta"] != ""


def test_summary_json_reports_plan_maker_latency_and_beta_for_assisted_runs(written_assisted_run_dir):
    run_dir, _config, _result = written_assisted_run_dir
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_consultations"] > 0
    assert summary["mean_plan_maker_latency_ms"] == 42.0
    assert summary["mean_beta"] is not None


def test_updates_csv_populates_query_fields_for_assisted_runs(written_assisted_run_dir):
    run_dir, _config, _result = written_assisted_run_dir
    rows = _read_csv(run_dir / "updates.csv")
    assert all(row["mean_query_probability"] != "" for row in rows)
    assert all(0.0 <= float(row["actual_query_rate"]) <= 1.0 for row in rows)


@pytest.mark.integration
def test_generate_plots_includes_consultation_plots_for_assisted_runs(written_assisted_run_dir):
    from marla.metrics.plots import generate_plots

    run_dir, _config, _result = written_assisted_run_dir
    written = generate_plots(run_dir, run_dir / "plots")
    names = {p.name for p in written}
    assert "consultation_activity.png" in names
    assert "query_behavior.png" in names
    assert "policy_confidence.png" in names
