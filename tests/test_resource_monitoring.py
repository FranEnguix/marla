"""Resource telemetry (CPU/RAM/GPU) -- see monitoring/resources.py.

Runs on whatever hardware this test happens to execute on: CI/CPU-only
machines must pass with every GPU field null, and a machine with a real
GPU (this repo's own dev box has one) exercises the real NVML/torch.cuda
path. Neither case is skipped -- the assertions are written to hold in
both.
"""

from __future__ import annotations

import time

import pytest
import torch

from marla.monitoring.resources import ResourceMonitor, ResourceSample, machine_report


def test_disabled_monitor_collects_nothing_and_every_method_is_a_no_op():
    monitor = ResourceMonitor(enabled=False)
    monitor.start()
    time.sleep(0.05)
    monitor.set_context(phase="rollout_collection")
    monitor.stop()
    assert monitor.samples == []
    assert monitor.to_rows() == []
    assert monitor.summarize() == {}


def test_enabled_monitor_collects_samples_at_roughly_the_configured_interval():
    monitor = ResourceMonitor(sampling_interval_seconds=0.1, enabled=True)
    monitor.start()
    time.sleep(0.45)
    monitor.stop()
    # At least 3 samples in ~0.45s at a 0.1s interval; generous lower bound
    # to avoid flaking under CI scheduling jitter.
    assert len(monitor.samples) >= 3
    assert all(isinstance(s, ResourceSample) for s in monitor.samples)


def test_set_context_tags_subsequent_samples_not_earlier_ones():
    monitor = ResourceMonitor(sampling_interval_seconds=0.05, enabled=True)
    monitor.set_context(phase="rollout_collection", global_env_step=0, ppo_update=1)
    monitor.start()
    time.sleep(0.15)
    monitor.set_context(phase="ppo_update")
    time.sleep(0.15)
    monitor.stop()
    phases = {s.phase for s in monitor.samples}
    assert "rollout_collection" in phases
    assert "ppo_update" in phases
    # Every sample tagged "ppo_update" keeps the ppo_update index set earlier
    # (set_context only overwrites fields explicitly passed).
    for s in monitor.samples:
        if s.phase == "ppo_update":
            assert s.ppo_update == 1


def test_every_sample_has_finite_cpu_and_ram_fields():
    monitor = ResourceMonitor(sampling_interval_seconds=0.05, enabled=True)
    monitor.start()
    time.sleep(0.2)
    monitor.stop()
    assert monitor.samples
    for s in monitor.samples:
        assert s.cpu_process_pct is not None and s.cpu_process_pct >= 0.0
        assert s.cpu_system_pct is not None and s.cpu_system_pct >= 0.0
        assert s.cpu_logical_count is not None and s.cpu_logical_count >= 1
        assert s.ram_rss_mb is not None and s.ram_rss_mb > 0.0
        assert s.ram_system_total_mb is not None and s.ram_system_total_mb > 0.0
        assert 0.0 <= s.ram_system_pct <= 100.0


def test_gpu_fields_are_consistent_with_cuda_availability():
    monitor = ResourceMonitor(sampling_interval_seconds=0.05, enabled=True)
    monitor.start()
    time.sleep(0.15)
    monitor.stop()
    assert monitor.samples
    for s in monitor.samples:
        if not torch.cuda.is_available():
            # CPU-only: torch.cuda.* fields must be None, never a crash or
            # a fabricated zero pretending to be a real reading.
            assert s.torch_cuda_allocated_mb is None
            assert s.torch_cuda_reserved_mb is None
        else:
            assert s.torch_cuda_allocated_mb is not None
            assert s.torch_cuda_allocated_mb >= 0.0
            assert s.torch_cuda_reserved_mb is not None
        # gpu_util_pct/gpu_memory_used_mb are populated together (real NVML
        # reading) or both None (no NVML/no GPU) -- never one without the
        # other, which would imply a half-successful query.
        assert (s.gpu_util_pct is None) == (s.gpu_memory_used_mb is None)


def test_summarize_reports_overall_and_per_phase_stats():
    monitor = ResourceMonitor(sampling_interval_seconds=0.05, enabled=True)
    monitor.set_context(phase="rollout_collection")
    monitor.start()
    time.sleep(0.15)
    monitor.set_context(phase="ppo_update")
    time.sleep(0.15)
    monitor.stop()
    summary = monitor.summarize()
    assert summary["sample_count"] == len(monitor.samples)
    assert summary["sampling_interval_seconds"] == pytest.approx(0.05)
    assert "cpu_process_pct" in summary["overall"]
    for stat in ("mean", "p95", "max"):
        assert stat in summary["overall"]["cpu_process_pct"]
    assert set(summary["by_phase"]) <= {"rollout_collection", "ppo_update"}


def test_negative_or_zero_sampling_interval_rejected():
    with pytest.raises(ValueError):
        ResourceMonitor(sampling_interval_seconds=0.0)
    with pytest.raises(ValueError):
        ResourceMonitor(sampling_interval_seconds=-1.0)


def test_stop_before_start_and_double_stop_are_harmless():
    monitor = ResourceMonitor(sampling_interval_seconds=0.1, enabled=True)
    monitor.stop()  # never started
    monitor.start()
    time.sleep(0.05)
    monitor.stop()
    monitor.stop()  # already stopped
    assert monitor.samples  # at least the collection from the one start/stop cycle


def test_machine_report_has_expected_keys_and_no_private_paths_or_hostnames():
    report = machine_report()
    for key in (
        "os", "python_version", "marla_version", "git_commit", "torch_version",
        "cuda_available", "cuda_version", "cudnn_version", "gpu_count", "gpu_models",
        "nvml_available", "cpu_model", "cpu_logical_count", "cpu_physical_count", "total_ram_mb",
    ):
        assert key in report
    assert report["cuda_available"] == torch.cuda.is_available()
    assert report["gpu_count"] == (torch.cuda.device_count() if torch.cuda.is_available() else 0)
    # No private/machine-identifying leakage: hostname or an absolute
    # filesystem path must never appear anywhere in the report's values.
    import getpass
    import socket

    hostname = socket.gethostname()
    text_blob = str(report)
    assert hostname not in text_blob
    assert getpass.getuser() not in text_blob
    assert "/home/" not in text_blob


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resource_artifacts_and_plots_generate_from_a_real_tiny_run(tmp_path):
    """End-to-end: a real (tiny) training run with resource monitoring
    enabled writes resources.csv/resource_summary.json, and
    metrics.plots.generate_plots renders all 4 resource plots from them.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    from marla.config.loader import load_config, parse_config
    from marla.learning.trainer import run_baseline_training
    from marla.metrics.plots import generate_plots
    from marla.metrics.writer import write_run_artifacts
    from marla.runtime.device import resolve_device

    REPO_ROOT = Path(__file__).resolve().parent.parent
    SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())

    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    data["policy"]["ppo"]["steps_per_env"] = 12
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["device"] = "cpu"
    data["metrics"]["eval_episodes"] = 0
    data["metrics"]["resource_monitoring"] = {"enabled": True, "sampling_interval_seconds": 0.05}
    config = parse_config(data)

    result = await run_baseline_training(config, SMALL_SCENARIO, num_rollouts=3, seed=1)
    assert result.resource_monitor is not None
    assert result.resource_monitor.samples

    run_dir = tmp_path / "run"
    resolved_device = resolve_device(config.device)
    now = datetime.now(timezone.utc)
    write_run_artifacts(run_dir, config, result, resolved_device, now, now, status="completed")

    assert (run_dir / "resources.csv").is_file()
    assert (run_dir / "resource_summary.json").is_file()

    written = generate_plots(run_dir, run_dir / "plots")
    names = {p.name for p in written}
    assert "resource_cpu_over_time.png" in names
    assert "resource_ram_over_time.png" in names
    # GPU plots are conditionally present (depend on this test machine's
    # hardware) -- assert presence only when torch.cuda.is_available().
    if torch.cuda.is_available():
        assert "resource_gpu_memory_over_time.png" in names
