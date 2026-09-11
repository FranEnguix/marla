"""Per-agent CodeCarbon energy/CO2eq tracking -- see monitoring/carbon.py.

Most tests here use the REAL codecarbon package (it is a declared,
installed dependency of this dev environment, and running it directly is
both faster to write and a stronger guarantee than mocking its internals)
-- CI does not need real carbon-measurement HARDWARE for any of these
(CodeCarbon's own TDP/CPU-load estimation fallback works identically on
any machine). A few tests specifically simulate "codecarbon not
installed" by monkeypatching the module's own import-success flag,
matching the spec's "mock where appropriate" -- there is no other
meaningful thing to mock here.
"""

from __future__ import annotations

import time

import pytest

from marla.config.models import CarbonConfig
from marla.monitoring import carbon as carbon_module
from marla.monitoring.carbon import (
    CarbonSummary,
    CarbonTracker,
    CarbonTrackerUnavailableError,
    derived_efficiency_metrics,
    write_carbon_summary,
)


def _cfg(**overrides) -> CarbonConfig:
    return CarbonConfig(enabled=True, measure_power_secs=1.0, **overrides)


def test_carbon_disabled_has_zero_behavioral_impact(tmp_path):
    tracker = CarbonTracker(CarbonConfig(enabled=False), tmp_path / "carbon", "proj")
    tracker.start()
    tracker.start_task("training")
    tracker.stop_task("training")
    summary = tracker.stop()
    assert summary.enabled is False
    assert all(v is None for k, v in summary.to_dict().items() if k != "enabled")
    # No directory/files created at all -- a fully inert no-op.
    assert not (tmp_path / "carbon").exists()


def test_enabled_tracker_lifecycle_is_correct(tmp_path):
    tracker = CarbonTracker(_cfg(), tmp_path / "carbon", "proj")
    tracker.start()
    tracker.start_task("training")
    time.sleep(0.1)
    tracker.stop_task("training")
    summary = tracker.stop()
    assert summary.enabled is True
    assert summary.codecarbon_version is not None
    assert summary.tracker_class == "EmissionsTracker"
    assert summary.duration_seconds is not None and summary.duration_seconds > 0
    assert summary.energy_consumed_kwh is not None and summary.energy_consumed_kwh >= 0
    assert summary.training_energy_kwh is not None
    assert summary.evaluation_energy_kwh is None  # never opened


def test_training_task_closed_correctly_and_aggregated(tmp_path):
    tracker = CarbonTracker(_cfg(), tmp_path / "carbon", "proj")
    tracker.start()
    tracker.start_task("training")
    time.sleep(0.05)
    tracker.stop_task("training")
    tracker.start_task("training")  # a SECOND training segment -- the interleaved-eval pattern
    time.sleep(0.05)
    tracker.stop_task("training")
    summary = tracker.stop()
    assert summary.training_energy_kwh is not None and summary.training_energy_kwh > 0
    assert summary.evaluation_energy_kwh is None
    # Both segments' energy contributed (sum >= a single-segment amount) --
    # not overwritten by the second start_task("training") call.
    assert len(tracker._segments) == 2
    assert all(s.task_name == "training" for s in tracker._segments)


def test_evaluation_task_closed_correctly(tmp_path):
    tracker = CarbonTracker(_cfg(), tmp_path / "carbon", "proj")
    tracker.start()
    tracker.start_task("training")
    time.sleep(0.05)
    tracker.stop_task("training")
    tracker.start_task("evaluation")
    time.sleep(0.05)
    tracker.stop_task("evaluation")
    summary = tracker.stop()
    assert summary.training_energy_kwh is not None
    assert summary.evaluation_energy_kwh is not None
    assert summary.evaluation_energy_kwh > 0


def test_repeated_same_named_task_across_the_run_does_not_corrupt_totals(tmp_path):
    """Regression test: CodeCarbon's own task API does not support
    reusing one task name within a tracker instance (observed directly --
    a second start_task("training") after an earlier stop_task("training")
    corrupted that segment's delta before CarbonTracker's unique-name
    workaround). training/evaluation/training must all measure real,
    nonzero, non-corrupted energy.
    """
    tracker = CarbonTracker(_cfg(), tmp_path / "carbon", "proj")
    tracker.start()
    tracker.start_task("training")
    time.sleep(0.1)
    tracker.stop_task("training")
    tracker.start_task("evaluation")
    time.sleep(0.05)
    tracker.stop_task("evaluation")
    tracker.start_task("training")
    time.sleep(0.05)
    tracker.stop_task("training")
    summary = tracker.stop()
    assert summary.duration_seconds == pytest.approx(0.2, abs=0.15)
    # Independently-measured segments (each its own CodeCarbon
    # measurement window) need not sum to EXACTLY the top-level total --
    # a loose relative tolerance confirms "not wildly corrupted", not
    # bit-exact equality.
    assert summary.energy_consumed_kwh == pytest.approx(
        (summary.training_energy_kwh or 0) + (summary.evaluation_energy_kwh or 0), rel=0.05
    )


def test_exception_mid_task_still_closes_the_tracker(tmp_path):
    tracker = CarbonTracker(_cfg(), tmp_path / "carbon", "proj")
    tracker.start()
    tracker.start_task("training")
    try:
        raise RuntimeError("simulated training crash")
    except RuntimeError:
        pass
    finally:
        summary = tracker.stop()
    assert summary.enabled is True
    assert summary.training_energy_kwh is not None  # the open task was closed, not dropped


def test_summary_has_the_documented_schema(tmp_path):
    tracker = CarbonTracker(_cfg(), tmp_path / "carbon", "proj")
    tracker.start()
    tracker.start_task("training")
    tracker.stop_task("training")
    summary = tracker.stop()
    expected_keys = {
        "enabled", "codecarbon_version", "tracker_class", "tracking_mode", "measure_power_secs",
        "duration_seconds", "energy_consumed_kwh", "emissions_kg_co2eq", "emissions_g_co2eq",
        "training_energy_kwh", "training_emissions_kg_co2eq", "evaluation_energy_kwh",
        "evaluation_emissions_kg_co2eq", "country_name", "country_iso_code", "region",
        "cloud_provider", "cloud_region", "carbon_intensity_source", "cpu_model", "gpu_model",
        "ram_total_gb",
    }
    assert set(summary.to_dict()) == expected_keys


def test_zero_successful_finishes_gives_null_not_a_crash_or_sentinel():
    summary = CarbonSummary(
        enabled=True, codecarbon_version="3.3.1", tracker_class="EmissionsTracker", tracking_mode="process",
        measure_power_secs=1.0, duration_seconds=10.0, energy_consumed_kwh=0.001, emissions_kg_co2eq=0.0004,
        emissions_g_co2eq=0.4, training_energy_kwh=0.001, training_emissions_kg_co2eq=0.0004,
        evaluation_energy_kwh=None, evaluation_emissions_kg_co2eq=None, country_name="Switzerland",
        country_iso_code="CHE", region="zurich", cloud_provider=None, cloud_region=None,
        carbon_intensity_source="codecarbon_auto", cpu_model="x", gpu_model=None, ram_total_gb=32.0,
    )
    metrics = derived_efficiency_metrics(summary, environment_steps=2048, root_auc=0.6, successful_finish_count=0)
    assert metrics["g_co2eq_per_successful_finish"] is None
    # Every other metric with a real, nonzero denominator IS populated.
    assert metrics["g_co2eq_per_1k_transitions"] is not None
    assert metrics["root_auc_per_kwh"] is not None


def test_zero_environment_steps_gives_null_not_a_crash():
    summary = CarbonSummary(
        enabled=True, codecarbon_version=None, tracker_class=None, tracking_mode=None, measure_power_secs=None,
        duration_seconds=None, energy_consumed_kwh=None, emissions_kg_co2eq=None, emissions_g_co2eq=None,
        training_energy_kwh=None, training_emissions_kg_co2eq=None, evaluation_energy_kwh=None,
        evaluation_emissions_kg_co2eq=None, country_name=None, country_iso_code=None, region=None,
        cloud_provider=None, cloud_region=None, carbon_intensity_source=None, cpu_model=None, gpu_model=None,
        ram_total_gb=None,
    )
    metrics = derived_efficiency_metrics(summary, environment_steps=0, root_auc=None, successful_finish_count=0)
    assert all(v is None for v in metrics.values())


def test_write_carbon_summary_includes_agent_id(tmp_path):
    summary = CarbonSummary(
        enabled=False, **{f: None for f in CarbonSummary.__dataclass_fields__ if f != "enabled"}
    )
    write_carbon_summary(tmp_path, summary, agent_id="trial_0007-seed909")
    import json

    payload = json.loads((tmp_path / "carbon" / "carbon_summary.json").read_text())
    assert payload["agent_id"] == "trial_0007-seed909"
    assert payload["enabled"] is False


def test_write_carbon_summary_includes_efficiency_when_given(tmp_path):
    summary = CarbonSummary(
        enabled=False, **{f: None for f in CarbonSummary.__dataclass_fields__ if f != "enabled"}
    )
    write_carbon_summary(tmp_path, summary, efficiency={"g_co2eq_per_1k_transitions": None}, agent_id="a")
    import json

    payload = json.loads((tmp_path / "carbon" / "carbon_summary.json").read_text())
    assert "efficiency" in payload


def test_no_external_api_output_and_local_csv_only(tmp_path, monkeypatch):
    """Asserts the exact kwargs CarbonTracker passes to CodeCarbon:
    output_methods=[CSV] only (never OutputMethod.API, and never the
    deprecated save_to_api kwarg at all) regardless of config -- never
    configurable to leak to CodeCarbon's hosted API.
    """
    captured = {}
    real_tracker_cls = carbon_module.EmissionsTracker

    class _SpyTracker(real_tracker_cls):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(carbon_module, "EmissionsTracker", _SpyTracker)
    tracker = CarbonTracker(_cfg(), tmp_path / "carbon", "proj")
    tracker.start()
    tracker.stop()
    assert "save_to_api" not in captured
    assert captured["output_methods"] == [carbon_module.OutputMethod.CSV]


def test_explicit_location_config_uses_offline_tracker(tmp_path):
    tracker = CarbonTracker(_cfg(country_iso_code="CHE", region="zurich"), tmp_path / "carbon", "proj")
    tracker.start()
    summary = tracker.stop()
    assert tracker.tracker_class == "OfflineEmissionsTracker"
    assert summary.carbon_intensity_source == "explicit_config"


def test_no_explicit_location_uses_auto_resolution_tracker(tmp_path):
    tracker = CarbonTracker(_cfg(), tmp_path / "carbon", "proj")
    tracker.start()
    summary = tracker.stop()
    assert tracker.tracker_class == "EmissionsTracker"
    assert summary.carbon_intensity_source == "codecarbon_auto"


def test_codecarbon_not_installed_raises_a_clear_error_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(carbon_module, "_CODECARBON_IMPORTABLE", False)
    tracker = CarbonTracker(_cfg(), tmp_path / "carbon", "proj")
    with pytest.raises(CarbonTrackerUnavailableError):
        tracker.start()


def test_codecarbon_not_installed_is_harmless_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(carbon_module, "_CODECARBON_IMPORTABLE", False)
    tracker = CarbonTracker(CarbonConfig(enabled=False), tmp_path / "carbon", "proj")
    tracker.start()  # must NOT raise -- disabled means codecarbon is never touched
    summary = tracker.stop()
    assert summary.enabled is False


def test_per_agent_identifier_is_never_shared_across_two_trackers(tmp_path):
    """Two independently-constructed CarbonTrackers (e.g. seed909 and
    seed919 of the same Optuna trial) must never share underlying
    CodeCarbon run/task state -- each is its own agent.
    """
    t1 = CarbonTracker(_cfg(), tmp_path / "carbon909", "trial7-seed909")
    t2 = CarbonTracker(_cfg(), tmp_path / "carbon919", "trial7-seed919")
    t1.start()
    t2.start()
    t1.start_task("training")
    t2.start_task("training")
    time.sleep(0.05)
    t1.stop_task("training")
    t2.stop_task("training")
    s1, s2 = t1.stop(), t2.stop()
    assert s1.training_energy_kwh is not None
    assert s2.training_energy_kwh is not None
    # Independent output directories -- no shared/overwritten CSV.
    assert (tmp_path / "carbon909" / "emissions.csv").exists()
    assert (tmp_path / "carbon919" / "emissions.csv").exists()
