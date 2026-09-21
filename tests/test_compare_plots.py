"""marla.metrics.compare_plots -- multi-run variability comparison plots
(never a single run's own time series; see marla.metrics.plots for that).
"""

from __future__ import annotations

import json

import pytest

from marla.metrics.compare_plots import (
    COMPARISON_METRICS,
    generate_comparison_plots,
    load_run_metrics,
    plot_metric_comparison,
)


def _write_run(
    tmp_path, name: str, *, total_training_seconds=None, cpu_mean=None, energy_kwh=None, carbon_enabled=True,
    peak_rss_mib=None, cpu_total_seconds=None,
):
    run_dir = tmp_path / name
    run_dir.mkdir()
    if total_training_seconds is not None:
        (run_dir / "summary.json").write_text(json.dumps({"total_training_seconds": total_training_seconds}), encoding="utf-8")
    if cpu_mean is not None or peak_rss_mib is not None or cpu_total_seconds is not None:
        resource_summary = {"overall": {}}
        if cpu_mean is not None:
            resource_summary["overall"]["cpu_process_pct"] = {"mean": cpu_mean, "p95": cpu_mean, "max": cpu_mean}
        if peak_rss_mib is not None:
            resource_summary["peak_rss_mib"] = peak_rss_mib
            resource_summary["mean_rss_mib"] = peak_rss_mib * 0.8
        if cpu_total_seconds is not None:
            resource_summary["process_cpu_total_seconds"] = cpu_total_seconds
        (run_dir / "resource_summary.json").write_text(json.dumps(resource_summary), encoding="utf-8")
    if energy_kwh is not None:
        carbon_dir = run_dir / "carbon"
        carbon_dir.mkdir()
        (carbon_dir / "carbon_summary.json").write_text(
            json.dumps({"enabled": carbon_enabled, "energy_consumed_kwh": energy_kwh, "emissions_kg_co2eq": energy_kwh * 0.035}),
            encoding="utf-8",
        )
    return run_dir


def test_load_run_metrics_reads_all_three_sources(tmp_path):
    run_dir = _write_run(tmp_path, "seed1", total_training_seconds=100.0, cpu_mean=42.0, energy_kwh=0.01)
    m = load_run_metrics(run_dir)
    assert m.total_training_seconds == 100.0
    assert m.cpu_process_pct_mean == 42.0
    assert m.energy_kwh == 0.01
    assert m.co2eq_kg == pytest.approx(0.00035)


def test_load_run_metrics_reads_absolute_units_as_primary_fields(tmp_path):
    """Absolute quantities (RSS MiB, CPU-seconds) are the primary
    reporting fields -- read directly as top-level resource_summary.json
    keys, distinct from the secondary cpu_process_pct_mean percentage."""
    run_dir = _write_run(tmp_path, "seed1", peak_rss_mib=2048.0, cpu_total_seconds=305.5, cpu_mean=42.0)
    m = load_run_metrics(run_dir)
    assert m.peak_rss_mib == 2048.0
    assert m.mean_rss_mib == pytest.approx(2048.0 * 0.8)
    assert m.cpu_process_total_seconds == 305.5
    # both the absolute and the (still-present, secondary) percentage field
    # are available side by side.
    assert m.cpu_process_pct_mean == 42.0


def test_generate_comparison_plots_prioritizes_absolute_metrics_first(tmp_path):
    """COMPARISON_METRICS orders absolute quantities before percentage
    appendix metrics -- the reporting-emphasis rule this module exists to
    enforce."""
    absolute_metrics = {"total_training_seconds", "cpu_process_total_seconds", "peak_rss_mib", "mean_rss_mib", "energy_kwh", "co2eq_kg"}
    percentage_metrics = {"cpu_process_pct_mean", "gpu_util_pct_mean"}
    names = [m[0] for m in COMPARISON_METRICS]
    last_absolute_index = max(names.index(m) for m in absolute_metrics if m in names)
    first_percentage_index = min(names.index(m) for m in percentage_metrics if m in names)
    assert last_absolute_index < first_percentage_index


def test_generate_comparison_plots_writes_absolute_unit_comparison_files(tmp_path):
    run_dirs = [
        _write_run(tmp_path, "seed909", total_training_seconds=100.0, peak_rss_mib=1500.0, cpu_total_seconds=90.0),
        _write_run(tmp_path, "seed919", total_training_seconds=110.0, peak_rss_mib=1600.0, cpu_total_seconds=95.0),
    ]
    plots_dir = tmp_path / "compare_plots"
    written = generate_comparison_plots(run_dirs, plots_dir)
    names = {p.name for p in written}
    assert "compare_peak_rss_mib.png" in names
    assert "compare_cpu_process_total_seconds.png" in names


def test_load_run_metrics_missing_files_are_none_not_zero(tmp_path):
    run_dir = tmp_path / "bare"
    run_dir.mkdir()
    m = load_run_metrics(run_dir)
    assert m.total_training_seconds is None
    assert m.cpu_process_pct_mean is None
    assert m.energy_kwh is None
    assert m.co2eq_kg is None


def test_load_run_metrics_carbon_disabled_is_none_not_a_fabricated_number(tmp_path):
    run_dir = _write_run(tmp_path, "run", energy_kwh=0.02, carbon_enabled=False)
    m = load_run_metrics(run_dir)
    assert m.energy_kwh is None


def test_load_run_metrics_default_label_disambiguates_same_named_run_dirs(tmp_path):
    trial_a = tmp_path / "trial_a" / "seed909"
    trial_a.mkdir(parents=True)
    trial_b = tmp_path / "trial_b" / "seed909"
    trial_b.mkdir(parents=True)
    label_a = load_run_metrics(trial_a).label
    label_b = load_run_metrics(trial_b).label
    assert label_a != label_b
    assert "trial_a" in label_a and "trial_b" in label_b


def test_load_run_metrics_explicit_label_overrides_default(tmp_path):
    run_dir = tmp_path / "seed909"
    run_dir.mkdir()
    assert load_run_metrics(run_dir, label="my custom label").label == "my custom label"


def test_plot_metric_comparison_returns_none_when_no_run_has_the_metric(tmp_path):
    runs = [load_run_metrics(_write_run(tmp_path, "a")), load_run_metrics(_write_run(tmp_path, "b"))]
    result = plot_metric_comparison(runs, "energy_kwh", "Energy (kWh)", "Estimated energy consumed", tmp_path)
    assert result is None


def test_plot_metric_comparison_single_run_still_plots_a_point(tmp_path, monkeypatch):
    import marla.metrics.compare_plots as compare_module

    figs = []
    real_save = compare_module._save

    def spy(fig, path):
        figs.append(fig)
        return real_save(fig, path)

    monkeypatch.setattr(compare_module, "_save", spy)

    runs = [load_run_metrics(_write_run(tmp_path, "only-run", total_training_seconds=250.0))]
    result = plot_metric_comparison(runs, "total_training_seconds", "Wall-clock seconds", "Total training time", tmp_path)
    assert result is not None and result.is_file()
    ax = figs[-1].axes[0]
    assert ax.get_title() == "Total training time"
    # No mean+/-std marker with only one point -- nothing to average.
    assert not any("mean" in t.get_text().lower() for t in ax.get_legend().get_texts())


def test_plot_metric_comparison_multiple_runs_shows_mean_and_std(tmp_path, monkeypatch):
    import marla.metrics.compare_plots as compare_module

    figs = []
    real_save = compare_module._save

    def spy(fig, path):
        figs.append(fig)
        return real_save(fig, path)

    monkeypatch.setattr(compare_module, "_save", spy)

    runs = [
        load_run_metrics(_write_run(tmp_path, f"seed{i}", total_training_seconds=100.0 + i))
        for i in range(3)
    ]
    plot_metric_comparison(runs, "total_training_seconds", "Wall-clock seconds", "Total training time", tmp_path)
    ax = figs[-1].axes[0]
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "mean +/- std" in legend_labels
    assert len(ax.collections) >= 1  # the per-run scatter


def test_generate_comparison_plots_end_to_end(tmp_path):
    run_dirs = [
        _write_run(tmp_path, "seed909", total_training_seconds=100.0, cpu_mean=50.0, energy_kwh=0.01),
        _write_run(tmp_path, "seed919", total_training_seconds=110.0, cpu_mean=55.0, energy_kwh=0.011),
    ]
    plots_dir = tmp_path / "compare_plots"
    written = generate_comparison_plots(run_dirs, plots_dir)
    assert written
    names = {p.name for p in written}
    assert "compare_total_training_seconds.png" in names
    assert "compare_cpu_process_pct_mean.png" in names
    assert "compare_energy_kwh.png" in names
    # Metrics no run has (evaluation seconds, GPU fields) are simply
    # absent, not a fabricated empty file.
    assert "compare_total_evaluation_seconds.png" not in names
    assert len(written) <= len(COMPARISON_METRICS)


def test_generate_comparison_plots_rejects_mismatched_label_count(tmp_path):
    run_dirs = [_write_run(tmp_path, "a", total_training_seconds=1.0), _write_run(tmp_path, "b", total_training_seconds=2.0)]
    with pytest.raises(ValueError):
        generate_comparison_plots(run_dirs, tmp_path / "out", labels=["only-one-label"])
