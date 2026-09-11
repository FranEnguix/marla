"""Multi-run comparison plots -- variability ACROSS runs/seeds/algorithms,
never a single run's own time series (that stays in :mod:`marla.metrics.plots`).

Each comparable metric here is already a single scalar summary PER RUN
(total wall-clock seconds, mean CPU%, total energy kWh, ...) --
comparing N runs means comparing N such scalars, which is a genuinely
different statistical question from "how did this one run change over
time." A per-run scalar with a handful of comparison runs (a paper's
typical 3 seeds, say) is exactly the case this project has already
decided, elsewhere, never to overclaim: no boxplot quartile structure is
presented for it (a boxplot needs enough points for quartiles to mean
anything), and no fabricated confidence interval either. Every
comparison here is instead: each run's own value as a visible point,
plus the mean +/- population std across those points as a simple visual
summary -- accurate for any N >= 1, and it never implies more precision
than N points actually support.

Used by ``marla compare RUN_DIR [RUN_DIR ...]``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RunMetrics:
    """One run's scalar comparison metrics, each ``None`` when that run's
    artifacts don't have it (an older run directory, a disabled feature,
    ...) -- never a fabricated 0.0.
    """

    label: str
    total_training_seconds: float | None = None
    total_collection_seconds: float | None = None
    total_optimization_seconds: float | None = None
    total_evaluation_seconds: float | None = None
    cpu_process_pct_mean: float | None = None
    ram_rss_mb_mean: float | None = None
    gpu_util_pct_mean: float | None = None
    gpu_memory_used_mb_mean: float | None = None
    energy_kwh: float | None = None
    co2eq_kg: float | None = None


# (attribute name, axis label incl. unit, short title)
COMPARISON_METRICS: list[tuple[str, str, str]] = [
    ("total_training_seconds", "Wall-clock seconds", "Total training time"),
    ("total_collection_seconds", "Wall-clock seconds", "Rollout collection time"),
    ("total_optimization_seconds", "Wall-clock seconds", "PPO optimization time"),
    ("total_evaluation_seconds", "Wall-clock seconds", "Evaluation time"),
    ("cpu_process_pct_mean", "Process CPU % (can exceed 100%)", "Mean process CPU utilization"),
    ("ram_rss_mb_mean", "RAM (MB)", "Mean process RSS"),
    ("gpu_util_pct_mean", "GPU utilization %", "Mean GPU utilization"),
    ("gpu_memory_used_mb_mean", "GPU memory (MB)", "Mean GPU memory used"),
    ("energy_kwh", "Energy (kWh)", "Estimated energy consumed"),
    ("co2eq_kg", "CO2eq (kg)", "Estimated CO2eq emissions"),
]


def load_run_metrics(run_dir: Path, label: str | None = None) -> RunMetrics:
    """Reads whichever of summary.json/resource_summary.json/carbon
    summary.json exist for ``run_dir`` -- missing files or fields become
    ``None``, never 0.0 or a crash (a run directory is never required to
    have every optional artifact: carbon tracking and resource monitoring
    are both individually optional).
    """
    run_dir = Path(run_dir)
    # Default label includes the parent directory name too -- a bare
    # run_dir.name (e.g. "seed909") is ambiguous across two different
    # experiments/trials that happen to reuse the same seed-named
    # subdirectory, which real run layouts in this project do.
    label = label or f"{run_dir.parent.name}/{run_dir.name}"

    summary: dict = {}
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    resource_summary: dict = {}
    resource_path = run_dir / "resource_summary.json"
    if resource_path.is_file():
        resource_summary = json.loads(resource_path.read_text(encoding="utf-8")).get("overall", {})

    carbon_summary: dict = {}
    carbon_path = run_dir / "carbon" / "carbon_summary.json"
    if carbon_path.is_file():
        carbon_summary = json.loads(carbon_path.read_text(encoding="utf-8"))
        if not carbon_summary.get("enabled"):
            carbon_summary = {}

    def _res(key: str) -> float | None:
        return resource_summary.get(key, {}).get("mean") if key in resource_summary else None

    return RunMetrics(
        label=label,
        total_training_seconds=summary.get("total_training_seconds"),
        total_collection_seconds=summary.get("total_collection_seconds"),
        total_optimization_seconds=summary.get("total_optimization_seconds"),
        total_evaluation_seconds=summary.get("total_evaluation_seconds"),
        cpu_process_pct_mean=_res("cpu_process_pct"),
        ram_rss_mb_mean=_res("ram_rss_mb"),
        gpu_util_pct_mean=_res("gpu_util_pct"),
        gpu_memory_used_mb_mean=_res("gpu_memory_used_mb"),
        energy_kwh=carbon_summary.get("energy_consumed_kwh"),
        co2eq_kg=carbon_summary.get("emissions_kg_co2eq"),
    )


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_comparison(runs: list[RunMetrics], metric: str, ylabel: str, title: str, plots_dir: Path) -> Path | None:
    """Per-run points (so N=1 or N=2 still shows real data, not a
    degenerate box) plus a mean +/- std marker when N >= 2 -- never a
    traditional box-and-whisker plot here, since this project's own
    typical N (a handful of seeds) is too small for quartiles to mean
    anything (see this module's own docstring). Returns ``None`` (no file
    written) when no run has this metric at all.
    """
    values = [(r.label, getattr(r, metric)) for r in runs]
    valid = [(label, v) for label, v in values if v is not None]
    if not valid:
        return None

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(valid) + 3), 4.5))
    xs = list(range(len(valid)))
    ys = [v for _, v in valid]
    ax.scatter(xs, ys, color="tab:blue", zorder=3, label="per-run value")
    tick_positions = list(xs)
    tick_labels = [label for label, _ in valid]
    if len(valid) >= 2:
        mean = float(np.mean(ys))
        std = float(np.std(ys))
        summary_x = len(xs)
        ax.errorbar(
            [summary_x], [mean], yerr=[std],
            fmt="D", color="tab:red", capsize=5, markersize=7, zorder=4,
            label="mean +/- std",
        )
        tick_positions.append(summary_x)
        tick_labels.append("mean +/- std")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    note = (
        f"n={len(valid)} run(s) -- points only, too few for meaningful quartiles"
        if len(valid) < 5 else f"n={len(valid)} runs"
    )
    from marla.metrics.plots import _set_title_with_note  # local import: avoids a circular import at module load time

    _set_title_with_note(ax, title, note)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, plots_dir / f"compare_{metric}.png")


def generate_comparison_plots(run_dirs: list[Path], plots_dir: Path, labels: list[str] | None = None) -> list[Path]:
    """One comparison plot per metric in :data:`COMPARISON_METRICS`,
    across every run in ``run_dirs`` -- skips a metric entirely (no file
    written) when not a single run has it.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if labels is not None and len(labels) != len(run_dirs):
        raise ValueError("labels must have the same length as run_dirs when given")
    runs = [
        load_run_metrics(run_dir, label=(labels[i] if labels else None))
        for i, run_dir in enumerate(run_dirs)
    ]
    written = []
    for metric, ylabel, title in COMPARISON_METRICS:
        path = plot_metric_comparison(runs, metric, ylabel, title, plots_dir)
        if path is not None:
            written.append(path)
    return written
