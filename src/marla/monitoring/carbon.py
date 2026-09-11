"""Per-agent energy / CO2-equivalent emissions tracking, backed by
CodeCarbon (https://github.com/mlco2/codecarbon), version pinned per
``pyproject.toml``'s ``carbon`` extra.

**"Per agent" (project convention, see research/OPTUNA_STUDY.md)**: one
agent = one complete MARLA training/evaluation run, for one configuration
and one training seed. :class:`CarbonTracker` covers exactly one such
run -- never one instance per environment stream, never one shared
instance across multiple agents.

**Tracking mode**: ``process`` (the default -- see
:class:`marla.config.models.CarbonConfig`) isolates CPU/RAM energy
estimates to this process. GPU power, however, is measured at the
DEVICE level by CodeCarbon/NVML regardless of ``tracking_mode`` -- there
is no per-process GPU power API on consumer/datacenter NVIDIA hardware
via NVML. This means:

    ALL HPO agents/trials MUST run sequentially on the same device, never
    concurrently -- two agents sharing a GPU at once would each attribute
    the OTHER's GPU power draw to themselves, corrupting per-agent carbon
    attribution. This is enforced procedurally (the Optuna study driver
    uses ``n_jobs=1`` and never launches a second agent before the first
    finishes), not by this module itself.

**Local output only**: ``output_methods=[OutputMethod.CSV]`` (and nothing
else, in particular never ``OutputMethod.API``) -- nothing is ever
uploaded to CodeCarbon's hosted dashboard/API. Location
resolution (country/region, needed for carbon-intensity) still normally
performs one lightweight geo-IP lookup unless explicit location fields
are configured (:class:`marla.config.models.CarbonConfig`'s
``country_iso_code``/``region``/``cloud_provider``/``cloud_region``) --
that is a DIFFERENT thing from uploading experiment *data*, and is what
"CodeCarbon's normal online location resolution" means. When explicit
location fields are set, :class:`codecarbon.OfflineEmissionsTracker` is
used instead, and no network call is made at all.

**Estimates, not exact measurements**: every number here is CodeCarbon's
own best-effort estimate (TDP-based CPU power modeling was used on this
project's own dev machine -- no RAPL access in this sandboxed
environment; see the ``tracker_class``/hardware fields persisted in the
summary for exactly what was actually used on a given machine). Report
and document accordingly -- never call these numbers exact physical
measurements. See research/CARBON_ACCOUNTING.md for the full
methodological writeup and CodeCarbon's own citation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from marla.config.models import CarbonConfig

logger = logging.getLogger(__name__)

try:
    import codecarbon
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    from codecarbon.output_methods.base_output import OutputMethod

    _CODECARBON_IMPORTABLE = True
except ImportError:  # pragma: no cover -- exercised by the "codecarbon not installed" test
    codecarbon = None  # type: ignore[assignment]
    EmissionsTracker = None  # type: ignore[assignment,misc]
    OfflineEmissionsTracker = None  # type: ignore[assignment,misc]
    OutputMethod = None  # type: ignore[assignment,misc]
    _CODECARBON_IMPORTABLE = False


class CarbonTrackerUnavailableError(Exception):
    """Raised when ``carbon.enabled=True`` but the ``codecarbon`` package
    is not importable -- fails clearly (a study that explicitly asked for
    carbon tracking must not silently run without it) rather than
    degrading quietly. Set ``carbon.enabled: false`` to opt out
    explicitly instead.
    """


@dataclass
class _TaskSegment:
    task_name: str  # "training" | "evaluation" -- the ONLY two names this module uses
    duration_s: float
    energy_kwh: float
    emissions_kg: float | None  # None only if CodeCarbon genuinely could not compute emissions for this segment


@dataclass
class CarbonSummary:
    """The compact, MARLA-owned schema persisted as carbon_summary.json --
    deliberately NOT just a dump of CodeCarbon's own EmissionsData (which
    has a different, CodeCarbon-versioned shape); this is the stable
    contract the rest of MARLA (Optuna objective/user-attrs, plots,
    reports) reads.
    """

    enabled: bool
    codecarbon_version: str | None
    tracker_class: str | None  # "EmissionsTracker" | "OfflineEmissionsTracker" | None
    tracking_mode: str | None
    measure_power_secs: float | None
    duration_seconds: float | None
    energy_consumed_kwh: float | None
    emissions_kg_co2eq: float | None
    emissions_g_co2eq: float | None
    training_energy_kwh: float | None
    training_emissions_kg_co2eq: float | None
    evaluation_energy_kwh: float | None
    evaluation_emissions_kg_co2eq: float | None
    country_name: str | None
    country_iso_code: str | None
    region: str | None
    cloud_provider: str | None
    cloud_region: str | None
    carbon_intensity_source: str | None  # "explicit_config" | "codecarbon_auto" | None
    cpu_model: str | None
    gpu_model: str | None
    ram_total_gb: float | None

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


class CarbonTracker:
    """Wraps exactly one CodeCarbon tracker instance for exactly one
    MARLA agent (one run, one config, one seed). Two named tasks only:
    "training" and "evaluation" -- called possibly MULTIPLE times each
    across one agent's lifetime (MARLA's periodic evaluation is
    interleaved with training, not a single trailing phase), with
    segments accumulated and summed here rather than relying on
    CodeCarbon's own single-task semantics.
    """

    def __init__(self, config: "CarbonConfig", output_dir: Path, project_name: str) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.project_name = project_name
        self._tracker: Any = None
        self._segments: list[_TaskSegment] = []
        self._active_task: str | None = None  # logical name ("training"/"evaluation"), for callers
        self._active_task_unique_name: str | None = None  # the actual name passed to CodeCarbon
        self._task_call_counts: dict[str, int] = {}
        self.tracker_class: str | None = None
        self.carbon_intensity_source: str | None = None

    def start(self) -> None:
        if not self.config.enabled:
            return
        if not _CODECARBON_IMPORTABLE:
            raise CarbonTrackerUnavailableError(
                "carbon.enabled=True but the codecarbon package is not installed "
                "(pip install 'marla-agents[carbon]', or install codecarbon directly). "
                "Set carbon.enabled: false to opt out explicitly instead of leaving this unresolved."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        common_kwargs: dict[str, Any] = dict(
            project_name=self.project_name,
            measure_power_secs=self.config.measure_power_secs,
            tracking_mode=self.config.tracking_mode,
            output_dir=str(self.output_dir),
            # output_methods=[CSV] alone is authoritative for "local only,
            # never uploaded" -- CodeCarbon derives save_to_api=False from
            # OutputMethod.API simply being absent from this list; passing
            # the (now-deprecated) save_to_api kwarg on top would only
            # trigger a DeprecationWarning for a redundant guarantee.
            output_methods=[OutputMethod.CSV],
            log_level="warning",
            allow_multiple_runs=True,
        )
        has_explicit_location = any(
            (self.config.country_iso_code, self.config.region, self.config.cloud_provider, self.config.cloud_region)
        )
        if has_explicit_location:
            self._tracker = OfflineEmissionsTracker(
                country_iso_code=self.config.country_iso_code,
                region=self.config.region,
                cloud_provider=self.config.cloud_provider,
                cloud_region=self.config.cloud_region,
                **common_kwargs,
            )
            self.tracker_class = "OfflineEmissionsTracker"
            self.carbon_intensity_source = "explicit_config"
        else:
            self._tracker = EmissionsTracker(**common_kwargs)
            self.tracker_class = "EmissionsTracker"
            self.carbon_intensity_source = "codecarbon_auto"
        self._tracker.start()

    @property
    def enabled_and_active(self) -> bool:
        return self._tracker is not None

    def start_task(self, task_name: str) -> None:
        """``task_name`` is a LOGICAL name ("training"/"evaluation"),
        callable repeatedly across one agent's lifetime (MARLA's periodic
        evaluation is interleaved with training, not one trailing phase).
        CodeCarbon's own task API does not support reusing one task name
        within a single tracker instance (a second ``start_task("training")``
        after an earlier ``stop_task("training")`` corrupts that task's
        delta computation) -- so each call gets its own unique underlying
        CodeCarbon task name (``f"{task_name}#{n}"``), and segments are
        aggregated back onto the logical name in :meth:`stop`.
        """
        if not self.enabled_and_active:
            return
        if self._active_task is not None:
            # Defensive: never leave a task dangling if a caller forgets
            # to close one before opening the next.
            self.stop_task(self._active_task)
        self._task_call_counts[task_name] = self._task_call_counts.get(task_name, 0) + 1
        unique_name = f"{task_name}#{self._task_call_counts[task_name]}"
        self._tracker.start_task(unique_name)
        self._active_task = task_name
        self._active_task_unique_name = unique_name

    def stop_task(self, task_name: str) -> None:
        if not self.enabled_and_active:
            return
        unique_name = self._active_task_unique_name if self._active_task == task_name else task_name
        data = self._tracker.stop_task(unique_name)
        if self._active_task == task_name:
            self._active_task = None
            self._active_task_unique_name = None
        if data is not None:
            self._segments.append(
                _TaskSegment(
                    task_name=task_name,
                    duration_s=float(data.duration or 0.0),
                    energy_kwh=float(data.energy_consumed or 0.0),
                    emissions_kg=(float(data.emissions) if data.emissions is not None else None),
                )
            )

    def stop(self) -> CarbonSummary:
        """Always safe to call, including after an exception mid-run
        (closes any still-open task first) -- returns a CarbonSummary
        that is all-``None`` (except ``enabled=False``) when carbon
        tracking was disabled, never a crash from "nothing to summarize".
        """
        if not self.enabled_and_active:
            return CarbonSummary(
                enabled=False, **{f: None for f in CarbonSummary.__dataclass_fields__ if f != "enabled"}
            )
        if self._active_task is not None:
            self.stop_task(self._active_task)
        self._tracker.stop()
        final = getattr(self._tracker, "final_emissions_data", None)

        training_energy = sum(s.energy_kwh for s in self._segments if s.task_name == "training")
        training_emissions = sum(
            (s.emissions_kg for s in self._segments if s.task_name == "training" and s.emissions_kg is not None),
            start=0.0,
        )
        eval_energy = sum(s.energy_kwh for s in self._segments if s.task_name == "evaluation")
        eval_emissions = sum(
            (s.emissions_kg for s in self._segments if s.task_name == "evaluation" and s.emissions_kg is not None),
            start=0.0,
        )
        any_training_emissions = any(s.task_name == "training" for s in self._segments)
        any_eval_emissions = any(s.task_name == "evaluation" for s in self._segments)

        # NOT final.duration: CodeCarbon's own top-level "duration" is
        # unreliable once named tasks were used (observed directly --
        # it reflects only the last measurement interval, not the true
        # cumulative wall-clock since .start()), while its top-level
        # energy_consumed/emissions DO correctly equal the sum of every
        # task segment (verified directly too). Summing segment durations
        # ourselves is the reliable value; if no tasks were ever used at
        # all, fall back to CodeCarbon's own total.
        segment_duration_total = sum(s.duration_s for s in self._segments)
        duration_seconds = segment_duration_total if self._segments else getattr(final, "duration", None)

        return CarbonSummary(
            enabled=True,
            codecarbon_version=getattr(codecarbon, "__version__", None),
            tracker_class=self.tracker_class,
            tracking_mode=self.config.tracking_mode,
            measure_power_secs=self.config.measure_power_secs,
            duration_seconds=duration_seconds,
            energy_consumed_kwh=getattr(final, "energy_consumed", None),
            emissions_kg_co2eq=getattr(final, "emissions", None),
            emissions_g_co2eq=(final.emissions * 1000.0 if final is not None and final.emissions is not None else None),
            training_energy_kwh=(training_energy if any_training_emissions else None),
            training_emissions_kg_co2eq=(training_emissions if any_training_emissions else None),
            evaluation_energy_kwh=(eval_energy if any_eval_emissions else None),
            evaluation_emissions_kg_co2eq=(eval_emissions if any_eval_emissions else None),
            country_name=getattr(final, "country_name", None),
            country_iso_code=getattr(final, "country_iso_code", None),
            region=getattr(final, "region", None),
            cloud_provider=getattr(final, "cloud_provider", None) or None,
            cloud_region=getattr(final, "cloud_region", None) or None,
            carbon_intensity_source=self.carbon_intensity_source,
            cpu_model=getattr(final, "cpu_model", None),
            gpu_model=getattr(final, "gpu_model", None),
            ram_total_gb=getattr(final, "ram_total_size", None),
        )


def derived_efficiency_metrics(
    summary: CarbonSummary,
    environment_steps: int,
    root_auc: float | None,
    successful_finish_count: int,
) -> dict[str, float | None]:
    """Spec section 11: per-agent carbon-efficiency metrics. Every ratio
    is ``None`` (never a ZeroDivisionError, never a fabricated sentinel)
    when its denominator is genuinely zero/unavailable -- e.g.
    ``g_co2eq_per_successful_finish`` is None when
    ``successful_finish_count == 0``, not 0.0 or inf.
    """

    def _safe_div(numerator: float | None, denominator: float) -> float | None:
        if numerator is None or denominator == 0:
            return None
        return numerator / denominator

    energy_kwh = summary.energy_consumed_kwh
    emissions_g = summary.emissions_g_co2eq
    duration_h = (summary.duration_seconds / 3600.0) if summary.duration_seconds else None

    per_1k_steps = environment_steps / 1000.0 if environment_steps else 0.0

    return {
        "g_co2eq_per_1k_transitions": _safe_div(emissions_g, per_1k_steps),
        "kwh_per_1k_transitions": _safe_div(energy_kwh, per_1k_steps),
        "g_co2eq_per_wall_clock_hour": _safe_div(emissions_g, duration_h) if duration_h else None,
        "root_auc_per_kwh": _safe_div(root_auc, energy_kwh) if energy_kwh else None,
        "root_auc_per_g_co2eq": _safe_div(root_auc, emissions_g) if emissions_g else None,
        "g_co2eq_per_successful_finish": _safe_div(emissions_g, float(successful_finish_count)),
    }


def write_carbon_summary(
    run_dir: Path,
    summary: CarbonSummary,
    efficiency: dict[str, float | None] | None = None,
    agent_id: str | None = None,
) -> None:
    """Writes ``carbon/carbon_summary.json`` (MARLA's own compact schema,
    plus derived efficiency metrics and an optional agent identifier) into
    ``run_dir``. CodeCarbon's own ``carbon/emissions.csv`` (and
    ``carbon/emissions_base_<run_id>.csv``, the raw per-task rows) are
    already written directly into that same directory by
    :class:`CarbonTracker` itself (its ``output_dir`` -- see how callers
    construct it), so this function only adds the summary layer, never
    duplicates the raw CodeCarbon CSVs.
    """
    import json

    carbon_dir = run_dir / "carbon"
    carbon_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"agent_id": agent_id, **summary.to_dict()}
    if efficiency is not None:
        payload["efficiency"] = efficiency
    (carbon_dir / "carbon_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
