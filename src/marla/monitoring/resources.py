"""CPU/RAM/GPU resource telemetry for training and evaluation runs.

Reports computational cost alongside sample efficiency/performance (a
research-readiness requirement: results are not just "did it learn" but
also "what did it cost"). Samples on a background thread at a configurable
interval (default ~1s -- see ``ResourceMonitoringConfig``), tagged with
whatever the caller last set as the current phase/step via
:meth:`ResourceMonitor.set_context` (``rollout_collection`` /
``ppo_update`` / ``evaluation``, matching ``learning/trainer.py``'s own
phase names) -- never sampled per environment step, which would both be
far too fine-grained to be meaningful and would materially slow training.

**Absolute vs. relative units (paper-reporting hardening)**: percentages
(``cpu_process_pct``, ``cpu_system_pct``, ``gpu_util_pct``) are strongly
hardware-relative -- the same workload on a different core count or a
different GPU reports a different percentage for the same real work. Every
metric with a genuine absolute physical unit is ALSO recorded that way:
process memory in MiB (never converted from/to a CPU-percent -- memory and
compute are different physical quantities), GPU memory in MiB (device-level
NVML, PyTorch-allocated, and PyTorch-reserved kept as three DISTINCT
fields, never merged into one ambiguous "GPU memory"), and CPU *compute*
in absolute seconds (``cpu_process_user_seconds`` / ``..._system_seconds``
/ ``..._total_seconds``, from :meth:`psutil.Process.cpu_times`, cumulative
since process start -- unaffected by core count the way a percentage is).
Percentages remain in the raw per-sample telemetry as a secondary
diagnostic (useful for spotting throttling/contention within one run), but
:meth:`ResourceMonitor.summarize` surfaces the absolute quantities as the
primary run-level summary fields a paper table should read.

GPU *compute* has no reliable absolute measure available here: NVML's
``utilization.gpu`` is a coarse, driver-reported percentage-busy-over-a-
recent-window sample, not a kernel-time accounting API, and PyTorch
exposes no elapsed-kernel-time counter either. :func:`_gpu_utilization_equivalent_seconds`
derives a documented, clearly-labeled *approximation* (utilization
fraction integrated over the actual wall-clock gap between consecutive
samples) for readers who want a single number, but this is never presented
as exact GPU compute time -- see that function's own docstring.

**CPU-percent semantics** -- ``cpu_process_pct`` and ``cpu_system_pct``
use TWO DIFFERENT conventions, both :mod:`psutil`'s own, neither
normalized here:

- ``cpu_process_pct`` (``psutil.Process.cpu_percent``) is *summed across
  logical cores*: a fully single-threaded process pinning one core of an
  8-core machine reports ~12.5%, while a process actually using all 8
  cores reports ~800% in principle. In practice PyTorch's own internal
  thread pool means MARLA's process CPU percent commonly exceeds 100%
  even for otherwise "single-threaded" Python code -- this is expected,
  not a bug.
- ``cpu_system_pct`` (``psutil.cpu_percent()``, no ``percpu``) is instead
  an AVERAGE across logical cores, capped to ``[0, 100]`` -- an 8-core
  machine fully busy on every core reports 100%, not 800%. It is **not**
  directly comparable to ``cpu_process_pct``'s scale; do not divide one by
  the other expecting a "fraction of system CPU this process used" number
  -- that would need ``cpu_process_pct / (100 * cpu_logical_count)``
  instead. Both scales are documented here specifically because
  conflating them would silently misrepresent utilization. Prefer the
  absolute ``cpu_process_*_seconds`` fields for cross-run/cross-hardware
  comparison; keep the percentages for within-run diagnostics.

Logical vs. physical core counts are both reported specifically so a
percentage can be interpreted against either basis by whoever reads
``resources.csv``.

**GPU semantics**: NVML (via the optional ``nvidia-ml-py`` dependency,
imported as ``pynvml``) provides hardware utilization/memory numbers when
available; ``torch.cuda.memory_allocated``/``memory_reserved``/
``max_memory_allocated``/``max_memory_reserved`` are reported whenever
CUDA is available REGARDLESS of NVML (they come from PyTorch's own CUDA
caching allocator, not from any driver query) -- so a CUDA-without-NVML
environment still gets PyTorch-side memory numbers, just with
``gpu_util_pct``/``gpu_memory_used_mib`` left ``None``. On a CPU-only
machine every GPU field is ``None`` and nothing about training changes;
this module never repeatedly shells out to ``nvidia-smi`` (a real,
measurable per-call subprocess-spawn cost at 1s sampling intervals) -- it
calls NVML's C library bindings directly, once per sample, if available.
"""

from __future__ import annotations

import platform
import threading
import time
from dataclasses import asdict, dataclass, field

import psutil
import torch

try:
    import pynvml

    _NVML_IMPORTABLE = True
except ImportError:  # pragma: no cover -- exercised by the "no NVML" smoke path
    pynvml = None  # type: ignore[assignment]
    _NVML_IMPORTABLE = False


def _bytes_to_mib(n: int | float) -> float:
    """Binary mebibytes (1 MiB = 1024*1024 bytes) -- this is what every
    "_mib"-suffixed field in this module actually computes and always has
    (the prior "_mb" naming computed the identical binary value, which is
    MiB, not decimal MB; this is a labeling correction, not a value
    change)."""
    return n / (1024 * 1024)


class _NvmlHandles:
    """Process-wide lazy NVML init/shutdown, so a monitor that samples
    every ~1s never re-initializes the driver connection per sample.
    ``available`` is False (not an exception) whenever NVML can't be used
    for ANY reason -- not installed, no NVIDIA driver, no GPU -- so
    callers never need their own try/except around every sample.
    """

    _initialized = False
    _available = False
    _handles: list = []

    @classmethod
    def ensure_init(cls) -> bool:
        if cls._initialized:
            return cls._available
        cls._initialized = True
        if not _NVML_IMPORTABLE:
            return False
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            cls._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            cls._available = count > 0
        except Exception:  # noqa: BLE001 -- any NVML failure degrades to "unavailable", never a training crash
            cls._available = False
            cls._handles = []
        return cls._available

    @classmethod
    def handles(cls) -> list:
        cls.ensure_init()
        return cls._handles


@dataclass
class ResourceSample:
    timestamp: float  # time.time(), seconds since epoch
    phase: str | None  # "rollout_collection" | "ppo_update" | "evaluation" | None (not yet set)
    global_env_step: int | None
    ppo_update: int | None

    cpu_process_pct: float | None
    cpu_system_pct: float | None
    cpu_logical_count: int | None
    cpu_physical_count: int | None
    # Absolute CPU compute, cumulative since process start (psutil.Process
    # .cpu_times()) -- unlike the percentages above, these are directly
    # comparable across runs/hardware with different core counts.
    cpu_process_user_seconds: float | None
    cpu_process_system_seconds: float | None
    cpu_process_total_seconds: float | None

    ram_rss_mib: float | None
    ram_peak_rss_mib: float | None  # None where the platform doesn't reliably expose peak RSS (see _process_peak_rss_mib)
    ram_vms_mib: float | None  # secondary: virtual memory size, not the primary process-memory metric (that's RSS)
    ram_system_used_mib: float | None
    ram_system_available_mib: float | None
    ram_system_total_mib: float | None
    ram_system_pct: float | None

    gpu_index: int | None
    gpu_util_pct: float | None  # NVML-reported, relative -- see module docstring
    gpu_memory_used_mib: float | None  # device-level (NVML): everything resident on the GPU, not just this process/PyTorch
    gpu_memory_total_mib: float | None

    torch_cuda_allocated_mib: float | None  # PyTorch's own caching allocator: tensors actually in use
    torch_cuda_reserved_mib: float | None  # PyTorch's own caching allocator: reserved from the driver (>= allocated)
    torch_cuda_max_allocated_mib: float | None
    torch_cuda_max_reserved_mib: float | None


def _process_peak_rss_mib(process: psutil.Process) -> float | None:
    """Peak RSS is not part of psutil's cross-platform API (it varies by
    OS: ``ru_maxrss`` via ``resource.getrusage`` on Linux/macOS, nothing
    directly comparable on Windows) -- reported "if reliably available"
    per the spec, ``None`` otherwise, never a fabricated number.
    """
    try:
        import resource  # POSIX only

        # ru_maxrss is KB on Linux, bytes on macOS -- both give the SAME
        # constant conversion mistake if not handled; Linux is this
        # project's only supported/tested platform (pyproject.toml has no
        # Windows classifier), so KB is assumed here and documented.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:  # noqa: BLE001
        return None


def _sample_once(process: psutil.Process, phase: str | None, global_env_step: int | None, ppo_update: int | None) -> ResourceSample:
    cpu_process_pct = process.cpu_percent(interval=None)
    cpu_system_pct = psutil.cpu_percent(interval=None)
    vmem = psutil.virtual_memory()
    mem_info = process.memory_info()
    cpu_times = process.cpu_times()  # cumulative seconds since process start -- an absolute measure, not a rate

    gpu_index = gpu_util = gpu_mem_used = gpu_mem_total = None
    if _NvmlHandles.ensure_init():
        handles = _NvmlHandles.handles()
        if handles:
            # Single-GPU reporting for now (this repo's own hardware and
            # every planned experiment use exactly one GPU) -- handle 0
            # only; a genuinely multi-GPU run would need one row per GPU
            # per sample, not attempted here (spec: don't over-build this).
            handle = handles[0]
            gpu_index = 0
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = float(util.gpu)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_used = _bytes_to_mib(mem.used)
                gpu_mem_total = _bytes_to_mib(mem.total)
            except Exception:  # noqa: BLE001 -- a transient NVML query failure degrades this one sample, not the run
                pass

    torch_allocated = torch_reserved = torch_max_allocated = torch_max_reserved = None
    if torch.cuda.is_available():
        torch_allocated = _bytes_to_mib(torch.cuda.memory_allocated())
        torch_reserved = _bytes_to_mib(torch.cuda.memory_reserved())
        torch_max_allocated = _bytes_to_mib(torch.cuda.max_memory_allocated())
        torch_max_reserved = _bytes_to_mib(torch.cuda.max_memory_reserved())

    return ResourceSample(
        timestamp=time.time(), phase=phase, global_env_step=global_env_step, ppo_update=ppo_update,
        cpu_process_pct=cpu_process_pct, cpu_system_pct=cpu_system_pct,
        cpu_logical_count=psutil.cpu_count(logical=True), cpu_physical_count=psutil.cpu_count(logical=False),
        cpu_process_user_seconds=cpu_times.user, cpu_process_system_seconds=cpu_times.system,
        cpu_process_total_seconds=cpu_times.user + cpu_times.system,
        ram_rss_mib=_bytes_to_mib(mem_info.rss), ram_peak_rss_mib=_process_peak_rss_mib(process),
        ram_vms_mib=_bytes_to_mib(mem_info.vms),
        ram_system_used_mib=_bytes_to_mib(vmem.used), ram_system_available_mib=_bytes_to_mib(vmem.available),
        ram_system_total_mib=_bytes_to_mib(vmem.total), ram_system_pct=vmem.percent,
        gpu_index=gpu_index, gpu_util_pct=gpu_util, gpu_memory_used_mib=gpu_mem_used, gpu_memory_total_mib=gpu_mem_total,
        torch_cuda_allocated_mib=torch_allocated, torch_cuda_reserved_mib=torch_reserved,
        torch_cuda_max_allocated_mib=torch_max_allocated, torch_cuda_max_reserved_mib=torch_max_reserved,
    )


def _gpu_utilization_equivalent_seconds(rows: list[dict]) -> float | None:
    r"""An approximate GPU-busy-time in seconds, derived ONLY from periodic
    utilization sampling -- NOT exact kernel time (NVML exposes no
    absolute kernel-time counter, and PyTorch exposes none either).

    Defined explicitly as: for each pair of consecutive samples with a
    non-null ``gpu_util_pct``, accumulate
    ``(util_fraction_i / 100) * (timestamp_{i+1} - timestamp_i)`` -- i.e.
    utilization-fraction integrated over the REAL wall-clock gap between
    samples (never assumed to equal the configured sampling interval,
    since a slow sample or a paused monitor would otherwise silently bias
    this). ``None`` (not 0.0) when fewer than 2 samples carry a GPU
    utilization value, since there is then no time gap to integrate over.
    """
    timed = [(r["timestamp"], r["gpu_util_pct"]) for r in rows if r.get("gpu_util_pct") is not None]
    if len(timed) < 2:
        return None
    timed.sort(key=lambda pair: pair[0])
    total = 0.0
    for (t0, u0), (t1, _u1) in zip(timed, timed[1:]):
        dt = t1 - t0
        if dt > 0:
            total += (u0 / 100.0) * dt
    return total


class ResourceMonitor:
    """Background-thread sampler. ``start()``/``stop()`` bracket a run (or
    a smoke test); :meth:`set_context` is called by the training loop
    whenever its phase/step changes (cheap -- just updates 3 attributes
    read by the sampling thread, no lock needed since Python attribute
    assignment/read is already atomic for these primitive types).

    ``enabled=False`` makes every method a no-op and ``samples`` stays
    empty -- the overhead-comparison smoke test (spec section 22) toggles
    this directly rather than needing two different code paths at the
    call site.
    """

    def __init__(self, sampling_interval_seconds: float = 1.0, enabled: bool = True):
        if sampling_interval_seconds <= 0:
            raise ValueError("sampling_interval_seconds must be > 0")
        self.sampling_interval_seconds = sampling_interval_seconds
        self.enabled = enabled
        self.samples: list[ResourceSample] = []
        self._phase: str | None = None
        self._global_env_step: int | None = None
        self._ppo_update: int | None = None
        self._process = psutil.Process()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def set_context(self, phase: str | None = None, global_env_step: int | None = None, ppo_update: int | None = None) -> None:
        if phase is not None:
            self._phase = phase
        if global_env_step is not None:
            self._global_env_step = global_env_step
        if ppo_update is not None:
            self._ppo_update = ppo_update

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        # Priming calls, done here (synchronously, once, before the
        # background thread starts) rather than letting the FIRST sample
        # pay these costs on the sampling thread:
        # - psutil.Process.cpu_percent's FIRST call always returns 0.0 (no
        #   prior interval to measure against) -- calling it once here
        #   means the first reported sample already reflects a real
        #   interval instead of a guaranteed-wrong 0.0.
        # - NVML's OWN first initialization (opening the driver
        #   connection) is measurably slow (~1s on this project's own dev
        #   machine) -- without warming it here, the sampling thread's
        #   first .wait(sampling_interval_seconds) call would effectively
        #   be preceded by an extra ~1s of unaccounted latency, most
        #   visible at short sampling intervals (see
        #   tests/test_resource_monitoring.py's interval test).
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        _NvmlHandles.ensure_init()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="marla-resource-monitor")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.sampling_interval_seconds * 2 + 1)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.samples.append(_sample_once(self._process, self._phase, self._global_env_step, self._ppo_update))
            except Exception:  # noqa: BLE001 -- a monitoring failure must never take down training
                pass
            self._stop_event.wait(self.sampling_interval_seconds)

    def to_rows(self) -> list[dict]:
        return [asdict(s) for s in self.samples]

    def summarize(self) -> dict:
        """Run-level resource summary, prioritizing ABSOLUTE, dimensionally
        correct quantities (spec: paper reporting should not rely on
        hardware-relative percentages as the primary measure). Returns
        ``{}`` if no samples were collected (monitor disabled, or a run
        shorter than one sampling interval).

        Top-level keys (present only when the underlying telemetry is
        available -- never a fabricated 0.0):

        - ``peak_rss_mib`` / ``mean_rss_mib`` -- process RSS, the primary
          process-memory metric.
        - ``peak_system_memory_used_mib`` -- system-wide RAM used.
        - ``peak_gpu_device_memory_mib`` -- NVML device-level GPU memory
          (everything resident on the GPU, not just this process).
        - ``peak_torch_allocated_mib`` / ``peak_torch_reserved_mib`` --
          PyTorch's own caching-allocator memory, distinct from the NVML
          device-level figure above (reserved >= allocated by
          construction; reserved approaches the device figure only when
          nothing else shares the GPU).
        - ``process_cpu_user_seconds`` / ``..._system_seconds`` /
          ``..._total_seconds`` -- absolute CPU compute consumed by this
          process, cumulative since process start (the last sample's
          cumulative ``psutil.Process.cpu_times()`` reading) -- directly
          comparable across hardware with different core counts, unlike
          ``cpu_process_pct``.
        - ``gpu_utilization_equivalent_seconds`` -- see
          :func:`_gpu_utilization_equivalent_seconds`'s docstring: an
          explicitly-approximate integral of sampled utilization over
          wall-clock time, never exact GPU kernel time.

        ``overall``/``by_phase`` (mean/p95/max per raw telemetry field,
        including the still-useful percentage diagnostics) are kept
        unchanged as secondary/diagnostic detail -- not the primary
        reporting surface.
        """
        if not self.samples:
            return {}
        import pandas as pd

        df = pd.DataFrame(self.to_rows())
        metrics = [
            "cpu_process_pct", "cpu_system_pct", "ram_rss_mib", "ram_system_pct",
            "gpu_util_pct", "gpu_memory_used_mib", "torch_cuda_allocated_mib", "torch_cuda_reserved_mib",
        ]
        summary: dict = {"overall": {}, "by_phase": {}}
        for metric in metrics:
            series = df[metric].dropna()
            if series.empty:
                continue
            summary["overall"][metric] = {"mean": float(series.mean()), "p95": float(series.quantile(0.95)), "max": float(series.max())}
        for phase, group in df.groupby("phase", dropna=True):
            summary["by_phase"][str(phase)] = {}
            for metric in metrics:
                series = group[metric].dropna()
                if series.empty:
                    continue
                summary["by_phase"][str(phase)][metric] = {
                    "mean": float(series.mean()), "p95": float(series.quantile(0.95)), "max": float(series.max()),
                }
        summary["sample_count"] = len(self.samples)
        summary["sampling_interval_seconds"] = self.sampling_interval_seconds

        def _peak(col: str) -> float | None:
            s = df[col].dropna()
            return float(s.max()) if not s.empty else None

        def _mean(col: str) -> float | None:
            s = df[col].dropna()
            return float(s.mean()) if not s.empty else None

        summary["peak_rss_mib"] = _peak("ram_rss_mib")
        summary["mean_rss_mib"] = _mean("ram_rss_mib")
        summary["peak_system_memory_used_mib"] = _peak("ram_system_used_mib")
        summary["peak_gpu_device_memory_mib"] = _peak("gpu_memory_used_mib")
        summary["peak_torch_allocated_mib"] = _peak("torch_cuda_allocated_mib")
        summary["peak_torch_reserved_mib"] = _peak("torch_cuda_reserved_mib")

        cpu_total = df["cpu_process_total_seconds"].dropna()
        if not cpu_total.empty:
            # Cumulative-since-process-start counters: the LAST sample already
            # reports the total up to that point -- not summed across samples
            # (summing would massively over-count a monotonically increasing
            # cumulative quantity).
            summary["process_cpu_user_seconds"] = float(df["cpu_process_user_seconds"].dropna().iloc[-1])
            summary["process_cpu_system_seconds"] = float(df["cpu_process_system_seconds"].dropna().iloc[-1])
            summary["process_cpu_total_seconds"] = float(cpu_total.iloc[-1])

        summary["gpu_utilization_equivalent_seconds"] = _gpu_utilization_equivalent_seconds(self.to_rows())

        return summary


def _cpu_model_name() -> str | None:
    """``platform.processor()`` is frequently empty on Linux; ``/proc/cpuinfo``'s
    ``model name`` line is the reliable source there. Returns ``None``
    (never a fabricated value) if neither works, e.g. non-Linux platforms.
    """
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    name = platform.processor()
    return name or None


def machine_report() -> dict:
    """Compact, PRIVACY-CONSCIOUS machine/software metadata for a run
    (spec section 20): OS/Python/MARLA/git/PyTorch/CUDA/cuDNN versions,
    GPU model(s)/count, CPU model/core counts, total RAM. Deliberately
    excludes hostname, username, and any absolute filesystem path --
    ``platform.node()``/``os.getlogin()``/``Path.cwd()`` are never called
    here.
    """
    from marla import __version__ as marla_version
    from marla.utils.versions import git_commit

    gpu_names: list[str] = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    vmem = psutil.virtual_memory()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "marla_version": marla_version,
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_models": gpu_names,
        "nvml_available": _NvmlHandles.ensure_init(),
        "cpu_model": _cpu_model_name(),
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "total_ram_mib": _bytes_to_mib(vmem.total),
    }
