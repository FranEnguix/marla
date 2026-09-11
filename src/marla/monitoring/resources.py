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
  conflating them would silently misrepresent utilization.

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
``gpu_util_pct``/``gpu_memory_used_mb`` left ``None``. On a CPU-only
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


def _bytes_to_mb(n: int | float) -> float:
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

    ram_rss_mb: float | None
    ram_peak_rss_mb: float | None  # None where the platform doesn't reliably expose peak RSS (see _process_peak_rss_mb)
    ram_system_used_mb: float | None
    ram_system_total_mb: float | None
    ram_system_pct: float | None

    gpu_index: int | None
    gpu_util_pct: float | None
    gpu_memory_used_mb: float | None
    gpu_memory_total_mb: float | None

    torch_cuda_allocated_mb: float | None
    torch_cuda_reserved_mb: float | None
    torch_cuda_max_allocated_mb: float | None
    torch_cuda_max_reserved_mb: float | None


def _process_peak_rss_mb(process: psutil.Process) -> float | None:
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
                gpu_mem_used = _bytes_to_mb(mem.used)
                gpu_mem_total = _bytes_to_mb(mem.total)
            except Exception:  # noqa: BLE001 -- a transient NVML query failure degrades this one sample, not the run
                pass

    torch_allocated = torch_reserved = torch_max_allocated = torch_max_reserved = None
    if torch.cuda.is_available():
        torch_allocated = _bytes_to_mb(torch.cuda.memory_allocated())
        torch_reserved = _bytes_to_mb(torch.cuda.memory_reserved())
        torch_max_allocated = _bytes_to_mb(torch.cuda.max_memory_allocated())
        torch_max_reserved = _bytes_to_mb(torch.cuda.max_memory_reserved())

    return ResourceSample(
        timestamp=time.time(), phase=phase, global_env_step=global_env_step, ppo_update=ppo_update,
        cpu_process_pct=cpu_process_pct, cpu_system_pct=cpu_system_pct,
        cpu_logical_count=psutil.cpu_count(logical=True), cpu_physical_count=psutil.cpu_count(logical=False),
        ram_rss_mb=_bytes_to_mb(mem_info.rss), ram_peak_rss_mb=_process_peak_rss_mb(process),
        ram_system_used_mb=_bytes_to_mb(vmem.used), ram_system_total_mb=_bytes_to_mb(vmem.total),
        ram_system_pct=vmem.percent,
        gpu_index=gpu_index, gpu_util_pct=gpu_util, gpu_memory_used_mb=gpu_mem_used, gpu_memory_total_mb=gpu_mem_total,
        torch_cuda_allocated_mb=torch_allocated, torch_cuda_reserved_mb=torch_reserved,
        torch_cuda_max_allocated_mb=torch_max_allocated, torch_cuda_max_reserved_mb=torch_max_reserved,
    )


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
        """Phase-level mean/p95/max for the fields a compute-budget table
        (spec section 47) actually needs. Returns ``{}`` if no samples
        were collected (monitor disabled, or a run shorter than one
        sampling interval).
        """
        if not self.samples:
            return {}
        import pandas as pd

        df = pd.DataFrame(self.to_rows())
        metrics = [
            "cpu_process_pct", "cpu_system_pct", "ram_rss_mb", "ram_system_pct",
            "gpu_util_pct", "gpu_memory_used_mb", "torch_cuda_allocated_mb", "torch_cuda_reserved_mb",
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
        "total_ram_mb": _bytes_to_mb(vmem.total),
    }
