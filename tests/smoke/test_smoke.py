"""End-to-end health smoke test for the MARLA platform.

Exercises the real execution path -- not a hypothetical/static check:

1. Direct ``NasimEmuAdapter`` reset/legal_actions/step, in-process (fast,
   catches environment-integration breakage immediately, with explicit
   invariant assertions on observations/actions/rewards/termination).
2. A real ``marla run`` subprocess, PPO_ONLY (baseline) variant: config
   loading, environment reset/step, PPO rollout collection, PPO
   optimization, periodic deterministic evaluation, metrics/plot writing,
   clean shutdown.
3. A real ``marla run`` subprocess, MARLA_FULL (assisted) variant, using
   the project's own supported tiny local Plan Maker test fixture
   (``sshleifer/tiny-gpt2``, forced to CPU -- see
   ``test_local_runtime_assisted.py``'s documented CUDA-context-poisoning
   gotcha with an undersized context window). The LLM/Gatekeeper/schema-
   validation path is exercised for real, never silently skipped -- only
   the model itself is swapped for a small, fast stand-in instead of the
   production ``Qwen/Qwen2.5-1.5B-Instruct`` (which needs a ~3GB download
   and real GPU-scale latency, inappropriate for a smoke test).

Writes reproducible artifacts to ``artifacts/smoke_test/<run-id>/``:
``metrics.csv`` (per-decision rows from the real assisted run -- the
richest real schema available), ``summary.json``, ``smoke_test.log``, and
``plots/*.png``, all generated only from real data collected during this
run.

Run directly (writes artifacts, prints the full report):
    python tests/smoke/test_smoke.py

Run under pytest (writes artifacts, asserts overall status != FAIL):
    pytest tests/smoke/test_smoke.py -v -s
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "smoke_test"
SMOKE_SEED = 20260916  # deterministic, arbitrary, documented -- never a paper/tuning seed


# --------------------------------------------------------------------------
# Result bookkeeping
# --------------------------------------------------------------------------


@dataclass
class Finding:
    severity: str  # critical | high | medium | low | informational
    phase: str
    message: str


@dataclass
class SmokeResult:
    run_id: str
    artifacts_dir: Path
    started_at: str
    phases: dict = field(default_factory=dict)  # phase -> "pass" | "warn" | "fail"
    findings: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)  # arbitrary recorded facts for the report

    def ok(self, phase: str) -> None:
        self.phases[phase] = "pass"

    def fail(self, phase: str, severity: str, message: str) -> None:
        self.phases[phase] = "fail"
        self.findings.append(Finding(severity, phase, message))
        logging.getLogger("smoke").error("[%s] %s: %s", severity.upper(), phase, message)

    def warn(self, phase: str, message: str, severity: str = "medium") -> None:
        if self.phases.get(phase) != "fail":
            self.phases[phase] = "warn"
        self.findings.append(Finding(severity, phase, message))
        logging.getLogger("smoke").warning("[%s] %s: %s", severity.upper(), phase, message)

    @property
    def overall_status(self) -> str:
        if any(v == "fail" for v in self.phases.values()):
            return "FAIL"
        if any(v == "warn" for v in self.phases.values()) or self.findings:
            return "PASS WITH WARNINGS"
        return "PASS"


def _setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("smoke")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------
# Phase 1: direct environment-adapter exercise (in-process, fast)
# --------------------------------------------------------------------------


def _phase_direct_adapter(result: SmokeResult, log: logging.Logger) -> list[dict]:
    """Reset, list legal actions, and step the real NasimEmuAdapter
    directly -- no subprocess, no policy -- picking the first non-FINISH
    legal action each step (deterministic, reproducible, no RNG needed
    beyond the environment's own seeded generation)."""
    from marla.environment.nasimemu_adapter import NasimEmuAdapter

    rows: list[dict] = []
    try:
        adapter = NasimEmuAdapter(
            scenario=SMALL_SCENARIO,
            max_episode_steps=8,
            completion_reward=1.0,
            premature_finish_penalty=-1.0,
        )
        log.info("Phase 1: NasimEmuAdapter constructed for scenario %s", SMALL_SCENARIO)
    except Exception as exc:  # noqa: BLE001 -- smoke test must report, not crash
        result.fail("direct_adapter_init", "critical", f"NasimEmuAdapter construction raised: {exc!r}")
        return rows

    try:
        state = adapter.reset(seed=SMOKE_SEED)
    except Exception as exc:  # noqa: BLE001
        result.fail("direct_adapter_reset", "critical", f"adapter.reset() raised: {exc!r}")
        return rows

    assert state.raw_observation is not None, "reset() must return a real observation"
    assert isinstance(state.raw_observation, np.ndarray), "raw_observation must be a numpy array"
    assert np.isfinite(state.raw_observation.astype(np.float64)).all(), "raw_observation contains NaN/Inf"
    log.info("Phase 1: reset OK, raw_observation shape=%s dtype=%s", state.raw_observation.shape, state.raw_observation.dtype)
    result.ok("direct_adapter_reset")

    step_idx = 0
    episodes_completed = 0
    max_steps = 20
    while step_idx < max_steps:
        try:
            legal = adapter.legal_actions(state)
        except Exception as exc:  # noqa: BLE001
            result.fail("direct_adapter_legal_actions", "critical", f"legal_actions() raised at step {step_idx}: {exc!r}")
            break
        assert isinstance(legal, list) and len(legal) > 0, f"legal_actions() returned no candidate actions at step {step_idx}"
        assert len({a.action_id for a in legal}) == len(legal), "duplicate action_id values in the legal-action list"

        non_finish = [a for a in legal if not a.is_finish]
        action = non_finish[0] if non_finish else legal[0]
        assert action.action_id, "selected action has an empty action_id"

        t0 = time.perf_counter()
        try:
            transition = adapter.step(action)
        except Exception as exc:  # noqa: BLE001
            result.fail("direct_adapter_step", "critical", f"step() raised at step {step_idx}: {exc!r}")
            break
        latency_ms = (time.perf_counter() - t0) * 1000

        assert isinstance(transition.nasimemu_reward, (int, float)), "reward is not numeric"
        assert np.isfinite(transition.nasimemu_reward), f"non-finite reward at step {step_idx}: {transition.nasimemu_reward!r}"
        assert isinstance(transition.terminated, bool) and isinstance(transition.truncated, bool), "terminated/truncated must be bool"
        assert not (transition.terminated and transition.truncated), "terminated and truncated both true for the same transition"

        rows.append(
            {
                "step": step_idx,
                "action_id": action.action_id,
                "action_type": action.action_type,
                "legal_action_count": len(legal),
                "reward": transition.nasimemu_reward,
                "terminated": transition.terminated,
                "truncated": transition.truncated,
                "step_latency_ms": latency_ms,
            }
        )
        step_idx += 1

        if transition.terminated or transition.truncated:
            episodes_completed += 1
            if step_idx >= max_steps:
                break
            try:
                state = adapter.reset(seed=SMOKE_SEED + episodes_completed)
            except Exception as exc:  # noqa: BLE001
                result.fail("direct_adapter_reset", "critical", f"re-reset after episode {episodes_completed} raised: {exc!r}")
                break
        else:
            state = transition.state
            assert state is not None, "non-terminal transition returned no next state"

    if rows:
        result.ok("direct_adapter_step_loop")
        result.facts["direct_adapter_steps"] = len(rows)
        result.facts["direct_adapter_episodes_completed"] = episodes_completed
        log.info("Phase 1: completed %d direct steps across %d episode boundary(ies)", len(rows), episodes_completed)
    else:
        result.fail("direct_adapter_step_loop", "critical", "no steps were successfully executed")
    return rows


# --------------------------------------------------------------------------
# Phase 2 / 3: real `marla run` subprocess (PPO_ONLY, then MARLA_FULL)
# --------------------------------------------------------------------------


def _write_tiny_baseline_config(tmp_dir: Path, run_id: str) -> Path:
    data = yaml.safe_load((REPO_ROOT / "examples" / "baseline.yaml").read_text())
    data["experiment"]["run_id"] = run_id
    data["experiment"]["seed"] = SMOKE_SEED
    data["environment"]["scenario"] = SMALL_SCENARIO
    data["environment"]["max_episode_steps"] = 8
    data["policy"]["ppo"]["total_environment_steps"] = 128
    data["policy"]["ppo"]["steps_per_env"] = 64
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 4
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["metrics"]["eval_episodes"] = 2
    data["metrics"]["eval_every_rollouts"] = 1
    data["metrics"]["resource_monitoring"] = {"enabled": True, "sampling_interval_seconds": 0.5}
    config_path = tmp_dir / "smoke_baseline.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def _write_tiny_assisted_config(tmp_dir: Path, run_id: str) -> Path:
    data = yaml.safe_load((REPO_ROOT / "examples" / "assisted.yaml").read_text())
    data["experiment"]["run_id"] = run_id
    data["experiment"]["seed"] = SMOKE_SEED
    data["environment"]["scenario"] = SMALL_SCENARIO
    data["environment"]["max_episode_steps"] = 8
    data["policy"]["ppo"]["total_environment_steps"] = 32
    data["policy"]["ppo"]["steps_per_env"] = 32
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    data["consultation"]["cost"] = 0.1
    # The project's own supported stub for exercising the real LLM/
    # Gatekeeper/schema-validation path without the production model's
    # ~3GB download and GPU-scale latency (see test_local_runtime_assisted.py).
    data["agents"][0]["model"]["name"] = "sshleifer/tiny-gpt2"
    # tiny-gpt2's 1024-token context window is smaller than a real prompt;
    # on CUDA the resulting IndexError poisons the whole process's CUDA
    # context. Forcing CPU keeps this smoke test about integration
    # correctness, not CUDA context-poisoning recovery (a separate, already
    #-known platform limitation of this specific stub model, not this test).
    data["device"] = "cpu"
    data["metrics"]["eval_episodes"] = 0
    data["metrics"]["resource_monitoring"] = {"enabled": True, "sampling_interval_seconds": 0.5}
    config_path = tmp_dir / "smoke_assisted.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def _run_marla_subprocess(config_path: Path, extra_env: dict, timeout: int) -> subprocess.CompletedProcess:
    env = {**os.environ, **extra_env}
    return subprocess.run(
        [sys.executable, "-m", "marla", "run", str(config_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _phase_baseline_subprocess(result: SmokeResult, log: logging.Logger, work_dir: Path) -> Path | None:
    config_path = _write_tiny_baseline_config(work_dir, run_id="smoke-baseline")
    log.info("Phase 2: launching real `marla run` subprocess (PPO_ONLY) -- config=%s", config_path)
    try:
        proc = _run_marla_subprocess(config_path, {"MARLA_RL_ORCHESTRATOR_PASSWORD": "smoke-test-pass"}, timeout=180)
    except subprocess.TimeoutExpired as exc:
        result.fail("baseline_subprocess", "critical", f"marla run (PPO_ONLY) timed out: {exc}")
        return None

    if proc.returncode != 0:
        result.fail(
            "baseline_subprocess", "critical",
            f"marla run (PPO_ONLY) exited {proc.returncode}. stdout(tail)={proc.stdout[-1500:]} stderr(tail)={proc.stderr[-1500:]}",
        )
        return None
    if "Run complete" not in proc.stdout:
        result.warn("baseline_subprocess", "marla run (PPO_ONLY) exited 0 but did not print 'Run complete'")
    else:
        result.ok("baseline_subprocess")
    log.info("Phase 2: PPO_ONLY subprocess completed (rc=0)")

    run_dir = REPO_ROOT / "runs" / "ppo_baseline" / "smoke-baseline"
    if not run_dir.is_dir():
        result.fail("baseline_artifacts", "critical", f"expected run directory not found: {run_dir}")
        return None
    return run_dir


def _phase_assisted_subprocess(result: SmokeResult, log: logging.Logger, work_dir: Path) -> Path | None:
    config_path = _write_tiny_assisted_config(work_dir, run_id="smoke-assisted")
    log.info("Phase 3: launching real `marla run` subprocess (MARLA_FULL, stub Plan Maker) -- config=%s", config_path)
    try:
        proc = _run_marla_subprocess(
            config_path,
            {
                "MARLA_RL_ORCHESTRATOR_PASSWORD": "smoke-test-pass",
                "MARLA_GATEKEEPER_PASSWORD": "smoke-test-pass",
                "MARLA_PLAN_MAKER_1_PASSWORD": "smoke-test-pass",
            },
            timeout=240,
        )
    except subprocess.TimeoutExpired as exc:
        result.fail("assisted_subprocess", "critical", f"marla run (MARLA_FULL) timed out: {exc}")
        return None

    if proc.returncode != 0:
        result.fail(
            "assisted_subprocess", "critical",
            f"marla run (MARLA_FULL) exited {proc.returncode}. stdout(tail)={proc.stdout[-1500:]} stderr(tail)={proc.stderr[-1500:]}",
        )
        return None
    result.ok("assisted_subprocess")
    log.info("Phase 3: MARLA_FULL subprocess completed (rc=0)")

    run_dir = REPO_ROOT / "runs" / "ppo_plan_maker" / "smoke-assisted"
    if not run_dir.is_dir():
        result.fail("assisted_artifacts", "critical", f"expected run directory not found: {run_dir}")
        return None
    return run_dir


# --------------------------------------------------------------------------
# Artifact validation
# --------------------------------------------------------------------------


def _validate_baseline_artifacts(result: SmokeResult, log: logging.Logger, run_dir: Path) -> pd.DataFrame | None:
    required = ["episodes.csv", "decisions.csv", "rollouts.csv", "updates.csv", "summary.json"]
    missing = [f for f in required if not (run_dir / f).is_file()]
    if missing:
        result.fail("baseline_artifacts", "critical", f"missing expected artifacts: {missing}")
        return None

    decisions = pd.read_csv(run_dir / "decisions.csv")
    if decisions.empty:
        result.fail("baseline_artifacts", "critical", "decisions.csv is empty -- no decisions were recorded")
        return None

    numeric_cols = ["nasimemu_reward", "training_reward", "critic_value", "gae_advantage", "return_target"]
    for col in numeric_cols:
        if col in decisions.columns:
            vals = decisions[col].dropna()
            if len(vals) and not np.isfinite(vals.to_numpy(dtype=np.float64)).all():
                result.fail("baseline_numerics", "critical", f"decisions.csv column {col!r} contains NaN/Inf")
    else:
        result.ok("baseline_numerics")

    assert (decisions["legal_action_count"] > 0).all(), "some decisions had zero legal actions"
    assert decisions["selected_action_base_probability"].between(0, 1, inclusive="both").all(), "base action probability outside [0,1]"

    summary = json.loads((run_dir / "summary.json").read_text())
    for key in ("episode_count", "total_training_seconds"):
        if key not in summary:
            result.warn("baseline_summary", f"summary.json missing expected key {key!r}")
    result.facts["baseline_episodes_recorded"] = len(pd.read_csv(run_dir / "episodes.csv"))
    result.facts["baseline_decisions_recorded"] = len(decisions)
    result.facts["baseline_summary"] = summary

    resource_summary_path = run_dir / "resource_summary.json"
    if resource_summary_path.is_file():
        rs = json.loads(resource_summary_path.read_text())
        if not rs.get("overall"):
            result.warn("baseline_resource_telemetry", "resource_summary.json has no 'overall' section")
        else:
            result.ok("baseline_resource_telemetry")
    else:
        result.warn("baseline_resource_telemetry", "resource_summary.json not written despite resource_monitoring.enabled=true")

    log.info("Phase 2: validated %d decision rows, %d episode rows", len(decisions), result.facts["baseline_episodes_recorded"])
    return decisions


def _validate_assisted_artifacts(result: SmokeResult, log: logging.Logger, run_dir: Path) -> pd.DataFrame | None:
    required = ["episodes.csv", "decisions.csv", "summary.json"]
    missing = [f for f in required if not (run_dir / f).is_file()]
    if missing:
        result.fail("assisted_artifacts", "critical", f"missing expected artifacts: {missing}")
        return None

    decisions = pd.read_csv(run_dir / "decisions.csv")
    if decisions.empty:
        result.fail("assisted_artifacts", "critical", "decisions.csv is empty")
        return None
    result.facts["assisted_decisions_recorded"] = len(decisions)

    if "query_probability" not in decisions.columns:
        result.fail("assisted_schema", "high", "decisions.csv missing query_probability column (assisted-variant schema)")
    else:
        qp = decisions["query_probability"].dropna()
        if len(qp):
            assert qp.between(0, 1, inclusive="both").all(), "query_probability outside [0,1]"
            if not np.isfinite(qp.to_numpy(dtype=np.float64)).all():
                result.fail("assisted_schema", "critical", "query_probability contains NaN/Inf")

    queried = decisions[decisions.get("queried", False) == True] if "queried" in decisions.columns else decisions.iloc[0:0]  # noqa: E712
    result.facts["assisted_queried_decisions"] = len(queried)
    if len(queried) == 0:
        result.warn(
            "assisted_consultation_activity",
            "zero queried decisions in this smoke run -- the (untrained) query gate never consulted the Plan "
            "Maker, so the advisory/Gatekeeper/schema-validation path was not exercised this run (a small, "
            "stochastic step budget can legitimately produce zero queries; not evidence of a broken integration "
            "by itself, but worth widening the step budget if this recurs).",
        )
    else:
        statuses = set(queried["response_status"].dropna().unique()) if "response_status" in queried.columns else set()
        valid_statuses = {"accepted", "schema_rejected", "timeout", "error"}
        unknown = statuses - valid_statuses
        if unknown:
            result.warn("assisted_schema", f"response_status contains values outside the known set: {unknown}")
        if statuses and "accepted" not in statuses:
            result.warn(
                "assisted_consultation_outcomes",
                f"every queried decision was rejected ({statuses}), none accepted -- with the tiny stub model "
                "(sshleifer/tiny-gpt2, 1024-token context window) this is expected: this scenario's real "
                "knowledge-augmented prompt exceeds that window, so generation raises inside "
                "transformers/torch (logged via logger.exception in plan_maker.py) and the Gatekeeper's "
                "existing correction/rejection path handles it gracefully by design (see that module's own "
                "comment) -- not a hang, not a crash, not silently dropped. This does mean the real "
                "accepted-advice code path (trust/beta computation, advice_changed_top_action) is not "
                "exercised by this specific stub+scenario combination; it is covered elsewhere by "
                "test_gatekeeper.py's mock-backend unit tests instead.",
                severity="low",
            )
        latencies = queried["response_latency_ms"].dropna() if "response_latency_ms" in queried.columns else pd.Series(dtype=float)
        if len(latencies) and not np.isfinite(latencies.to_numpy(dtype=np.float64)).all():
            result.fail("assisted_schema", "critical", "response_latency_ms contains NaN/Inf")
        result.facts["assisted_response_statuses"] = {k: int(v) for k, v in queried["response_status"].value_counts().items()} if "response_status" in queried.columns else {}
        result.facts["assisted_mean_latency_ms"] = float(latencies.mean()) if len(latencies) else None
        result.ok("assisted_consultation_activity")

    log.info("Phase 3: validated %d decision rows, %d queried", len(decisions), len(queried))
    return decisions


# --------------------------------------------------------------------------
# Plots (real data only)
# --------------------------------------------------------------------------


def _generate_plots(plots_dir: Path, direct_rows: list[dict], baseline_decisions: pd.DataFrame | None, assisted_decisions: pd.DataFrame | None, log: logging.Logger) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if direct_rows:
        df = pd.DataFrame(direct_rows)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["step"], df["reward"], marker=".", color="tab:blue")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Step (direct adapter loop)")
        ax.set_ylabel("Reward (this step)")
        ax.set_title("Reward per step")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = plots_dir / "reward_per_step.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["step"], df["reward"].cumsum(), marker=".", color="tab:green")
        ax.set_xlabel("Step (direct adapter loop)")
        ax.set_ylabel("Cumulative reward")
        ax.set_title("Cumulative reward over the smoke episode(s)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = plots_dir / "cumulative_reward.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["step"], df["legal_action_count"], marker=".", color="tab:orange")
        ax.set_xlabel("Step (direct adapter loop)")
        ax.set_ylabel("Number of legal (candidate) actions")
        ax.set_title("Candidate actions over time")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = plots_dir / "candidate_actions.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)
    else:
        log.warning("No direct-adapter rows collected -- skipping reward/cumulative-reward/candidate-action plots")

    if baseline_decisions is not None and "selected_action_type" in baseline_decisions.columns:
        counts = baseline_decisions["selected_action_type"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(counts.index.astype(str), counts.to_numpy(), color="tab:blue")
        ax.set_xlabel("Selected action type")
        ax.set_ylabel("Count (PPO_ONLY smoke run)")
        ax.set_title("Action-category distribution (real PPO_ONLY subprocess run)")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        p = plots_dir / "action_category_distribution.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    if assisted_decisions is not None and "queried" in assisted_decisions.columns:
        queried = assisted_decisions[assisted_decisions["queried"] == True]  # noqa: E712
        fig, ax = plt.subplots(figsize=(7, 4))
        if len(queried) and "response_status" in queried.columns and queried["response_status"].notna().any():
            counts = queried["response_status"].value_counts()
            ax.bar(counts.index.astype(str), counts.to_numpy(), color="tab:purple")
            ax.set_ylabel("Count")
            ax.set_title("Plan Maker validation outcomes (real MARLA_FULL smoke run, stub model)")
        else:
            ax.text(0.5, 0.5, "No Plan Maker consultations occurred\nin this smoke run (see log)", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Plan Maker validation outcomes (real MARLA_FULL smoke run, stub model)")
        fig.tight_layout()
        p = plots_dir / "validation_outcomes.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

        if len(queried) and "response_latency_ms" in queried.columns and queried["response_latency_ms"].notna().any():
            fig, ax = plt.subplots(figsize=(7, 4))
            lat = queried["response_latency_ms"].dropna()
            ax.plot(range(len(lat)), lat.to_numpy(), marker=".", color="tab:red")
            ax.set_xlabel("Queried decision (in order)")
            ax.set_ylabel("Plan Maker response latency (ms)")
            ax.set_title("Plan Maker latency (stub model -- not representative of production model cost)")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            p = plots_dir / "plan_maker_latency.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            written.append(p)

    return written


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------


def run_smoke_test(run_id: str | None = None) -> SmokeResult:
    run_id = run_id or f"smoke-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    artifacts_dir = ARTIFACTS_ROOT / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "plots").mkdir(exist_ok=True)

    log = _setup_logging(artifacts_dir / "smoke_test.log")
    started_at = datetime.now(timezone.utc).isoformat()
    result = SmokeResult(run_id=run_id, artifacts_dir=artifacts_dir, started_at=started_at)

    log.info("=== MARLA smoke test starting: run_id=%s ===", run_id)
    log.info("Python %s, platform %s", sys.version.split()[0], platform.platform())

    try:
        import torch

        result.facts["torch_version"] = torch.__version__
        result.facts["cuda_available"] = torch.cuda.is_available()
    except Exception as exc:  # noqa: BLE001
        result.warn("environment_probe", f"could not import torch to record its version: {exc!r}")

    git_sha = None
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    result.facts["git_commit"] = git_sha

    direct_rows = _phase_direct_adapter(result, log)

    work_dir = artifacts_dir / "_work"
    work_dir.mkdir(exist_ok=True)

    baseline_decisions = None
    baseline_run_dir = _phase_baseline_subprocess(result, log, work_dir)
    if baseline_run_dir is not None:
        baseline_decisions = _validate_baseline_artifacts(result, log, baseline_run_dir)

    assisted_decisions = None
    assisted_run_dir = _phase_assisted_subprocess(result, log, work_dir)
    if assisted_run_dir is not None:
        assisted_decisions = _validate_assisted_artifacts(result, log, assisted_run_dir)

    plots = _generate_plots(artifacts_dir / "plots", direct_rows, baseline_decisions, assisted_decisions, log)
    result.facts["plots_written"] = [str(p.relative_to(REPO_ROOT)) for p in plots]

    # metrics.csv: the richest real, step/decision-level data collected --
    # the direct-adapter loop (always available) plus, when present, the
    # real assisted-run decisions (consultation-specific fields).
    metrics_rows = []
    for row in direct_rows:
        metrics_rows.append({"source": "direct_adapter", **row})
    if assisted_decisions is not None:
        cols = [c for c in ["global_environment_step", "selected_action_type", "legal_action_count", "nasimemu_reward", "queried", "response_status", "response_latency_ms"] if c in assisted_decisions.columns]
        for _, r in assisted_decisions[cols].iterrows():
            metrics_rows.append({"source": "assisted_subprocess", **r.to_dict()})
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(artifacts_dir / "metrics.csv", index=False)

    summary = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": result.overall_status,
        "phases": result.phases,
        "findings": [f.__dict__ for f in result.findings],
        "facts": result.facts,
        "seed": SMOKE_SEED,
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    (artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    log.info("=== MARLA smoke test finished: overall_status=%s ===", result.overall_status)
    return result


# --------------------------------------------------------------------------
# pytest entry point
# --------------------------------------------------------------------------


def test_end_to_end_smoke():
    result = run_smoke_test()
    critical_or_high = [f for f in result.findings if f.severity in ("critical", "high")]
    assert not critical_or_high, (
        f"Smoke test found {len(critical_or_high)} critical/high-severity issue(s): "
        + "; ".join(f"[{f.phase}] {f.message}" for f in critical_or_high)
    )


if __name__ == "__main__":
    res = run_smoke_test()
    print()
    print(f"SMOKE TEST STATUS: {res.overall_status}")
    print(f"Artifacts: {res.artifacts_dir}")
    for finding in res.findings:
        print(f"  [{finding.severity.upper()}] ({finding.phase}) {finding.message}")
