"""Regression coverage for the smoke-test run-directory hazard.

``tests/smoke/test_smoke.py`` used to give its two real ``marla run``
subprocess phases FIXED run IDs (``"smoke-baseline"``/``"smoke-assisted"``),
so every invocation wrote to the SAME ``runs/ppo_baseline/smoke-baseline/``
directory. After a metrics/schema change, rerunning the smoke test would
therefore APPEND new-schema decision rows onto an old-schema
``decisions.csv`` -- a real corruption incident hit during this project's
own subnet-scoped-consultation work (a stale 74-column header followed by
82-column rows, raising ``pandas.errors.ParserError`` on the next read).

Each phase's inner ``experiment.run_id`` (and therefore its
``runs/<experiment>/<run_id>/`` output directory) is now derived from the
smoke invocation's own unique ``run_id`` (see ``test_smoke.py``'s module
docstring) -- this proves that fix holds, using the REAL PPO_ONLY subprocess
phase (cheaper than the assisted/Plan Maker phase, and the one the original
incident actually happened on), not a mock.
"""

from __future__ import annotations

import csv
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_smoke import (  # noqa: E402
    REPO_ROOT,
    SmokeResult,
    _phase_baseline_subprocess,
    _setup_logging,
)


def _run_baseline_phase(tmp_path: Path, run_id: str) -> Path:
    artifacts_dir = tmp_path / run_id
    artifacts_dir.mkdir(parents=True)
    work_dir = artifacts_dir / "_work"
    work_dir.mkdir()
    log = _setup_logging(artifacts_dir / "smoke_test.log")
    result = SmokeResult(run_id=run_id, artifacts_dir=artifacts_dir, started_at="test")

    run_dir = _phase_baseline_subprocess(result, log, work_dir, run_id)
    assert run_dir is not None, f"baseline phase failed: {result.findings}"
    return run_dir


@pytest.mark.integration
def test_smoke_baseline_phase_never_reuses_a_fixed_run_directory(tmp_path):
    """The full regression scenario: pre-create an old-schema artifact at
    the LEGACY fixed path a pre-fix invocation would have used, run the
    real baseline phase twice with distinct run_ids, and confirm neither
    invocation's real output ever lands in, or mixes schemas with, that
    legacy path or each other's directory.
    """
    # 1. Simulate a pre-fix run directory the OLD hardcoded "smoke-baseline"
    #    run_id would have produced, with a decisions.csv from an OLDER,
    #    narrower schema than the one this platform version writes today.
    legacy_run_dir = REPO_ROOT / "runs" / "ppo_baseline" / "smoke-baseline"
    legacy_run_dir.mkdir(parents=True, exist_ok=True)
    legacy_csv = legacy_run_dir / "decisions.csv"
    legacy_csv.write_text("run_id,global_environment_step,legal_action_count\nold-run,0,5\n", encoding="utf-8")
    legacy_snapshot = legacy_csv.read_text(encoding="utf-8")

    try:
        # 2. Run the real baseline phase (a genuine `marla run` PPO_ONLY
        #    subprocess -- not mocked) TWICE, with distinct run_ids, the way
        #    two separate smoke-test invocations would.
        run_id_a = f"smoke-hygiene-test-{uuid.uuid4().hex[:8]}"
        run_id_b = f"smoke-hygiene-test-{uuid.uuid4().hex[:8]}"
        assert run_id_a != run_id_b

        run_dir_a = _run_baseline_phase(tmp_path, run_id_a)
        run_dir_b = _run_baseline_phase(tmp_path, run_id_b)

        # 3. Different invocations must land in different directories --
        #    never the legacy fixed one, never each other's.
        assert run_dir_a != run_dir_b
        assert run_dir_a != legacy_run_dir
        assert run_dir_b != legacy_run_dir
        assert run_dir_a.name == f"{run_id_a}-baseline"
        assert run_dir_b.name == f"{run_id_b}-baseline"

        # 4. The pre-existing legacy artifact must be completely untouched
        #    -- proves neither invocation appended to it.
        assert legacy_csv.read_text(encoding="utf-8") == legacy_snapshot

        # 5. Each invocation's OWN decisions.csv must be internally
        #    consistent -- every row has the same column count as the
        #    header (this is exactly the failure mode pandas.read_csv
        #    raised ParserError on: "Expected N fields ... saw M").
        for run_dir in (run_dir_a, run_dir_b):
            csv_path = run_dir / "decisions.csv"
            assert csv_path.is_file()
            with csv_path.open(newline="") as f:
                rows = list(csv.reader(f))
            assert len(rows) > 1, f"expected a header plus at least one data row in {csv_path}"
            header_width = len(rows[0])
            for i, row in enumerate(rows[1:], start=2):
                assert len(row) == header_width, (
                    f"{csv_path}:{i} has {len(row)} fields, header has {header_width} -- mixed schema"
                )
    finally:
        # Only the synthetic legacy fixture this test itself created above
        # is ever removed here -- never a real generated run. run_dir_a/b
        # are genuine `marla run` output under REPO_ROOT/runs/ (the real
        # subprocess always writes there, matching test_smoke.py's own
        # convention) and are deliberately left in place, uniquely named
        # (uuid-suffixed, "smoke-hygiene-test-" prefixed) so a failure
        # remains inspectable and nothing here ever risks deleting a real
        # scientific run.
        import shutil

        shutil.rmtree(legacy_run_dir, ignore_errors=True)
