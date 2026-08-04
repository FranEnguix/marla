"""Gatekeeper integration tests: accept / correction-loop / exhausted-rejection.

Runs a stub orchestrator + real Gatekeeper + deterministic mock Plan Maker
together via SPADE (see tests/support/). Each scenario is a fresh subprocess
(SPADE's Container is a process-wide singleton, see runtime/local.py).

KNOWN FLAKINESS: SPADE's bundled embedded XMPP server (pyjabber) has an
observed race condition in its roster/presence-subscription handling under
concurrent multi-agent startup -- empirically roughly 1-in-4 runs raise an
unhandled sqlite/asyncio error during startup or, less harmfully, during
teardown after the scenario already completed and printed its result. This
is a pyjabber robustness issue, not a MARLA logic bug: retrying is
appropriate here in a way it would not be for a functional flake. For real
experiments where a Gatekeeper/Plan Maker are involved, pointing
`xmpp.server` at a real deployed XMPP server (not the embedded one) avoids
this risk entirely.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_SCRIPT_ARGS = [sys.executable, "-m", "tests.support.run_gatekeeper_scenario"]


def _run_scenario(strategy: str, max_schema_revisions: int, attempts: int = 6) -> dict:
    last_error = None
    for _ in range(attempts):
        try:
            result = subprocess.run(
                [*SCENARIO_SCRIPT_ARGS, strategy, str(max_schema_revisions)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = f"TIMEOUT: {exc.stdout}\n{exc.stderr}"
            continue
        # The scenario's JSON result is the last line printed, regardless of
        # whether a pyjabber teardown-phase exception polluted stdout after it.
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if lines:
            try:
                return json.loads(lines[-1])
            except json.JSONDecodeError:
                pass
        last_error = result.stdout + result.stderr
    pytest.fail(f"Scenario never produced a valid result after {attempts} attempts:\n{last_error}")


@pytest.mark.integration
def test_gatekeeper_accepts_valid_response_immediately():
    result = _run_scenario("always_valid", max_schema_revisions=3)
    assert result["failure"] is None
    assert result["status"] == "accepted"
    assert result["scores"] == {"service-scan:host-1-0": 0.5, "finish": 0.5}


@pytest.mark.integration
def test_gatekeeper_correction_loop_recovers_on_second_attempt():
    result = _run_scenario("invalid_then_valid", max_schema_revisions=3)
    assert result["failure"] is None
    assert result["status"] == "accepted"
    assert result["scores"] == {"service-scan:host-1-0": 0.5, "finish": 0.5}


@pytest.mark.integration
def test_gatekeeper_rejects_after_exhausting_max_schema_revisions():
    result = _run_scenario("always_invalid", max_schema_revisions=2)
    assert result["failure"] is None
    assert result["status"] == "schema_rejected"
    assert result["scores"] is None
