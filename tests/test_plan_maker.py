"""Real PlanMakerAgent integration tests: message handling, knowledge
retrieval wiring, and response parsing/coercion -- using fake (injected)
backends for speed and determinism. A real small-model end-to-end check
lives in test_plan_maker_real_model.py.

Runs in a subprocess (SPADE's Container is a process-wide singleton) and is
tolerant of pyjabber's known presence-subscription flakiness (see
tests/test_gatekeeper.py's module docstring).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_ARGS = [sys.executable, "-m", "tests.support.run_real_plan_maker_scenario"]


def _run_scenario(backend_kind: str, max_schema_revisions: int, attempts: int = 6, timeout: int = 60) -> dict:
    last_output = None
    for _ in range(attempts):
        try:
            result = subprocess.run(
                [*SCRIPT_ARGS, backend_kind, str(max_schema_revisions)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            last_output = f"TIMEOUT after {timeout}s: {exc.stdout}\n{exc.stderr}"
            continue
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if lines:
            try:
                return json.loads(lines[-1])
            except json.JSONDecodeError:
                pass
        last_output = result.stdout + result.stderr
    pytest.fail(f"Scenario never produced a valid result after {attempts} attempts:\n{last_output}")


@pytest.mark.integration
def test_plan_maker_accepts_valid_json_response():
    result = _run_scenario("fake_valid", max_schema_revisions=3)
    assert result["failure"] is None
    assert result["status"] == "accepted"
    assert result["scores"] == {"service-scan:host-1-0": 0.8, "finish": 0.1}
    assert isinstance(result["retrieved_rule_ids"], list)


@pytest.mark.integration
def test_plan_maker_extracts_json_from_markdown_fence():
    result = _run_scenario("fake_fenced_json", max_schema_revisions=3)
    assert result["failure"] is None
    assert result["status"] == "accepted"
    assert result["scores"] == {"service-scan:host-1-0": 0.6, "finish": 0.2}


@pytest.mark.integration
def test_plan_maker_garbage_output_exhausts_corrections_and_rejects():
    result = _run_scenario("fake_garbage", max_schema_revisions=2)
    assert result["failure"] is None
    assert result["status"] == "schema_rejected"
    assert result["scores"] is None


@pytest.mark.integration
def test_plan_maker_real_tiny_model_end_to_end():
    """A genuinely tiny (~untrained-quality) real HF model, end-to-end through
    the real LocalTransformersBackend, GatekeeperAgent, and PlanMakerAgent.

    Its output is essentially random text, so it reliably exercises the
    correction-loop-to-rejection path rather than the accept path -- that's
    fine, this test's purpose is to confirm the full pipeline (model load,
    prompt build, generate, parse, respond) runs without error or hanging,
    not to check response quality.
    """
    result = _run_scenario("real_tiny", max_schema_revisions=3, timeout=45)
    assert result["failure"] is None
    assert result["status"] in ("accepted", "schema_rejected")
