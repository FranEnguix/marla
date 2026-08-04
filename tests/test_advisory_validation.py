import pytest

from marla.messaging.advisory_validation import AdvisoryValidationError, validate_advisory_response_body

VALID_BODY = {
    "schema_version": "1.0",
    "run_id": "run-1",
    "request_id": "request-1",
    "scores": {"service-scan:host-1-0": 0.8, "finish": 0.1},
    "model_version": "plan-maker-v1",
    "prompt_version": "prompt-v1",
    "knowledge_version": "knowledge-v1",
    "inference_latency_ms": 12.5,
    "retrieved_rule_ids": ["scan-before-exploit"],
}
EXPECTED_ACTION_IDS = {"service-scan:host-1-0", "finish"}


def test_valid_response_passes():
    response = validate_advisory_response_body(VALID_BODY, "run-1", "request-1", EXPECTED_ACTION_IDS)
    assert response.scores["finish"] == 0.1


def test_missing_field_is_invalid_schema():
    body = {k: v for k, v in VALID_BODY.items() if k != "model_version"}
    with pytest.raises(AdvisoryValidationError) as excinfo:
        validate_advisory_response_body(body, "run-1", "request-1", EXPECTED_ACTION_IDS)
    assert excinfo.value.reason == "invalid_schema"


def test_score_out_of_bounds_is_invalid_schema():
    body = {**VALID_BODY, "scores": {**VALID_BODY["scores"], "finish": 1.5}}
    with pytest.raises(AdvisoryValidationError) as excinfo:
        validate_advisory_response_body(body, "run-1", "request-1", EXPECTED_ACTION_IDS)
    assert excinfo.value.reason == "invalid_schema"


def test_non_numeric_score_is_invalid_schema():
    body = {**VALID_BODY, "scores": {**VALID_BODY["scores"], "finish": "high"}}
    with pytest.raises(AdvisoryValidationError) as excinfo:
        validate_advisory_response_body(body, "run-1", "request-1", EXPECTED_ACTION_IDS)
    assert excinfo.value.reason == "invalid_schema"


def test_run_id_mismatch():
    body = {**VALID_BODY, "run_id": "different-run"}
    with pytest.raises(AdvisoryValidationError) as excinfo:
        validate_advisory_response_body(body, "run-1", "request-1", EXPECTED_ACTION_IDS)
    assert excinfo.value.reason == "run_id_mismatch"


def test_request_id_mismatch():
    body = {**VALID_BODY, "request_id": "different-request"}
    with pytest.raises(AdvisoryValidationError) as excinfo:
        validate_advisory_response_body(body, "run-1", "request-1", EXPECTED_ACTION_IDS)
    assert excinfo.value.reason == "request_id_mismatch"


def test_missing_action_id_coverage():
    body = {**VALID_BODY, "scores": {"finish": 0.1}}  # missing service-scan entry
    with pytest.raises(AdvisoryValidationError) as excinfo:
        validate_advisory_response_body(body, "run-1", "request-1", EXPECTED_ACTION_IDS)
    assert excinfo.value.reason == "action_id_coverage_mismatch"
    assert "service-scan:host-1-0" in excinfo.value.detail


def test_unknown_extra_action_id():
    body = {**VALID_BODY, "scores": {**VALID_BODY["scores"], "unknown-action": 0.5}}
    with pytest.raises(AdvisoryValidationError) as excinfo:
        validate_advisory_response_body(body, "run-1", "request-1", EXPECTED_ACTION_IDS)
    assert excinfo.value.reason == "action_id_coverage_mismatch"
    assert "unknown-action" in excinfo.value.detail


def test_exact_coverage_with_reordered_keys_still_passes():
    body = {**VALID_BODY, "scores": {"finish": 0.1, "service-scan:host-1-0": 0.8}}
    response = validate_advisory_response_body(body, "run-1", "request-1", EXPECTED_ACTION_IDS)
    assert set(response.scores) == EXPECTED_ACTION_IDS
