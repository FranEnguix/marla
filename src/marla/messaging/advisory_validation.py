"""Gatekeeper-side advisory response validation (spec sections 9-10), SPADE-free.

Separated from the schema models themselves because "exact legal action-ID
coverage" and "expected sender/request correlation" checks need context
(the original request's action IDs, the expected requester alias) that a
standalone Pydantic model can't carry. Kept independent of SPADE so it is
directly unit-testable.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from marla.messaging.schemas import AdvisoryResponsePayload


class AdvisoryValidationError(Exception):
    """A single reason an advisory response was rejected, matching spec section 10's checklist."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def validate_advisory_response_body(
    raw_body: dict[str, Any],
    expected_run_id: str,
    expected_request_id: str,
    expected_action_ids: set[str],
) -> AdvisoryResponsePayload:
    """Validate a Plan Maker response body against its originating request's context.

    Raises :class:`AdvisoryValidationError` with a specific ``reason`` on the
    first failing check: ``invalid_schema``, ``run_id_mismatch``,
    ``request_id_mismatch``, or ``action_id_coverage_mismatch``. Score
    numeric-type and ``[0,1]`` bound checks happen inside
    :class:`AdvisoryResponsePayload` itself and surface as ``invalid_schema``.
    """
    try:
        response = AdvisoryResponsePayload.model_validate(raw_body)
    except ValidationError as exc:
        raise AdvisoryValidationError("invalid_schema", str(exc)) from exc

    if response.run_id != expected_run_id:
        raise AdvisoryValidationError(
            "run_id_mismatch", f"{response.run_id!r} != {expected_run_id!r}"
        )
    if response.request_id != expected_request_id:
        raise AdvisoryValidationError(
            "request_id_mismatch", f"{response.request_id!r} != {expected_request_id!r}"
        )

    actual_ids = set(response.scores.keys())
    if actual_ids != expected_action_ids:
        missing = sorted(expected_action_ids - actual_ids)
        unknown = sorted(actual_ids - expected_action_ids)
        raise AdvisoryValidationError(
            "action_id_coverage_mismatch", f"missing={missing} unknown={unknown}"
        )

    return response
