"""Best-effort extraction of a scores mapping from raw (possibly noisy) LM output.

An unparsable or malformed response is a normal, expected outcome -- the
Gatekeeper's correction loop exists precisely to handle it -- so this module
never raises; it returns ``None``/drops entries it can't make sense of.
"""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(raw_text: str) -> dict | None:
    """Extract the first top-level JSON object found in ``raw_text``, if any."""
    text = raw_text.strip()

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def coerce_scores(raw_scores: dict) -> dict[str, float]:
    """Best-effort coercion of a raw {action_id: value} mapping to floats.

    Entries that can't be coerced are dropped (not defaulted) -- the
    Gatekeeper's exact-coverage check will then correctly flag the response
    as invalid rather than MARLA silently inventing a score.
    """
    coerced: dict[str, float] = {}
    for key, value in raw_scores.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            coerced[key] = float(value)
            continue
        if isinstance(value, str):
            try:
                coerced[key] = float(value)
            except ValueError:
                continue
    return coerced
