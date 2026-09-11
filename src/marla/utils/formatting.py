"""Reusable "quantity" text formatting for CLI/display code -- one place
that gets the ``None`` case right, so a missing value always prints as a
clean ``n/a`` rather than a unit suffix glued onto the literal string
"n/a" (e.g. ``n/ams``, ``n/a%``). See ``marla.cli``'s ``summarize`` command
for callers.
"""

from __future__ import annotations


def format_quantity(value: float | int | None, unit: str = "", digits: int = 3) -> str:
    """``"n/a"`` when ``value`` is ``None`` (a unit suffix is only ever
    appended to an actual number); ``f"{value:.{digits}f}{unit}"``
    otherwise. Non-float values (e.g. an already-formatted ``int``) are
    stringified as-is, with the unit still appended.
    """
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"
