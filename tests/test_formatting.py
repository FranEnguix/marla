"""marla.utils.formatting.format_quantity -- a unit suffix must only ever
be appended to an actual number, never glued onto the literal "n/a"
string (regression for a real observed bug: "mean Plan Maker latency:
n/ams" when the value was None and "ms" was concatenated unconditionally).
"""

from __future__ import annotations

from marla.utils.formatting import format_quantity


def test_none_formats_as_bare_na_regardless_of_unit():
    assert format_quantity(None, "ms") == "n/a"
    assert format_quantity(None, "%") == "n/a"
    assert format_quantity(None, " kWh") == "n/a"
    assert format_quantity(None) == "n/a"


def test_float_value_gets_unit_suffix_and_requested_precision():
    assert format_quantity(12.3456, "ms") == "12.346ms"
    assert format_quantity(12.3456, "ms", digits=1) == "12.3ms"
    assert format_quantity(0.00054591, " kg", digits=6) == "0.000546 kg"


def test_zero_is_a_real_value_not_treated_as_missing():
    assert format_quantity(0.0, "ms") == "0.000ms"


def test_int_value_is_stringified_with_unit_no_forced_decimal():
    assert format_quantity(3, " steps") == "3 steps"
