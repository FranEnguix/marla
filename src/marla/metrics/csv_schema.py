"""One authoritative dtype schema for reading MARLA's metrics CSVs back
with pandas -- decisions.csv in particular carries several genuinely
three-valued columns (``True`` / ``False`` / "not applicable"), and
letting pandas *infer* a dtype for a mixed True/False/empty column
produces ``object`` (silently losing the semantic distinction between
"False" and "not applicable") for a short/early-truncated read, or a
``DtypeWarning: Columns (N) have mixed types`` for a column whose
True/False/empty mix isn't resolved until a later chunk (``low_memory``'s
default chunked type-sniffing) -- neither is what a downstream analysis
should build on.

The fix is a real dtype, not a warning suppression:
:func:`read_decisions_csv` reads every present nullable-boolean column as
pandas' nullable ``"boolean"`` dtype (``True``/``False``/``pandas.NA`` --
never silently coerced to ``0.0``/``1.0``/``NaN``, and never conflated
with "False") and every present nullable-integer column as ``"Int64"``.
Passing ``dtype=`` up front also means pandas never needs its own
multi-pass type-sniffing for those columns, so the ``DtypeWarning`` never
fires -- ``low_memory=False`` alone would silence the warning but leave
the column as ``object``, which is exactly the "merely silence it"
outcome this module exists to avoid.

Old run directories written before a given column existed are handled by
construction: :func:`_present_dtypes` only requests a dtype for a column
that is actually in the file's header, so a missing column is simply
absent from the returned frame (pandas' normal behavior), never a
``KeyError``/``ValueError`` from requesting a dtype for a nonexistent
column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# True/False/"not applicable" -- pandas nullable "boolean" dtype
# (pd.NA, never coerced to 0.0/1.0/NaN, never confused with False).
DECISIONS_NULLABLE_BOOLEAN_COLUMNS: tuple[str, ...] = (
    # None whenever the selected action has no such requirement at all
    # (a scan/FINISH, or an OS-less exploit) -- see
    # marla.metrics.writer._decision_row's own comment.
    "selected_action_known_service_match",
    "selected_action_known_process_match",
    "selected_action_known_os_match",
    "selected_action_known_os_mismatch",
    # Defensive-None: MARLA always offers exactly one FINISH candidate in
    # practice, but these are read without asserting that.
    "base_finish_is_argmax",
    "final_finish_is_argmax",
    "finish_selected",
    # None unless a query was actually sampled this decision (assisted
    # variant only; always None for baseline/PPO_ONLY).
    "advice_changed_top_action",
    # The central case this module exists for: None means "no sensitive
    # target is currently visible", which is a different state from
    # "visible and not all rooted" (False) -- see
    # marla.environment.visible_facts.all_visible_sensitive_targets_rooted's
    # own docstring for why "no visible target" must never collapse into
    # True *or* False.
    "all_visible_sensitive_targets_rooted_before_action",
)

# Structurally always populated (never None) -- plain pandas "bool" is
# safe and unambiguous for these; listed explicitly so a reader/writer
# change that accidentally starts leaving one blank is caught by
# tests/test_csv_schema.py rather than silently reinterpreted as False.
DECISIONS_ORDINARY_BOOLEAN_COLUMNS: tuple[str, ...] = (
    "terminated",
    "truncated",
    "objective_satisfied",
    "objective_became_satisfied",
    "queried",
    "objective_satisfied_before_action",
    "known_exploration_frontier_remaining_before_action",
    "has_visible_sensitive_target_before_action",
)

# Nullable integers -- pandas "Int64" (capital I: the nullable extension
# dtype, pd.NA-aware), never plain "int64" (which cannot hold a null at
# all) or "float64" (which silently turns e.g. sensitive_targets_total
# into 3.0).
DECISIONS_NULLABLE_INTEGER_COLUMNS: tuple[str, ...] = (
    # Multi-environment PPO collection (spec: the 2048-transition, 4-
    # independent-environment-stream ablation) -- None for every single-
    # environment run.
    "env_index",
    "base_finish_rank",
    "final_finish_rank",
    "sensitive_targets_total",
    "sensitive_targets_with_root",
    "sensitive_targets_remaining",
    "sensitive_targets_remaining_before_action",
    "sensitive_targets_with_root_before_action",
    "plan_maker_input_tokens",
    "plan_maker_output_tokens",
    "plan_maker_total_tokens",
)

# Nullable floats -- ordinary "float64" already represents these
# correctly (NaN for "no value"), listed here only so this module is a
# complete, documented audit of decisions.csv's non-trivially-typed
# columns (spec: "do not special-case only column 72").
DECISIONS_NULLABLE_FLOAT_COLUMNS: tuple[str, ...] = (
    "base_finish_probability",
    "final_finish_probability",
    "beta",
    "alpha",
    "query_probability",
    "response_latency_ms",
)

# String/status columns that are sometimes empty (None) -- left as
# pandas' default "object" dtype (a nullable string dtype would be
# stricter, but these are never compared as booleans/numbers, so object
# +pandas.isna() is unambiguous and requires no behavior change here).
DECISIONS_NULLABLE_STRING_COLUMNS: tuple[str, ...] = (
    "request_id",
    "response_status",
    "plan_maker_top_action_id",
)


def _present_dtypes(csv_path: Path, dtype_map: dict[str, str]) -> dict[str, str]:
    """Restricts ``dtype_map`` to columns that actually exist in this
    file's header -- an older run directory missing a newer column must
    keep summarizing (spec: "old runs missing newer columns must continue
    to summarize"), never raise for a dtype requested on a column pandas
    never sees.
    """
    header = pd.read_csv(csv_path, nrows=0)
    return {column: dtype for column, dtype in dtype_map.items() if column in header.columns}


def decisions_dtype_map() -> dict[str, str]:
    """The full column -> pandas dtype mapping this module knows about
    (nullable-boolean and nullable-integer columns only -- ordinary
    booleans and floats are left to pandas' own correct-by-default
    inference, and string/status columns stay ``object``).
    """
    mapping: dict[str, str] = {}
    for column in DECISIONS_NULLABLE_BOOLEAN_COLUMNS:
        mapping[column] = "boolean"
    for column in DECISIONS_NULLABLE_INTEGER_COLUMNS:
        mapping[column] = "Int64"
    return mapping


def read_decisions_csv(csv_path: Path | str) -> pd.DataFrame:
    """Reads a ``decisions.csv`` with every present nullable-boolean
    column as pandas' nullable ``"boolean"`` dtype and every present
    nullable-integer column as ``"Int64"`` -- never ``object``, never a
    silent ``0.0``/``1.0``/``NaN`` float coercion, and never a
    ``DtypeWarning`` (passing ``dtype=`` up front means pandas never needs
    to sniff chunks to resolve a column's type).
    """
    csv_path = Path(csv_path)
    dtypes = _present_dtypes(csv_path, decisions_dtype_map())
    return pd.read_csv(csv_path, dtype=dtypes)
