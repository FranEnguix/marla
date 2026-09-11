"""The portable ``marla://`` scenario reference scheme.

MARLA's own packaged, solvability-checked scenarios (:mod:`marla.scenarios`)
need a way to be named in a config file that resolves identically no
matter *how* MARLA is installed or run from -- a source checkout, an
editable install, a ``pip``-installed wheel, a ``marla init``-generated
config, a saved run directory's ``config.yaml``, or a checkpoint-evaluation
config on a different machine entirely. A filesystem path baked into a
config (``/home/ubuntu/work/marla/NASimEmu/scenarios/...``) satisfies none
of these; it is specific to whoever happened to write it.

``marla://<filename>`` fixes this: it always resolves through
:func:`marla.scenarios.solvable_scenario_path`, i.e. against MARLA's own
installed package data via ``importlib.resources`` -- never relative to the
current working directory, a config file's directory, or any particular
checkout path. A persisted config keeps the URI itself; only a runtime
call site that actually needs a real filesystem path (constructing a
``NasimEmuAdapter``, invoking the solvability checker) materializes one,
and only transiently.

This module is the *one* place that interprets a ``config.environment
.scenario`` string as one of the three forms MARLA has always supported --
see :func:`resolve_scenario_reference`:

1. ``marla://<filename>``       -- a MARLA-owned packaged scenario.
2. anything ending in ``.yaml`` -- a filesystem path to a NASimEmu scenario
   file, resolved relative to ``config_dir`` if not already absolute
   (unchanged, pre-existing behavior).
3. anything else                -- passed through unchanged: a NASimEmu
   named/procedural scenario reference (e.g. ``"uniform-small-gen"``),
   which NASimEmu itself resolves, not MARLA.
"""

from __future__ import annotations

from pathlib import Path

from marla.scenarios import solvable_scenario_path

MARLA_SCENARIO_URI_PREFIX = "marla://"


class ScenarioReferenceError(ValueError):
    """Raised for a malformed, unresolvable, or unsafe ``marla://`` scenario
    reference -- never silently falls back to any other interpretation."""


def resolve_scenario_reference(reference: str, config_dir: Path) -> str:
    """Resolves a ``config.environment.scenario`` string to whatever a
    ``NasimEmuAdapter``/the solvability checker actually needs: a real
    filesystem path (``str``) for a ``marla://`` reference or a ``.yaml``
    filesystem path, or the untouched reference for a NASimEmu named/
    procedural scenario.

    ``config_dir``: the directory a relative filesystem ``.yaml`` path is
    resolved against (a config file's own directory) -- irrelevant for
    ``marla://`` references, which never depend on it.
    """
    if reference.startswith(MARLA_SCENARIO_URI_PREFIX):
        name = reference[len(MARLA_SCENARIO_URI_PREFIX) :]
        return str(_resolve_marla_scenario_name(name, reference))
    if reference.endswith(".yaml"):
        path = Path(reference)
        if not path.is_absolute():
            path = (config_dir / path).resolve()
        return str(path)
    return reference


def _resolve_marla_scenario_name(name: str, original_reference: str) -> Path:
    if not name:
        raise ScenarioReferenceError(
            f"Malformed marla:// scenario reference {original_reference!r}: no scenario name after 'marla://'."
        )
    if "/" in name or "\\" in name or ".." in name:
        raise ScenarioReferenceError(
            f"Invalid marla:// scenario reference {original_reference!r}: the scenario name must be a bare "
            "filename with no path separators or '..' components -- path traversal outside MARLA's packaged "
            "scenario resources is never permitted. Use e.g. 'marla://sm_entry_user_three_subnets.solvable.v2.yaml'."
        )
    if not name.endswith(".yaml"):
        raise ScenarioReferenceError(
            f"Invalid marla:// scenario reference {original_reference!r}: scenario names must end in '.yaml'."
        )
    try:
        return solvable_scenario_path(name)
    except FileNotFoundError as exc:
        raise ScenarioReferenceError(str(exc)) from exc
