"""Enforce the exact Python version range MARLA is validated against."""

from __future__ import annotations

import sys


class UnsupportedPythonVersionError(RuntimeError):
    """Raised when MARLA is run under an unsupported Python interpreter."""


REQUIRED_MAJOR = 3
REQUIRED_MINOR = 10


def check_python_version(version_info: tuple[int, int, int, str, int] | None = None) -> None:
    """Raise :class:`UnsupportedPythonVersionError` unless running Python 3.10.x.

    Parameters
    ----------
    version_info:
        Defaults to ``sys.version_info``. Overridable for unit testing.
    """
    info = version_info if version_info is not None else sys.version_info
    major, minor = info[0], info[1]

    if (major, minor) != (REQUIRED_MAJOR, REQUIRED_MINOR):
        raise UnsupportedPythonVersionError(
            "MARLA requires Python >=3.10,<3.11 "
            f"(running {major}.{minor}.{info[2]})."
        )
