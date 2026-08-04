"""Best-effort version/environment info, shared by ``marla version`` and metadata.json."""

from __future__ import annotations

import subprocess
import sys

_TRACKED_MODULES = ("torch", "torch_geometric", "spade", "pydantic", "typer", "nasimemu")


def collect_dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in _TRACKED_MODULES:
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[module_name] = "not installed"
    return versions


def python_version() -> str:
    return sys.version.split()[0]


def git_commit() -> str | None:
    """Best-effort current commit hash; ``None`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
