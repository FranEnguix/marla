"""Sphinx configuration for MARLA's documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# So autodoc can `import marla` without an editable install being present
# (e.g. on Read the Docs' build image).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

project = "MARLA"
copyright = "2026, Fran Enguix"
author = "Fran Enguix"

try:
    from marla import __version__ as release
except Exception:  # pragma: no cover - docs must still build if the import chain is broken
    release = "0.2.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
]

# NASimEmu/torch/transformers/etc. are heavy, sometimes GPU-dependent
# imports that a docs build has no business requiring -- autodoc_mock_imports
# lets `automodule`/`autoclass` render signatures and docstrings without
# actually importing (or successfully running) these packages.
autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "spade",
    "slixmpp",
    "nasimemu",
    "transformers",
    "accelerate",
    "gym",
]
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- HTML output -------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
