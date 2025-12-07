"""Sphinx configuration for Streamlined CRISPR documentation."""

from __future__ import annotations

import sys
from pathlib import Path
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

with (PROJECT_ROOT / "pyproject.toml").open("rb") as fh:
    project_meta = tomllib.load(fh)

project = "crispyx"
version = project_meta.get("project", {}).get("version", "0.0.0")
release = version
copyright = "2024, Streamlining CRISPR Team"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

html_theme = "alabaster"
html_static_path: list[str] = []
