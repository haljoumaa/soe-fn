"""Smoke-test imports for the public reconstruction namespace layout."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_imports_smoke() -> None:
    """Import top-level and key subpackages."""
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    modules = [
        "soe",
        "soe.adapters",
        "soe.analysis.reconstruction_artifacts",
        "soe.analysis.reconstruction_diagnostics",
        "soe.analysis.reconstruction_run",
        "soe.core",
        "soe.geometry",
        "soe.soe",
        "soe.analysis",
    ]

    for name in modules:
        importlib.import_module(name)
