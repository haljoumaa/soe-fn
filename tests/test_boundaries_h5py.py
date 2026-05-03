"""Enforce that h5py imports are isolated to the NGImager reader boundary."""

from __future__ import annotations

import ast
from pathlib import Path


ALLOWED = Path("src/soe/adapters/hdf5_ngimager/reader.py")
SCAN_ROOTS = (Path("src/soe"), Path("cli"))


def iter_python_files(root: Path) -> list[Path]:
    """Return all Python files under a root if it exists."""
    if not root.exists():
        return []
    return sorted(root.rglob("*.py"))


def imports_h5py(path: Path) -> bool:
    """Detect direct h5py imports using AST parsing."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "h5py" for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom):
            if node.module == "h5py":
                return True
    return False


def test_h5py_import_boundary() -> None:
    """Only the allowed module may import h5py; no import is also acceptable."""
    root = Path(__file__).resolve().parents[1]
    violations: list[str] = []

    for scan_root in SCAN_ROOTS:
        for file_path in iter_python_files(root / scan_root):
            rel = file_path.relative_to(root)
            if imports_h5py(file_path) and rel != ALLOWED:
                violations.append(str(rel))

    assert not violations, f"h5py import found outside allowed boundary: {violations}"
