"""Require any h5py.File call to provide literal read-only mode \"r\"."""

from __future__ import annotations

import ast
from pathlib import Path


SCAN_ROOTS = (Path("src/soe"), Path("cli"))


def iter_python_files(root: Path) -> list[Path]:
    """Return all Python files under a root if it exists."""
    if not root.exists():
        return []
    return sorted(root.rglob("*.py"))


def is_h5py_file_call(node: ast.Call) -> bool:
    """Check whether a call is h5py.File(...)."""
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "File"
        and isinstance(func.value, ast.Name)
        and func.value.id == "h5py"
    )


def mode_is_literal_r(node: ast.Call) -> bool:
    """Require explicit literal mode \"r\" as positional arg 2 or keyword mode."""
    if len(node.args) >= 2:
        arg = node.args[1]
        return isinstance(arg, ast.Constant) and arg.value == "r"

    for kw in node.keywords:
        if kw.arg == "mode":
            return isinstance(kw.value, ast.Constant) and kw.value.value == "r"

    return False


def test_h5py_file_open_is_readonly_literal_r() -> None:
    """If h5py.File calls exist, each must use literal read-only mode \"r\"."""
    repo_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []

    for scan_root in SCAN_ROOTS:
        for file_path in iter_python_files(repo_root / scan_root):
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and is_h5py_file_call(node):
                    if not mode_is_literal_r(node):
                        rel = file_path.relative_to(repo_root)
                        violations.append(f"{rel}:{node.lineno}")

    assert not violations, (
        "h5py.File calls must use literal read-only mode 'r'. Violations: "
        + ", ".join(violations)
    )
