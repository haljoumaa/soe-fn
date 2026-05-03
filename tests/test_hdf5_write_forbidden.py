"""Forbid HDF5 write-like API usage in source and CLI code."""

from pathlib import Path


FORBIDDEN_SNIPPETS = [
    "create_dataset",
    "require_group",
    "require_dataset",
    ".attrs[",
    "resize(",
]
SCAN_ROOTS = (Path("src/soe"), Path("cli"))


def iter_python_files(root: Path) -> list[Path]:
    """Return all Python files under a root if it exists."""
    if not root.exists():
        return []
    return sorted(root.rglob("*.py"))


def test_hdf5_write_apis_forbidden() -> None:
    """Source and CLI code must not include forbidden write-ish HDF5 snippets."""
    repo_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []

    for scan_root in SCAN_ROOTS:
        for file_path in iter_python_files(repo_root / scan_root):
            content = file_path.read_text(encoding="utf-8")
            for snippet in FORBIDDEN_SNIPPETS:
                if snippet in content:
                    rel = file_path.relative_to(repo_root)
                    violations.append(f"{rel}: contains '{snippet}'")

    assert not violations, "Forbidden HDF5 write APIs detected:\n" + "\n".join(violations)
