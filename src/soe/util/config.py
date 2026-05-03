"""TOML config loader utility for infra-only config entrypoint conventions."""

from __future__ import annotations

from pathlib import Path
import tomllib


def load_toml(path: Path) -> dict:
    """Load a TOML file and return a dictionary payload."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return data
