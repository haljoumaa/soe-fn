"""Run provenance helpers for infra-only metadata capture and run folder bookkeeping."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_git_sha(cwd: Path) -> str | None:
    """Return the current git commit SHA for a working tree, or None when unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None

    sha = result.stdout.strip()
    return sha or None


def _is_under_public_data_inputs(path: Path) -> bool:
    """Check whether a path points inside the public data input hierarchy."""
    parts = path.resolve().parts
    for idx in range(len(parts) - 1):
        if parts[idx] == "data" and parts[idx + 1] in {"toy_cases", "water_phantom"}:
            return True
    return False


def write_run_meta(run_dir: Path, *, argv: list[str], input_paths: list[str] = []) -> Path:
    """Create a run directory and write required provenance metadata to meta.json."""
    if _is_under_public_data_inputs(run_dir):
        raise ValueError("run_dir must not be inside public data inputs")

    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "meta.json"
    payload = {
        "run_id": run_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(Path.cwd()),
        "argv": list(argv),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "input_paths": list(input_paths),
    }
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return meta_path
