#!/usr/bin/env python3
"""Thin authoritative reconstruction entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from soe.analysis import reconstruction_run as reconstruction_analysis
from soe.util.config import load_toml


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run reconstruction from an explicit TOML config and write the "
            "authoritative reconstruction artifact bundle. Relative paths "
            "inside the TOML are resolved from the TOML file's directory."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to the reconstruction TOML config. Recommended convention: "
            "keep reconstruction TOMLs under orchestration/configs/. The active "
            "reconstruction run surface lives under orchestration/, not under "
            "dataset examples."
        ),
    )
    parser.add_argument(
        "--admission-only",
        action="store_true",
        help=(
            "Run only the live reconstruction admission boundary, writing "
            "survivor provenance sidecars and skipping MH plus image outputs."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config_payload = load_toml(config_path)
    result = reconstruction_analysis.run_reconstruction_job(
        config_payload,
        base_dir=config_path.parent,
        admission_only=args.admission_only,
    )
    if args.admission_only:
        print(result.artifact_bundle.admitted_event_ids_path)
        print(result.artifact_bundle.metadata_path)
    else:
        print(result.artifact_bundle.retained_mean_path)
        print(result.artifact_bundle.terminal_state_path)
        print(result.artifact_bundle.metadata_path)
        print(result.artifact_bundle.diagnostics_path)
        print(result.artifact_bundle.chain_health_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
