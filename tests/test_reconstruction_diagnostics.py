"""Focused tests for reconstruction diagnostics sidecars."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from soe.analysis.reconstruction_artifacts import load_reconstruction_metadata
from soe.analysis.reconstruction_diagnostics import (
    RECONSTRUCTION_DIAGNOSTICS_FILENAME,
    load_reconstruction_diagnostics,
    validate_reconstruction_diagnostics_payload,
)
from soe.analysis.reconstruction_run import (
    ReconstructionArtifactsConfig,
    ReconstructionRunConfig,
    ReconstructionSeeds,
    run_reconstruction,
)
from soe.contracts import EventObj, ReconstructionInput, VOIConfig
from soe.soe.run_protocol import FiniteRunConfig
from soe.soe.state import SampledSurvivalConfig


def _synthetic_reconstruction_input() -> ReconstructionInput:
    voi = VOIConfig(
        bounds=np.asarray(
            [
                [-1.0, 1.0],
                [1.0, 3.0],
                [-1.0, 1.0],
            ],
            dtype=float,
        ),
        grid_shape=(4, 4, 4),
    )
    event = EventObj(
        apex=np.asarray([0.0, 0.0, 0.0], dtype=float),
        axis=np.asarray([0.0, 1.0, 0.0], dtype=float),
        theta=math.pi / 4.0,
    )
    return ReconstructionInput(events=(event,), voi=voi)


def test_run_reconstruction_emits_separate_diagnostics_sidecar(tmp_path: Path) -> None:
    result = run_reconstruction(
        ReconstructionRunConfig(
            reconstruction_input=_synthetic_reconstruction_input(),
            artifacts=ReconstructionArtifactsConfig(artifacts_dir=tmp_path / "artifacts"),
            survival_config=SampledSurvivalConfig.fixed_full_box(K=8),
            run_config=FiniteRunConfig(
                num_steps=4,
                burn_in_steps=1,
                thin_every=2,
                capture_terminal_state=True,
            ),
            seeds=ReconstructionSeeds(
                event_index_selector=11,
                proposal_backend=12,
                acceptance_uniform=13,
            ),
            surviving_event_cap=1,
        )
    )

    artifacts_dir = tmp_path / "artifacts"
    diagnostics_path = artifacts_dir / RECONSTRUCTION_DIAGNOSTICS_FILENAME
    metadata_path = artifacts_dir / "reconstruction_metadata.json"

    assert diagnostics_path.is_file()
    diagnostics = load_reconstruction_diagnostics(artifacts_dir)
    metadata = load_reconstruction_metadata(artifacts_dir)
    raw_diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert diagnostics.counts.surviving_event_count == len(result.initial_state.events)
    assert diagnostics.counts.filtered_valid_event_count == len(
        result.config.reconstruction_input.events
    )
    assert diagnostics.counts.dropped_event_count == (
        diagnostics.counts.filtered_valid_event_count
        - diagnostics.counts.surviving_event_count
    )
    assert diagnostics.preprocessing.survival_mode == "fixed"
    assert diagnostics.preprocessing.gate_mode == "full_box"
    assert diagnostics.preprocessing.K == 8
    assert diagnostics.preprocessing.runtime_seconds is not None
    assert diagnostics.preprocessing.runtime_seconds >= 0.0
    assert diagnostics.chain.burn_in_steps == 1
    assert diagnostics.chain.thin_every == 2
    assert diagnostics.chain.retained_count == len(result.chain_result.retained_states)
    assert diagnostics.chain.total_steps_attempted == result.chain_result.total_steps
    assert diagnostics.chain.accepted_step_count == result.chain_result.accepted_step_count
    assert diagnostics.chain.rejected_step_count == (
        diagnostics.chain.total_steps_attempted - diagnostics.chain.accepted_step_count
    )
    assert diagnostics.chain.runtime_seconds is not None
    assert diagnostics.chain.runtime_seconds >= 0.0
    assert diagnostics.chain.chain_core_runtime_seconds is not None
    assert diagnostics.chain.chain_core_runtime_seconds >= 0.0
    assert diagnostics.chain.retained_mean_accumulation_runtime_seconds is not None
    assert diagnostics.chain.retained_mean_accumulation_runtime_seconds >= 0.0
    assert diagnostics.chain.artifact_write_runtime_seconds is not None
    assert diagnostics.chain.artifact_write_runtime_seconds >= 0.0
    assert diagnostics.chain.runtime_seconds >= diagnostics.chain.chain_core_runtime_seconds
    assert diagnostics.chain.total_runtime_seconds is not None
    assert diagnostics.chain.total_runtime_seconds >= diagnostics.chain.runtime_seconds
    assert diagnostics.chain.total_runtime_seconds >= diagnostics.chain.artifact_write_runtime_seconds
    if diagnostics.chain.total_steps_attempted > 0:
        assert diagnostics.chain.acceptance_rate == pytest.approx(
            diagnostics.chain.accepted_step_count / diagnostics.chain.total_steps_attempted
        )
    else:
        assert diagnostics.chain.acceptance_rate is None
    assert diagnostics.state_summary.grid_shape == (4, 4, 4)
    assert diagnostics.state_summary.total_voxel_count == 64
    assert diagnostics.state_summary.terminal_occupancy_sum == int(
        np.sum(result.terminal_state, dtype=np.int64)
    )
    assert diagnostics.state_summary.retained_mean_occupancy_sum == pytest.approx(
        float(np.sum(result.retained_mean, dtype=float))
    )
    assert diagnostics.seeds.event_index_selector == 11
    assert diagnostics.seeds.proposal_backend == 12
    assert diagnostics.seeds.acceptance_uniform == 13

    assert "counts" not in raw_metadata
    assert "preprocessing" not in raw_metadata
    assert "chain" not in raw_metadata
    assert "state_summary" not in raw_metadata
    assert "accepted_step_count" not in raw_metadata["provenance"]
    assert raw_diagnostics["counts"]["surviving_event_count"] == 1
    assert raw_diagnostics["chain"]["accepted_step_count"] == result.chain_result.accepted_step_count
    assert "chain_core_runtime_seconds" in raw_diagnostics["chain"]
    assert "retained_mean_accumulation_runtime_seconds" in raw_diagnostics["chain"]
    assert "artifact_write_runtime_seconds" in raw_diagnostics["chain"]
    assert metadata.provenance.retained_count == diagnostics.chain.retained_count


def test_validate_reconstruction_diagnostics_rejects_inconsistent_chain_metrics() -> None:
    with pytest.raises(ValueError, match="acceptance_rate"):
        validate_reconstruction_diagnostics_payload(
            {
                "schema_name": "soe_reconstruction_diagnostics",
                "schema_version": 1,
                "counts": {
                    "raw_event_count": None,
                    "filtered_valid_event_count": 2,
                    "surviving_event_count": 1,
                    "dropped_event_count": 1,
                },
                "preprocessing": {
                    "survival_mode": "fixed",
                    "gate_mode": "full_box",
                    "K": 8,
                    "surviving_event_cap": None,
                    "runtime_seconds": 0.1,
                },
                "chain": {
                    "total_steps_attempted": 4,
                    "accepted_step_count": 1,
                    "rejected_step_count": 3,
                    "acceptance_rate": 0.5,
                    "burn_in_steps": 0,
                    "thin_every": 1,
                    "retained_count": 4,
                    "terminal_step_index": 4,
                    "runtime_seconds": 0.2,
                    "chain_core_runtime_seconds": 0.1,
                    "retained_mean_accumulation_runtime_seconds": 0.05,
                    "artifact_write_runtime_seconds": 0.01,
                    "total_runtime_seconds": 0.3,
                },
                "state_summary": {
                    "grid_shape": [1, 1, 1],
                    "total_voxel_count": 1,
                    "terminal_occupancy_sum": 1,
                    "retained_mean_occupancy_sum": 1.0,
                    "terminal_nonzero_voxel_count": 1,
                    "retained_mean_nonzero_voxel_count": 1,
                },
                "seeds": {
                    "event_index_selector": 11,
                    "proposal_backend": 12,
                    "acceptance_uniform": 13,
                },
            }
        )
