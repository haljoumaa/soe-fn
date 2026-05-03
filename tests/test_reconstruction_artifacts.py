"""Focused tests for reconstruction artifact metadata sidecars."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from soe.analysis.reconstruction_artifacts import (
    ADMITTED_EVENT_IDS_FILENAME,
    RECONSTRUCTION_METADATA_FILENAME,
    RECONSTRUCTION_EXECUTION_MODE_ADMISSION_ONLY,
    RECONSTRUCTION_EXECUTION_MODE_FULL_RUN,
    ReconstructionAdmissionMetadata,
    ReconstructionAdmissionArtifactBundle,
    ReconstructionAdmittedEventIdentity,
    ReconstructionAdmittedEventIds,
    ReconstructionGeometryMetadata,
    TERMINAL_OCCUPANCY_COUNTS_QUANTITY_LABEL,
    load_reconstruction_admitted_event_ids,
    load_reconstruction_metadata,
    validate_reconstruction_admitted_event_ids_payload,
    validate_reconstruction_metadata_payload,
    write_reconstruction_admission_artifact_bundle,
    write_reconstruction_artifact_bundle,
    write_reconstruction_metadata_sidecar,
)
from soe.analysis.reconstruction_diagnostics import (
    RECONSTRUCTION_DIAGNOSTICS_FILENAME,
    ReconstructionChainDiagnostics,
    ReconstructionCountsDiagnostics,
    ReconstructionDiagnostics,
    ReconstructionPreprocessingDiagnostics,
    ReconstructionSeedDiagnostics,
    ReconstructionStateSummaryDiagnostics,
    load_reconstruction_diagnostics,
)
from soe.soe.chain_health import (
    CHAIN_HEALTH_FILENAME,
    ChainHealthCollector,
    load_chain_health,
    validate_chain_health_payload,
)


def _example_diagnostics() -> ReconstructionDiagnostics:
    return ReconstructionDiagnostics(
        counts=ReconstructionCountsDiagnostics(
            raw_event_count=None,
            filtered_valid_event_count=6,
            surviving_event_count=4,
            dropped_event_count=2,
        ),
        preprocessing=ReconstructionPreprocessingDiagnostics(
            survival_mode="fixed",
            gate_mode="full_box",
            K=8,
            surviving_event_cap=None,
            runtime_seconds=0.125,
        ),
        chain=ReconstructionChainDiagnostics(
            total_steps_attempted=20,
            accepted_step_count=7,
            rejected_step_count=13,
            acceptance_rate=0.35,
            burn_in_steps=2,
            thin_every=3,
            retained_count=4,
            terminal_step_index=20,
            runtime_seconds=0.75,
            total_runtime_seconds=0.875,
        ),
        state_summary=ReconstructionStateSummaryDiagnostics(
            grid_shape=(1, 2, 3),
            total_voxel_count=6,
            terminal_occupancy_sum=4,
            retained_mean_occupancy_sum=4.0,
            terminal_nonzero_voxel_count=6,
            retained_mean_nonzero_voxel_count=6,
        ),
        seeds=ReconstructionSeedDiagnostics(
            event_index_selector=11,
            proposal_backend=12,
            acceptance_uniform=13,
        ),
    )


def _example_chain_health():
    collector = ChainHealthCollector()
    collector.record_step(
        same_voxel=True,
        old_count=2,
        new_count=2,
        accepted=True,
        proposal_ratio_was_omitted=True,
    )
    collector.record_step(
        same_voxel=False,
        old_count=1,
        new_count=0,
        accepted=True,
        proposal_ratio_was_omitted=True,
    )
    collector.record_step(
        same_voxel=False,
        old_count=2,
        new_count=0,
        accepted=False,
        proposal_ratio_was_omitted=True,
    )
    return collector.build_report(
        run_id="artifact-roundtrip",
        artifact_dir=Path("/tmp/example-artifacts"),
        total_attempted_steps=3,
        burn_in_steps=2,
        thin_every=3,
        surviving_event_count=4,
        total_voxel_count=6,
        grid_shape=(1, 2, 3),
        alpha=1.0,
        terminal_nonzero_voxel_count=2,
        retained_mean_nonzero_voxel_count=3,
    )


def _example_admitted_event_ids(artifacts_dir: Path) -> ReconstructionAdmittedEventIds:
    return ReconstructionAdmittedEventIds(
        admitted_events=(
            ReconstructionAdmittedEventIdentity(reconstruction_input_index=1),
            ReconstructionAdmittedEventIdentity(reconstruction_input_index=3),
        ),
        surviving_event_count=2,
        admission=ReconstructionAdmissionMetadata(
            gate_mode="full_box",
            survival_mode="fixed",
            K=8,
            surviving_event_cap=2,
        ),
        geometry=ReconstructionGeometryMetadata(
            bounds_cm=np.asarray(
                [
                    [0.0, 1.0],
                    [2.0, 4.0],
                    [5.0, 11.0],
                ],
                dtype=float,
            ),
            grid_shape=(1, 2, 3),
            voxel_size_cm=(1.0, 1.0, 2.0),
        ),
        artifacts_dir=artifacts_dir,
        run_id="artifact-roundtrip",
    )


def test_reconstruction_artifact_bundle_roundtrip(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    bundle = write_reconstruction_artifact_bundle(
        artifacts_dir,
        retained_mean=np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=float),
        terminal_state=np.asarray([[[3, 2, 1], [6, 5, 4]]], dtype=np.int64),
        retained_mean_filename="retained_mean.npy",
        terminal_state_filename="terminal_state.npy",
        bounds_cm=np.asarray(
            [
                [0.0, 1.0],
                [2.0, 4.0],
                [5.0, 11.0],
            ],
            dtype=float,
        ),
        grid_shape=(1, 2, 3),
        burn_in_steps=2,
        thin_every=3,
        retained_count=4,
        diagnostics=_example_diagnostics(),
        chain_health=_example_chain_health(),
        total_steps=20,
        terminal_step_index=20,
    )

    assert bundle.retained_mean_path.is_file()
    assert bundle.terminal_state_path.is_file()
    assert bundle.diagnostics_path.is_file()
    assert bundle.chain_health_path.is_file()
    np.testing.assert_allclose(
        np.load(bundle.retained_mean_path),
        np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=float),
    )
    np.testing.assert_array_equal(
        np.load(bundle.terminal_state_path),
        np.asarray([[[3, 2, 1], [6, 5, 4]]], dtype=np.int64),
    )

    metadata_path = artifacts_dir / RECONSTRUCTION_METADATA_FILENAME
    diagnostics_path = artifacts_dir / RECONSTRUCTION_DIAGNOSTICS_FILENAME
    chain_health_path = artifacts_dir / CHAIN_HEALTH_FILENAME
    assert metadata_path.is_file()
    assert diagnostics_path.is_file()
    assert chain_health_path.is_file()

    loaded = load_reconstruction_metadata(artifacts_dir)
    diagnostics = load_reconstruction_diagnostics(artifacts_dir)
    chain_health = load_chain_health(artifacts_dir)
    validated = validate_reconstruction_metadata_payload(
        json.loads(metadata_path.read_text(encoding="utf-8"))
    )
    validated_chain_health = validate_chain_health_payload(
        json.loads(chain_health_path.read_text(encoding="utf-8"))
    )
    assert validated.to_payload() == loaded.to_payload()
    assert diagnostics.to_payload() == _example_diagnostics().to_payload()
    assert validated_chain_health.to_payload() == chain_health.to_payload()
    assert chain_health.to_payload() == _example_chain_health().to_payload()
    np.testing.assert_allclose(
        loaded.geometry.bounds_cm,
        np.asarray(
            [
                [0.0, 1.0],
                [2.0, 4.0],
                [5.0, 11.0],
            ],
            dtype=float,
        ),
    )
    assert loaded.schema_name == "soe_reconstruction_metadata"
    assert loaded.schema_version == 1
    assert loaded.execution_mode == RECONSTRUCTION_EXECUTION_MODE_FULL_RUN
    assert loaded.artifacts.retained_mean.filename == "retained_mean.npy"
    assert loaded.artifacts.retained_mean.quantity_label == "mean_occupancy"
    assert loaded.artifacts.terminal_state.filename == "terminal_state.npy"
    assert loaded.artifacts.terminal_state.quantity_label == TERMINAL_OCCUPANCY_COUNTS_QUANTITY_LABEL
    assert loaded.geometry.grid_shape == (1, 2, 3)
    assert loaded.geometry.voxel_size_cm == (1.0, 1.0, 2.0)
    assert loaded.geometry.axis_order == ("x", "y", "z")
    assert loaded.provenance.burn_in_steps == 2
    assert loaded.provenance.thin_every == 3
    assert loaded.provenance.retained_count == 4
    assert loaded.provenance.total_steps == 20
    assert loaded.provenance.terminal_step_index == 20

    raw_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    diagnostics_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert "beam_axis" not in raw_payload
    assert "r_ref" not in raw_payload
    assert "evaluation_roi" not in raw_payload
    assert "accepted_step_count" not in raw_payload["provenance"]
    assert "counts" not in raw_payload
    assert raw_payload["artifacts"]["retained_mean"]["filename"] == "retained_mean.npy"
    assert raw_payload["artifacts"]["terminal_state"]["filename"] == "terminal_state.npy"
    assert diagnostics_payload["counts"]["surviving_event_count"] == 4
    assert diagnostics_payload["chain"]["accepted_step_count"] == 7
    assert diagnostics_payload["preprocessing"]["gate_mode"] == "full_box"
    chain_health_payload = json.loads(chain_health_path.read_text(encoding="utf-8"))
    assert chain_health_payload["metadata"]["run_id"] == "artifact-roundtrip"
    assert chain_health_payload["metadata"]["total_attempted_steps"] == 3
    assert chain_health_payload["rates"]["overall_acceptance_rate"] == pytest.approx(2.0 / 3.0)
    assert chain_health_payload["histograms"]["old_count"] == {"1": 1, "2": 2}
    assert chain_health_payload["pair_table"]["2"]["0"]["rejected_count"] == 1


def test_reconstruction_artifact_bundle_writes_admitted_event_ids_sidecar(
    tmp_path: Path,
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    bundle = write_reconstruction_artifact_bundle(
        artifacts_dir,
        retained_mean=np.asarray([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=float),
        terminal_state=np.asarray([[[3, 2, 1], [6, 5, 4]]], dtype=np.int64),
        retained_mean_filename="retained_mean.npy",
        terminal_state_filename="terminal_state.npy",
        bounds_cm=np.asarray(
            [
                [0.0, 1.0],
                [2.0, 4.0],
                [5.0, 11.0],
            ],
            dtype=float,
        ),
        grid_shape=(1, 2, 3),
        burn_in_steps=2,
        thin_every=3,
        retained_count=4,
        admitted_event_ids=_example_admitted_event_ids(artifacts_dir),
        diagnostics=_example_diagnostics(),
        chain_health=_example_chain_health(),
        total_steps=20,
        terminal_step_index=20,
    )

    admitted_event_ids_path = artifacts_dir / ADMITTED_EVENT_IDS_FILENAME
    assert admitted_event_ids_path.is_file()
    assert bundle.admitted_event_ids_path == admitted_event_ids_path.resolve()
    assert bundle.metadata.artifacts.admitted_event_ids_filename == ADMITTED_EVENT_IDS_FILENAME
    assert bundle.admitted_event_ids is not None
    assert bundle.admitted_event_ids.to_payload() == _example_admitted_event_ids(artifacts_dir).to_payload()

    loaded = load_reconstruction_admitted_event_ids(artifacts_dir)
    validated = validate_reconstruction_admitted_event_ids_payload(
        json.loads(admitted_event_ids_path.read_text(encoding="utf-8"))
    )
    assert loaded.to_payload() == validated.to_payload()
    assert loaded.to_payload() == _example_admitted_event_ids(artifacts_dir).to_payload()

    metadata_payload = json.loads(
        (artifacts_dir / RECONSTRUCTION_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert metadata_payload["execution_mode"] == RECONSTRUCTION_EXECUTION_MODE_FULL_RUN
    assert metadata_payload["artifacts"]["admitted_event_ids_filename"] == ADMITTED_EVENT_IDS_FILENAME

    admitted_payload = json.loads(admitted_event_ids_path.read_text(encoding="utf-8"))
    assert admitted_payload["run_id"] == "artifact-roundtrip"
    assert admitted_payload["artifacts_dir"] == str(artifacts_dir.resolve())
    assert admitted_payload["surviving_event_count"] == 2
    assert admitted_payload["admission"]["gate_mode"] == "full_box"
    assert admitted_payload["admission"]["survival_mode"] == "fixed"
    assert admitted_payload["admission"]["K"] == 8
    assert admitted_payload["admitted_events"] == [
        {"reconstruction_input_index": 1},
        {"reconstruction_input_index": 3},
    ]


@pytest.mark.parametrize(
    ("payload", "error_pattern"),
    [
        (
            {
                "schema_name": "soe_reconstruction_metadata",
                "schema_version": 1,
                "artifacts": {
                    "retained_mean": {
                        "filename": "retained_mean.npy",
                        "quantity_label": "mean_occupancy",
                    },
                    "terminal_state": {
                        "filename": "terminal_state.npy",
                        "quantity_label": TERMINAL_OCCUPANCY_COUNTS_QUANTITY_LABEL,
                    },
                },
                "geometry": {
                    "grid_shape": [1, 2, 3],
                    "voxel_size_cm": [1.0, 1.0, 2.0],
                    "axis_order": ["x", "y", "z"],
                },
                "provenance": {
                    "burn_in_steps": 0,
                    "thin_every": 1,
                    "retained_count": 1,
                },
            },
            "geometry.bounds_cm",
        ),
        (
            {
                "schema_name": "soe_reconstruction_metadata",
                "schema_version": 1,
                "artifacts": {
                    "retained_mean": {
                        "filename": "retained_mean.npy",
                        "quantity_label": "mean_occupancy",
                    },
                    "terminal_state": {
                        "filename": "terminal_state.npy",
                        "quantity_label": TERMINAL_OCCUPANCY_COUNTS_QUANTITY_LABEL,
                    },
                },
                "geometry": {
                    "bounds_cm": [[0.0, 1.0], [2.0, 4.0], [5.0, 11.0]],
                    "grid_shape": [1, 2, 3],
                    "voxel_size_cm": [1.0, 1.0, 2.0],
                    "axis_order": ["z", "y", "x"],
                },
                "provenance": {
                    "burn_in_steps": 0,
                    "thin_every": 1,
                    "retained_count": 1,
                },
            },
            "geometry.axis_order",
        ),
    ],
)
def test_load_reconstruction_metadata_rejects_incomplete_or_malformed_payload(
    tmp_path: Path,
    payload: dict[str, object],
    error_pattern: str,
) -> None:
    metadata_path = tmp_path / RECONSTRUCTION_METADATA_FILENAME
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=error_pattern):
        load_reconstruction_metadata(metadata_path)


def test_write_reconstruction_admission_artifact_bundle_writes_only_sidecars(
    tmp_path: Path,
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    bundle = write_reconstruction_admission_artifact_bundle(
        artifacts_dir,
        bounds_cm=np.asarray(
            [
                [0.0, 1.0],
                [2.0, 4.0],
                [5.0, 11.0],
            ],
            dtype=float,
        ),
        grid_shape=(1, 2, 3),
        admitted_event_ids=_example_admitted_event_ids(artifacts_dir),
        burn_in_steps=2,
        thin_every=3,
    )

    assert isinstance(bundle, ReconstructionAdmissionArtifactBundle)
    assert bundle.admitted_event_ids_path.is_file()
    assert bundle.metadata_path.is_file()
    assert not (artifacts_dir / "retained_mean.npy").exists()
    assert not (artifacts_dir / "terminal_state.npy").exists()
    assert not (artifacts_dir / RECONSTRUCTION_DIAGNOSTICS_FILENAME).exists()
    assert not (artifacts_dir / CHAIN_HEALTH_FILENAME).exists()

    metadata = load_reconstruction_metadata(artifacts_dir)
    admitted_event_ids = load_reconstruction_admitted_event_ids(artifacts_dir)
    assert metadata.execution_mode == RECONSTRUCTION_EXECUTION_MODE_ADMISSION_ONLY
    assert metadata.artifacts.retained_mean.filename is None
    assert metadata.artifacts.terminal_state.filename is None
    assert metadata.artifacts.admitted_event_ids_filename == ADMITTED_EVENT_IDS_FILENAME
    assert metadata.provenance.burn_in_steps == 2
    assert metadata.provenance.thin_every == 3
    assert metadata.provenance.retained_count == 0
    assert metadata.provenance.total_steps is None
    assert metadata.provenance.terminal_step_index is None
    assert admitted_event_ids.to_payload() == _example_admitted_event_ids(artifacts_dir).to_payload()


def test_write_reconstruction_metadata_sidecar_defaults_terminal_quantity_label_honestly(
    tmp_path: Path,
) -> None:
    metadata = write_reconstruction_metadata_sidecar(
        tmp_path,
        retained_mean_filename="retained_mean.npy",
        terminal_state_filename="terminal_state.npy",
        bounds_cm=np.asarray(
            [
                [0.0, 1.0],
                [2.0, 4.0],
                [5.0, 11.0],
            ],
            dtype=float,
        ),
        grid_shape=(1, 2, 3),
        burn_in_steps=2,
        thin_every=3,
        retained_count=4,
    )

    assert metadata.artifacts.retained_mean.quantity_label == "mean_occupancy"
    assert metadata.artifacts.terminal_state.quantity_label == TERMINAL_OCCUPANCY_COUNTS_QUANTITY_LABEL
    assert metadata.artifacts.admitted_event_ids_filename is None
    assert metadata.execution_mode == RECONSTRUCTION_EXECUTION_MODE_FULL_RUN
