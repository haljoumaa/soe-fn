"""Focused tests for the reconstruction-side run surface."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestration import run_reconstruction as reconstruction_entrypoint
from soe.analysis import reconstruction_run as reconstruction_run_module
from soe.analysis.reconstruction_artifacts import (
    ADMITTED_EVENT_IDS_FILENAME,
    RECONSTRUCTION_EXECUTION_MODE_ADMISSION_ONLY,
    load_reconstruction_admitted_event_ids,
    load_reconstruction_metadata,
)
from soe.analysis.reconstruction_diagnostics import load_reconstruction_diagnostics
from soe.analysis.reconstruction_run import (
    ReconstructionArtifactsConfig,
    ReconstructionAdmissionOnlyResult,
    ReconstructionRunConfig,
    ReconstructionSeeds,
    run_reconstruction_admission_only,
    run_reconstruction,
)
from soe.contracts import EventObj, ReconstructionInput, VOIConfig
from soe.soe.estimators import RetainedStateAccumulator
from soe.soe.chain_health import CHAIN_HEALTH_FILENAME, load_chain_health
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


def _config_toml() -> str:
    return """
run_id = "synthetic_reconstruction"

[input]
hdf5_path = "synthetic.h5"

[reconstruction]
bounds_cm = [[-1.0, 1.0], [1.0, 3.0], [-1.0, 1.0]]
grid_shape = [4, 4, 4]
artifacts_dir = "artifacts"
surviving_event_cap = 1

[survival]
mode = "fixed"
K = 8
gate_mode = "full_box"

[run]
num_steps = 2
burn_in_steps = 0
thin_every = 1
capture_terminal_state = true

[seeds]
event_index_selector = 11
proposal_backend = 12
acceptance_uniform = 13
""".strip()


def _multi_event_reconstruction_input() -> ReconstructionInput:
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
    return ReconstructionInput(
        events=(
            EventObj(
                apex=np.asarray([-0.25, 0.0, 0.0], dtype=float),
                axis=np.asarray([0.0, 1.0, 0.0], dtype=float),
                theta=math.pi / 4.0,
            ),
            EventObj(
                apex=np.asarray([0.0, 0.0, 0.0], dtype=float),
                axis=np.asarray([0.0, 1.0, 0.0], dtype=float),
                theta=math.pi / 4.0,
            ),
            EventObj(
                apex=np.asarray([0.25, 0.0, 0.0], dtype=float),
                axis=np.asarray([0.0, 1.0, 0.0], dtype=float),
                theta=math.pi / 4.0,
            ),
        ),
        voi=voi,
    )


def test_run_reconstruction_writes_authoritative_artifact_bundle(tmp_path: Path) -> None:
    result = run_reconstruction(
        ReconstructionRunConfig(
            reconstruction_input=_synthetic_reconstruction_input(),
            artifacts=ReconstructionArtifactsConfig(artifacts_dir=tmp_path / "artifacts"),
            survival_config=SampledSurvivalConfig.fixed_full_box(K=8),
            run_config=FiniteRunConfig(
                num_steps=2,
                burn_in_steps=0,
                thin_every=1,
                capture_terminal_state=True,
            ),
            seeds=ReconstructionSeeds(
                event_index_selector=11,
                proposal_backend=12,
                acceptance_uniform=13,
            ),
            run_id="synthetic_direct_run",
            surviving_event_cap=1,
        )
    )

    artifacts_dir = tmp_path / "artifacts"
    retained_mean_path = artifacts_dir / "retained_mean.npy"
    terminal_state_path = artifacts_dir / "terminal_state.npy"
    metadata_path = artifacts_dir / "reconstruction_metadata.json"
    diagnostics_path = artifacts_dir / "reconstruction_diagnostics.json"
    chain_health_path = artifacts_dir / CHAIN_HEALTH_FILENAME
    admitted_event_ids_path = artifacts_dir / ADMITTED_EVENT_IDS_FILENAME

    assert retained_mean_path.is_file()
    assert terminal_state_path.is_file()
    assert metadata_path.is_file()
    assert diagnostics_path.is_file()
    assert chain_health_path.is_file()
    assert admitted_event_ids_path.is_file()
    retained_mean = np.load(retained_mean_path)
    terminal_state = np.load(terminal_state_path)
    metadata = load_reconstruction_metadata(artifacts_dir)
    diagnostics = load_reconstruction_diagnostics(artifacts_dir)
    chain_health = load_chain_health(artifacts_dir)
    admitted_event_ids = load_reconstruction_admitted_event_ids(artifacts_dir)

    assert result.artifact_bundle.retained_mean_path == retained_mean_path.resolve()
    assert result.artifact_bundle.terminal_state_path == terminal_state_path.resolve()
    assert result.artifact_bundle.admitted_event_ids_path == admitted_event_ids_path.resolve()
    assert result.artifact_bundle.metadata_path == metadata_path.resolve()
    assert result.artifact_bundle.diagnostics_path == diagnostics_path.resolve()
    assert result.artifact_bundle.chain_health_path == chain_health_path.resolve()
    assert metadata.execution_mode == "full_run"
    assert metadata.artifacts.retained_mean.filename == "retained_mean.npy"
    assert metadata.artifacts.terminal_state.filename == "terminal_state.npy"
    assert metadata.artifacts.admitted_event_ids_filename == ADMITTED_EVENT_IDS_FILENAME
    assert metadata.provenance.burn_in_steps == 0
    assert metadata.provenance.thin_every == 1
    assert metadata.provenance.retained_count == len(result.chain_result.retained_states)
    assert metadata.provenance.total_steps == result.chain_result.total_steps
    assert metadata.provenance.terminal_step_index == result.chain_result.terminal_step_index
    assert diagnostics.counts.surviving_event_count == len(result.initial_state.events)
    assert diagnostics.chain.accepted_step_count == result.chain_result.accepted_step_count
    assert diagnostics.chain.total_steps_attempted == result.chain_result.total_steps
    assert chain_health.metadata.run_id == "synthetic_direct_run"
    assert chain_health.metadata.artifact_dir == str(artifacts_dir.resolve())
    assert chain_health.metadata.total_attempted_steps == result.chain_result.total_steps
    assert chain_health.metadata.burn_in_steps == 0
    assert chain_health.metadata.thin_every == 1
    assert chain_health.metadata.surviving_event_count == len(result.initial_state.events)
    assert chain_health.metadata.total_voxel_count == 64
    assert chain_health.metadata.grid_shape == (4, 4, 4)
    assert chain_health.metadata.alpha == 1.0
    assert chain_health.metadata.proposal_ratio_was_omitted_for_all_steps is True
    assert chain_health.metadata.proposal_ratio_treated_as_one_for_all_steps is True
    assert admitted_event_ids.run_id == "synthetic_direct_run"
    assert admitted_event_ids.artifacts_dir == artifacts_dir.resolve()
    assert admitted_event_ids.surviving_event_count == len(result.initial_state.events)
    assert admitted_event_ids.admission.gate_mode == "full_box"
    assert admitted_event_ids.admission.survival_mode == "fixed"
    assert admitted_event_ids.admission.K == 8
    assert admitted_event_ids.admission.surviving_event_cap == 1
    assert [event.reconstruction_input_index for event in admitted_event_ids.admitted_events] == [0]
    assert chain_health.occupancy_support.terminal_nonzero_voxel_count == int(
        np.count_nonzero(result.terminal_state)
    )
    assert chain_health.occupancy_support.retained_mean_nonzero_voxel_count == int(
        np.count_nonzero(result.retained_mean)
    )
    assert retained_mean.shape == (4, 4, 4)
    assert terminal_state.shape == (4, 4, 4)
    assert np.issubdtype(retained_mean.dtype, np.floating)
    assert np.issubdtype(terminal_state.dtype, np.integer)
    np.testing.assert_allclose(retained_mean.sum(), 1.0, rtol=0.0, atol=1e-12)
    assert int(terminal_state.sum()) == 1

    raw_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "truth" not in raw_payload
    assert "beam_axis" not in raw_payload
    assert "r_ref" not in raw_payload
    assert "evaluation_roi" not in raw_payload
    assert "counts" not in raw_payload


def test_run_reconstruction_online_retained_mean_matches_postpass_and_preserves_terminal_state(
    tmp_path: Path,
) -> None:
    result = run_reconstruction(
        ReconstructionRunConfig(
            reconstruction_input=_multi_event_reconstruction_input(),
            artifacts=ReconstructionArtifactsConfig(artifacts_dir=tmp_path / "artifacts"),
            survival_config=SampledSurvivalConfig.fixed_full_box(K=8),
            run_config=FiniteRunConfig(
                num_steps=6,
                burn_in_steps=1,
                thin_every=2,
                capture_terminal_state=True,
            ),
            seeds=ReconstructionSeeds(
                event_index_selector=21,
                proposal_backend=22,
                acceptance_uniform=23,
            ),
            run_id="online-retained-mean",
            surviving_event_cap=3,
        )
    )

    postpass_accumulator = RetainedStateAccumulator(result.grid)
    for state in result.chain_result.retained_states:
        postpass_accumulator.add_state(state)
    expected_retained_mean = postpass_accumulator.mean_occupancy()

    np.testing.assert_allclose(result.retained_mean, expected_retained_mean, rtol=0.0, atol=1e-12)
    assert result.chain_result.terminal_occupancy_counts is not None
    np.testing.assert_array_equal(result.terminal_state, result.chain_result.terminal_occupancy_counts)
    np.testing.assert_allclose(
        np.sum(result.retained_mean, dtype=float),
        float(len(result.initial_state.events)),
        rtol=0.0,
        atol=1e-12,
    )
    assert int(np.sum(result.terminal_state, dtype=np.int64)) == len(result.initial_state.events)


def test_run_reconstruction_admission_only_writes_sidecars_and_skips_mh(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _fail_run_chain(*args, **kwargs):
        raise AssertionError("admission-only mode must not run MH")

    monkeypatch.setattr(reconstruction_run_module, "run_chain", _fail_run_chain)

    result = run_reconstruction_admission_only(
        ReconstructionRunConfig(
            reconstruction_input=_synthetic_reconstruction_input(),
            artifacts=ReconstructionArtifactsConfig(artifacts_dir=tmp_path / "artifacts"),
            survival_config=SampledSurvivalConfig.fixed_full_box(K=8),
            run_config=FiniteRunConfig(
                num_steps=2,
                burn_in_steps=0,
                thin_every=1,
                capture_terminal_state=True,
            ),
            seeds=ReconstructionSeeds(
                event_index_selector=11,
                proposal_backend=12,
                acceptance_uniform=13,
            ),
            run_id="synthetic_admission_only",
            surviving_event_cap=1,
        )
    )

    artifacts_dir = tmp_path / "artifacts"
    metadata_path = artifacts_dir / "reconstruction_metadata.json"
    admitted_event_ids_path = artifacts_dir / ADMITTED_EVENT_IDS_FILENAME
    diagnostics_path = artifacts_dir / "reconstruction_diagnostics.json"
    chain_health_path = artifacts_dir / CHAIN_HEALTH_FILENAME
    retained_mean_path = artifacts_dir / "retained_mean.npy"
    terminal_state_path = artifacts_dir / "terminal_state.npy"

    assert isinstance(result, ReconstructionAdmissionOnlyResult)
    assert metadata_path.is_file()
    assert admitted_event_ids_path.is_file()
    assert not diagnostics_path.exists()
    assert not chain_health_path.exists()
    assert not retained_mean_path.exists()
    assert not terminal_state_path.exists()

    metadata = load_reconstruction_metadata(artifacts_dir)
    admitted_event_ids = load_reconstruction_admitted_event_ids(artifacts_dir)
    assert result.artifact_bundle.metadata_path == metadata_path.resolve()
    assert result.artifact_bundle.admitted_event_ids_path == admitted_event_ids_path.resolve()
    assert metadata.execution_mode == RECONSTRUCTION_EXECUTION_MODE_ADMISSION_ONLY
    assert metadata.artifacts.retained_mean.filename is None
    assert metadata.artifacts.terminal_state.filename is None
    assert metadata.artifacts.admitted_event_ids_filename == ADMITTED_EVENT_IDS_FILENAME
    assert metadata.provenance.retained_count == 0
    assert metadata.provenance.total_steps is None
    assert metadata.provenance.terminal_step_index is None
    assert admitted_event_ids.run_id == "synthetic_admission_only"
    assert [event.reconstruction_input_index for event in admitted_event_ids.admitted_events] == [0]


def test_build_initial_state_with_provenance_tracks_exact_post_gate_survivors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reconstruction_input = _multi_event_reconstruction_input()
    original_representative_point = reconstruction_run_module._strict_open_representative_point

    def _fake_check_event_usability(event, allowed_region, *, survival_config):
        del allowed_region, survival_config
        return SimpleNamespace(usable=float(event.apex[0]) >= 0.0)

    def _fake_representative_point(event, *, allowed_region, azimuths):
        if np.isclose(float(event.apex[0]), 0.25):
            raise ValueError("no strict-open representative point")
        return original_representative_point(
            event,
            allowed_region=allowed_region,
            azimuths=azimuths,
        )

    monkeypatch.setattr(
        reconstruction_run_module,
        "check_event_usability",
        _fake_check_event_usability,
    )
    monkeypatch.setattr(
        reconstruction_run_module,
        "_strict_open_representative_point",
        _fake_representative_point,
    )

    build_result = reconstruction_run_module._build_initial_state_with_provenance(
        ReconstructionRunConfig(
            reconstruction_input=reconstruction_input,
            artifacts=ReconstructionArtifactsConfig(artifacts_dir=tmp_path / "artifacts"),
            survival_config=SampledSurvivalConfig.fixed_full_box(K=8),
            run_config=FiniteRunConfig(
                num_steps=2,
                burn_in_steps=0,
                thin_every=1,
                capture_terminal_state=True,
            ),
            surviving_event_cap=2,
        )
    )

    assert len(build_result.state.events) == 1
    assert build_result.state.events[0] is reconstruction_input.events[1]
    assert build_result.admitted_event_input_indices == (1,)


def test_orchestration_entrypoint_accepts_admission_only_flag(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "reconstruction.toml"
    config_path.write_text(_config_toml() + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "soe.analysis.reconstruction_run.assemble_reconstruction_input_from_hdf5",
        lambda *args, **kwargs: _synthetic_reconstruction_input(),
    )
    monkeypatch.setattr(
        reconstruction_run_module,
        "run_chain",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("admission-only CLI must not run MH")
        ),
    )

    exit_code = reconstruction_entrypoint.main(
        ["--config", str(config_path), "--admission-only"]
    )

    assert exit_code == 0
    stdout_lines = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(stdout_lines) == 2
    assert stdout_lines[0].endswith(f"/artifacts/{ADMITTED_EVENT_IDS_FILENAME}")
    assert stdout_lines[1].endswith("/artifacts/reconstruction_metadata.json")
    assert (tmp_path / "artifacts" / ADMITTED_EVENT_IDS_FILENAME).is_file()
    assert (tmp_path / "artifacts" / "reconstruction_metadata.json").is_file()
    assert not (tmp_path / "artifacts" / "retained_mean.npy").exists()
    assert not (tmp_path / "artifacts" / "terminal_state.npy").exists()
    assert not (tmp_path / "artifacts" / "reconstruction_diagnostics.json").exists()
    assert not (tmp_path / "artifacts" / "chain_health.json").exists()


def test_orchestration_entrypoint_writes_bundle_from_explicit_config(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config_path = tmp_path / "reconstruction.toml"
    config_path.write_text(_config_toml() + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "soe.analysis.reconstruction_run.assemble_reconstruction_input_from_hdf5",
        lambda *args, **kwargs: _synthetic_reconstruction_input(),
    )

    exit_code = reconstruction_entrypoint.main(["--config", str(config_path)])

    assert exit_code == 0
    stdout_lines = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(stdout_lines) == 5
    assert stdout_lines[0].endswith("/artifacts/retained_mean.npy")
    assert stdout_lines[1].endswith("/artifacts/terminal_state.npy")
    assert stdout_lines[2].endswith("/artifacts/reconstruction_metadata.json")
    assert stdout_lines[3].endswith("/artifacts/reconstruction_diagnostics.json")
    assert stdout_lines[4].endswith("/artifacts/chain_health.json")
    assert (tmp_path / "artifacts" / "retained_mean.npy").is_file()
    assert (tmp_path / "artifacts" / "terminal_state.npy").is_file()
    assert (tmp_path / "artifacts" / "reconstruction_metadata.json").is_file()
    assert (tmp_path / "artifacts" / "reconstruction_diagnostics.json").is_file()
    assert (tmp_path / "artifacts" / "chain_health.json").is_file()
    assert (tmp_path / "artifacts" / ADMITTED_EVENT_IDS_FILENAME).is_file()
