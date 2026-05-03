"""Reconstruction-stage execution surface.

This module owns the reconstruction-only run helper used by the authoritative
`orchestration/run_reconstruction.py` entrypoint. It starts from the `ReconstructionInput(events, voi)` handoff, reuses
the preprocessing and MH kernel components in `src/soe/`, and writes only the
reconstruction artifact bundle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import operator
from pathlib import Path
import time

import numpy as np

from soe.adapters.voi_voxel import VoiBounds, VoxelGrid
from soe.analysis.reconstruction_diagnostics import (
    ReconstructionChainDiagnostics,
    ReconstructionCountsDiagnostics,
    ReconstructionDiagnostics,
    ReconstructionPreprocessingDiagnostics,
    ReconstructionSeedDiagnostics,
    ReconstructionStateSummaryDiagnostics,
    write_reconstruction_diagnostics_sidecar,
)
from soe.analysis.reconstruction_artifacts import (
    ReconstructionAdmissionMetadata,
    ReconstructionAdmissionArtifactBundle,
    ReconstructionArtifactBundle,
    ReconstructionAdmittedEventIdentity,
    ReconstructionAdmittedEventIds,
    ReconstructionGeometryMetadata,
    write_reconstruction_admission_artifact_bundle,
    write_reconstruction_artifact_bundle,
)
from soe.contracts import (
    EventFilterConfig,
    EventObj,
    ReconstructionInput,
    RegistrationTransform,
    VOIConfig,
)
from soe.core import assemble_reconstruction_input_from_hdf5
from soe.geometry.cone import cone_generator_direction, cone_surface_point, is_admissible_point
from soe.geometry.ray_box import open_box_ray_interval
from soe.geometry.sampled_surrogate import midpoint_azimuth_grid
from soe.soe.estimators import RetainedStateAccumulator
from soe.soe.chain_health import ChainHealthCollector
from soe.soe.run_protocol import ChainRunResult, FiniteRunConfig, run_chain
from soe.soe.state import (
    SAMPLED_SURVIVAL_GATE_FULL_BOX,
    SampledSurvivalConfig,
    SurvivingEventState,
    check_event_usability,
)
from soe.soe.uniform_surface_reference import UniformSurfaceReferenceProposalBackend

DEFAULT_RECONSTRUCTION_CONFIG_FILENAME = "reconstruction.toml"
DEFAULT_RETAINED_MEAN_FILENAME = "retained_mean.npy"
DEFAULT_TERMINAL_STATE_FILENAME = "terminal_state.npy"
DEFAULT_EVENT_INDEX_SELECTOR_SEED = 20260406
DEFAULT_PROPOSAL_BACKEND_SEED = 20260407
DEFAULT_ACCEPTANCE_UNIFORM_SEED = 20260408
_FORBIDDEN_RUN_SURFACE_KEYS = frozenset(
    {
        "truth",
        "truth_path",
        "beam_axis",
        "r_ref",
        "evaluation",
        "evaluation_roi",
        "evaluation_grid",
        "metrics",
        "report",
    }
)


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _as_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _as_optional_name(value: object | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _as_name(value, field_name=field_name)


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool")
    return value


def _as_nonnegative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a nonnegative integer")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a nonnegative integer") from exc
    if normalized < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return int(normalized)


def _as_positive_int(value: object, *, field_name: str) -> int:
    normalized = _as_nonnegative_int(value, field_name=field_name)
    if normalized < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return normalized


def _as_optional_positive_int(value: object | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_positive_int(value, field_name=field_name)


def _as_bounds_cm(value: object, *, field_name: str) -> np.ndarray:
    bounds_cm = np.asarray(value, dtype=float)
    if bounds_cm.shape != (3, 2):
        raise ValueError(f"{field_name} must have shape (3, 2)")
    if not np.all(np.isfinite(bounds_cm)):
        raise ValueError(f"{field_name} must be finite")
    if not np.all(bounds_cm[:, 0] < bounds_cm[:, 1]):
        raise ValueError(f"{field_name} must satisfy min < max for x, y, z")
    return bounds_cm


def _as_grid_shape(value: object, *, field_name: str) -> tuple[int, int, int]:
    try:
        grid_shape = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must contain 3 positive integers") from exc
    if len(grid_shape) != 3:
        raise ValueError(f"{field_name} must contain 3 entries")
    normalized: list[int] = []
    for axis_index, entry in enumerate(grid_shape):
        if isinstance(entry, bool):
            raise ValueError(f"{field_name}[{axis_index}] must be an integer")
        try:
            component = operator.index(entry)
        except TypeError as exc:
            raise ValueError(f"{field_name}[{axis_index}] must be an integer") from exc
        if component <= 0:
            raise ValueError(f"{field_name}[{axis_index}] must be positive")
        normalized.append(int(component))
    return (normalized[0], normalized[1], normalized[2])


def _resolve_path(base_dir: Path, value: object, *, field_name: str) -> Path:
    path = Path(_as_name(value, field_name=field_name)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _identity_registration() -> RegistrationTransform:
    return RegistrationTransform(Q=np.eye(3), t=np.zeros(3))


def _retained_state_count(run_config: FiniteRunConfig) -> int:
    return sum(
        1 for step_index in range(1, run_config.num_steps + 1) if run_config.retain_post_step(step_index)
    )


def _whole_voi_open_box(
    voi: VoiBounds,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    return (
        (float(voi.xmin), float(voi.xmax)),
        (float(voi.ymin), float(voi.ymax)),
        (float(voi.zmin), float(voi.zmax)),
    )


def _is_strict_open_box_interior(point: np.ndarray, voi: VoiBounds) -> bool:
    x = float(point[0])
    y = float(point[1])
    z = float(point[2])
    return voi.xmin < x < voi.xmax and voi.ymin < y < voi.ymax and voi.zmin < z < voi.zmax


def _strict_open_representative_point(
    event: EventObj,
    *,
    allowed_region: VoiBounds,
    azimuths: tuple[float, ...],
) -> np.ndarray:
    apex = tuple(float(component) for component in event.apex)
    voi_box = _whole_voi_open_box(allowed_region)
    for xi in azimuths:
        direction = tuple(float(component) for component in cone_generator_direction(event, xi))
        interval = open_box_ray_interval(apex, direction, voi_box)
        if interval is None:
            continue
        ell_minus, ell_plus = interval
        point = cone_surface_point(event, ell=0.5 * (ell_minus + ell_plus), xi=xi)
        if _is_strict_open_box_interior(point, allowed_region) and is_admissible_point(
            point,
            event,
            allowed_region,
        ):
            return point
    raise ValueError(
        "event passed the sampled full-box preprocessing gate but no strict-open "
        "representative point could be constructed on the current bounded box"
    )


def _reject_forbidden_keys(mapping: Mapping[str, object], *, field_name: str) -> None:
    forbidden = sorted(key for key in mapping if key in _FORBIDDEN_RUN_SURFACE_KEYS)
    if forbidden:
        forbidden_fields = ", ".join(forbidden)
        raise ValueError(
            f"{field_name} contains downstream evaluation/truth fields that do not belong "
            f"on the reconstruction run surface: {forbidden_fields}"
        )


def _survival_config_from_payload(payload: Mapping[str, object]) -> SampledSurvivalConfig:
    mode = str(payload.get("mode", "fixed"))
    gate_mode = str(payload.get("gate_mode", SAMPLED_SURVIVAL_GATE_FULL_BOX))
    if mode == "fixed":
        return SampledSurvivalConfig.fixed(
            K=_as_positive_int(payload.get("K", 8), field_name="survival.K"),
            gate_mode=gate_mode,
        )
    if mode == "adaptive":
        return SampledSurvivalConfig.adaptive(
            K0=_as_positive_int(payload.get("K0"), field_name="survival.K0"),
            J_max=_as_nonnegative_int(payload.get("J_max"), field_name="survival.J_max"),
            gate_mode=gate_mode,
        )
    raise ValueError("survival.mode must be 'fixed' or 'adaptive'")


def _run_config_from_payload(payload: Mapping[str, object]) -> FiniteRunConfig:
    return FiniteRunConfig(
        num_steps=_as_nonnegative_int(payload.get("num_steps"), field_name="run.num_steps"),
        burn_in_steps=_as_nonnegative_int(payload.get("burn_in_steps", 0), field_name="run.burn_in_steps"),
        thin_every=_as_positive_int(payload.get("thin_every", 1), field_name="run.thin_every"),
        capture_terminal_state=_as_bool(
            payload.get("capture_terminal_state", True),
            field_name="run.capture_terminal_state",
        ),
        capture_step_results=_as_bool(
            payload.get("capture_step_results", False),
            field_name="run.capture_step_results",
        ),
    )


@dataclass(frozen=True, slots=True)
class ReconstructionSeeds:
    """Deterministic seeds for the reconstruction run surface."""

    event_index_selector: int = DEFAULT_EVENT_INDEX_SELECTOR_SEED
    proposal_backend: int = DEFAULT_PROPOSAL_BACKEND_SEED
    acceptance_uniform: int = DEFAULT_ACCEPTANCE_UNIFORM_SEED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_index_selector",
            _as_nonnegative_int(
                self.event_index_selector,
                field_name="seeds.event_index_selector",
            ),
        )
        object.__setattr__(
            self,
            "proposal_backend",
            _as_nonnegative_int(self.proposal_backend, field_name="seeds.proposal_backend"),
        )
        object.__setattr__(
            self,
            "acceptance_uniform",
            _as_nonnegative_int(
                self.acceptance_uniform,
                field_name="seeds.acceptance_uniform",
            ),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionArtifactsConfig:
    """Filesystem target for the reconstruction artifact bundle."""

    artifacts_dir: Path
    retained_mean_filename: str = DEFAULT_RETAINED_MEAN_FILENAME
    terminal_state_filename: str = DEFAULT_TERMINAL_STATE_FILENAME

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts_dir", Path(self.artifacts_dir).resolve())
        object.__setattr__(
            self,
            "retained_mean_filename",
            _as_name(self.retained_mean_filename, field_name="retained_mean_filename"),
        )
        object.__setattr__(
            self,
            "terminal_state_filename",
            _as_name(self.terminal_state_filename, field_name="terminal_state_filename"),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionRunConfig:
    """Reconstruction execution config following input assembly."""

    reconstruction_input: ReconstructionInput
    artifacts: ReconstructionArtifactsConfig
    survival_config: SampledSurvivalConfig = field(
        default_factory=lambda: SampledSurvivalConfig.fixed_full_box(K=8)
    )
    run_config: FiniteRunConfig = field(
        default_factory=lambda: FiniteRunConfig(
            num_steps=6,
            burn_in_steps=1,
            thin_every=2,
            capture_terminal_state=True,
        )
    )
    seeds: ReconstructionSeeds = field(default_factory=ReconstructionSeeds)
    run_id: str | None = None
    surviving_event_cap: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reconstruction_input, ReconstructionInput):
            raise TypeError("reconstruction_input must be a ReconstructionInput")
        if not isinstance(self.artifacts, ReconstructionArtifactsConfig):
            raise TypeError("artifacts must be a ReconstructionArtifactsConfig")
        if not isinstance(self.survival_config, SampledSurvivalConfig):
            raise TypeError("survival_config must be a SampledSurvivalConfig")
        if not isinstance(self.run_config, FiniteRunConfig):
            raise TypeError("run_config must be a FiniteRunConfig")
        if not isinstance(self.seeds, ReconstructionSeeds):
            raise TypeError("seeds must be a ReconstructionSeeds")
        surviving_event_cap = _as_optional_positive_int(
            self.surviving_event_cap,
            field_name="surviving_event_cap",
        )
        run_id = _as_optional_name(self.run_id, field_name="run_id")
        if not self.run_config.capture_terminal_state:
            raise ValueError(
                "run_config.capture_terminal_state must remain True because "
                "the authoritative reconstruction bundle always writes terminal_state.npy"
            )
        if _retained_state_count(self.run_config) == 0:
            raise ValueError(
                "run_config must retain at least one post-step state to write retained_mean.npy"
            )
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "surviving_event_cap", surviving_event_cap)


@dataclass(frozen=True, slots=True)
class ReconstructionJobConfig:
    """Thin pre-handoff config used by the orchestration entrypoint."""

    input_h5_path: Path
    registration: RegistrationTransform
    filter_config: EventFilterConfig
    reconstruction_voi: VOIConfig
    artifacts: ReconstructionArtifactsConfig
    survival_config: SampledSurvivalConfig = field(
        default_factory=lambda: SampledSurvivalConfig.fixed_full_box(K=8)
    )
    run_config: FiniteRunConfig = field(
        default_factory=lambda: FiniteRunConfig(
            num_steps=6,
            burn_in_steps=1,
            thin_every=2,
            capture_terminal_state=True,
        )
    )
    seeds: ReconstructionSeeds = field(default_factory=ReconstructionSeeds)
    run_id: str | None = None
    surviving_event_cap: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_h5_path", Path(self.input_h5_path).resolve())
        if not isinstance(self.registration, RegistrationTransform):
            raise TypeError("registration must be a RegistrationTransform")
        if not isinstance(self.filter_config, EventFilterConfig):
            raise TypeError("filter_config must be an EventFilterConfig")
        if not isinstance(self.reconstruction_voi, VOIConfig):
            raise TypeError("reconstruction_voi must be a VOIConfig")
        if not isinstance(self.artifacts, ReconstructionArtifactsConfig):
            raise TypeError("artifacts must be a ReconstructionArtifactsConfig")
        if not isinstance(self.survival_config, SampledSurvivalConfig):
            raise TypeError("survival_config must be a SampledSurvivalConfig")
        if not isinstance(self.run_config, FiniteRunConfig):
            raise TypeError("run_config must be a FiniteRunConfig")
        if not isinstance(self.seeds, ReconstructionSeeds):
            raise TypeError("seeds must be a ReconstructionSeeds")
        object.__setattr__(self, "run_id", _as_optional_name(self.run_id, field_name="run_id"))
        object.__setattr__(
            self,
            "surviving_event_cap",
            _as_optional_positive_int(self.surviving_event_cap, field_name="surviving_event_cap"),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionRunResult:
    """Completed reconstruction run plus the saved artifact bundle."""

    config: ReconstructionRunConfig
    allowed_region: VoiBounds
    grid: VoxelGrid
    initial_state: SurvivingEventState
    chain_result: ChainRunResult
    retained_mean: np.ndarray
    terminal_state: np.ndarray
    artifact_bundle: ReconstructionArtifactBundle

    def __post_init__(self) -> None:
        if not isinstance(self.config, ReconstructionRunConfig):
            raise TypeError("config must be a ReconstructionRunConfig")
        if not isinstance(self.allowed_region, VoiBounds):
            raise TypeError("allowed_region must be a VoiBounds")
        if not isinstance(self.grid, VoxelGrid):
            raise TypeError("grid must be a VoxelGrid")
        if not isinstance(self.initial_state, SurvivingEventState):
            raise TypeError("initial_state must be a SurvivingEventState")
        if not isinstance(self.chain_result, ChainRunResult):
            raise TypeError("chain_result must be a ChainRunResult")
        if not isinstance(self.artifact_bundle, ReconstructionArtifactBundle):
            raise TypeError("artifact_bundle must be a ReconstructionArtifactBundle")


@dataclass(frozen=True, slots=True)
class ReconstructionAdmissionOnlyResult:
    """Completed admission-only reconstruction preprocessing plus sidecars."""

    config: ReconstructionRunConfig
    allowed_region: VoiBounds
    grid: VoxelGrid
    initial_state: SurvivingEventState
    artifact_bundle: ReconstructionAdmissionArtifactBundle

    def __post_init__(self) -> None:
        if not isinstance(self.config, ReconstructionRunConfig):
            raise TypeError("config must be a ReconstructionRunConfig")
        if not isinstance(self.allowed_region, VoiBounds):
            raise TypeError("allowed_region must be a VoiBounds")
        if not isinstance(self.grid, VoxelGrid):
            raise TypeError("grid must be a VoxelGrid")
        if not isinstance(self.initial_state, SurvivingEventState):
            raise TypeError("initial_state must be a SurvivingEventState")
        if not isinstance(self.artifact_bundle, ReconstructionAdmissionArtifactBundle):
            raise TypeError(
                "artifact_bundle must be a ReconstructionAdmissionArtifactBundle"
            )


@dataclass(frozen=True, slots=True)
class _InitialStateBuildResult:
    """Internal reconstruction preprocessing result including survivor provenance."""

    allowed_region: VoiBounds
    grid: VoxelGrid
    state: SurvivingEventState
    admitted_event_input_indices: tuple[int, ...]


def _reconstruction_geometry_metadata(reconstruction_input: ReconstructionInput) -> ReconstructionGeometryMetadata:
    bounds_cm = np.asarray(reconstruction_input.voi.bounds, dtype=float)
    grid_shape = tuple(reconstruction_input.voi.grid_shape)
    spans_cm = bounds_cm[:, 1] - bounds_cm[:, 0]
    voxel_size_cm = tuple(
        float(component)
        for component in (spans_cm / np.asarray(grid_shape, dtype=float))
    )
    return ReconstructionGeometryMetadata(
        bounds_cm=bounds_cm,
        grid_shape=grid_shape,
        voxel_size_cm=voxel_size_cm,
    )


def reconstruction_job_config_from_payload(
    payload: Mapping[str, object],
    *,
    base_dir: Path,
) -> ReconstructionJobConfig:
    """Validate and normalize one reconstruction TOML payload."""
    mapping = _as_mapping(payload, field_name="reconstruction_run")
    _reject_forbidden_keys(mapping, field_name="reconstruction_run")
    input_config = _as_mapping(mapping.get("input"), field_name="input")
    _reject_forbidden_keys(input_config, field_name="input")
    reconstruction = _as_mapping(mapping.get("reconstruction"), field_name="reconstruction")
    _reject_forbidden_keys(reconstruction, field_name="reconstruction")
    survival = _as_mapping(mapping.get("survival", {}), field_name="survival")
    run = _as_mapping(mapping.get("run"), field_name="run")
    seeds_payload = _as_mapping(mapping.get("seeds", {}), field_name="seeds")
    registration_payload = _as_mapping(mapping.get("registration", {}), field_name="registration")
    filter_payload = _as_mapping(mapping.get("filter", {}), field_name="filter")

    return ReconstructionJobConfig(
        input_h5_path=_resolve_path(base_dir, input_config.get("hdf5_path"), field_name="input.hdf5_path"),
        registration=RegistrationTransform(
            Q=np.asarray(registration_payload.get("Q", np.eye(3)), dtype=float),
            t=np.asarray(registration_payload.get("t", np.zeros(3)), dtype=float),
        )
        if registration_payload
        else _identity_registration(),
        filter_config=EventFilterConfig(
            lambda_min=filter_payload.get("lambda_min", 0.001),
            species=filter_payload.get("species"),
        ),
        reconstruction_voi=VOIConfig(
            bounds=_as_bounds_cm(reconstruction.get("bounds_cm"), field_name="reconstruction.bounds_cm"),
            grid_shape=_as_grid_shape(
                reconstruction.get("grid_shape"),
                field_name="reconstruction.grid_shape",
            ),
        ),
        artifacts=ReconstructionArtifactsConfig(
            artifacts_dir=_resolve_path(
                base_dir,
                reconstruction.get("artifacts_dir", "artifacts"),
                field_name="reconstruction.artifacts_dir",
            ),
            retained_mean_filename=reconstruction.get(
                "retained_mean_filename",
                DEFAULT_RETAINED_MEAN_FILENAME,
            ),
            terminal_state_filename=reconstruction.get(
                "terminal_state_filename",
                DEFAULT_TERMINAL_STATE_FILENAME,
            ),
        ),
        survival_config=_survival_config_from_payload(survival),
        run_config=_run_config_from_payload(run),
        seeds=ReconstructionSeeds(
            event_index_selector=seeds_payload.get(
                "event_index_selector",
                DEFAULT_EVENT_INDEX_SELECTOR_SEED,
            ),
            proposal_backend=seeds_payload.get(
                "proposal_backend",
                DEFAULT_PROPOSAL_BACKEND_SEED,
            ),
            acceptance_uniform=seeds_payload.get(
                "acceptance_uniform",
                DEFAULT_ACCEPTANCE_UNIFORM_SEED,
            ),
        ),
        run_id=mapping.get("run_id"),
        surviving_event_cap=_as_optional_positive_int(
            reconstruction.get("surviving_event_cap"),
            field_name="reconstruction.surviving_event_cap",
        ),
    )


def _build_initial_state_with_provenance(
    config: ReconstructionRunConfig,
) -> _InitialStateBuildResult:
    reconstruction_input = config.reconstruction_input
    if config.survival_config.gate_mode != SAMPLED_SURVIVAL_GATE_FULL_BOX:
        raise ValueError(
            "run_reconstruction currently supports only the sampled full-box preprocessing "
            "gate because the representative-point initialization path stays on the "
            "current bounded-box reconstruction authority"
        )
    if config.survival_config.mode != "fixed" or config.survival_config.K is None:
        raise ValueError(
            "run_reconstruction currently supports only fixed sampled full-box preprocessing"
        )
    if len(reconstruction_input.events) == 0:
        raise ValueError("reconstruction_input.events must contain at least one canonical event")

    allowed_region = VoiBounds.from_config(reconstruction_input.voi)
    grid = VoxelGrid.from_config(reconstruction_input.voi)
    azimuths = midpoint_azimuth_grid(config.survival_config.K)
    selected_events: list[EventObj] = []
    representative_points: list[np.ndarray] = []
    admitted_event_input_indices: list[int] = []

    for reconstruction_input_index, event in enumerate(reconstruction_input.events):
        decision = check_event_usability(
            event,
            allowed_region,
            survival_config=config.survival_config,
        )
        if not decision.usable:
            continue
        try:
            representative_point = _strict_open_representative_point(
                event,
                allowed_region=allowed_region,
                azimuths=azimuths,
            )
        except ValueError:
            continue
        selected_events.append(event)
        representative_points.append(representative_point)
        admitted_event_input_indices.append(reconstruction_input_index)
        if (
            config.surviving_event_cap is not None
            and len(selected_events) >= config.surviving_event_cap
        ):
            break

    if not selected_events:
        raise ValueError(
            "no events survived the reconstruction-side sampled full-box preprocessing gate"
        )
    state = SurvivingEventState._from_already_admitted_events(
        selected_events,
        allowed_region,
        representative_points,
    )
    return _InitialStateBuildResult(
        allowed_region=allowed_region,
        grid=grid,
        state=state,
        admitted_event_input_indices=tuple(admitted_event_input_indices),
    )


def _build_initial_state(
    config: ReconstructionRunConfig,
) -> tuple[VoiBounds, VoxelGrid, SurvivingEventState]:
    build_result = _build_initial_state_with_provenance(config)
    return build_result.allowed_region, build_result.grid, build_result.state


def _build_admitted_event_ids_payload(
    *,
    config: ReconstructionRunConfig,
    build_result: _InitialStateBuildResult,
) -> ReconstructionAdmittedEventIds:
    return ReconstructionAdmittedEventIds(
        admitted_events=tuple(
            ReconstructionAdmittedEventIdentity(
                reconstruction_input_index=reconstruction_input_index
            )
            for reconstruction_input_index in build_result.admitted_event_input_indices
        ),
        surviving_event_count=len(build_result.state.events),
        admission=ReconstructionAdmissionMetadata(
            gate_mode=config.survival_config.gate_mode,
            survival_mode=config.survival_config.mode,
            K=config.survival_config.K,
            K0=config.survival_config.K0,
            J_max=config.survival_config.J_max,
            surviving_event_cap=config.surviving_event_cap,
        ),
        geometry=_reconstruction_geometry_metadata(config.reconstruction_input),
        artifacts_dir=config.artifacts.artifacts_dir,
        run_id=config.run_id,
    )


def _build_reconstruction_diagnostics(
    *,
    config: ReconstructionRunConfig,
    grid: VoxelGrid,
    initial_state: SurvivingEventState,
    chain_result: ChainRunResult,
    retained_mean: np.ndarray,
    terminal_state: np.ndarray,
    preprocessing_runtime_seconds: float,
    chain_runtime_seconds: float,
    chain_core_runtime_seconds: float | None,
    retained_mean_accumulation_runtime_seconds: float | None,
    artifact_write_runtime_seconds: float | None,
    total_runtime_seconds: float,
) -> ReconstructionDiagnostics:
    filtered_valid_event_count = len(config.reconstruction_input.events)
    surviving_event_count = len(initial_state.events)
    # This run boundary does not split preprocessing rejection from cap truncation.
    dropped_event_count = filtered_valid_event_count - surviving_event_count
    total_steps_attempted = chain_result.total_steps
    accepted_step_count = chain_result.accepted_step_count
    rejected_step_count = total_steps_attempted - accepted_step_count
    acceptance_rate = (
        None
        if total_steps_attempted == 0
        else accepted_step_count / float(total_steps_attempted)
    )
    # The authoritative handoff does not retain raw HDF5 row counts at this boundary.
    raw_event_count = None

    return ReconstructionDiagnostics(
        counts=ReconstructionCountsDiagnostics(
            raw_event_count=raw_event_count,
            filtered_valid_event_count=filtered_valid_event_count,
            surviving_event_count=surviving_event_count,
            dropped_event_count=dropped_event_count,
        ),
        preprocessing=ReconstructionPreprocessingDiagnostics(
            survival_mode=config.survival_config.mode,
            gate_mode=config.survival_config.gate_mode,
            K=config.survival_config.K,
            surviving_event_cap=config.surviving_event_cap,
            runtime_seconds=preprocessing_runtime_seconds,
        ),
        chain=ReconstructionChainDiagnostics(
            total_steps_attempted=total_steps_attempted,
            accepted_step_count=accepted_step_count,
            rejected_step_count=rejected_step_count,
            acceptance_rate=acceptance_rate,
            burn_in_steps=config.run_config.burn_in_steps,
            thin_every=config.run_config.thin_every,
            retained_count=len(chain_result.retained_states),
            terminal_step_index=chain_result.terminal_step_index,
            runtime_seconds=chain_runtime_seconds,
            chain_core_runtime_seconds=chain_core_runtime_seconds,
            retained_mean_accumulation_runtime_seconds=retained_mean_accumulation_runtime_seconds,
            artifact_write_runtime_seconds=artifact_write_runtime_seconds,
            total_runtime_seconds=total_runtime_seconds,
        ),
        state_summary=ReconstructionStateSummaryDiagnostics(
            grid_shape=grid.grid_shape,
            total_voxel_count=int(np.prod(np.asarray(grid.grid_shape, dtype=np.int64))),
            terminal_occupancy_sum=int(np.sum(terminal_state, dtype=np.int64)),
            retained_mean_occupancy_sum=float(np.sum(retained_mean, dtype=float)),
            terminal_nonzero_voxel_count=int(np.count_nonzero(terminal_state)),
            retained_mean_nonzero_voxel_count=int(np.count_nonzero(retained_mean)),
        ),
        seeds=ReconstructionSeedDiagnostics(
            event_index_selector=config.seeds.event_index_selector,
            proposal_backend=config.seeds.proposal_backend,
            acceptance_uniform=config.seeds.acceptance_uniform,
        ),
    )


def run_reconstruction_admission_only(
    config: ReconstructionRunConfig,
) -> ReconstructionAdmissionOnlyResult:
    """Run only reconstruction admission and write survivor provenance sidecars."""
    if not isinstance(config, ReconstructionRunConfig):
        raise TypeError("config must be a ReconstructionRunConfig")

    initial_state_build = _build_initial_state_with_provenance(config)
    admitted_event_ids = _build_admitted_event_ids_payload(
        config=config,
        build_result=initial_state_build,
    )
    artifact_bundle = write_reconstruction_admission_artifact_bundle(
        config.artifacts.artifacts_dir,
        bounds_cm=config.reconstruction_input.voi.bounds,
        grid_shape=config.reconstruction_input.voi.grid_shape,
        admitted_event_ids=admitted_event_ids,
        burn_in_steps=config.run_config.burn_in_steps,
        thin_every=config.run_config.thin_every,
    )
    return ReconstructionAdmissionOnlyResult(
        config=config,
        allowed_region=initial_state_build.allowed_region,
        grid=initial_state_build.grid,
        initial_state=initial_state_build.state,
        artifact_bundle=artifact_bundle,
    )


def run_reconstruction(config: ReconstructionRunConfig) -> ReconstructionRunResult:
    """Run reconstruction from the authoritative `ReconstructionInput` handoff."""
    if not isinstance(config, ReconstructionRunConfig):
        raise TypeError("config must be a ReconstructionRunConfig")

    total_start = time.perf_counter()
    preprocessing_start = total_start
    initial_state_build = _build_initial_state_with_provenance(config)
    allowed_region = initial_state_build.allowed_region
    grid = initial_state_build.grid
    initial_state = initial_state_build.state
    preprocessing_runtime_seconds = time.perf_counter() - preprocessing_start
    initial_counts = initial_state.occupancy_counts(grid)
    alpha = 1.0
    event_selector_rng = np.random.default_rng(config.seeds.event_index_selector)
    proposal_backend = UniformSurfaceReferenceProposalBackend(
        allowed_region=allowed_region,
        event_index_selector=lambda state: int(event_selector_rng.integers(len(state.events))),
        rng=np.random.default_rng(config.seeds.proposal_backend),
    )
    chain_health_collector = ChainHealthCollector()
    retained_accumulator = RetainedStateAccumulator(grid)
    retained_mean_accumulation_runtime_seconds = 0.0

    def _on_retained(
        _state: SurvivingEventState,
        counts: np.ndarray,
        _step_index: int,
    ) -> None:
        nonlocal retained_mean_accumulation_runtime_seconds
        accumulation_start = time.perf_counter()
        retained_accumulator.add_occupancy_counts(counts)
        retained_mean_accumulation_runtime_seconds += time.perf_counter() - accumulation_start

    chain_start = time.perf_counter()
    chain_result = run_chain(
        initial_state,
        initial_counts,
        grid=grid,
        proposal_backend=proposal_backend,
        alpha=alpha,
        uniform01=np.random.default_rng(config.seeds.acceptance_uniform).random,
        run_config=config.run_config,
        on_retained=_on_retained,
        chain_health_collector=chain_health_collector,
    )
    chain_runtime_seconds = time.perf_counter() - chain_start
    chain_core_runtime_seconds = chain_result.chain_core_runtime_seconds
    retained_mean = retained_accumulator.mean_occupancy()
    if chain_result.terminal_occupancy_counts is None:
        raise RuntimeError(
            "terminal_state.npy could not be written because the chain did not capture "
            "terminal occupancy counts"
        )
    terminal_state = np.array(chain_result.terminal_occupancy_counts, copy=True)
    chain_health = chain_health_collector.build_report(
        run_id=config.run_id,
        artifact_dir=config.artifacts.artifacts_dir,
        total_attempted_steps=chain_result.total_steps,
        burn_in_steps=config.run_config.burn_in_steps,
        thin_every=config.run_config.thin_every,
        surviving_event_count=len(initial_state.events),
        total_voxel_count=int(np.prod(np.asarray(grid.grid_shape, dtype=np.int64))),
        grid_shape=grid.grid_shape,
        alpha=alpha,
        terminal_nonzero_voxel_count=int(np.count_nonzero(terminal_state)),
        retained_mean_nonzero_voxel_count=int(np.count_nonzero(retained_mean)),
    )
    admitted_event_ids = _build_admitted_event_ids_payload(
        config=config,
        build_result=initial_state_build,
    )
    diagnostics = _build_reconstruction_diagnostics(
        config=config,
        grid=grid,
        initial_state=initial_state,
        chain_result=chain_result,
        retained_mean=retained_mean,
        terminal_state=terminal_state,
        preprocessing_runtime_seconds=preprocessing_runtime_seconds,
        chain_runtime_seconds=chain_runtime_seconds,
        chain_core_runtime_seconds=chain_core_runtime_seconds,
        retained_mean_accumulation_runtime_seconds=retained_mean_accumulation_runtime_seconds,
        artifact_write_runtime_seconds=None,
        total_runtime_seconds=time.perf_counter() - total_start,
    )
  
    artifact_write_start = time.perf_counter()
    artifact_bundle = write_reconstruction_artifact_bundle(
        config.artifacts.artifacts_dir,
        retained_mean=retained_mean,
        terminal_state=terminal_state,
        retained_mean_filename=config.artifacts.retained_mean_filename,
        terminal_state_filename=config.artifacts.terminal_state_filename,
        bounds_cm=config.reconstruction_input.voi.bounds,
        grid_shape=config.reconstruction_input.voi.grid_shape,
        burn_in_steps=config.run_config.burn_in_steps,
        thin_every=config.run_config.thin_every,
        retained_count=len(chain_result.retained_states),
        admitted_event_ids=admitted_event_ids,
        diagnostics=diagnostics,
        chain_health=chain_health,
        total_steps=chain_result.total_steps,
        terminal_step_index=chain_result.terminal_step_index,
    )
    artifact_write_runtime_seconds = time.perf_counter() - artifact_write_start
    diagnostics = _build_reconstruction_diagnostics(
        config=config,
        grid=grid,
        initial_state=initial_state,
        chain_result=chain_result,
        retained_mean=retained_mean,
        terminal_state=terminal_state,
        preprocessing_runtime_seconds=preprocessing_runtime_seconds,
        chain_runtime_seconds=chain_runtime_seconds,
        chain_core_runtime_seconds=chain_core_runtime_seconds,
        retained_mean_accumulation_runtime_seconds=retained_mean_accumulation_runtime_seconds,
        artifact_write_runtime_seconds=artifact_write_runtime_seconds,
        total_runtime_seconds=time.perf_counter() - total_start,
    )
    diagnostics = write_reconstruction_diagnostics_sidecar(
        config.artifacts.artifacts_dir,
        diagnostics,
    )
    artifact_bundle = ReconstructionArtifactBundle(
        artifacts_dir=artifact_bundle.artifacts_dir,
        retained_mean_path=artifact_bundle.retained_mean_path,
        terminal_state_path=artifact_bundle.terminal_state_path,
        admitted_event_ids_path=artifact_bundle.admitted_event_ids_path,
        metadata_path=artifact_bundle.metadata_path,
        diagnostics_path=artifact_bundle.diagnostics_path,
        chain_health_path=artifact_bundle.chain_health_path,
        admitted_event_ids=artifact_bundle.admitted_event_ids,
        metadata=artifact_bundle.metadata,
        diagnostics=diagnostics,
        chain_health=artifact_bundle.chain_health,
    )
    return ReconstructionRunResult(
        config=config,
        allowed_region=allowed_region,
        grid=grid,
        initial_state=initial_state,
        chain_result=chain_result,
        retained_mean=retained_mean,
        terminal_state=terminal_state,
        artifact_bundle=artifact_bundle,
    )


def run_reconstruction_job(
    config: ReconstructionJobConfig | Mapping[str, object],
    *,
    base_dir: Path | None = None,
    admission_only: bool = False,
) -> ReconstructionRunResult | ReconstructionAdmissionOnlyResult:
    """Assemble reconstruction input from explicit config, then run reconstruction."""
    if isinstance(config, Mapping):
        if base_dir is None:
            raise ValueError("base_dir is required when config is a mapping payload")
        job_config = reconstruction_job_config_from_payload(config, base_dir=Path(base_dir).resolve())
    elif isinstance(config, ReconstructionJobConfig):
        job_config = config
    else:
        raise TypeError("config must be a ReconstructionJobConfig or mapping payload")

    reconstruction_input = assemble_reconstruction_input_from_hdf5(
        job_config.input_h5_path,
        registration=job_config.registration,
        filter_config=job_config.filter_config,
        voi=job_config.reconstruction_voi,
    )
    run_config = ReconstructionRunConfig(
        reconstruction_input=reconstruction_input,
        artifacts=job_config.artifacts,
        survival_config=job_config.survival_config,
        run_config=job_config.run_config,
        seeds=job_config.seeds,
        run_id=job_config.run_id,
        surviving_event_cap=job_config.surviving_event_cap,
    )
    if admission_only:
        return run_reconstruction_admission_only(run_config)
    return run_reconstruction(run_config)


__all__ = [
    "DEFAULT_ACCEPTANCE_UNIFORM_SEED",
    "DEFAULT_EVENT_INDEX_SELECTOR_SEED",
    "DEFAULT_PROPOSAL_BACKEND_SEED",
    "DEFAULT_RECONSTRUCTION_CONFIG_FILENAME",
    "DEFAULT_RETAINED_MEAN_FILENAME",
    "DEFAULT_TERMINAL_STATE_FILENAME",
    "ReconstructionArtifactsConfig",
    "ReconstructionAdmissionOnlyResult",
    "ReconstructionJobConfig",
    "ReconstructionRunConfig",
    "ReconstructionRunResult",
    "ReconstructionSeeds",
    "reconstruction_job_config_from_payload",
    "run_reconstruction_admission_only",
    "run_reconstruction",
    "run_reconstruction_job",
]
