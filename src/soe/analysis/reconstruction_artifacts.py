"""Reconstruction artifact sidecar helpers.

This module owns the JSON sidecars written next to serialized reconstruction
arrays. `reconstruction_metadata.json` stays limited to artifact identity,
reconstruction geometry, and the retained-state provenance needed to interpret
the arrays. Run-behavior instrumentation is written separately to
`reconstruction_diagnostics.json`, and the exact admitted-event pool used to
seed the chain is exported separately to `admitted_event_ids.json`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from soe.analysis.reconstruction_diagnostics import (
    RECONSTRUCTION_DIAGNOSTICS_FILENAME,
    ReconstructionDiagnostics,
    write_reconstruction_diagnostics_sidecar,
)
from soe.soe.chain_health import (
    CHAIN_HEALTH_FILENAME,
    ChainHealthReport,
    write_chain_health_sidecar,
)

RECONSTRUCTION_METADATA_FILENAME = "reconstruction_metadata.json"
RECONSTRUCTION_METADATA_SCHEMA_NAME = "soe_reconstruction_metadata"
RECONSTRUCTION_METADATA_SCHEMA_VERSION = 1
RECONSTRUCTION_EXECUTION_MODE_FULL_RUN = "full_run"
RECONSTRUCTION_EXECUTION_MODE_ADMISSION_ONLY = "admission_only"
ADMITTED_EVENT_IDS_FILENAME = "admitted_event_ids.json"
ADMITTED_EVENT_IDS_SCHEMA_NAME = "soe_reconstruction_admitted_event_ids"
ADMITTED_EVENT_IDS_SCHEMA_VERSION = 1
ADMITTED_EVENT_IDS_SOURCE_POOL = (
    "selected_events_after_sampled_gate_and_strict_open_representative_point"
)
ADMITTED_EVENT_IDS_PRIMARY_IDENTITY_FIELD = "reconstruction_input_index"
ADMITTED_EVENT_IDS_IDENTITY_NOTE = (
    "Stable per-run indices are aligned with reconstruction_input.events. Raw "
    "HDF5 identifiers are not retained on the authoritative reconstruction input handoff "
    "and are therefore unavailable in this sidecar."
)
AXIS_ORDER = ("x", "y", "z")
SOE_PRIMARY_RECONSTRUCTED_QUANTITY_LABEL = "mean_occupancy"
TERMINAL_OCCUPANCY_COUNTS_QUANTITY_LABEL = "occupancy_counts"


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


def _as_nonnegative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} must be a nonnegative integer")
    int_value = int(value)
    if int_value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return int_value


def _as_positive_int(value: object, *, field_name: str) -> int:
    int_value = _as_nonnegative_int(value, field_name=field_name)
    if int_value < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return int_value


def _as_optional_nonnegative_int(value: object | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_nonnegative_int(value, field_name=field_name)


def _as_optional_positive_int(value: object | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_positive_int(value, field_name=field_name)


def _as_schema_name(value: object) -> str:
    schema_name = _as_name(value, field_name="schema_name")
    if schema_name != RECONSTRUCTION_METADATA_SCHEMA_NAME:
        raise ValueError(
            "schema_name must be "
            f"{RECONSTRUCTION_METADATA_SCHEMA_NAME!r}, got {schema_name!r}"
        )
    return schema_name


def _as_schema_version(value: object) -> int:
    schema_version = _as_nonnegative_int(value, field_name="schema_version")
    if schema_version != RECONSTRUCTION_METADATA_SCHEMA_VERSION:
        raise ValueError(
            "schema_version must be "
            f"{RECONSTRUCTION_METADATA_SCHEMA_VERSION}, got {schema_version}"
        )
    return schema_version


def _as_execution_mode(value: object) -> str:
    execution_mode = _as_name(value, field_name="execution_mode")
    if execution_mode not in (
        RECONSTRUCTION_EXECUTION_MODE_FULL_RUN,
        RECONSTRUCTION_EXECUTION_MODE_ADMISSION_ONLY,
    ):
        raise ValueError(
            "execution_mode must be "
            f"{RECONSTRUCTION_EXECUTION_MODE_FULL_RUN!r} or "
            f"{RECONSTRUCTION_EXECUTION_MODE_ADMISSION_ONLY!r}"
        )
    return execution_mode


def _as_admitted_event_ids_schema_name(value: object) -> str:
    schema_name = _as_name(value, field_name="schema_name")
    if schema_name != ADMITTED_EVENT_IDS_SCHEMA_NAME:
        raise ValueError(
            "schema_name must be "
            f"{ADMITTED_EVENT_IDS_SCHEMA_NAME!r}, got {schema_name!r}"
        )
    return schema_name


def _as_admitted_event_ids_schema_version(value: object) -> int:
    schema_version = _as_nonnegative_int(value, field_name="schema_version")
    if schema_version != ADMITTED_EVENT_IDS_SCHEMA_VERSION:
        raise ValueError(
            "schema_version must be "
            f"{ADMITTED_EVENT_IDS_SCHEMA_VERSION}, got {schema_version}"
        )
    return schema_version


def _as_bounds_cm(value: object, *, field_name: str = "bounds_cm") -> np.ndarray:
    bounds_cm = np.asarray(value, dtype=float)
    if bounds_cm.shape != (3, 2):
        raise ValueError(f"{field_name} must have shape (3, 2)")
    if not np.all(np.isfinite(bounds_cm)):
        raise ValueError(f"{field_name} must be finite")
    if not np.all(bounds_cm[:, 0] < bounds_cm[:, 1]):
        raise ValueError(f"{field_name} must satisfy min < max for x, y, z")
    return bounds_cm


def _as_grid_shape(value: object, *, field_name: str = "grid_shape") -> tuple[int, int, int]:
    try:
        grid_shape = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must contain 3 positive integers") from exc
    if len(grid_shape) != 3:
        raise ValueError(f"{field_name} must contain 3 entries")
    normalized: list[int] = []
    for axis_index, entry in enumerate(grid_shape):
        if isinstance(entry, bool) or not isinstance(entry, (int, np.integer)):
            raise ValueError(f"{field_name}[{axis_index}] must be an integer")
        axis_count = int(entry)
        if axis_count <= 0:
            raise ValueError(f"{field_name}[{axis_index}] must be positive")
        normalized.append(axis_count)
    return (normalized[0], normalized[1], normalized[2])


def _derived_voxel_size_cm(
    bounds_cm: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> tuple[float, float, float]:
    spans_cm = bounds_cm[:, 1] - bounds_cm[:, 0]
    voxel_size_cm = spans_cm / np.asarray(grid_shape, dtype=float)
    return (
        float(voxel_size_cm[0]),
        float(voxel_size_cm[1]),
        float(voxel_size_cm[2]),
    )


def _as_voxel_size_cm(
    value: object,
    *,
    bounds_cm: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> tuple[float, float, float]:
    voxel_size_cm = np.asarray(value, dtype=float)
    if voxel_size_cm.shape != (3,):
        raise ValueError("geometry.voxel_size_cm must have shape (3,)")
    if not np.all(np.isfinite(voxel_size_cm)):
        raise ValueError("geometry.voxel_size_cm must be finite")
    if not np.all(voxel_size_cm > 0.0):
        raise ValueError("geometry.voxel_size_cm must be positive")
    derived = np.asarray(_derived_voxel_size_cm(bounds_cm, grid_shape), dtype=float)
    if not np.allclose(voxel_size_cm, derived, rtol=0.0, atol=1e-12):
        raise ValueError(
            "geometry.voxel_size_cm must match the exact spacing implied by "
            "geometry.bounds_cm and geometry.grid_shape"
        )
    return (
        float(voxel_size_cm[0]),
        float(voxel_size_cm[1]),
        float(voxel_size_cm[2]),
    )


def _as_axis_order(value: object) -> tuple[str, str, str]:
    try:
        axis_order = tuple(value)
    except TypeError as exc:
        raise ValueError(f"geometry.axis_order must be the fixed {AXIS_ORDER} order") from exc
    if axis_order != AXIS_ORDER:
        raise ValueError(f"geometry.axis_order must be the fixed {AXIS_ORDER} order")
    return AXIS_ORDER


def _metadata_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.name == RECONSTRUCTION_METADATA_FILENAME:
        return candidate
    return candidate / RECONSTRUCTION_METADATA_FILENAME


def _admitted_event_ids_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.name == ADMITTED_EVENT_IDS_FILENAME:
        return candidate
    if candidate.suffix != "":
        return candidate
    return candidate / ADMITTED_EVENT_IDS_FILENAME


def _as_saved_array(
    value: object,
    *,
    grid_shape: tuple[int, int, int],
    field_name: str,
    integer_only: bool,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != grid_shape:
        raise ValueError(f"{field_name} must have shape {grid_shape}, got {array.shape}")
    if integer_only:
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{field_name} must be integer-valued")
    else:
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{field_name} must be numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must be finite")
    return array


@dataclass(frozen=True, slots=True)
class ReconstructionArtifactEntry:
    """Identity for one serialized reconstruction array."""

    filename: str | None
    quantity_label: str | None

    def __post_init__(self) -> None:
        filename = _as_optional_name(self.filename, field_name="filename")
        quantity_label = _as_optional_name(
            self.quantity_label,
            field_name="quantity_label",
        )
        if (filename is None) != (quantity_label is None):
            raise ValueError(
                "artifact entries must provide both filename and quantity_label or neither"
            )
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "quantity_label", quantity_label)

    def to_payload(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "quantity_label": self.quantity_label,
        }


@dataclass(frozen=True, slots=True)
class ReconstructionArtifactsMetadata:
    """Serialized artifact identities for retained-mean and terminal-state outputs."""

    retained_mean: ReconstructionArtifactEntry
    terminal_state: ReconstructionArtifactEntry
    admitted_event_ids_filename: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "admitted_event_ids_filename",
            _as_optional_name(
                self.admitted_event_ids_filename,
                field_name="artifacts.admitted_event_ids_filename",
            ),
        )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "retained_mean": self.retained_mean.to_payload(),
            "terminal_state": self.terminal_state.to_payload(),
        }
        if self.admitted_event_ids_filename is not None:
            payload["admitted_event_ids_filename"] = self.admitted_event_ids_filename
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> "ReconstructionArtifactsMetadata":
        artifacts = _as_mapping(payload, field_name="artifacts")
        retained_mean = _as_mapping(
            artifacts.get("retained_mean"),
            field_name="artifacts.retained_mean",
        )
        terminal_state = _as_mapping(
            artifacts.get("terminal_state"),
            field_name="artifacts.terminal_state",
        )
        return cls(
            retained_mean=ReconstructionArtifactEntry(
                filename=retained_mean.get("filename"),
                quantity_label=retained_mean.get("quantity_label"),
            ),
            terminal_state=ReconstructionArtifactEntry(
                filename=terminal_state.get("filename"),
                quantity_label=terminal_state.get("quantity_label"),
            ),
            admitted_event_ids_filename=artifacts.get("admitted_event_ids_filename"),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionGeometryMetadata:
    """Reconstruction-space geometry required to interpret serialized arrays."""

    bounds_cm: np.ndarray
    grid_shape: tuple[int, int, int]
    voxel_size_cm: tuple[float, float, float]
    axis_order: tuple[str, str, str] = AXIS_ORDER

    def __post_init__(self) -> None:
        bounds_cm = _as_bounds_cm(self.bounds_cm, field_name="geometry.bounds_cm")
        grid_shape = _as_grid_shape(self.grid_shape, field_name="geometry.grid_shape")
        voxel_size_cm = _as_voxel_size_cm(
            self.voxel_size_cm,
            bounds_cm=bounds_cm,
            grid_shape=grid_shape,
        )
        axis_order = _as_axis_order(self.axis_order)
        object.__setattr__(self, "bounds_cm", bounds_cm)
        object.__setattr__(self, "grid_shape", grid_shape)
        object.__setattr__(self, "voxel_size_cm", voxel_size_cm)
        object.__setattr__(self, "axis_order", axis_order)

    def to_payload(self) -> dict[str, object]:
        return {
            "bounds_cm": self.bounds_cm.tolist(),
            "grid_shape": list(self.grid_shape),
            "voxel_size_cm": list(self.voxel_size_cm),
            "axis_order": list(self.axis_order),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ReconstructionGeometryMetadata":
        geometry = _as_mapping(payload, field_name="geometry")
        return cls(
            bounds_cm=geometry.get("bounds_cm"),
            grid_shape=geometry.get("grid_shape"),
            voxel_size_cm=geometry.get("voxel_size_cm"),
            axis_order=geometry.get("axis_order"),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionAdmittedEventIdentity:
    """Stable identity for one admitted event on the current run input."""

    reconstruction_input_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reconstruction_input_index",
            _as_nonnegative_int(
                self.reconstruction_input_index,
                field_name="admitted_events[].reconstruction_input_index",
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {"reconstruction_input_index": self.reconstruction_input_index}

    @classmethod
    def from_payload(cls, payload: object) -> "ReconstructionAdmittedEventIdentity":
        mapping = _as_mapping(payload, field_name="admitted_events[]")
        return cls(
            reconstruction_input_index=mapping.get("reconstruction_input_index"),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionAdmissionMetadata:
    """Admission settings required to rebuild the same run-local survivor pool."""

    gate_mode: str
    survival_mode: str
    K: int | None = None
    K0: int | None = None
    J_max: int | None = None
    surviving_event_cap: int | None = None

    def __post_init__(self) -> None:
        gate_mode = _as_name(self.gate_mode, field_name="admission.gate_mode")
        survival_mode = _as_name(self.survival_mode, field_name="admission.survival_mode")
        surviving_event_cap = _as_optional_positive_int(
            self.surviving_event_cap,
            field_name="admission.surviving_event_cap",
        )
        if survival_mode == "fixed":
            K = _as_positive_int(self.K, field_name="admission.K")
            if self.K0 is not None or self.J_max is not None:
                raise ValueError("fixed admission metadata accepts only K")
            object.__setattr__(self, "gate_mode", gate_mode)
            object.__setattr__(self, "survival_mode", survival_mode)
            object.__setattr__(self, "K", K)
            object.__setattr__(self, "K0", None)
            object.__setattr__(self, "J_max", None)
            object.__setattr__(self, "surviving_event_cap", surviving_event_cap)
            return

        if survival_mode != "adaptive":
            raise ValueError("admission.survival_mode must be 'fixed' or 'adaptive'")
        if self.K is not None:
            raise ValueError("adaptive admission metadata accepts only K0 and J_max")
        object.__setattr__(self, "gate_mode", gate_mode)
        object.__setattr__(self, "survival_mode", survival_mode)
        object.__setattr__(self, "K", None)
        object.__setattr__(self, "K0", _as_positive_int(self.K0, field_name="admission.K0"))
        object.__setattr__(
            self,
            "J_max",
            _as_nonnegative_int(self.J_max, field_name="admission.J_max"),
        )
        object.__setattr__(self, "surviving_event_cap", surviving_event_cap)

    def to_payload(self) -> dict[str, object]:
        return {
            "gate_mode": self.gate_mode,
            "survival_mode": self.survival_mode,
            "K": self.K,
            "K0": self.K0,
            "J_max": self.J_max,
            "surviving_event_cap": self.surviving_event_cap,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ReconstructionAdmissionMetadata":
        admission = _as_mapping(payload, field_name="admission")
        return cls(
            gate_mode=admission.get("gate_mode"),
            survival_mode=admission.get("survival_mode"),
            K=admission.get("K"),
            K0=admission.get("K0"),
            J_max=admission.get("J_max"),
            surviving_event_cap=admission.get("surviving_event_cap"),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionAdmittedEventIds:
    """Standalone reconstruction-side survivor provenance sidecar payload."""

    admitted_events: tuple[ReconstructionAdmittedEventIdentity, ...]
    surviving_event_count: int
    admission: ReconstructionAdmissionMetadata
    geometry: ReconstructionGeometryMetadata
    artifacts_dir: Path
    run_id: str | None = None
    source_pool: str = ADMITTED_EVENT_IDS_SOURCE_POOL
    event_identity_field: str = ADMITTED_EVENT_IDS_PRIMARY_IDENTITY_FIELD
    event_identity_note: str = ADMITTED_EVENT_IDS_IDENTITY_NOTE
    schema_name: str = ADMITTED_EVENT_IDS_SCHEMA_NAME
    schema_version: int = ADMITTED_EVENT_IDS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_name",
            _as_admitted_event_ids_schema_name(self.schema_name),
        )
        object.__setattr__(
            self,
            "schema_version",
            _as_admitted_event_ids_schema_version(self.schema_version),
        )
        admitted_events = tuple(self.admitted_events)
        if any(not isinstance(event, ReconstructionAdmittedEventIdentity) for event in admitted_events):
            raise TypeError(
                "admitted_events must contain only ReconstructionAdmittedEventIdentity values"
            )
        surviving_event_count = _as_nonnegative_int(
            self.surviving_event_count,
            field_name="surviving_event_count",
        )
        if surviving_event_count != len(admitted_events):
            raise ValueError(
                "surviving_event_count must match the number of admitted_events"
            )
        if not isinstance(self.admission, ReconstructionAdmissionMetadata):
            raise TypeError("admission must be a ReconstructionAdmissionMetadata")
        if not isinstance(self.geometry, ReconstructionGeometryMetadata):
            raise TypeError("geometry must be a ReconstructionGeometryMetadata")
        source_pool = _as_name(self.source_pool, field_name="source_pool")
        event_identity_field = _as_name(
            self.event_identity_field,
            field_name="event_identity_field",
        )
        event_identity_note = _as_name(
            self.event_identity_note,
            field_name="event_identity_note",
        )
        object.__setattr__(self, "admitted_events", admitted_events)
        object.__setattr__(self, "surviving_event_count", surviving_event_count)
        object.__setattr__(self, "artifacts_dir", Path(self.artifacts_dir).resolve())
        object.__setattr__(self, "run_id", _as_optional_name(self.run_id, field_name="run_id"))
        object.__setattr__(self, "source_pool", source_pool)
        object.__setattr__(self, "event_identity_field", event_identity_field)
        object.__setattr__(self, "event_identity_note", event_identity_note)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "artifacts_dir": str(self.artifacts_dir),
            "source_pool": self.source_pool,
            "event_identity_field": self.event_identity_field,
            "event_identity_note": self.event_identity_note,
            "surviving_event_count": self.surviving_event_count,
            "admission": self.admission.to_payload(),
            "geometry": self.geometry.to_payload(),
            "admitted_events": [
                event.to_payload() for event in self.admitted_events
            ],
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ReconstructionAdmittedEventIds":
        mapping = _as_mapping(payload, field_name="admitted_event_ids")
        admitted_events_payload = mapping.get("admitted_events")
        if not isinstance(admitted_events_payload, list):
            raise TypeError("admitted_events must be a list")
        return cls(
            schema_name=mapping.get("schema_name"),
            schema_version=mapping.get("schema_version"),
            run_id=mapping.get("run_id"),
            artifacts_dir=Path(_as_name(mapping.get("artifacts_dir"), field_name="artifacts_dir")),
            source_pool=mapping.get("source_pool"),
            event_identity_field=mapping.get("event_identity_field"),
            event_identity_note=mapping.get("event_identity_note"),
            surviving_event_count=mapping.get("surviving_event_count"),
            admission=ReconstructionAdmissionMetadata.from_payload(mapping.get("admission")),
            geometry=ReconstructionGeometryMetadata.from_payload(mapping.get("geometry")),
            admitted_events=tuple(
                ReconstructionAdmittedEventIdentity.from_payload(item)
                for item in admitted_events_payload
            ),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionProvenanceMetadata:
    """Retained-state/run provenance needed to interpret saved arrays."""

    burn_in_steps: int
    thin_every: int
    retained_count: int
    total_steps: int | None = None
    terminal_step_index: int | None = None

    def __post_init__(self) -> None:
        burn_in_steps = _as_nonnegative_int(
            self.burn_in_steps,
            field_name="provenance.burn_in_steps",
        )
        thin_every = _as_positive_int(self.thin_every, field_name="provenance.thin_every")
        retained_count = _as_nonnegative_int(
            self.retained_count,
            field_name="provenance.retained_count",
        )
        total_steps = _as_optional_nonnegative_int(
            self.total_steps,
            field_name="provenance.total_steps",
        )
        terminal_step_index = _as_optional_nonnegative_int(
            self.terminal_step_index,
            field_name="provenance.terminal_step_index",
        )
        if total_steps is not None and terminal_step_index is not None:
            if terminal_step_index > total_steps:
                raise ValueError(
                    "provenance.terminal_step_index must be <= provenance.total_steps"
                )
        object.__setattr__(self, "burn_in_steps", burn_in_steps)
        object.__setattr__(self, "thin_every", thin_every)
        object.__setattr__(self, "retained_count", retained_count)
        object.__setattr__(self, "total_steps", total_steps)
        object.__setattr__(self, "terminal_step_index", terminal_step_index)

    def to_payload(self) -> dict[str, object]:
        return {
            "burn_in_steps": self.burn_in_steps,
            "thin_every": self.thin_every,
            "retained_count": self.retained_count,
            "total_steps": self.total_steps,
            "terminal_step_index": self.terminal_step_index,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ReconstructionProvenanceMetadata":
        provenance = _as_mapping(payload, field_name="provenance")
        return cls(
            burn_in_steps=provenance.get("burn_in_steps"),
            thin_every=provenance.get("thin_every"),
            retained_count=provenance.get("retained_count"),
            total_steps=provenance.get("total_steps"),
            terminal_step_index=provenance.get("terminal_step_index"),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionMetadata:
    """Validated reconstruction-only metadata sidecar payload."""

    artifacts: ReconstructionArtifactsMetadata
    geometry: ReconstructionGeometryMetadata
    provenance: ReconstructionProvenanceMetadata
    execution_mode: str = RECONSTRUCTION_EXECUTION_MODE_FULL_RUN
    schema_name: str = RECONSTRUCTION_METADATA_SCHEMA_NAME
    schema_version: int = RECONSTRUCTION_METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_name", _as_schema_name(self.schema_name))
        object.__setattr__(self, "schema_version", _as_schema_version(self.schema_version))
        execution_mode = _as_execution_mode(self.execution_mode)
        retained_mean_present = (
            self.artifacts.retained_mean.filename is not None
            and self.artifacts.retained_mean.quantity_label is not None
        )
        terminal_state_present = (
            self.artifacts.terminal_state.filename is not None
            and self.artifacts.terminal_state.quantity_label is not None
        )
        admitted_event_ids_present = self.artifacts.admitted_event_ids_filename is not None
        if execution_mode == RECONSTRUCTION_EXECUTION_MODE_FULL_RUN:
            if not retained_mean_present:
                raise ValueError(
                    "artifacts.retained_mean must provide filename and quantity_label "
                    "for full_run metadata"
                )
        else:
            if retained_mean_present or terminal_state_present:
                raise ValueError(
                    "admission_only metadata must not claim retained_mean or terminal_state artifacts"
                )
            if not admitted_event_ids_present:
                raise ValueError(
                    "admission_only metadata must provide artifacts.admitted_event_ids_filename"
                )
            if self.provenance.retained_count != 0:
                raise ValueError(
                    "admission_only metadata must report provenance.retained_count == 0"
                )
            if self.provenance.total_steps is not None:
                raise ValueError(
                    "admission_only metadata must not report provenance.total_steps"
                )
            if self.provenance.terminal_step_index is not None:
                raise ValueError(
                    "admission_only metadata must not report provenance.terminal_step_index"
                )
        object.__setattr__(self, "execution_mode", execution_mode)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "execution_mode": self.execution_mode,
            "artifacts": self.artifacts.to_payload(),
            "geometry": self.geometry.to_payload(),
            "provenance": self.provenance.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ReconstructionMetadata":
        mapping = _as_mapping(payload, field_name="reconstruction_metadata")
        return cls(
            schema_name=mapping.get("schema_name"),
            schema_version=mapping.get("schema_version"),
            execution_mode=mapping.get("execution_mode", RECONSTRUCTION_EXECUTION_MODE_FULL_RUN),
            artifacts=ReconstructionArtifactsMetadata.from_payload(mapping.get("artifacts")),
            geometry=ReconstructionGeometryMetadata.from_payload(mapping.get("geometry")),
            provenance=ReconstructionProvenanceMetadata.from_payload(mapping.get("provenance")),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionArtifactBundle:
    """Concrete paths for one saved reconstruction artifact bundle."""

    artifacts_dir: Path
    retained_mean_path: Path
    terminal_state_path: Path
    admitted_event_ids_path: Path | None
    metadata_path: Path
    diagnostics_path: Path
    chain_health_path: Path
    admitted_event_ids: ReconstructionAdmittedEventIds | None
    metadata: ReconstructionMetadata
    diagnostics: ReconstructionDiagnostics
    chain_health: ChainHealthReport

    def __post_init__(self) -> None:
        artifacts_dir = Path(self.artifacts_dir).resolve()
        retained_mean_path = Path(self.retained_mean_path).resolve()
        terminal_state_path = Path(self.terminal_state_path).resolve()
        admitted_event_ids_path = (
            None if self.admitted_event_ids_path is None else Path(self.admitted_event_ids_path).resolve()
        )
        metadata_path = Path(self.metadata_path).resolve()
        diagnostics_path = Path(self.diagnostics_path).resolve()
        chain_health_path = Path(self.chain_health_path).resolve()
        if retained_mean_path.parent != artifacts_dir:
            raise ValueError("retained_mean_path must live under artifacts_dir")
        if terminal_state_path.parent != artifacts_dir:
            raise ValueError("terminal_state_path must live under artifacts_dir")
        if admitted_event_ids_path is not None and admitted_event_ids_path.parent != artifacts_dir:
            raise ValueError("admitted_event_ids_path must live under artifacts_dir")
        if metadata_path.parent != artifacts_dir:
            raise ValueError("metadata_path must live under artifacts_dir")
        if diagnostics_path.parent != artifacts_dir:
            raise ValueError("diagnostics_path must live under artifacts_dir")
        if chain_health_path.parent != artifacts_dir:
            raise ValueError("chain_health_path must live under artifacts_dir")
        if (admitted_event_ids_path is None) != (self.admitted_event_ids is None):
            raise ValueError(
                "admitted_event_ids_path and admitted_event_ids must either both be present or both be absent"
            )
        if self.admitted_event_ids is not None and not isinstance(
            self.admitted_event_ids,
            ReconstructionAdmittedEventIds,
        ):
            raise TypeError("admitted_event_ids must be a ReconstructionAdmittedEventIds")
        if not isinstance(self.metadata, ReconstructionMetadata):
            raise TypeError("metadata must be a ReconstructionMetadata")
        if not isinstance(self.diagnostics, ReconstructionDiagnostics):
            raise TypeError("diagnostics must be a ReconstructionDiagnostics")
        if not isinstance(self.chain_health, ChainHealthReport):
            raise TypeError("chain_health must be a ChainHealthReport")
        object.__setattr__(self, "artifacts_dir", artifacts_dir)
        object.__setattr__(self, "retained_mean_path", retained_mean_path)
        object.__setattr__(self, "terminal_state_path", terminal_state_path)
        object.__setattr__(self, "admitted_event_ids_path", admitted_event_ids_path)
        object.__setattr__(self, "metadata_path", metadata_path)
        object.__setattr__(self, "diagnostics_path", diagnostics_path)
        object.__setattr__(self, "chain_health_path", chain_health_path)


@dataclass(frozen=True, slots=True)
class ReconstructionAdmissionArtifactBundle:
    """Concrete paths for one admission-only reconstruction artifact bundle."""

    artifacts_dir: Path
    admitted_event_ids_path: Path
    metadata_path: Path
    admitted_event_ids: ReconstructionAdmittedEventIds
    metadata: ReconstructionMetadata

    def __post_init__(self) -> None:
        artifacts_dir = Path(self.artifacts_dir).resolve()
        admitted_event_ids_path = Path(self.admitted_event_ids_path).resolve()
        metadata_path = Path(self.metadata_path).resolve()
        if admitted_event_ids_path.parent != artifacts_dir:
            raise ValueError("admitted_event_ids_path must live under artifacts_dir")
        if metadata_path.parent != artifacts_dir:
            raise ValueError("metadata_path must live under artifacts_dir")
        if not isinstance(self.admitted_event_ids, ReconstructionAdmittedEventIds):
            raise TypeError("admitted_event_ids must be a ReconstructionAdmittedEventIds")
        if not isinstance(self.metadata, ReconstructionMetadata):
            raise TypeError("metadata must be a ReconstructionMetadata")
        if self.metadata.execution_mode != RECONSTRUCTION_EXECUTION_MODE_ADMISSION_ONLY:
            raise ValueError(
                "ReconstructionAdmissionArtifactBundle requires admission_only metadata"
            )
        object.__setattr__(self, "artifacts_dir", artifacts_dir)
        object.__setattr__(self, "admitted_event_ids_path", admitted_event_ids_path)
        object.__setattr__(self, "metadata_path", metadata_path)


def write_reconstruction_metadata_sidecar(
    artifacts_dir: Path,
    *,
    retained_mean_filename: str | None,
    bounds_cm: object,
    grid_shape: object,
    burn_in_steps: object,
    thin_every: object,
    retained_count: object,
    retained_mean_quantity_label: str | None = SOE_PRIMARY_RECONSTRUCTED_QUANTITY_LABEL,
    terminal_state_filename: str | None = None,
    terminal_state_quantity_label: str | None = None,
    admitted_event_ids_filename: str | None = None,
    execution_mode: str = RECONSTRUCTION_EXECUTION_MODE_FULL_RUN,
    total_steps: object | None = None,
    terminal_step_index: object | None = None,
) -> ReconstructionMetadata:
    """Write the reconstruction-only metadata sidecar next to saved arrays.

    `total_steps` and the lightweight auxiliary provenance fields stay nullable
    because future artifact writers may not surface them at the write boundary.
    This helper validates what is present and does not invent missing authority.
    """
    retained_entry = ReconstructionArtifactEntry(
        filename=retained_mean_filename,
        quantity_label=retained_mean_quantity_label,
    )
    terminal_entry = ReconstructionArtifactEntry(
        filename=terminal_state_filename,
        quantity_label=(
            None
            if terminal_state_filename is None
            else (
                terminal_state_quantity_label
                if terminal_state_quantity_label is not None
                else TERMINAL_OCCUPANCY_COUNTS_QUANTITY_LABEL
            )
        ),
    )
    bounds_cm_array = _as_bounds_cm(bounds_cm)
    grid_shape_tuple = _as_grid_shape(grid_shape)
    metadata = ReconstructionMetadata(
        artifacts=ReconstructionArtifactsMetadata(
            retained_mean=retained_entry,
            terminal_state=terminal_entry,
            admitted_event_ids_filename=admitted_event_ids_filename,
        ),
        geometry=ReconstructionGeometryMetadata(
            bounds_cm=bounds_cm_array,
            grid_shape=grid_shape_tuple,
            voxel_size_cm=_derived_voxel_size_cm(bounds_cm_array, grid_shape_tuple),
        ),
        provenance=ReconstructionProvenanceMetadata(
            burn_in_steps=burn_in_steps,
            thin_every=thin_every,
            retained_count=retained_count,
            total_steps=total_steps,
            terminal_step_index=terminal_step_index,
        ),
        execution_mode=execution_mode,
    )
    metadata_path = _metadata_path(Path(artifacts_dir).resolve())
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def validate_reconstruction_metadata_payload(payload: object) -> ReconstructionMetadata:
    """Validate one in-memory reconstruction metadata payload."""
    return ReconstructionMetadata.from_payload(payload)


def write_reconstruction_admitted_event_ids_sidecar(
    path: Path,
    admitted_event_ids: ReconstructionAdmittedEventIds,
) -> ReconstructionAdmittedEventIds:
    """Write the standalone admitted-event survivor provenance sidecar."""
    resolved_path = _admitted_event_ids_path(Path(path).resolve())
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    if admitted_event_ids.artifacts_dir != resolved_path.parent:
        raise ValueError(
            "admitted_event_ids.artifacts_dir must match the sidecar directory"
        )
    resolved_path.write_text(
        json.dumps(admitted_event_ids.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return admitted_event_ids


def validate_reconstruction_admitted_event_ids_payload(
    payload: object,
) -> ReconstructionAdmittedEventIds:
    """Validate one in-memory admitted-event survivor provenance payload."""
    return ReconstructionAdmittedEventIds.from_payload(payload)


def write_reconstruction_admission_artifact_bundle(
    artifacts_dir: Path,
    *,
    bounds_cm: object,
    grid_shape: object,
    admitted_event_ids: ReconstructionAdmittedEventIds,
    burn_in_steps: object = 0,
    thin_every: object = 1,
    admitted_event_ids_filename: str = ADMITTED_EVENT_IDS_FILENAME,
) -> ReconstructionAdmissionArtifactBundle:
    """Write the admission-only survivor provenance sidecars without MH outputs."""
    resolved_artifacts_dir = Path(artifacts_dir).resolve()
    resolved_artifacts_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(admitted_event_ids, ReconstructionAdmittedEventIds):
        raise TypeError("admitted_event_ids must be a ReconstructionAdmittedEventIds")
    admitted_event_ids_path = resolved_artifacts_dir / _as_name(
        admitted_event_ids_filename,
        field_name="admitted_event_ids_filename",
    )
    admitted_event_ids_payload = write_reconstruction_admitted_event_ids_sidecar(
        admitted_event_ids_path,
        admitted_event_ids,
    )
    metadata = write_reconstruction_metadata_sidecar(
        resolved_artifacts_dir,
        retained_mean_filename=None,
        retained_mean_quantity_label=None,
        terminal_state_filename=None,
        terminal_state_quantity_label=None,
        admitted_event_ids_filename=admitted_event_ids_path.name,
        execution_mode=RECONSTRUCTION_EXECUTION_MODE_ADMISSION_ONLY,
        bounds_cm=bounds_cm,
        grid_shape=grid_shape,
        burn_in_steps=burn_in_steps,
        thin_every=thin_every,
        retained_count=0,
        total_steps=None,
        terminal_step_index=None,
    )
    return ReconstructionAdmissionArtifactBundle(
        artifacts_dir=resolved_artifacts_dir,
        admitted_event_ids_path=admitted_event_ids_path,
        metadata_path=_metadata_path(resolved_artifacts_dir),
        admitted_event_ids=admitted_event_ids_payload,
        metadata=metadata,
    )


def write_reconstruction_artifact_bundle(
    artifacts_dir: Path,
    *,
    retained_mean: object,
    terminal_state: object,
    bounds_cm: object,
    grid_shape: object,
    burn_in_steps: object,
    thin_every: object,
    retained_count: object,
    retained_mean_filename: str = "retained_mean.npy",
    terminal_state_filename: str = "terminal_state.npy",
    retained_mean_quantity_label: str = SOE_PRIMARY_RECONSTRUCTED_QUANTITY_LABEL,
    terminal_state_quantity_label: str = TERMINAL_OCCUPANCY_COUNTS_QUANTITY_LABEL,
    admitted_event_ids: ReconstructionAdmittedEventIds | None = None,
    admitted_event_ids_filename: str = ADMITTED_EVENT_IDS_FILENAME,
    diagnostics: ReconstructionDiagnostics,
    chain_health: ChainHealthReport,
    total_steps: object | None = None,
    terminal_step_index: object | None = None,
) -> ReconstructionArtifactBundle:
    """Write the authoritative reconstruction arrays plus metadata/diagnostics sidecars."""
    grid_shape_tuple = _as_grid_shape(grid_shape)
    retained_mean_array = _as_saved_array(
        retained_mean,
        grid_shape=grid_shape_tuple,
        field_name="retained_mean",
        integer_only=False,
    )
    terminal_state_array = _as_saved_array(
        terminal_state,
        grid_shape=grid_shape_tuple,
        field_name="terminal_state",
        integer_only=True,
    )
    resolved_artifacts_dir = Path(artifacts_dir).resolve()
    resolved_artifacts_dir.mkdir(parents=True, exist_ok=True)

    retained_mean_path = resolved_artifacts_dir / _as_name(
        retained_mean_filename,
        field_name="retained_mean_filename",
    )
    terminal_state_path = resolved_artifacts_dir / _as_name(
        terminal_state_filename,
        field_name="terminal_state_filename",
    )
    np.save(retained_mean_path, retained_mean_array, allow_pickle=False)
    np.save(terminal_state_path, terminal_state_array, allow_pickle=False)

    admitted_event_ids_path: Path | None = None
    admitted_event_ids_payload: ReconstructionAdmittedEventIds | None = None
    if admitted_event_ids is not None:
        if not isinstance(admitted_event_ids, ReconstructionAdmittedEventIds):
            raise TypeError("admitted_event_ids must be a ReconstructionAdmittedEventIds")
        admitted_event_ids_path = resolved_artifacts_dir / _as_name(
            admitted_event_ids_filename,
            field_name="admitted_event_ids_filename",
        )
        admitted_event_ids_payload = write_reconstruction_admitted_event_ids_sidecar(
            admitted_event_ids_path,
            admitted_event_ids,
        )

    metadata = write_reconstruction_metadata_sidecar(
        resolved_artifacts_dir,
        retained_mean_filename=retained_mean_path.name,
        terminal_state_filename=terminal_state_path.name,
        bounds_cm=bounds_cm,
        grid_shape=grid_shape_tuple,
        burn_in_steps=burn_in_steps,
        thin_every=thin_every,
        retained_count=retained_count,
        retained_mean_quantity_label=retained_mean_quantity_label,
        terminal_state_quantity_label=terminal_state_quantity_label,
        admitted_event_ids_filename=(
            None if admitted_event_ids_path is None else admitted_event_ids_path.name
        ),
        total_steps=total_steps,
        terminal_step_index=terminal_step_index,
    )
    diagnostics_payload = write_reconstruction_diagnostics_sidecar(
        resolved_artifacts_dir,
        diagnostics,
    )
    chain_health_payload = write_chain_health_sidecar(
        resolved_artifacts_dir,
        chain_health,
    )
    return ReconstructionArtifactBundle(
        artifacts_dir=resolved_artifacts_dir,
        retained_mean_path=retained_mean_path,
        terminal_state_path=terminal_state_path,
        admitted_event_ids_path=admitted_event_ids_path,
        metadata_path=_metadata_path(resolved_artifacts_dir),
        diagnostics_path=(resolved_artifacts_dir / RECONSTRUCTION_DIAGNOSTICS_FILENAME),
        chain_health_path=(resolved_artifacts_dir / CHAIN_HEALTH_FILENAME),
        admitted_event_ids=admitted_event_ids_payload,
        metadata=metadata,
        diagnostics=diagnostics_payload,
        chain_health=chain_health_payload,
    )


def load_reconstruction_metadata(path: Path) -> ReconstructionMetadata:
    """Load and validate one reconstruction metadata sidecar."""
    metadata_path = _metadata_path(Path(path).resolve())
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing reconstruction metadata sidecar: {metadata_path}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return validate_reconstruction_metadata_payload(payload)


def load_reconstruction_admitted_event_ids(path: Path) -> ReconstructionAdmittedEventIds:
    """Load and validate one admitted-event survivor provenance sidecar."""
    admitted_event_ids_path = _admitted_event_ids_path(Path(path).resolve())
    if not admitted_event_ids_path.is_file():
        raise FileNotFoundError(
            f"Missing reconstruction admitted-event ids sidecar: {admitted_event_ids_path}"
        )
    payload = json.loads(admitted_event_ids_path.read_text(encoding="utf-8"))
    return validate_reconstruction_admitted_event_ids_payload(payload)


__all__ = [
    "ADMITTED_EVENT_IDS_FILENAME",
    "ADMITTED_EVENT_IDS_SCHEMA_NAME",
    "ADMITTED_EVENT_IDS_SCHEMA_VERSION",
    "AXIS_ORDER",
    "RECONSTRUCTION_EXECUTION_MODE_ADMISSION_ONLY",
    "RECONSTRUCTION_EXECUTION_MODE_FULL_RUN",
    "RECONSTRUCTION_METADATA_FILENAME",
    "RECONSTRUCTION_METADATA_SCHEMA_NAME",
    "RECONSTRUCTION_METADATA_SCHEMA_VERSION",
    "TERMINAL_OCCUPANCY_COUNTS_QUANTITY_LABEL",
    "ReconstructionAdmissionMetadata",
    "ReconstructionAdmissionArtifactBundle",
    "ReconstructionAdmittedEventIdentity",
    "ReconstructionAdmittedEventIds",
    "ReconstructionArtifactBundle",
    "ReconstructionArtifactEntry",
    "ReconstructionArtifactsMetadata",
    "ReconstructionGeometryMetadata",
    "ReconstructionMetadata",
    "ReconstructionProvenanceMetadata",
    "load_reconstruction_admitted_event_ids",
    "load_reconstruction_metadata",
    "validate_reconstruction_admitted_event_ids_payload",
    "validate_reconstruction_metadata_payload",
    "write_reconstruction_admission_artifact_bundle",
    "write_reconstruction_artifact_bundle",
    "write_reconstruction_admitted_event_ids_sidecar",
    "write_reconstruction_metadata_sidecar",
]
