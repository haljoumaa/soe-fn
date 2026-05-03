"""Reconstruction diagnostics sidecar helpers.

This module owns the separate reconstruction diagnostics JSON written next to
the authoritative reconstruction arrays. The diagnostics payload is
intentionally decoupled from `reconstruction_metadata.json`: it records
instrumentation and run-behavior metrics only, not the artifact-interpretation
contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import operator
from pathlib import Path

import numpy as np

RECONSTRUCTION_DIAGNOSTICS_FILENAME = "reconstruction_diagnostics.json"
RECONSTRUCTION_DIAGNOSTICS_SCHEMA_NAME = "soe_reconstruction_diagnostics"
RECONSTRUCTION_DIAGNOSTICS_SCHEMA_VERSION = 1


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


def _as_optional_nonnegative_int(value: object | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_nonnegative_int(value, field_name=field_name)


def _as_optional_positive_int(value: object | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_positive_int(value, field_name=field_name)


def _as_optional_seconds(value: object | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"{field_name} must be a finite nonnegative float")
    return seconds


def _as_optional_nonnegative_float(value: object | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be a finite nonnegative float")
    return number


def _as_optional_rate(value: object | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    rate = float(value)
    if not math.isfinite(rate) or not (0.0 <= rate <= 1.0):
        raise ValueError(f"{field_name} must be a finite rate in [0, 1]")
    return rate


def _as_grid_shape(value: object, *, field_name: str = "grid_shape") -> tuple[int, int, int]:
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


def _as_schema_name(value: object) -> str:
    schema_name = _as_name(value, field_name="schema_name")
    if schema_name != RECONSTRUCTION_DIAGNOSTICS_SCHEMA_NAME:
        raise ValueError(
            "schema_name must be "
            f"{RECONSTRUCTION_DIAGNOSTICS_SCHEMA_NAME!r}, got {schema_name!r}"
        )
    return schema_name


def _as_schema_version(value: object) -> int:
    schema_version = _as_nonnegative_int(value, field_name="schema_version")
    if schema_version != RECONSTRUCTION_DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError(
            "schema_version must be "
            f"{RECONSTRUCTION_DIAGNOSTICS_SCHEMA_VERSION}, got {schema_version}"
        )
    return schema_version


def _diagnostics_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.name == RECONSTRUCTION_DIAGNOSTICS_FILENAME:
        return candidate
    return candidate / RECONSTRUCTION_DIAGNOSTICS_FILENAME


@dataclass(frozen=True, slots=True)
class ReconstructionCountsDiagnostics:
    """Reconstruction-boundary event counts."""

    surviving_event_count: int
    raw_event_count: int | None = None
    filtered_valid_event_count: int | None = None
    dropped_event_count: int | None = None

    def __post_init__(self) -> None:
        surviving_event_count = _as_nonnegative_int(
            self.surviving_event_count,
            field_name="counts.surviving_event_count",
        )
        raw_event_count = _as_optional_nonnegative_int(
            self.raw_event_count,
            field_name="counts.raw_event_count",
        )
        filtered_valid_event_count = _as_optional_nonnegative_int(
            self.filtered_valid_event_count,
            field_name="counts.filtered_valid_event_count",
        )
        dropped_event_count = _as_optional_nonnegative_int(
            self.dropped_event_count,
            field_name="counts.dropped_event_count",
        )
        if raw_event_count is not None and filtered_valid_event_count is not None:
            if filtered_valid_event_count > raw_event_count:
                raise ValueError(
                    "counts.filtered_valid_event_count must be <= counts.raw_event_count"
                )
        if filtered_valid_event_count is not None:
            if surviving_event_count > filtered_valid_event_count:
                raise ValueError(
                    "counts.surviving_event_count must be <= counts.filtered_valid_event_count"
                )
            if dropped_event_count is not None:
                expected_dropped = filtered_valid_event_count - surviving_event_count
                if dropped_event_count != expected_dropped:
                    raise ValueError(
                        "counts.dropped_event_count must equal "
                        "counts.filtered_valid_event_count - counts.surviving_event_count"
                    )
        object.__setattr__(self, "surviving_event_count", surviving_event_count)
        object.__setattr__(self, "raw_event_count", raw_event_count)
        object.__setattr__(self, "filtered_valid_event_count", filtered_valid_event_count)
        object.__setattr__(self, "dropped_event_count", dropped_event_count)

    def to_payload(self) -> dict[str, object]:
        return {
            "raw_event_count": self.raw_event_count,
            "filtered_valid_event_count": self.filtered_valid_event_count,
            "surviving_event_count": self.surviving_event_count,
            "dropped_event_count": self.dropped_event_count,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ReconstructionCountsDiagnostics":
        counts = _as_mapping(payload, field_name="counts")
        return cls(
            raw_event_count=counts.get("raw_event_count"),
            filtered_valid_event_count=counts.get("filtered_valid_event_count"),
            surviving_event_count=counts.get("surviving_event_count"),
            dropped_event_count=counts.get("dropped_event_count"),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionPreprocessingDiagnostics:
    """Preprocessing-mode diagnostics surfaced at the run boundary."""

    survival_mode: str | None = None
    gate_mode: str | None = None
    K: int | None = None
    surviving_event_cap: int | None = None
    runtime_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "survival_mode",
            _as_optional_name(self.survival_mode, field_name="preprocessing.survival_mode"),
        )
        object.__setattr__(
            self,
            "gate_mode",
            _as_optional_name(self.gate_mode, field_name="preprocessing.gate_mode"),
        )
        object.__setattr__(
            self,
            "K",
            _as_optional_positive_int(self.K, field_name="preprocessing.K"),
        )
        object.__setattr__(
            self,
            "surviving_event_cap",
            _as_optional_positive_int(
                self.surviving_event_cap,
                field_name="preprocessing.surviving_event_cap",
            ),
        )
        object.__setattr__(
            self,
            "runtime_seconds",
            _as_optional_seconds(
                self.runtime_seconds,
                field_name="preprocessing.runtime_seconds",
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "survival_mode": self.survival_mode,
            "gate_mode": self.gate_mode,
            "K": self.K,
            "surviving_event_cap": self.surviving_event_cap,
            "runtime_seconds": self.runtime_seconds,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ReconstructionPreprocessingDiagnostics":
        preprocessing = _as_mapping(payload, field_name="preprocessing")
        return cls(
            survival_mode=preprocessing.get("survival_mode"),
            gate_mode=preprocessing.get("gate_mode"),
            K=preprocessing.get("K"),
            surviving_event_cap=preprocessing.get("surviving_event_cap"),
            runtime_seconds=preprocessing.get("runtime_seconds"),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionChainDiagnostics:
    """Finite-run chain diagnostics."""

    burn_in_steps: int
    thin_every: int
    retained_count: int
    total_steps_attempted: int | None = None
    accepted_step_count: int | None = None
    rejected_step_count: int | None = None
    acceptance_rate: float | None = None
    terminal_step_index: int | None = None
    runtime_seconds: float | None = None
    chain_core_runtime_seconds: float | None = None
    retained_mean_accumulation_runtime_seconds: float | None = None
    artifact_write_runtime_seconds: float | None = None
    total_runtime_seconds: float | None = None

    def __post_init__(self) -> None:
        burn_in_steps = _as_nonnegative_int(
            self.burn_in_steps,
            field_name="chain.burn_in_steps",
        )
        thin_every = _as_positive_int(self.thin_every, field_name="chain.thin_every")
        retained_count = _as_nonnegative_int(
            self.retained_count,
            field_name="chain.retained_count",
        )
        total_steps_attempted = _as_optional_nonnegative_int(
            self.total_steps_attempted,
            field_name="chain.total_steps_attempted",
        )
        accepted_step_count = _as_optional_nonnegative_int(
            self.accepted_step_count,
            field_name="chain.accepted_step_count",
        )
        rejected_step_count = _as_optional_nonnegative_int(
            self.rejected_step_count,
            field_name="chain.rejected_step_count",
        )
        acceptance_rate = _as_optional_rate(
            self.acceptance_rate,
            field_name="chain.acceptance_rate",
        )
        terminal_step_index = _as_optional_nonnegative_int(
            self.terminal_step_index,
            field_name="chain.terminal_step_index",
        )
        runtime_seconds = _as_optional_seconds(
            self.runtime_seconds,
            field_name="chain.runtime_seconds",
        )
        chain_core_runtime_seconds = _as_optional_seconds(
            self.chain_core_runtime_seconds,
            field_name="chain.chain_core_runtime_seconds",
        )
        retained_mean_accumulation_runtime_seconds = _as_optional_seconds(
            self.retained_mean_accumulation_runtime_seconds,
            field_name="chain.retained_mean_accumulation_runtime_seconds",
        )
        artifact_write_runtime_seconds = _as_optional_seconds(
            self.artifact_write_runtime_seconds,
            field_name="chain.artifact_write_runtime_seconds",
        )
        total_runtime_seconds = _as_optional_seconds(
            self.total_runtime_seconds,
            field_name="chain.total_runtime_seconds",
        )
        if total_steps_attempted is not None and terminal_step_index is not None:
            if terminal_step_index > total_steps_attempted:
                raise ValueError(
                    "chain.terminal_step_index must be <= chain.total_steps_attempted"
                )
        if total_steps_attempted is not None and accepted_step_count is not None:
            if accepted_step_count > total_steps_attempted:
                raise ValueError(
                    "chain.accepted_step_count must be <= chain.total_steps_attempted"
                )
        if total_steps_attempted is not None and rejected_step_count is not None:
            if rejected_step_count > total_steps_attempted:
                raise ValueError(
                    "chain.rejected_step_count must be <= chain.total_steps_attempted"
                )
        if total_steps_attempted is not None and accepted_step_count is not None:
            derived_rejected = total_steps_attempted - accepted_step_count
            if rejected_step_count is not None and rejected_step_count != derived_rejected:
                raise ValueError(
                    "chain.rejected_step_count must equal "
                    "chain.total_steps_attempted - chain.accepted_step_count"
                )
            if total_steps_attempted == 0:
                if acceptance_rate is not None:
                    raise ValueError(
                        "chain.acceptance_rate must be null when no steps were attempted"
                    )
            else:
                derived_acceptance_rate = accepted_step_count / float(total_steps_attempted)
                if acceptance_rate is not None and not math.isclose(
                    acceptance_rate,
                    derived_acceptance_rate,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "chain.acceptance_rate must equal "
                        "chain.accepted_step_count / chain.total_steps_attempted"
                    )
        if runtime_seconds is not None and chain_core_runtime_seconds is not None:
            if chain_core_runtime_seconds > runtime_seconds:
                raise ValueError(
                    "chain.chain_core_runtime_seconds must be <= chain.runtime_seconds"
                )
        if runtime_seconds is not None and retained_mean_accumulation_runtime_seconds is not None:
            if retained_mean_accumulation_runtime_seconds > runtime_seconds:
                raise ValueError(
                    "chain.retained_mean_accumulation_runtime_seconds must be <= "
                    "chain.runtime_seconds"
                )
        if total_runtime_seconds is not None and runtime_seconds is not None:
            if runtime_seconds > total_runtime_seconds:
                raise ValueError(
                    "chain.runtime_seconds must be <= chain.total_runtime_seconds"
                )
        if total_runtime_seconds is not None and artifact_write_runtime_seconds is not None:
            if artifact_write_runtime_seconds > total_runtime_seconds:
                raise ValueError(
                    "chain.artifact_write_runtime_seconds must be <= "
                    "chain.total_runtime_seconds"
                )
        object.__setattr__(self, "burn_in_steps", burn_in_steps)
        object.__setattr__(self, "thin_every", thin_every)
        object.__setattr__(self, "retained_count", retained_count)
        object.__setattr__(self, "total_steps_attempted", total_steps_attempted)
        object.__setattr__(self, "accepted_step_count", accepted_step_count)
        object.__setattr__(self, "rejected_step_count", rejected_step_count)
        object.__setattr__(self, "acceptance_rate", acceptance_rate)
        object.__setattr__(self, "terminal_step_index", terminal_step_index)
        object.__setattr__(self, "runtime_seconds", runtime_seconds)
        object.__setattr__(self, "chain_core_runtime_seconds", chain_core_runtime_seconds)
        object.__setattr__(
            self,
            "retained_mean_accumulation_runtime_seconds",
            retained_mean_accumulation_runtime_seconds,
        )
        object.__setattr__(
            self,
            "artifact_write_runtime_seconds",
            artifact_write_runtime_seconds,
        )
        object.__setattr__(self, "total_runtime_seconds", total_runtime_seconds)

    def to_payload(self) -> dict[str, object]:
        return {
            "total_steps_attempted": self.total_steps_attempted,
            "accepted_step_count": self.accepted_step_count,
            "rejected_step_count": self.rejected_step_count,
            "acceptance_rate": self.acceptance_rate,
            "burn_in_steps": self.burn_in_steps,
            "thin_every": self.thin_every,
            "retained_count": self.retained_count,
            "terminal_step_index": self.terminal_step_index,
            "runtime_seconds": self.runtime_seconds,
            "chain_core_runtime_seconds": self.chain_core_runtime_seconds,
            "retained_mean_accumulation_runtime_seconds": (
                self.retained_mean_accumulation_runtime_seconds
            ),
            "artifact_write_runtime_seconds": self.artifact_write_runtime_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ReconstructionChainDiagnostics":
        chain = _as_mapping(payload, field_name="chain")
        return cls(
            total_steps_attempted=chain.get("total_steps_attempted"),
            accepted_step_count=chain.get("accepted_step_count"),
            rejected_step_count=chain.get("rejected_step_count"),
            acceptance_rate=chain.get("acceptance_rate"),
            burn_in_steps=chain.get("burn_in_steps"),
            thin_every=chain.get("thin_every"),
            retained_count=chain.get("retained_count"),
            terminal_step_index=chain.get("terminal_step_index"),
            runtime_seconds=chain.get("runtime_seconds"),
            chain_core_runtime_seconds=chain.get("chain_core_runtime_seconds"),
            retained_mean_accumulation_runtime_seconds=chain.get(
                "retained_mean_accumulation_runtime_seconds"
            ),
            artifact_write_runtime_seconds=chain.get("artifact_write_runtime_seconds"),
            total_runtime_seconds=chain.get("total_runtime_seconds"),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionStateSummaryDiagnostics:
    """Cheap summary statistics over the emitted reconstruction arrays."""

    grid_shape: tuple[int, int, int]
    total_voxel_count: int
    terminal_occupancy_sum: int | None = None
    retained_mean_occupancy_sum: float | None = None
    terminal_nonzero_voxel_count: int | None = None
    retained_mean_nonzero_voxel_count: int | None = None

    def __post_init__(self) -> None:
        grid_shape = _as_grid_shape(self.grid_shape, field_name="state_summary.grid_shape")
        total_voxel_count = _as_nonnegative_int(
            self.total_voxel_count,
            field_name="state_summary.total_voxel_count",
        )
        if total_voxel_count != int(np.prod(np.asarray(grid_shape, dtype=np.int64))):
            raise ValueError(
                "state_summary.total_voxel_count must match the product of "
                "state_summary.grid_shape"
            )
        terminal_occupancy_sum = _as_optional_nonnegative_int(
            self.terminal_occupancy_sum,
            field_name="state_summary.terminal_occupancy_sum",
        )
        retained_mean_occupancy_sum = _as_optional_nonnegative_float(
            self.retained_mean_occupancy_sum,
            field_name="state_summary.retained_mean_occupancy_sum",
        )
        terminal_nonzero_voxel_count = _as_optional_nonnegative_int(
            self.terminal_nonzero_voxel_count,
            field_name="state_summary.terminal_nonzero_voxel_count",
        )
        retained_mean_nonzero_voxel_count = _as_optional_nonnegative_int(
            self.retained_mean_nonzero_voxel_count,
            field_name="state_summary.retained_mean_nonzero_voxel_count",
        )
        object.__setattr__(self, "grid_shape", grid_shape)
        object.__setattr__(self, "total_voxel_count", total_voxel_count)
        object.__setattr__(self, "terminal_occupancy_sum", terminal_occupancy_sum)
        object.__setattr__(
            self,
            "retained_mean_occupancy_sum",
            retained_mean_occupancy_sum,
        )
        object.__setattr__(
            self,
            "terminal_nonzero_voxel_count",
            terminal_nonzero_voxel_count,
        )
        object.__setattr__(
            self,
            "retained_mean_nonzero_voxel_count",
            retained_mean_nonzero_voxel_count,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "grid_shape": list(self.grid_shape),
            "total_voxel_count": self.total_voxel_count,
            "terminal_occupancy_sum": self.terminal_occupancy_sum,
            "retained_mean_occupancy_sum": self.retained_mean_occupancy_sum,
            "terminal_nonzero_voxel_count": self.terminal_nonzero_voxel_count,
            "retained_mean_nonzero_voxel_count": self.retained_mean_nonzero_voxel_count,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ReconstructionStateSummaryDiagnostics":
        state_summary = _as_mapping(payload, field_name="state_summary")
        return cls(
            grid_shape=state_summary.get("grid_shape"),
            total_voxel_count=state_summary.get("total_voxel_count"),
            terminal_occupancy_sum=state_summary.get("terminal_occupancy_sum"),
            retained_mean_occupancy_sum=state_summary.get("retained_mean_occupancy_sum"),
            terminal_nonzero_voxel_count=state_summary.get("terminal_nonzero_voxel_count"),
            retained_mean_nonzero_voxel_count=state_summary.get(
                "retained_mean_nonzero_voxel_count"
            ),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionSeedDiagnostics:
    """Seeds used by one reconstruction run."""

    event_index_selector: int | None = None
    proposal_backend: int | None = None
    acceptance_uniform: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_index_selector",
            _as_optional_nonnegative_int(
                self.event_index_selector,
                field_name="seeds.event_index_selector",
            ),
        )
        object.__setattr__(
            self,
            "proposal_backend",
            _as_optional_nonnegative_int(
                self.proposal_backend,
                field_name="seeds.proposal_backend",
            ),
        )
        object.__setattr__(
            self,
            "acceptance_uniform",
            _as_optional_nonnegative_int(
                self.acceptance_uniform,
                field_name="seeds.acceptance_uniform",
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "event_index_selector": self.event_index_selector,
            "proposal_backend": self.proposal_backend,
            "acceptance_uniform": self.acceptance_uniform,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ReconstructionSeedDiagnostics":
        seeds = _as_mapping(payload, field_name="seeds")
        return cls(
            event_index_selector=seeds.get("event_index_selector"),
            proposal_backend=seeds.get("proposal_backend"),
            acceptance_uniform=seeds.get("acceptance_uniform"),
        )


@dataclass(frozen=True, slots=True)
class ReconstructionDiagnostics:
    """Validated reconstruction diagnostics payload."""

    counts: ReconstructionCountsDiagnostics
    preprocessing: ReconstructionPreprocessingDiagnostics
    chain: ReconstructionChainDiagnostics
    state_summary: ReconstructionStateSummaryDiagnostics
    seeds: ReconstructionSeedDiagnostics
    schema_name: str = RECONSTRUCTION_DIAGNOSTICS_SCHEMA_NAME
    schema_version: int = RECONSTRUCTION_DIAGNOSTICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.counts, ReconstructionCountsDiagnostics):
            raise TypeError("counts must be a ReconstructionCountsDiagnostics")
        if not isinstance(self.preprocessing, ReconstructionPreprocessingDiagnostics):
            raise TypeError("preprocessing must be a ReconstructionPreprocessingDiagnostics")
        if not isinstance(self.chain, ReconstructionChainDiagnostics):
            raise TypeError("chain must be a ReconstructionChainDiagnostics")
        if not isinstance(self.state_summary, ReconstructionStateSummaryDiagnostics):
            raise TypeError("state_summary must be a ReconstructionStateSummaryDiagnostics")
        if not isinstance(self.seeds, ReconstructionSeedDiagnostics):
            raise TypeError("seeds must be a ReconstructionSeedDiagnostics")
        object.__setattr__(self, "schema_name", _as_schema_name(self.schema_name))
        object.__setattr__(self, "schema_version", _as_schema_version(self.schema_version))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "counts": self.counts.to_payload(),
            "preprocessing": self.preprocessing.to_payload(),
            "chain": self.chain.to_payload(),
            "state_summary": self.state_summary.to_payload(),
            "seeds": self.seeds.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ReconstructionDiagnostics":
        mapping = _as_mapping(payload, field_name="reconstruction_diagnostics")
        return cls(
            schema_name=mapping.get("schema_name"),
            schema_version=mapping.get("schema_version"),
            counts=ReconstructionCountsDiagnostics.from_payload(mapping.get("counts")),
            preprocessing=ReconstructionPreprocessingDiagnostics.from_payload(
                mapping.get("preprocessing")
            ),
            chain=ReconstructionChainDiagnostics.from_payload(mapping.get("chain")),
            state_summary=ReconstructionStateSummaryDiagnostics.from_payload(
                mapping.get("state_summary")
            ),
            seeds=ReconstructionSeedDiagnostics.from_payload(mapping.get("seeds")),
        )


def write_reconstruction_diagnostics_sidecar(
    artifacts_dir: Path,
    diagnostics: ReconstructionDiagnostics,
) -> ReconstructionDiagnostics:
    """Write the reconstruction diagnostics sidecar next to the arrays."""
    if not isinstance(diagnostics, ReconstructionDiagnostics):
        raise TypeError("diagnostics must be a ReconstructionDiagnostics")
    diagnostics_path = _diagnostics_path(Path(artifacts_dir).resolve())
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(diagnostics.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return diagnostics


def validate_reconstruction_diagnostics_payload(payload: object) -> ReconstructionDiagnostics:
    """Validate one in-memory reconstruction diagnostics payload."""
    return ReconstructionDiagnostics.from_payload(payload)


def load_reconstruction_diagnostics(path: Path) -> ReconstructionDiagnostics:
    """Load and validate one reconstruction diagnostics sidecar."""
    diagnostics_path = _diagnostics_path(Path(path).resolve())
    if not diagnostics_path.is_file():
        raise FileNotFoundError(
            f"Missing reconstruction diagnostics sidecar: {diagnostics_path}"
        )
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    return validate_reconstruction_diagnostics_payload(payload)


__all__ = [
    "RECONSTRUCTION_DIAGNOSTICS_FILENAME",
    "RECONSTRUCTION_DIAGNOSTICS_SCHEMA_NAME",
    "RECONSTRUCTION_DIAGNOSTICS_SCHEMA_VERSION",
    "ReconstructionChainDiagnostics",
    "ReconstructionCountsDiagnostics",
    "ReconstructionDiagnostics",
    "ReconstructionPreprocessingDiagnostics",
    "ReconstructionSeedDiagnostics",
    "ReconstructionStateSummaryDiagnostics",
    "load_reconstruction_diagnostics",
    "validate_reconstruction_diagnostics_payload",
    "write_reconstruction_diagnostics_sidecar",
]
