"""Core VOI and event-geometry data contracts.

These types fix the ingestion and reconstruction boundary data.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

_VEC3_SHAPE = (3,)
_BOUNDS_SHAPE = (3, 2)
_RIGID_TOL = 1e-10


def _as_float_array(value: object, *, shape: tuple[int, ...], field_name: str) -> np.ndarray:
    """Validate and coerce a finite float array of the expected shape."""
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must be finite")
    return array


def _as_vec3(value: object, *, field_name: str) -> np.ndarray:
    """Validate a finite `(x, y, z)` vector."""
    return _as_float_array(value, shape=_VEC3_SHAPE, field_name=field_name)


def _as_bounds(value: object) -> np.ndarray:
    """Validate `(x, y, z)` VOI bounds as [[min, max], ...]."""
    bounds = _as_float_array(value, shape=_BOUNDS_SHAPE, field_name="bounds")
    if not np.all(bounds[:, 0] < bounds[:, 1]):
        raise ValueError("bounds must satisfy min < max for x, y, z")
    return bounds


def _as_grid_shape(value: Iterable[int]) -> tuple[int, int, int]:
    """Validate positive grid counts in `(x, y, z)` order."""
    grid_shape = tuple(value)
    if len(grid_shape) != 3:
        raise ValueError(f"grid_shape must contain 3 entries, got {len(grid_shape)}")
    if any(
        isinstance(entry, bool) or not isinstance(entry, (int, np.integer))
        for entry in grid_shape
    ):
        raise ValueError("grid_shape entries must be integers")
    if any(entry <= 0 for entry in grid_shape):
        raise ValueError("grid_shape entries must be positive")
    return grid_shape


def _as_unit_axis(value: object) -> np.ndarray:
    """Validate a finite unit axis without performing canonicalization."""
    axis = _as_vec3(value, field_name="axis")
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError("axis must have nonzero norm")
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("axis must be unit-length")
    return axis


def _as_theta(value: float) -> float:
    """Validate a canonical opening angle in `(0, pi/2]`."""
    theta = float(value)
    if not math.isfinite(theta) or not (0.0 < theta <= (math.pi / 2.0)):
        raise ValueError("theta must be finite and in (0, pi/2]")
    return theta


def _as_rotation_matrix(value: object) -> np.ndarray:
    """Validate a proper rigid rotation matrix `Q` in `SO(3)`."""
    q = _as_float_array(value, shape=(3, 3), field_name="Q")
    if not np.allclose(q.T @ q, np.eye(3), atol=_RIGID_TOL, rtol=0.0):
        raise ValueError("Q must be orthonormal")
    det = float(np.linalg.det(q))
    if not math.isclose(det, 1.0, rel_tol=0.0, abs_tol=_RIGID_TOL):
        raise ValueError("Q must have determinant +1 (same-handed SO(3) only)")
    return q


def _as_species_filter(value: str | Iterable[str] | None) -> tuple[str, ...] | None:
    """Validate an optional species allow-list."""
    if value is None:
        return None
    if isinstance(value, str):
        species = (value,)
    else:
        species = tuple(value)
    if any(not isinstance(entry, str) or not entry for entry in species):
        raise ValueError("species entries must be non-empty strings")
    return species


@dataclass(frozen=True, slots=True)
class VOIConfig:
    

    bounds: np.ndarray
    grid_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        bounds = _as_bounds(self.bounds)
        grid_shape = _as_grid_shape(self.grid_shape)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "grid_shape", grid_shape)


@dataclass(frozen=True, slots=True)
class EventObj:
    """Canonical event data only.

    Carries only `apex`, `axis`, `theta`, and optional `species` in `(x, y, z)`
    order. This excludes event id, event index, provenance, and run-level
    filtering configuration such as `lambda_min`. 0 < lambda_e < 1.
    """

    apex: np.ndarray
    axis: np.ndarray
    theta: float
    species: str | None = None

    def __post_init__(self) -> None:
        apex = _as_vec3(self.apex, field_name="apex")
        axis = _as_unit_axis(self.axis)
        theta = _as_theta(self.theta)
        if self.species is not None and (
            not isinstance(self.species, str) or not self.species
        ):
            raise ValueError("species must be a non-empty string when provided")
        object.__setattr__(self, "apex", apex)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "theta", theta)


@dataclass(frozen=True, slots=True)
class ReconstructionInput:
    """Minimal reconstruction input: canonical events plus a VOI."""

    events: tuple[EventObj, ...]
    voi: VOIConfig

    def __post_init__(self) -> None:
        events = tuple(self.events)
        if any(not isinstance(event, EventObj) for event in events):
            raise TypeError("events must contain only EventObj instances")
        if not isinstance(self.voi, VOIConfig):
            raise TypeError("voi must be a VOIConfig")
        object.__setattr__(self, "events", events)


@dataclass(frozen=True, slots=True)
class RegistrationTransform:
    """Ingestion-time rigid transform `(Q, t)`.
    """

    Q: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        q = _as_rotation_matrix(self.Q)
        t = _as_vec3(self.t, field_name="t")
        object.__setattr__(self, "Q", q)
        object.__setattr__(self, "t", t)


@dataclass(frozen=True, slots=True)
class EventFilterConfig:
    """Auxiliary event filtering inputs.
    """

    lambda_min: float = 0.001
    species: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        lambda_min = float(self.lambda_min)
        if not math.isfinite(lambda_min) or not (0.0 < lambda_min < 1.0):
            raise ValueError("lambda_min must be finite and in (0, 1)")
        species = _as_species_filter(self.species)
        object.__setattr__(self, "lambda_min", lambda_min)
        object.__setattr__(self, "species", species)


__all__ = [
    "EventFilterConfig",
    "EventObj",
    "ReconstructionInput",
    "RegistrationTransform",
    "VOIConfig",
]
