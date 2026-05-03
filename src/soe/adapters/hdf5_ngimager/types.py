"""Typed raw adapter-side objects for the NGImager HDF5 reader."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class MetaGeometry:
    """Geometry metadata extracted from /meta."""

    attrs: dict[str, object]
    plane: dict[str, list[float]] = field(default_factory=dict)
    grid: dict[str, float | int] = field(default_factory=dict)


@dataclass(slots=True)
class Cones:
    """Cone-event arrays extracted from /cones."""

    cone_id: np.ndarray
    apex_xyz: np.ndarray
    axis_xyz: np.ndarray
    theta: np.ndarray


@dataclass(frozen=True, slots=True)
class RawEventRecord:
    """Single raw adapter-side event row from the HDF5 boundary.

    This record keeps adapter-side provenance and bookkeeping separate from the
    canonical core `EventObj`. Raw axis values are stored exactly as read from
    `/cones/axis_xyz` before registration and one-nappe canonicalization.
    """

    apex_raw: np.ndarray
    axis_raw: np.ndarray
    theta_raw: float
    species: str | None = None
    cone_row_index: int | None = None
    event_index: int | None = None
    cone_id: int | None = None


__all__ = ["Cones", "MetaGeometry", "RawEventRecord"]
