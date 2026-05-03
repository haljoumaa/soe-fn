"""VOI bounds bookkeeping with half-open containment semantics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from soe.contracts import VOIConfig


def _as_point_xyz(value: object) -> np.ndarray:
    """Validate a world-space point in fixed `(x, y, z)` order."""
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"point must have shape (3,), got {point.shape}")
    if not np.all(np.isfinite(point)):
        raise ValueError("point must be finite")
    return point


def _as_private_grid_shape(value: object) -> tuple[int, int, int] | None:
    """Validate optional config-derived grid metadata without widening the contract."""
    if value is None:
        return None
    grid_shape = tuple(value)
    if len(grid_shape) != 3:
        raise ValueError(f"_grid_shape must contain 3 entries, got {len(grid_shape)}")
    if any(
        isinstance(entry, bool) or not isinstance(entry, (int, np.integer))
        for entry in grid_shape
    ):
        raise ValueError("_grid_shape entries must be integers")
    if any(int(entry) <= 0 for entry in grid_shape):
        raise ValueError("_grid_shape entries must be positive")
    return (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))


@dataclass(frozen=True, slots=True)
class VoiBounds:
    """Axis-aligned reconstruction VOI bounds in centimeters.

    The authoritative coordinate order is `(x, y, z)`. The VOI is the Cartesian
    product of the three half-open coordinate intervals.
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    _grid_shape: tuple[int, int, int] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate finite strictly ordered VOI bounds."""
        values = np.asarray(
            [self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("VOI bounds must be finite")
        if not (self.xmin < self.xmax):
            raise ValueError("VOI x interval must satisfy xmin < xmax")
        if not (self.ymin < self.ymax):
            raise ValueError("VOI y interval must satisfy ymin < ymax")
        if not (self.zmin < self.zmax):
            raise ValueError("VOI z interval must satisfy zmin < zmax")
        object.__setattr__(self, "_grid_shape", _as_private_grid_shape(self._grid_shape))

    @classmethod
    def from_config(cls, config: VOIConfig) -> "VoiBounds":
        """Bridge the frozen `VOIConfig(bounds, grid_shape)` contract."""
        if not isinstance(config, VOIConfig):
            raise TypeError("config must be a VOIConfig")
        bounds = np.asarray(config.bounds, dtype=float)
        return cls(
            xmin=float(bounds[0, 0]),
            xmax=float(bounds[0, 1]),
            ymin=float(bounds[1, 0]),
            ymax=float(bounds[1, 1]),
            zmin=float(bounds[2, 0]),
            zmax=float(bounds[2, 1]),
            _grid_shape=config.grid_shape,
        )

    @property
    def bounds(self) -> np.ndarray:
        """Return VOI bounds as `[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`."""
        return np.asarray(
            [
                [self.xmin, self.xmax],
                [self.ymin, self.ymax],
                [self.zmin, self.zmax],
            ],
            dtype=float,
        )

    def contains(self, r: np.ndarray) -> bool:
        """Return True if point r=(x,y,z) is inside the half-open VOI."""
        point = _as_point_xyz(r)
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        return (
            self.xmin <= x < self.xmax
            and self.ymin <= y < self.ymax
            and self.zmin <= z < self.zmax
        )

    def in_voi(self, r: np.ndarray) -> bool:
        """Compatibility alias for half-open VOI containment."""
        return self.contains(r)
