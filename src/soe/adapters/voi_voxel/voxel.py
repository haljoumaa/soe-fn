"""Voxel grid bookkeeping and deterministic world/index mappings."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np

from soe.contracts import VOIConfig

from .voi import VoiBounds

StrictOpenBox = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def _as_count(value: object, *, name: str) -> int:
    """Validate a positive voxel count."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    count = int(value)
    if count < 1:
        raise ValueError(f"{name} must be >= 1")
    return count


def _as_index_xyz(value: object) -> tuple[int, int, int]:
    """Validate a voxel index tuple in `(i, j, k)` == `(x, y, z)` order."""
    idx = tuple(value)
    if len(idx) != 3:
        raise ValueError(f"voxel index must have 3 entries, got {len(idx)}")
    if any(isinstance(entry, bool) or not isinstance(entry, (int, np.integer)) for entry in idx):
        raise ValueError("voxel index entries must be integers")
    return (int(idx[0]), int(idx[1]), int(idx[2]))


def _bounds_to_strict_open_box(bounds: object) -> StrictOpenBox:
    """Convert `[[xmin, xmax], ...]` bounds into a strict-open box tuple."""
    arr = np.asarray(bounds, dtype=float)
    if arr.shape != (3, 2):
        raise ValueError(f"bounds must have shape (3, 2), got {arr.shape}")
    return (
        (float(arr[0, 0]), float(arr[0, 1])),
        (float(arr[1, 0]), float(arr[1, 1])),
        (float(arr[2, 0]), float(arr[2, 1])),
    )


def _materialize_grid_boxes(
    grid: "VoxelGrid",
    indices: object,
) -> tuple[StrictOpenBox, ...]:
    """Return strict-open voxel boxes for the supplied in-range grid indices."""
    return tuple(
        _bounds_to_strict_open_box(grid.index_to_bounds((int(i), int(j), int(k))))
        for i, j, k in indices
    )


def _grid_shape_for_sampled_survival(voi: VoiBounds) -> tuple[int, int, int]:
    """Require config-derived grid metadata before voxel boxes are materialized."""
    if voi._grid_shape is None:
        raise ValueError(
            "sampled survival requires grid-aware voxel-cell semantics; "
            "bare VoiBounds without grid metadata are not allowed"
        )
    return voi._grid_shape


def materialize_allowed_boxes(allowed_region: object) -> tuple[StrictOpenBox, ...]:
    """Materialize sampled-surrogate boxes from the current VOI/voxel boundary."""
    if isinstance(allowed_region, VoiBounds):
        nx, ny, nz = _grid_shape_for_sampled_survival(allowed_region)
        return _materialize_grid_boxes(
            VoxelGrid(nx=nx, ny=ny, nz=nz, voi=allowed_region),
            (
                (i, j, k)
                for i in range(nx)
                for j in range(ny)
                for k in range(nz)
            ),
        )

    raise TypeError("allowed_region must be a VoiBounds")


@dataclass(frozen=True, slots=True)
class VoxelGrid:
    """Regular VOI voxel grid in fixed `(x, y, z)` order.

    `grid_shape = (nx, ny, nz)` tiles the VOI exactly. Spacings and all voxel
    edges/centres are derived from the VOI bounds rather than accepted as
    independent user inputs.
    """

    nx: int
    ny: int
    nz: int
    voi: VoiBounds

    def __post_init__(self) -> None:
        """Validate voxel counts and VOI extents."""
        nx = _as_count(self.nx, name="nx")
        ny = _as_count(self.ny, name="ny")
        nz = _as_count(self.nz, name="nz")
        if not isinstance(self.voi, VoiBounds):
            raise TypeError("voi must be a VoiBounds")
        object.__setattr__(self, "nx", nx)
        object.__setattr__(self, "ny", ny)
        object.__setattr__(self, "nz", nz)

    @classmethod
    def from_config(cls, config: VOIConfig) -> "VoxelGrid":
        """Bridge `VOIConfig(bounds, grid_shape)` to the internal grid."""
        if not isinstance(config, VOIConfig):
            raise TypeError("config must be a VOIConfig")
        voi = VoiBounds.from_config(config)
        nx, ny, nz = config.grid_shape
        return cls(nx=nx, ny=ny, nz=nz, voi=voi)

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Return the grid shape as `(nx, ny, nz)` in `(x, y, z)` order."""
        return (self.nx, self.ny, self.nz)

    @property
    def dx(self) -> float:
        """Voxel spacing along x in centimeters."""
        return (self.voi.xmax - self.voi.xmin) / self.nx

    @property
    def dy(self) -> float:
        """Voxel spacing along y in centimeters."""
        return (self.voi.ymax - self.voi.ymin) / self.ny

    @property
    def dz(self) -> float:
        """Voxel spacing along z in centimeters."""
        return (self.voi.zmax - self.voi.zmin) / self.nz

    @property
    def voxel_volume(self) -> float:
        """Return the exact voxel volume implied by VOI extents and grid shape."""
        return self.dx * self.dy * self.dz

    @property
    def x_edges(self) -> np.ndarray:
        """Return x-axis voxel edges including both VOI endpoints."""
        return np.linspace(self.voi.xmin, self.voi.xmax, num=self.nx + 1, dtype=float)

    @property
    def y_edges(self) -> np.ndarray:
        """Return y-axis voxel edges including both VOI endpoints."""
        return np.linspace(self.voi.ymin, self.voi.ymax, num=self.ny + 1, dtype=float)

    @property
    def z_edges(self) -> np.ndarray:
        """Return z-axis voxel edges including both VOI endpoints."""
        return np.linspace(self.voi.zmin, self.voi.zmax, num=self.nz + 1, dtype=float)

    def in_voi(self, r: np.ndarray) -> bool:
        """Return half-open VOI containment for a world-space point."""
        return self.voi.contains(r)

    def world_to_index(self, r: np.ndarray) -> tuple[int, int, int] | None:
        """Map `(x, y, z)` to `(i, j, k)` with no upper-bound clamping."""
        if not self.in_voi(r):
            return None

        point = np.asarray(r, dtype=float)
        x, y, z = float(point[0]), float(point[1]), float(point[2])

        i = floor((x - self.voi.xmin) / self.dx)
        j = floor((y - self.voi.ymin) / self.dy)
        k = floor((z - self.voi.zmin) / self.dz)
        idx = (i, j, k)
        if not self.index_in_range(idx):
            return None
        return idx

    def index_in_range(self, idx: tuple[int, int, int]) -> bool:
        """Return whether `(i, j, k)` is inside the grid bounds."""
        i, j, k = _as_index_xyz(idx)
        return 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz

    def index_to_bounds(self, idx: tuple[int, int, int]) -> np.ndarray:
        """Return half-open voxel bounds in `[[xmin, xmax], ...]` order."""
        i, j, k = _as_index_xyz(idx)
        if not self.index_in_range((i, j, k)):
            raise IndexError(f"voxel index out of range: {(i, j, k)}")

        return np.asarray(
            [
                [self.x_edges[i], self.x_edges[i + 1]],
                [self.y_edges[j], self.y_edges[j + 1]],
                [self.z_edges[k], self.z_edges[k + 1]],
            ],
            dtype=float,
        )

    def index_to_center(self, idx: tuple[int, int, int]) -> np.ndarray:
        """Return the world-space center of `(i, j, k)`."""
        bounds = self.index_to_bounds(idx)
        return bounds.mean(axis=1)


def world_to_index(grid: VoxelGrid, r: np.ndarray) -> tuple[int, int, int] | None:
    """Compatibility wrapper around the authoritative `VoxelGrid.world_to_index`."""
    return grid.world_to_index(r)


def index_to_center(grid: VoxelGrid, idx: tuple[int, int, int]) -> np.ndarray:
    """Compatibility wrapper around the authoritative `VoxelGrid.index_to_center`."""
    return grid.index_to_center(idx)
