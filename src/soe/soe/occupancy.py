"""Deterministic representative-point voxelisation and occupancy views."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from soe.adapters.voi_voxel import VoxelGrid

VoxelIndex = tuple[int, int, int]


def _as_grid(value: object) -> VoxelGrid:
    """Validate the explicit grid carrier used for occupancy-count views."""
    if isinstance(value, VoxelGrid):
        return value
    raise TypeError("grid must be a VoxelGrid")


def _as_point(value: object, *, field_name: str) -> np.ndarray:
    """Validate a finite world-space point in fixed `(x, y, z)` order."""
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{field_name} must have shape (3,), got {point.shape}")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{field_name} must be finite")
    return point


def _as_index(value: object) -> VoxelIndex:
    """Validate an integer voxel index tuple in fixed `(x, y, z)` order."""
    idx = tuple(value)
    if len(idx) != 3:
        raise ValueError(f"voxel index must have 3 entries, got {len(idx)}")
    if any(isinstance(entry, bool) or not isinstance(entry, (int, np.integer)) for entry in idx):
        raise ValueError("voxel index entries must be integers")
    return (int(idx[0]), int(idx[1]), int(idx[2]))


def _as_event_index(value: object, *, n_events: int) -> int:
    """Validate an event index against the current representative-point field."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("event_index must be an integer")
    event_index = int(value)
    if not 0 <= event_index < n_events:
        raise IndexError(f"event_index out of range: {event_index}")
    return event_index


def _as_counts(value: object, *, grid: VoxelGrid) -> np.ndarray:
    """Validate a full-grid integer occupancy array."""
    counts = np.asarray(value)
    if counts.shape != grid.grid_shape:
        raise ValueError(f"counts must have shape {grid.grid_shape}, got {counts.shape}")
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("counts must be integer-valued")
    counts_int64 = counts.astype(np.int64, copy=True)
    if np.any(counts_int64 < 0):
        raise ValueError("counts must be nonnegative")
    return counts_int64


def points_to_voxel_indices(
    representative_points: Iterable[object],
    grid: VoxelGrid,
) -> tuple[VoxelIndex, ...]:
    """Map representative points through the frozen VOI world/index path."""
    grid = _as_grid(grid)
    indices: list[VoxelIndex] = []

    for point_idx, point_value in enumerate(representative_points):
        point = _as_point(point_value, field_name="representative_point")
        voxel_idx = grid.world_to_index(point)
        if voxel_idx is None:
            raise ValueError(
                "representative point does not voxelise through the frozen VOI "
                f"world-to-index map at position {point_idx}: {point.tolist()}"
            )
        indices.append(voxel_idx)

    return tuple(indices)


def build_occupancy_counts(
    voxel_indices: Iterable[object],
    grid: VoxelGrid,
) -> np.ndarray:
    """Build the full-grid integer occupancy histogram from voxel assignments."""
    grid = _as_grid(grid)
    counts = np.zeros(grid.grid_shape, dtype=np.int64)

    for index_position, raw_idx in enumerate(voxel_indices):
        idx = _as_index(raw_idx)
        if not grid.index_in_range(idx):
            raise ValueError(
                f"voxel index out of range at position {index_position}: {idx}"
            )
        counts[idx] += 1

    return counts


def _apply_local_delta_to_owned_counts(
    counts: np.ndarray,
    *,
    old_voxel_index: VoxelIndex,
    new_voxel_index: VoxelIndex,
) -> np.ndarray:
    """Apply one exact local occupancy delta to an owned mutable counts buffer."""
    if old_voxel_index == new_voxel_index:
        return counts

    if counts[old_voxel_index] <= 0:
        raise ValueError(
            "cannot decrement occupancy count below zero at "
            f"{old_voxel_index}"
        )

    counts[old_voxel_index] -= 1
    counts[new_voxel_index] += 1
    return counts


def updated_voxel_index_for_event(
    representative_points: Iterable[object],
    event_index: object,
    candidate_point: object,
    grid: VoxelGrid,
) -> tuple[VoxelIndex, VoxelIndex]:
    """Return the exact old/new voxel indices for one event replacement."""
    grid = _as_grid(grid)
    points = tuple(representative_points)
    point_index = _as_event_index(event_index, n_events=len(points))

    old_point = _as_point(points[point_index], field_name="representative_point")
    new_point = _as_point(candidate_point, field_name="candidate_point")

    old_voxel_index = grid.world_to_index(old_point)
    if old_voxel_index is None:
        raise ValueError(
            "current representative point does not voxelise through the frozen "
            f"VOI world-to-index map for event {point_index}: {old_point.tolist()}"
        )

    new_voxel_index = grid.world_to_index(new_point)
    if new_voxel_index is None:
        raise ValueError(
            "candidate representative point does not voxelise through the frozen "
            f"VOI world-to-index map for event {point_index}: {new_point.tolist()}"
        )

    return old_voxel_index, new_voxel_index


def apply_single_event_delta(
    counts: object,
    representative_points: Iterable[object],
    event_index: object,
    candidate_point: object,
    grid: VoxelGrid,
) -> np.ndarray:
    """Apply the exact local occupancy delta for one representative-point replacement."""
    grid = _as_grid(grid)
    updated_counts = _as_counts(counts, grid=grid)
    old_voxel_index, new_voxel_index = updated_voxel_index_for_event(
        representative_points,
        event_index,
        candidate_point,
        grid,
    )
    return _apply_local_delta_to_owned_counts(
        updated_counts,
        old_voxel_index=old_voxel_index,
        new_voxel_index=new_voxel_index,
    )


def count_density_view(counts: object, grid: VoxelGrid) -> np.ndarray:
    """Return the derived count-density view `rho_v = n_v / |B_v|`."""
    grid = _as_grid(grid)
    counts_array = _as_counts(counts, grid=grid)
    return counts_array.astype(float, copy=False) / grid.voxel_volume


__all__ = [
    "apply_single_event_delta",
    "build_occupancy_counts",
    "count_density_view",
    "points_to_voxel_indices",
    "updated_voxel_index_for_event",
]
