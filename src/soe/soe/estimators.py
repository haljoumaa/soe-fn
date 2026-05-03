"""Deterministic retained-state accumulation over canonical occupancy images."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from soe.adapters.voi_voxel import VoxelGrid
from soe.soe.state import SurvivingEventState

GridSignature = tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]


def _as_grid(value: object) -> VoxelGrid:
    """Validate the explicit grid carrier used for retained-state reporting."""
    if isinstance(value, VoxelGrid):
        return value
    raise TypeError("grid must be a VoxelGrid")


def _grid_signature(grid: VoxelGrid) -> GridSignature:
    """Capture the canonical voxel tiling used by the accumulator."""
    return (
        tuple(float(value) for value in grid.x_edges),
        tuple(float(value) for value in grid.y_edges),
        tuple(float(value) for value in grid.z_edges),
    )


def _as_occupancy_counts(value: object, *, grid_shape: tuple[int, int, int]) -> np.ndarray:
    """Validate one canonical full-grid per-state occupancy image without copying."""
    counts = np.asarray(value)
    if counts.shape != grid_shape:
        raise ValueError(f"counts must have shape {grid_shape}, got {counts.shape}")
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("per-state occupancy counts must be integer-valued")
    if np.any(counts < 0):
        raise ValueError("per-state occupancy counts must be nonnegative")
    return counts


@dataclass(slots=True)
class RetainedStateAccumulator:
    """Accumulate already-retained canonical per-state occupancy images.

    This class does not decide which states are retained. It only consumes
    already-retained canonical occupancy images, or representative-point
    states that are deterministically converted to the same images through the
    frozen Phase 3.1 voxelisation path. Direct occupancy ingestion is the
    standard low-memory path; `add_state(...)` remains compatibility-only.
    """

    grid: VoxelGrid
    _grid: VoxelGrid = field(init=False, repr=False)
    _grid_signature: GridSignature = field(init=False, repr=False)
    _sum_counts: np.ndarray = field(init=False, repr=False)
    _last_counts: np.ndarray | None = field(init=False, default=None, repr=False)
    _num_retained: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Normalize the canonical grid and initialize empty accumulation state."""
        grid = _as_grid(self.grid)
        self._grid = grid
        self._grid_signature = _grid_signature(grid)
        self._sum_counts = np.zeros(grid.grid_shape, dtype=np.float64)

    @property
    def num_retained(self) -> int:
        """Return the number of retained per-state images added so far."""
        return self._num_retained

    def _ensure_grid_matches(self, grid: VoxelGrid) -> VoxelGrid:
        """Reject inconsistent voxel tilings even when shapes coincide."""
        grid = _as_grid(grid)
        if _grid_signature(grid) != self._grid_signature:
            raise ValueError("grid metadata mismatch for retained-state accumulation")
        return grid

    def _require_samples(self, *, query_name: str) -> None:
        """Reject aggregate queries before any retained state has been added."""
        if self._num_retained == 0:
            raise ValueError(f"{query_name} requires at least one retained state")

    def add_occupancy_counts(self, counts: object) -> None:
        """Add one canonical full-grid per-state occupancy image `D(X)`."""
        per_state_counts = _as_occupancy_counts(counts, grid_shape=self._grid.grid_shape)
        self._sum_counts += per_state_counts
        self._last_counts = np.array(per_state_counts, dtype=np.int64, copy=True)
        self._num_retained += 1

    def add_state(
        self,
        state: SurvivingEventState,
        *,
        grid: VoxelGrid | None = None,
    ) -> None:
        """Add one already-retained representative-point state via canonical voxelisation."""
        if not isinstance(state, SurvivingEventState):
            raise TypeError("state must be a SurvivingEventState")
        if grid is None:
            active_grid = self._grid
        else:
            active_grid = self._ensure_grid_matches(grid)
        self.add_occupancy_counts(state.canonical_image(active_grid))

    def mean_occupancy(self) -> np.ndarray:
        """Return the ensemble-average occupancy image over retained states."""
        self._require_samples(query_name="mean occupancy")
        return self._sum_counts / float(self._num_retained)

    def mean_density(self) -> np.ndarray:
        """Return the derived average density view from mean occupancy only."""
        return self.mean_occupancy() / self._grid.voxel_volume

    def final_state_image(self) -> np.ndarray:
        """Return the canonical occupancy image of the last retained state only."""
        self._require_samples(query_name="final state image")
        assert self._last_counts is not None
        return self._last_counts.copy()


__all__ = ["RetainedStateAccumulator"]
