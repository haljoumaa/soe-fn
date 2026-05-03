"""Focused tests for occupancy semantics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from soe.adapters.voi_voxel import VoxelGrid
from soe.contracts import EventObj, VOIConfig
from soe.soe.occupancy import (
    build_occupancy_counts,
    count_density_view,
    points_to_voxel_indices,
)
from soe.soe.state import SurvivingEventState


def _grid() -> VoxelGrid:
    """Build a simple grid with non-unit voxel volume."""
    return VoxelGrid.from_config(
        VOIConfig(
            bounds=np.asarray(
                [
                    [0.0, 4.0],
                    [-1.0, 1.0],
                    [0.0, 4.0],
                ],
                dtype=float,
            ),
            grid_shape=(4, 2, 2),
        )
    )


def _event() -> EventObj:
    """Build a placeholder canonical event for frozen-state tests."""
    return EventObj(
        apex=np.asarray([0.0, 0.0, 0.0], dtype=float),
        axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
        theta=math.pi / 4.0,
    )


def _state(points: list[tuple[float, float, float]]) -> SurvivingEventState:
    """Build frozen representative-point state for occupancy tests."""
    events = tuple(_event() for _ in points)
    representative_points = tuple(np.asarray(point, dtype=float) for point in points)
    return SurvivingEventState(events=events, representative_points=representative_points)


def test_representative_points_map_deterministically_to_voxel_indices() -> None:
    """Valid representative points should voxelise deterministically through `nu`."""
    grid = _grid()
    state = _state(
        [
            (0.25, -0.75, 0.25),
            (1.0, 0.0, 2.0),
            (3.999999999, 0.999999999, 3.999999999),
        ]
    )

    expected = (
        (0, 0, 0),
        (1, 1, 1),
        (3, 1, 1),
    )

    assert points_to_voxel_indices(state.representative_points, grid) == expected
    assert state.voxel_indices(grid) == expected


def test_histogram_counts_match_explicit_event_to_voxel_assignments() -> None:
    """The occupancy histogram should match the representative-point assignments."""
    grid = _grid()
    state = _state(
        [
            (0.25, -0.75, 0.25),
            (1.10, 0.10, 3.10),
            (1.90, 0.90, 3.99),
            (2.00, -0.20, 0.40),
        ]
    )
    expected_counts = np.zeros(grid.grid_shape, dtype=np.int64)
    expected_counts[0, 0, 0] = 1
    expected_counts[1, 1, 1] = 2
    expected_counts[2, 0, 0] = 1

    helper_counts = build_occupancy_counts(state.voxel_indices(grid), grid)
    state_counts = state.occupancy_counts(grid)

    assert np.issubdtype(state_counts.dtype, np.integer)
    np.testing.assert_array_equal(helper_counts, expected_counts)
    np.testing.assert_array_equal(state_counts, expected_counts)


def test_canonical_image_equals_histogram_exactly() -> None:
    """Occupancy must keep `D(X) = n(X)` exactly."""
    grid = _grid()
    state = _state(
        [
            (0.25, -0.75, 0.25),
            (1.10, 0.10, 3.10),
            (1.90, 0.90, 3.99),
        ]
    )

    np.testing.assert_array_equal(state.canonical_image(grid), state.occupancy_counts(grid))


def test_density_view_is_derived_exactly_from_counts_and_voxel_volume() -> None:
    """Density must remain a derived count view only."""
    grid = _grid()
    state = _state(
        [
            (0.25, -0.75, 0.25),
            (1.10, 0.10, 3.10),
            (1.90, 0.90, 3.99),
            (2.00, -0.20, 0.40),
        ]
    )
    counts = state.occupancy_counts(grid)
    expected_density = counts.astype(float) / grid.voxel_volume

    np.testing.assert_array_equal(count_density_view(counts, grid), expected_density)
    np.testing.assert_array_equal(state.count_density_view(grid), expected_density)


def test_per_state_occupancy_counts_conserve_the_number_of_surviving_events() -> None:
    """The full-grid histogram should conserve the number of representative points."""
    grid = _grid()
    state = _state(
        [
            (0.25, -0.75, 0.25),
            (1.10, 0.10, 3.10),
            (1.90, 0.90, 3.99),
            (2.00, -0.20, 0.40),
        ]
    )
    counts = state.occupancy_counts(grid)

    assert int(counts.sum()) == len(state.events)


def test_non_voxelisable_representative_point_raises_a_hard_error() -> None:
    """A point outside the half-open VOI must not be clamped or repaired."""
    grid = _grid()
    state = _state([(4.0, 0.0, 1.0)])

    with pytest.raises(
        ValueError,
        match="does not voxelise through the frozen VOI world-to-index map",
    ):
        state.voxel_indices(grid)
