"""Focused tests for local occupancy deltas."""

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
from soe.soe.occupancy import apply_single_event_delta, updated_voxel_index_for_event
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
    """Build frozen representative-point state for delta tests."""
    events = tuple(_event() for _ in points)
    representative_points = tuple(np.asarray(point, dtype=float) for point in points)
    return SurvivingEventState(events=events, representative_points=representative_points)


def _updated_state(
    state: SurvivingEventState,
    *,
    event_index: int,
    candidate_point: tuple[float, float, float],
) -> SurvivingEventState:
    """Return a new state with one representative point replaced."""
    updated_points = list(state.representative_points)
    updated_points[event_index] = np.asarray(candidate_point, dtype=float)
    return SurvivingEventState(events=state.events, representative_points=tuple(updated_points))


def test_single_event_move_changes_at_most_two_voxel_counts() -> None:
    """A local replacement should only touch the old and new voxel bins."""
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
    candidate_point = (3.25, -0.75, 0.25)

    old_idx, new_idx = state.updated_voxel_index_for_event(1, candidate_point, grid)
    updated_counts = state.apply_single_event_delta(
        counts,
        event_index=1,
        candidate_point=candidate_point,
        grid=grid,
    )
    delta = updated_counts - counts

    assert old_idx == (1, 1, 1)
    assert new_idx == (3, 0, 0)
    assert np.count_nonzero(delta) == 2
    assert delta[old_idx] == -1
    assert delta[new_idx] == 1


def test_same_voxel_replacement_leaves_histogram_unchanged_exactly() -> None:
    """If `v_- == v_+`, the occupancy histogram must stay unchanged."""
    grid = _grid()
    state = _state(
        [
            (0.25, -0.75, 0.25),
            (1.10, 0.10, 3.10),
            (1.90, 0.90, 3.99),
        ]
    )
    counts = state.occupancy_counts(grid)
    candidate_point = (1.75, 0.25, 2.50)

    old_idx, new_idx = updated_voxel_index_for_event(
        state.representative_points,
        1,
        candidate_point,
        grid,
    )
    updated_counts = apply_single_event_delta(
        counts,
        state.representative_points,
        1,
        candidate_point,
        grid,
    )

    assert old_idx == new_idx == (1, 1, 1)
    np.testing.assert_array_equal(updated_counts, counts)


def test_local_delta_matches_full_rebuild_from_updated_state() -> None:
    """The exact local delta must match a rebuild from the updated point field."""
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
    candidate_point = (3.25, -0.75, 0.25)

    delta_counts = state.apply_single_event_delta(
        counts,
        event_index=1,
        candidate_point=candidate_point,
        grid=grid,
    )
    rebuilt_counts = _updated_state(
        state,
        event_index=1,
        candidate_point=candidate_point,
    ).occupancy_counts(grid)

    np.testing.assert_array_equal(delta_counts, rebuilt_counts)


def test_per_state_conservation_holds_exactly_after_local_delta() -> None:
    """A single-event replacement must preserve total occupancy mass."""
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
    updated_counts = state.apply_single_event_delta(
        counts,
        event_index=1,
        candidate_point=(3.25, -0.75, 0.25),
        grid=grid,
    )

    assert int(updated_counts.sum()) == len(state.events)


def test_invalid_old_or_new_point_voxelisation_raises_a_hard_error() -> None:
    """Invalid current or candidate points must fail instead of being repaired."""
    grid = _grid()
    counts = np.zeros(grid.grid_shape, dtype=np.int64)

    with pytest.raises(
        ValueError,
        match="current representative point does not voxelise through the frozen VOI world-to-index map",
    ):
        _state([(4.0, 0.0, 1.0)]).apply_single_event_delta(
            counts,
            event_index=0,
            candidate_point=(3.25, -0.75, 0.25),
            grid=grid,
        )
    state = _state([(0.25, -0.75, 0.25)])
    valid_counts = state.occupancy_counts(grid)
    with pytest.raises(
        ValueError,
        match="candidate representative point does not voxelise through the frozen VOI world-to-index map",
    ):
        state.apply_single_event_delta(
            valid_counts,
            event_index=0,
            candidate_point=(4.0, 0.0, 1.0),
            grid=grid,
        )


def test_impossible_negative_count_transition_raises_a_hard_error() -> None:
    """Local updates must not silently permit decrements below zero."""
    grid = _grid()
    state = _state([(0.25, -0.75, 0.25)])
    empty_counts = np.zeros(grid.grid_shape, dtype=np.int64)

    with pytest.raises(ValueError, match="cannot decrement occupancy count below zero"):
        state.apply_single_event_delta(
            empty_counts,
            event_index=0,
            candidate_point=(3.25, -0.75, 0.25),
            grid=grid,
        )
