"""Focused tests for retained-state accumulation."""

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
from soe.soe.estimators import RetainedStateAccumulator
from soe.soe.state import SurvivingEventState


def _grid() -> VoxelGrid:
    """Build a common grid for retained-state tests."""
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


def _other_grid_same_shape() -> VoxelGrid:
    """Build a different grid signature with the same array shape."""
    return VoxelGrid.from_config(
        VOIConfig(
            bounds=np.asarray(
                [
                    [0.0, 8.0],
                    [-1.0, 1.0],
                    [0.0, 2.0],
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
    """Build frozen representative-point state for estimator tests."""
    events = tuple(_event() for _ in points)
    representative_points = tuple(np.asarray(point, dtype=float) for point in points)
    return SurvivingEventState(events=events, representative_points=representative_points)


def _retained_states() -> tuple[SurvivingEventState, SurvivingEventState, SurvivingEventState]:
    """Build a small retained-state ensemble with fixed event count."""
    return (
        _state(
            [
                (0.25, -0.75, 0.25),
                (1.10, 0.10, 3.10),
                (1.90, 0.90, 3.99),
                (2.00, -0.20, 0.40),
            ]
        ),
        _state(
            [
                (0.25, -0.75, 0.25),
                (3.25, -0.75, 0.25),
                (3.75, 0.75, 3.75),
                (2.00, -0.20, 0.40),
            ]
        ),
        _state(
            [
                (0.25, -0.75, 0.25),
                (1.50, 0.10, 3.10),
                (2.50, 0.90, 3.99),
                (3.00, -0.20, 0.40),
            ]
        ),
    )


def test_retained_state_average_matches_manual_average_of_canonical_images() -> None:
    """The primary reported image must be the mean of canonical per-state occupancies."""
    grid = _grid()
    states = _retained_states()
    accumulator = RetainedStateAccumulator(grid)
    canonical_images = [state.canonical_image(grid) for state in states]

    for counts in canonical_images:
        accumulator.add_occupancy_counts(counts)

    expected = sum(counts.astype(np.float64) for counts in canonical_images) / float(len(states))

    np.testing.assert_allclose(accumulator.mean_occupancy(), expected, rtol=0.0, atol=0.0)


def test_mean_density_is_derived_from_mean_occupancy_and_voxel_volume_only() -> None:
    """Average density must be computed from mean occupancy, not stored separately."""
    grid = _grid()
    states = _retained_states()
    accumulator = RetainedStateAccumulator(grid)

    for state in states:
        accumulator.add_state(state)

    expected_density = accumulator.mean_occupancy() / grid.voxel_volume

    np.testing.assert_allclose(accumulator.mean_density(), expected_density, rtol=0.0, atol=0.0)


def test_terminal_state_image_matches_last_retained_canonical_image() -> None:
    """The terminal-state helper should stay secondary and return the last canonical image."""
    grid = _grid()
    state_a, state_b, _ = _retained_states()
    accumulator = RetainedStateAccumulator(grid)

    accumulator.add_state(state_a)
    accumulator.add_state(state_b)

    np.testing.assert_array_equal(accumulator.final_state_image(), state_b.canonical_image(grid))


def test_mean_output_is_floating_point_while_per_state_occupancy_stays_integer() -> None:
    """Averaging should produce real-valued output without changing per-state image semantics."""
    grid = _grid()
    state_a, state_b, _ = _retained_states()
    accumulator = RetainedStateAccumulator(grid)
    per_state_counts = state_a.canonical_image(grid)

    accumulator.add_state(state_a)
    accumulator.add_state(state_b)
    mean_counts = accumulator.mean_occupancy()

    assert np.issubdtype(per_state_counts.dtype, np.integer)
    assert np.issubdtype(mean_counts.dtype, np.floating)


def test_average_conservation_holds_with_tight_tolerance() -> None:
    """The mean occupancy image should preserve the per-state total event count."""
    grid = _grid()
    states = _retained_states()
    accumulator = RetainedStateAccumulator(grid)

    for state in states:
        accumulator.add_state(state)

    assert np.isclose(
        accumulator.mean_occupancy().sum(),
        len(states[0].events),
        rtol=0.0,
        atol=1e-12,
    )


def test_zero_sample_mean_queries_raise_hard_errors() -> None:
    """Aggregate queries must not return junk before any retained state is added."""
    accumulator = RetainedStateAccumulator(_grid())

    with pytest.raises(ValueError, match="mean occupancy requires at least one retained state"):
        accumulator.mean_occupancy()
    with pytest.raises(ValueError, match="mean occupancy requires at least one retained state"):
        accumulator.mean_density()


def test_shape_mismatch_across_added_occupancy_arrays_raises_a_hard_error() -> None:
    """Direct occupancy accumulation should reject incompatible full-grid shapes."""
    accumulator = RetainedStateAccumulator(_grid())

    with pytest.raises(ValueError, match="counts must have shape"):
        accumulator.add_occupancy_counts(np.zeros((2, 2, 2), dtype=np.int64))


def test_final_state_image_isolated_from_later_input_array_mutation() -> None:
    """Compatibility last-state reporting should not alias caller-owned count arrays."""
    grid = _grid()
    accumulator = RetainedStateAccumulator(grid)
    counts = _retained_states()[0].canonical_image(grid).astype(np.int32, copy=True)
    expected = counts.astype(np.int64, copy=True)

    accumulator.add_occupancy_counts(counts)
    counts.fill(0)

    np.testing.assert_array_equal(accumulator.final_state_image(), expected)


def test_state_and_direct_occupancy_accumulation_paths_agree_exactly() -> None:
    """Adding retained states or their canonical images should produce the same average."""
    grid = _grid()
    states = _retained_states()
    state_accumulator = RetainedStateAccumulator(grid)
    counts_accumulator = RetainedStateAccumulator(grid)

    for state in states:
        state_accumulator.add_state(state)
        counts_accumulator.add_occupancy_counts(state.canonical_image(grid))

    np.testing.assert_allclose(
        state_accumulator.mean_occupancy(),
        counts_accumulator.mean_occupancy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        state_accumulator.final_state_image(),
        counts_accumulator.final_state_image(),
    )


def test_inconsistent_grid_metadata_raises_a_hard_error() -> None:
    """State accumulation must not silently voxelise against a different grid."""
    grid = _grid()
    accumulator = RetainedStateAccumulator(grid)
    state, _, _ = _retained_states()

    with pytest.raises(ValueError, match="grid metadata mismatch"):
        accumulator.add_state(state, grid=_other_grid_same_shape())
