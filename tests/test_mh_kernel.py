"""Focused tests for the toy MH kernel shell."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from soe.adapters.voi_voxel import VoxelGrid
from soe.contracts import EventObj, VOIConfig
from soe.soe.kernel import ProposalCandidate, single_event_mh_step
from soe.soe.state import SurvivingEventState


def _grid() -> VoxelGrid:
    """Build a simple one-dimensional toy grid embedded in `(x, y, z)` space."""
    return VoxelGrid.from_config(
        VOIConfig(
            bounds=np.asarray(
                [
                    [0.0, 4.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=float,
            ),
            grid_shape=(4, 1, 1),
        )
    )


def _event() -> EventObj:
    """Build a placeholder canonical event for frozen-state kernel tests."""
    return EventObj(
        apex=np.asarray([0.0, 0.0, 0.0], dtype=float),
        axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
        theta=math.pi / 4.0,
    )


def _state() -> SurvivingEventState:
    """Build a toy representative-point state with explicit voxel occupancy."""
    points = (
        np.asarray((0.25, 0.50, 0.50), dtype=float),
        np.asarray((1.25, 0.50, 0.50), dtype=float),
        np.asarray((1.75, 0.50, 0.50), dtype=float),
    )
    events = tuple(_event() for _ in points)
    return SurvivingEventState(events=events, representative_points=points)


@dataclass(slots=True)
class ToyDiscreteProposalBackend:
    """Finite-support toy backend for one-step kernel tests."""

    certifies_symmetric_proposal: ClassVar[bool] = True
    selected_event_index: int
    supports: dict[int, tuple[ProposalCandidate, ...]]
    support_choice: int = 0
    select_calls: int = 0
    propose_calls: int = 0
    proposed_event_indices: list[int] = field(default_factory=list)

    def select_event_index(self, state: SurvivingEventState) -> int:
        """Return the configured event index and track the selection count."""
        del state
        self.select_calls += 1
        return self.selected_event_index

    def propose_candidate(
        self,
        state: SurvivingEventState,
        event_index: int,
    ) -> ProposalCandidate:
        """Return one candidate from the configured exact finite support."""
        del state
        self.propose_calls += 1
        self.proposed_event_indices.append(event_index)
        return self.supports[event_index][self.support_choice]


def test_rejected_move_is_an_exact_noop_on_state_and_occupancy() -> None:
    """Rejected moves must leave both authoritative state and counts untouched."""
    grid = _grid()
    state = _state()
    counts = state.occupancy_counts(grid)
    backend = ToyDiscreteProposalBackend(
        selected_event_index=1,
        supports={
            1: (
                ProposalCandidate(
                    candidate_point=np.asarray((3.25, 0.50, 0.50), dtype=float),
                ),
            )
        },
    )

    result = single_event_mh_step(
        state,
        counts,
        grid=grid,
        proposal_backend=backend,
        alpha=1.0,
        uniform01=lambda: 0.90,
    )

    assert result.accepted is False
    assert result.state is state
    assert result.occupancy_counts is counts
    assert result.selected_event_index == 1
    assert result.same_voxel is False
    assert result.old_voxel == (1, 0, 0)
    assert result.new_voxel == (3, 0, 0)
    assert result.acceptance_probability == pytest.approx(0.5)
    np.testing.assert_array_equal(result.occupancy_counts, counts)


def test_accepted_move_mutates_exactly_one_event_representative_point() -> None:
    """Accepted moves should replace exactly one representative point."""
    grid = _grid()
    state = _state()
    counts = state.occupancy_counts(grid)
    candidate_point = np.asarray((3.25, 0.50, 0.50), dtype=float)
    backend = ToyDiscreteProposalBackend(
        selected_event_index=1,
        supports={1: (ProposalCandidate(candidate_point=candidate_point),)},
    )

    result = single_event_mh_step(
        state,
        counts,
        grid=grid,
        proposal_backend=backend,
        alpha=1.0,
        uniform01=lambda: 0.10,
    )

    assert result.accepted is True
    assert result.state is not state
    np.testing.assert_array_equal(result.state.representative_points[0], state.representative_points[0])
    np.testing.assert_array_equal(result.state.representative_points[1], candidate_point)
    np.testing.assert_array_equal(result.state.representative_points[2], state.representative_points[2])


def test_same_voxel_accepted_move_is_an_occupancy_noop() -> None:
    """Accepted same-voxel moves should preserve the occupancy image exactly."""
    grid = _grid()
    state = _state()
    counts = state.occupancy_counts(grid)
    candidate_point = np.asarray((1.10, 0.50, 0.50), dtype=float)
    backend = ToyDiscreteProposalBackend(
        selected_event_index=1,
        supports={1: (ProposalCandidate(candidate_point=candidate_point),)},
    )

    result = single_event_mh_step(
        state,
        counts,
        grid=grid,
        proposal_backend=backend,
        alpha=2.5,
        uniform01=lambda: 0.75,
    )

    assert result.accepted is True
    assert result.same_voxel is True
    assert result.old_voxel == result.new_voxel == (1, 0, 0)
    assert result.acceptance_probability == pytest.approx(1.0)
    np.testing.assert_array_equal(result.occupancy_counts, counts)
    np.testing.assert_array_equal(result.state.representative_points[1], candidate_point)


def test_cross_voxel_accepted_move_changes_exactly_two_bins_and_conserves_total_occupancy() -> None:
    """Accepted cross-voxel moves should apply one decrement and one increment."""
    grid = _grid()
    state = _state()
    counts = state.occupancy_counts(grid)
    backend = ToyDiscreteProposalBackend(
        selected_event_index=1,
        supports={
            1: (
                ProposalCandidate(
                    candidate_point=np.asarray((3.25, 0.50, 0.50), dtype=float),
                ),
            )
        },
    )

    result = single_event_mh_step(
        state,
        counts,
        grid=grid,
        proposal_backend=backend,
        alpha=1.0,
        uniform01=lambda: 0.10,
    )
    delta = result.occupancy_counts - counts

    assert result.accepted is True
    assert result.same_voxel is False
    assert np.count_nonzero(delta) == 2
    assert delta[result.old_voxel] == -1
    assert delta[result.new_voxel] == 1
    assert int(result.occupancy_counts.sum()) == int(counts.sum())


def test_only_one_event_is_selected_and_considered_per_step() -> None:
    """The one-step shell should query exactly one selected event and one proposal."""
    grid = _grid()
    state = _state()
    counts = state.occupancy_counts(grid)
    backend = ToyDiscreteProposalBackend(
        selected_event_index=2,
        supports={
            2: (
                ProposalCandidate(
                    candidate_point=np.asarray((2.25, 0.50, 0.50), dtype=float),
                    proposal_ratio=0.5,
                ),
            )
        },
    )

    result = single_event_mh_step(
        state,
        counts,
        grid=grid,
        proposal_backend=backend,
        alpha=1.0,
        uniform01=lambda: 0.90,
    )

    assert result.selected_event_index == 2
    assert backend.select_calls == 1
    assert backend.propose_calls == 1
    assert backend.proposed_event_indices == [2]
