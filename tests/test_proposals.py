"""Focused tests for the proposal contract."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from soe.adapters.voi_voxel import VoiBounds, VoxelGrid
from soe.contracts import EventObj, VOIConfig
from soe.soe.kernel import single_event_mh_step
from soe.soe.proposals import ProposalCandidate
from soe.soe.state import SurvivingEventState
from soe.soe.uniform_surface_reference import UniformSurfaceReferenceProposalBackend


def _grid() -> VoxelGrid:
    """Build a minimal toy grid for proposal-contract tests."""
    return VoxelGrid.from_config(
        VOIConfig(
            bounds=np.asarray(
                [
                    [0.0, 3.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=float,
            ),
            grid_shape=(3, 1, 1),
        )
    )


def _allowed_region() -> VoiBounds:
    """Build the matching bounded region for backend-construction checks."""
    return VoiBounds.from_config(
        VOIConfig(
            bounds=np.asarray(
                [
                    [0.0, 3.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=float,
            ),
            grid_shape=(3, 1, 1),
        )
    )


def _event() -> EventObj:
    """Build a placeholder canonical event for contract tests."""
    return EventObj(
        apex=np.asarray([0.0, 0.0, 0.0], dtype=float),
        axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
        theta=math.pi / 4.0,
    )


def _state() -> SurvivingEventState:
    """Build a one-event toy state with explicit voxel assignment."""
    return SurvivingEventState(
        events=(_event(),),
        representative_points=(np.asarray((0.25, 0.50, 0.50), dtype=float),),
    )


@dataclass(frozen=True, slots=True)
class FixedProposalBackend:
    """Return one preconfigured proposal candidate for the single toy event."""

    proposal: ProposalCandidate
    certifies_symmetric_proposal: bool = False

    def select_event_index(self, state: SurvivingEventState) -> int:
        """Always select the only event."""
        del state
        return 0

    def propose_candidate(
        self,
        state: SurvivingEventState,
        event_index: int,
    ) -> ProposalCandidate:
        """Return the configured proposal candidate."""
        del state
        del event_index
        return self.proposal


def test_omitted_ratio_metadata_is_distinguished_from_explicit_unit_ratio() -> None:
    """Omitted ratio metadata must stay distinguishable from an explicit `1.0`."""
    omitted = ProposalCandidate(
        candidate_point=np.asarray((1.25, 0.50, 0.50), dtype=float),
    )
    explicit = ProposalCandidate(
        candidate_point=np.asarray((1.25, 0.50, 0.50), dtype=float),
        proposal_ratio=1.0,
    )

    assert omitted.proposal_ratio == pytest.approx(1.0)
    assert omitted.proposal_ratio_was_omitted is True
    assert omitted.log_q_forward is None
    assert omitted.log_q_reverse is None
    assert explicit.proposal_ratio == pytest.approx(1.0)
    assert explicit.proposal_ratio_was_omitted is False
    assert explicit.log_q_forward is None
    assert explicit.log_q_reverse is None


def test_asymmetric_candidate_can_carry_nontrivial_log_density_metadata() -> None:
    """Non-symmetric proposals may supply forward/reverse log densities."""
    candidate = ProposalCandidate.from_log_proposal_densities(
        np.asarray((1.25, 0.50, 0.50), dtype=float),
        log_q_forward=math.log(0.8),
        log_q_reverse=math.log(0.2),
    )

    assert candidate.log_q_forward == pytest.approx(math.log(0.8))
    assert candidate.log_q_reverse == pytest.approx(math.log(0.2))
    assert candidate.proposal_ratio == pytest.approx(0.25)
    assert candidate.proposal_ratio_was_omitted is False


def test_incomplete_or_conflicting_asymmetric_metadata_is_rejected() -> None:
    """The contract should reject partial or over-specified ratio metadata."""
    with pytest.raises(
        ValueError,
        match="log_q_forward and log_q_reverse must be provided together",
    ):
        ProposalCandidate(
            np.asarray((1.25, 0.50, 0.50), dtype=float),
            log_q_forward=0.0,
        )

    with pytest.raises(
        ValueError,
        match="specify either proposal_ratio or both log_q_forward/log_q_reverse",
    ):
        ProposalCandidate(
            np.asarray((1.25, 0.50, 0.50), dtype=float),
            proposal_ratio=0.5,
            log_q_forward=0.0,
            log_q_reverse=-0.6931471805599453,
        )


def test_kernel_accepts_certified_symmetric_backend_with_omitted_ratio() -> None:
    """Certified symmetric backends may omit proposal-ratio metadata."""
    grid = _grid()
    state = _state()
    counts = state.occupancy_counts(grid)
    backend = FixedProposalBackend(
        proposal=ProposalCandidate(
            np.asarray((2.25, 0.50, 0.50), dtype=float),
        ),
        certifies_symmetric_proposal=True,
    )

    result = single_event_mh_step(
        state,
        counts,
        grid=grid,
        proposal_backend=backend,
        alpha=1.0,
        uniform01=lambda: 0.90,
    )

    assert result.accepted is True
    assert result.old_voxel == (0, 0, 0)
    assert result.new_voxel == (2, 0, 0)
    assert result.proposal_ratio == pytest.approx(1.0)
    assert result.acceptance_probability == pytest.approx(1.0)


def test_live_uniform_surface_backend_explicitly_marks_certified_symmetry() -> None:
    """The landed exact backend must opt in explicitly before omitting ratio metadata."""
    backend = UniformSurfaceReferenceProposalBackend(
        allowed_region=_allowed_region(),
        event_index_selector=lambda state: 0,
        rng=np.random.default_rng(0),
    )

    assert backend.certifies_symmetric_proposal is True
    assert isinstance(backend.certifies_symmetric_proposal, bool)


def test_kernel_rejects_noncertified_backend_with_omitted_ratio() -> None:
    """Non-certified omission should fail with a narrow contract error."""
    grid = _grid()
    state = _state()
    counts = state.occupancy_counts(grid)
    backend = FixedProposalBackend(
        proposal=ProposalCandidate(
            np.asarray((2.25, 0.50, 0.50), dtype=float),
        ),
    )

    with pytest.raises(
        ValueError,
        match="omitted proposal-ratio metadata without certifying symmetry",
    ):
        single_event_mh_step(
            state,
            counts,
            grid=grid,
            proposal_backend=backend,
            alpha=1.0,
            uniform01=lambda: 0.10,
        )


def test_kernel_accepts_explicit_ratio_from_noncertified_backend() -> None:
    """Non-certified backends still work when they provide explicit ratio data."""
    grid = _grid()
    state = _state()
    counts = state.occupancy_counts(grid)
    backend = FixedProposalBackend(
        proposal=ProposalCandidate(
            np.asarray((2.25, 0.50, 0.50), dtype=float),
            proposal_ratio=0.25,
        )
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
    assert result.old_voxel == (0, 0, 0)
    assert result.new_voxel == (2, 0, 0)
    assert result.proposal_ratio == pytest.approx(0.25)
    assert result.acceptance_probability == pytest.approx(0.25)
