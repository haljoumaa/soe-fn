"""Focused tests for explicit unsupported paths in the reference backend."""

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

from soe.adapters.voi_voxel import VoiBounds, VoxelGrid
from soe.contracts import EventObj, VOIConfig
from soe.geometry.cone import cone_surface_point
from soe.soe.kernel import single_event_mh_step
from soe.soe.state import SurvivingEventState
from soe.soe.uniform_surface_reference import (
    UniformSurfaceReferenceProposalBackend,
    UnsupportedUniformSurfaceProposal,
    _sample_radius_from_radial_endpoints,
)


def _event() -> EventObj:
    """Build a canonical event for uniform-surface backend tests."""
    return EventObj(
        apex=np.asarray([0.0, 0.0, 0.0], dtype=float),
        axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
        theta=math.pi / 4.0,
    )


def _voi_bounds() -> VoiBounds:
    """Build a simple bounded authoritative VOI with a reachable lower face."""
    return VoiBounds.from_config(
        VOIConfig(
            bounds=np.asarray(
                [
                    [-1.0, 2.0],
                    [-1.0, 1.0],
                    [0.0, 2.0],
                ],
                dtype=float,
            ),
            grid_shape=(3, 2, 2),
        )
    )


def _grid() -> VoxelGrid:
    """Build the matching grid for kernel-facing checks."""
    return VoxelGrid.from_config(
        VOIConfig(
            bounds=np.asarray(
                [
                    [-1.0, 2.0],
                    [-1.0, 1.0],
                    [0.0, 2.0],
                ],
                dtype=float,
            ),
            grid_shape=(3, 2, 2),
        )
    )


def _state(point: np.ndarray) -> SurvivingEventState:
    """Build a one-event authoritative state for independence checks."""
    return SurvivingEventState(events=(_event(),), representative_points=(point,))


def test_reference_backend_fails_explicitly_for_boundary_only_support_certificates() -> None:
    """Boundary-only current points should fail instead of silently broadening support."""
    backend = UniformSurfaceReferenceProposalBackend(
        allowed_region=_voi_bounds(),
        event_index_selector=lambda state: 0,
    )
    state = _state(cone_surface_point(_event(), math.sqrt(2.0), math.pi))

    with pytest.raises(
        UnsupportedUniformSurfaceProposal,
        match="strict-open VOI interior",
    ):
        backend.propose_candidate(state, 0)


def test_reference_backend_rejects_nonadmissible_current_points() -> None:
    """The narrow exact backend should reject states outside the authoritative support."""
    backend = UniformSurfaceReferenceProposalBackend(
        allowed_region=_voi_bounds(),
        event_index_selector=lambda state: 0,
    )
    state = _state(np.asarray((0.25, 0.25, 0.25), dtype=float))

    with pytest.raises(
        UnsupportedUniformSurfaceProposal,
        match="pointwise admissible",
    ):
        backend.propose_candidate(state, 0)


def test_exact_radial_helper_fails_closed_on_unbounded_or_degenerate_intervals() -> None:
    """The exact conditional radial law should reject unsupported intervals explicitly."""
    with pytest.raises(
        UnsupportedUniformSurfaceProposal,
        match="nondegenerate bounded radial interval",
    ):
        _sample_radius_from_radial_endpoints(
            ell_minus=1.0,
            ell_plus=math.inf,
            rng=np.random.default_rng(0),
        )

    with pytest.raises(
        UnsupportedUniformSurfaceProposal,
        match="nondegenerate bounded radial interval",
    ):
        _sample_radius_from_radial_endpoints(
            ell_minus=2.0,
            ell_plus=2.0,
            rng=np.random.default_rng(0),
        )


def test_kernel_surfaces_the_explicit_unsupported_reference_backend_failure() -> None:
    """The MH shell should propagate the explicit exactness-first failure."""
    grid = _grid()
    state = _state(cone_surface_point(_event(), math.sqrt(2.0), math.pi))
    backend = UniformSurfaceReferenceProposalBackend(
        allowed_region=_voi_bounds(),
        event_index_selector=lambda current_state: 0,
    )

    with pytest.raises(
        UnsupportedUniformSurfaceProposal,
        match="strict-open VOI interior",
    ):
        single_event_mh_step(
            state,
            state.occupancy_counts(grid),
            grid=grid,
            proposal_backend=backend,
            alpha=1.0,
            uniform01=lambda: 0.5,
        )
