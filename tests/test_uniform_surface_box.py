"""Focused tests for the narrow exact bounded-box proposal path."""

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
from soe.geometry.cone import cone_surface_point, is_admissible_point
from soe.soe.kernel import single_event_mh_step
from soe.soe.state import SurvivingEventState
from soe.soe.uniform_surface_reference import (
    UniformSurfaceReferenceProposalBackend,
    _sample_area_weighted_azimuth_and_interval,
    _sample_radius_from_radial_endpoints,
    _radial_endpoints_for_azimuth,
    _visible_area_profile_for_azimuth,
    _visible_ray_interval_for_azimuth,
)


def _event() -> EventObj:
    """Build a canonical event for the supported exact box case."""
    return EventObj(
        apex=np.asarray([0.0, 0.0, 0.0], dtype=float),
        axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
        theta=math.pi / 4.0,
    )


def _voi_bounds() -> VoiBounds:
    """Build a simple bounded VOI box."""
    return VoiBounds.from_config(
        VOIConfig(
            bounds=np.asarray(
                [
                    [0.0, 2.0],
                    [-1.0, 1.0],
                    [0.0, 2.0],
                ],
                dtype=float,
            ),
            grid_shape=(2, 2, 2),
        )
    )


def _grid() -> VoxelGrid:
    """Build the matching grid for kernel-facing checks."""
    return VoxelGrid.from_config(
        VOIConfig(
            bounds=np.asarray(
                [
                    [0.0, 2.0],
                    [-1.0, 1.0],
                    [0.0, 2.0],
                ],
                dtype=float,
            ),
            grid_shape=(2, 2, 2),
        )
    )


def _state(point: np.ndarray) -> SurvivingEventState:
    """Build a one-event authoritative state for proposal checks."""
    return SurvivingEventState(events=(_event(),), representative_points=(point,))


class _ScriptedRNG:
    """Minimal deterministic RNG stub for exact-law helper tests."""

    def __init__(self, draws: list[float]) -> None:
        self._draws = iter(draws)

    def random(self) -> float:
        return float(next(self._draws))


def test_nonvisible_azimuth_has_no_interval_or_area_profile() -> None:
    """An azimuth outside the visible half of the current certified box should be empty."""
    event = _event()
    voi = _voi_bounds()

    assert _visible_ray_interval_for_azimuth(event, voi, xi=math.pi) is None
    assert _radial_endpoints_for_azimuth(event, voi, xi=math.pi) is None
    assert _visible_area_profile_for_azimuth(event, voi, xi=math.pi) == pytest.approx(0.0)


def test_visible_azimuth_has_hand_checked_bounded_interval_and_area_profile() -> None:
    """The current certified subset should expose exact bounded radial data for visible `xi`."""
    event = _event()
    voi = _voi_bounds()

    interval = _visible_ray_interval_for_azimuth(event, voi, xi=math.pi / 4.0)
    endpoints = _radial_endpoints_for_azimuth(event, voi, xi=math.pi / 4.0)
    area_profile = _visible_area_profile_for_azimuth(event, voi, xi=math.pi / 4.0)

    assert interval == pytest.approx((0.0, 2.0))
    assert endpoints == pytest.approx((0.0, 2.0))
    assert endpoints is not None
    ell_minus, ell_plus = endpoints
    assert 0.0 <= ell_minus < ell_plus < math.inf
    assert area_profile == pytest.approx(4.0)


def test_area_profile_is_zero_exactly_when_the_visible_interval_is_empty_on_hand_checked_cases() -> None:
    """The local profile helper should agree with the current exact visibility interval."""
    event = _event()
    voi = _voi_bounds()

    for xi in (math.pi, 3.0 * math.pi / 4.0):
        assert _visible_ray_interval_for_azimuth(event, voi, xi=xi) is None
        assert _visible_area_profile_for_azimuth(event, voi, xi=xi) == pytest.approx(0.0)

    for xi in (0.0, math.pi / 4.0, 7.0 * math.pi / 4.0):
        assert _visible_ray_interval_for_azimuth(event, voi, xi=xi) is not None
        assert _visible_area_profile_for_azimuth(event, voi, xi=xi) > 0.0


def test_profile_layer_respects_the_strict_open_boundary_transition_at_exact_zero_azimuth() -> None:
    """The helper layer should stay aligned with strict-open ray-box boundary semantics."""
    event = _event()
    voi = VoiBounds.from_config(
        VOIConfig(
            bounds=np.asarray(
                [
                    [-1.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 2.0],
                ],
                dtype=float,
            ),
            grid_shape=(2, 1, 2),
        )
    )
    xi_boundary = 0.0
    xi_just_inside = math.nextafter(0.0, math.inf)

    assert _visible_ray_interval_for_azimuth(event, voi, xi=xi_boundary) is None
    assert _visible_area_profile_for_azimuth(event, voi, xi=xi_boundary) == pytest.approx(
        0.0
    )

    endpoints = _radial_endpoints_for_azimuth(event, voi, xi=xi_just_inside)
    assert endpoints is not None
    ell_minus, ell_plus = endpoints
    assert 0.0 <= ell_minus < ell_plus < math.inf
    assert _visible_area_profile_for_azimuth(event, voi, xi=xi_just_inside) > 0.0


def test_radial_sampler_is_exactly_uniform_in_squared_radius_on_bounded_intervals() -> None:
    """The conditional radial step must sample `ell^2` uniformly on the visible interval."""
    ell = _sample_radius_from_radial_endpoints(
        ell_minus=3.0,
        ell_plus=5.0,
        rng=_ScriptedRNG([0.25]),
    )

    assert ell == pytest.approx(math.sqrt(13.0))


def test_azimuth_sampler_uses_area_weighting_instead_of_uniform_azimuth() -> None:
    """A hand-checked box case should reject a smaller-area azimuth before a larger one."""
    event = _event()
    voi = _voi_bounds()

    area_low = _visible_area_profile_for_azimuth(event, voi, xi=math.pi / 4.0)
    area_high = _visible_area_profile_for_azimuth(event, voi, xi=0.0)
    assert area_low == pytest.approx(4.0)
    assert area_high == pytest.approx(8.0)

    xi, interval, area_profile = _sample_area_weighted_azimuth_and_interval(
        event=event,
        allowed_region=voi,
        rng=_ScriptedRNG(
            [
                1.0 / 8.0,
                0.5,
                0.0,
                0.5,
            ]
        ),
    )

    assert xi == pytest.approx(0.0)
    assert interval == pytest.approx((0.0, 2.0 * math.sqrt(2.0)))
    assert area_profile == pytest.approx(8.0)


def test_supported_exact_bounded_box_candidate_is_admissible_and_symmetric() -> None:
    """Supported exact proposals should stay on `X_k` and carry ratio `1`."""
    voi = _voi_bounds()
    backend = UniformSurfaceReferenceProposalBackend(
        allowed_region=voi,
        event_index_selector=lambda state: 0,
        rng=np.random.default_rng(20260313),
    )
    state = _state(cone_surface_point(_event(), 1.0, 0.0))

    proposal = backend.propose_candidate(state, 0)

    assert is_admissible_point(proposal.candidate_point, _event(), voi) is True
    assert proposal.proposal_ratio == pytest.approx(1.0)
    assert proposal.proposal_ratio_was_omitted is True
    assert proposal.log_q_forward is None
    assert proposal.log_q_reverse is None
    assert backend.certifies_symmetric_proposal is True


def test_supported_exact_bounded_box_is_independent_of_current_point() -> None:
    """The supported exact proposal law should ignore the current point value."""
    seed = 1701
    backend_a = UniformSurfaceReferenceProposalBackend(
        allowed_region=_voi_bounds(),
        event_index_selector=lambda state: 0,
        rng=np.random.default_rng(seed),
    )
    backend_b = UniformSurfaceReferenceProposalBackend(
        allowed_region=_voi_bounds(),
        event_index_selector=lambda state: 0,
        rng=np.random.default_rng(seed),
    )
    state_a = _state(cone_surface_point(_event(), 1.0, 0.0))
    state_b = _state(cone_surface_point(_event(), 1.0, math.pi / 4.0))

    proposal_a = backend_a.propose_candidate(state_a, 0)
    proposal_b = backend_b.propose_candidate(state_b, 0)

    assert np.allclose(proposal_a.candidate_point, proposal_b.candidate_point, atol=0.0)
    assert proposal_a.proposal_ratio == pytest.approx(1.0)
    assert proposal_b.proposal_ratio == pytest.approx(1.0)
    assert proposal_a.proposal_ratio_was_omitted is True
    assert proposal_b.proposal_ratio_was_omitted is True


def test_supported_exact_bounded_box_metadata_reaches_kernel_as_symmetric_ratio() -> None:
    """The generic MH shell should observe the proposal ratio `1` for supported cases."""
    grid = _grid()
    state = _state(cone_surface_point(_event(), 1.0, 0.0))
    backend = UniformSurfaceReferenceProposalBackend(
        allowed_region=_voi_bounds(),
        event_index_selector=lambda current_state: 0,
        rng=np.random.default_rng(99),
    )

    result = single_event_mh_step(
        state,
        state.occupancy_counts(grid),
        grid=grid,
        proposal_backend=backend,
        alpha=1.0,
        uniform01=lambda: 0.5,
    )

    assert result.proposal_ratio == pytest.approx(1.0)
    assert is_admissible_point(result.state.representative_points[0], _event(), _voi_bounds()) is True
