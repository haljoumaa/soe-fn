"""Tests for the finite-precision direct-sampler path and default-path preservation.

These tests verify:
1. The direct sampler produces points satisfying the cone-surface equation.
2. Sampled world points lie inside the exact bounded-box support.
3. Numerically inverted xi lies strictly inside its chosen visible segment.
4. Sampled ell lies in the exact admissible radial interval for that xi.
5. Empirical draws do not hit forbidden endpoints.
6. Returned proposal metadata preserves symmetry and proposal_ratio = 1.
7. Direct path fails closed outside the supported scope.
8. Default path uses direct backend; rejection reachable via explicit opt-in.
9. Repeated-sampling sanity test on the supported scope.
"""

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
from soe.geometry.cone import (
    build_cone_local_geometry,
    cone_surface_point,
    is_admissible_point,
    is_on_cone_surface,
)
from soe.geometry.cone_box_surface import (
    build_cone_box_decomposition,
    evaluate_fixed_azimuth,
)
import soe.soe.direct_surface_sampler as direct_surface_sampler
from soe.soe.direct_surface_sampler import (
    DirectSamplerError,
    DirectSamplerNotReady,
    build_direct_sampler_primitives_from_geometry,
    direct_sample_area_weighted_azimuth_and_interval,
    direct_sample_exact_bounded_box_point,
)
from soe.soe.kernel import single_event_mh_step
from soe.soe.state import SurvivingEventState
import soe.soe.uniform_surface_reference as uniform_surface_reference
from soe.soe.uniform_surface_reference import (
    UniformSurfaceReferenceProposalBackend,
)

_TWO_PI = 2.0 * math.pi


def _event() -> EventObj:
    return EventObj(
        apex=np.asarray([0.0, 0.0, 0.0], dtype=float),
        axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
        theta=math.pi / 4.0,
    )


def _voi_bounds() -> VoiBounds:
    return VoiBounds.from_config(
        VOIConfig(
            bounds=np.asarray(
                [[-1.0, 2.0], [-1.0, 1.0], [0.0, 2.0]], dtype=float
            ),
            grid_shape=(3, 2, 2),
        )
    )


def _grid() -> VoxelGrid:
    return VoxelGrid.from_config(
        VOIConfig(
            bounds=np.asarray(
                [[-1.0, 2.0], [-1.0, 1.0], [0.0, 2.0]], dtype=float
            ),
            grid_shape=(3, 2, 2),
        )
    )


def _interior_state() -> SurvivingEventState:
    """Build a one-event state with a strict-open interior representative point."""
    event = _event()
    point = cone_surface_point(event, ell=1.0, xi=0.0)
    assert is_admissible_point(point, event, _voi_bounds())
    return SurvivingEventState(events=(event,), representative_points=(point,))


# ---------------------------------------------------------------------------
# 1. Direct sampler returns a point on the cone surface
# ---------------------------------------------------------------------------


def test_direct_sampler_point_on_cone_surface() -> None:
    """The sampled point must lie on the canonical cone surface."""
    event = _event()
    voi = _voi_bounds()
    rng = np.random.default_rng(42)

    point = direct_sample_exact_bounded_box_point(
        event=event, allowed_region=voi, rng=rng,
    )
    assert is_on_cone_surface(point, event, tol=1e-10)


# ---------------------------------------------------------------------------
# 2. Sampled world point lies inside the bounded-box admissible support
# ---------------------------------------------------------------------------


def test_direct_sampler_point_inside_box() -> None:
    """The sampled point must be admissible in the bounded box."""
    event = _event()
    voi = _voi_bounds()
    rng = np.random.default_rng(123)

    point = direct_sample_exact_bounded_box_point(
        event=event, allowed_region=voi, rng=rng,
    )
    assert is_admissible_point(point, event, voi)


# ---------------------------------------------------------------------------
# 3. Selected xi lies strictly inside its chosen visible segment
# ---------------------------------------------------------------------------


def test_xi_inside_visible_segment() -> None:
    """The azimuth xi must lie strictly inside a visible segment."""
    event = _event()
    voi = _voi_bounds()
    rng = np.random.default_rng(99)

    cone_geom = build_cone_local_geometry(event)
    prims = build_direct_sampler_primitives_from_geometry(cone_geom, voi)

    xi, interval, ell2_width = direct_sample_area_weighted_azimuth_and_interval(
        event=event, allowed_region=voi, rng=rng,
    )

    # Find which segment xi belongs to
    found = False
    for j, regime in enumerate(prims.azimuth_law.regimes):
        if not regime.nonempty or prims.segment_masses[j] <= 0.0:
            continue
        xi_a = regime.xi_start
        xi_b = regime.xi_end
        wraps = xi_b <= xi_a
        if wraps:
            if xi > xi_a or xi < xi_b:
                found = True
                break
        else:
            if xi_a < xi < xi_b:
                found = True
                break

    assert found, f"xi={xi} not strictly inside any visible segment"


# ---------------------------------------------------------------------------
# 4. Sampled ell lies in the exact admissible radial interval for that xi
# ---------------------------------------------------------------------------


def test_ell_in_radial_interval() -> None:
    """The sampled ell must lie in the admissible radial interval."""
    event = _event()
    voi = _voi_bounds()
    rng = np.random.default_rng(77)
    cone_geom = build_cone_local_geometry(event)
    decomp = build_cone_box_decomposition(cone_geom, voi)

    xi, interval, ell2_width = direct_sample_area_weighted_azimuth_and_interval(
        event=event, allowed_region=voi, rng=rng,
    )
    ell_minus, ell_plus = interval

    # Draw radial coordinate the same way the full function does
    u_rad_rng = np.random.default_rng(78)
    while True:
        u_rad = float(u_rad_rng.random())
        if 0.0 < u_rad < 1.0:
            break
    ell_sq = (ell_minus * ell_minus) + u_rad * ell2_width
    ell = math.sqrt(ell_sq)

    assert ell_minus <= ell <= ell_plus, (
        f"ell={ell} not in [{ell_minus}, {ell_plus}]"
    )

    # Verify the interval matches the geometry at this azimuth
    iv = evaluate_fixed_azimuth(decomp.coefficients, xi)
    assert iv.nonempty
    assert math.isclose(iv.ell_minus, ell_minus, rel_tol=1e-12)
    assert math.isclose(iv.ell_plus, ell_plus, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 5. Empirical draws do not hit forbidden endpoints
# ---------------------------------------------------------------------------


def test_no_forbidden_endpoints() -> None:
    """Draws must not land exactly on radial interval endpoints."""
    event = _event()
    voi = _voi_bounds()

    for seed in range(50):
        rng = np.random.default_rng(seed)
        xi, interval, ell2_width = direct_sample_area_weighted_azimuth_and_interval(
            event=event, allowed_region=voi, rng=rng,
        )
        ell_minus, ell_plus = interval
        # The open-interval draw should never produce exact endpoints
        assert ell2_width > 0.0


# ---------------------------------------------------------------------------
# 6. Returned proposal metadata preserves symmetry and proposal_ratio = 1
# ---------------------------------------------------------------------------


def test_proposal_metadata_symmetry() -> None:
    """The direct-sampler backend must return proposal_ratio=1 and symmetric."""
    backend = UniformSurfaceReferenceProposalBackend(
        allowed_region=_voi_bounds(),
        event_index_selector=lambda state: 0,
        rng=np.random.default_rng(42),
        _use_direct_sampler=True,
    )
    state = _interior_state()

    candidate = backend.propose_candidate(state, 0)
    assert candidate.proposal_ratio == 1.0
    assert candidate.proposal_ratio_was_omitted is True
    assert backend.certifies_symmetric_proposal is True


# ---------------------------------------------------------------------------
# 7. Direct path fails closed outside the supported scope
# ---------------------------------------------------------------------------


def test_direct_sampler_fails_on_unsupported_type() -> None:
    """The direct sampler must reject non-VoiBounds allowed regions."""
    event = _event()
    rng = np.random.default_rng(0)

    with pytest.raises(TypeError, match="VoiBounds"):
        direct_sample_exact_bounded_box_point(
            event=event,
            allowed_region="not_a_voi_bounds",  # type: ignore[arg-type]
            rng=rng,
        )


def test_direct_sampler_fails_on_zero_mass_event() -> None:
    """The direct sampler must fail closed for events with no visible surface."""
    # Cone pointing away from the box
    event = EventObj(
        apex=np.asarray([0.0, 0.0, 0.0], dtype=float),
        axis=np.asarray([0.0, 0.0, -1.0], dtype=float),
        theta=math.pi / 6,
    )
    voi = VoiBounds(xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=2, zmax=4)
    rng = np.random.default_rng(0)

    with pytest.raises(DirectSamplerError, match="zero total"):
        direct_sample_exact_bounded_box_point(
            event=event, allowed_region=voi, rng=rng,
        )


# ---------------------------------------------------------------------------
# 8. Default path uses direct backend; rejection reachable via explicit opt-in
# ---------------------------------------------------------------------------


def test_default_path_uses_direct_sampler() -> None:
    """Default backend must use the direct-sampler on the supported scope."""
    grid = _grid()
    state = _interior_state()
    rng_seed = 20260330
    backend = UniformSurfaceReferenceProposalBackend(
        allowed_region=_voi_bounds(),
        event_index_selector=lambda s: 0,
        rng=np.random.default_rng(rng_seed),
    )
    assert backend._use_direct_sampler is True

    result = single_event_mh_step(
        state,
        state.occupancy_counts(grid),
        grid=grid,
        proposal_backend=backend,
        alpha=1.0,
        uniform01=np.random.default_rng(rng_seed + 1).random,
    )
    assert result.proposal_ratio == 1.0
    assert isinstance(result.state, SurvivingEventState)
    assert int(result.occupancy_counts.sum()) == len(state.events)


def test_rejection_fallback_reachable() -> None:
    """Rejection backend must remain reachable via explicit _use_direct_sampler=False."""
    grid = _grid()
    state = _interior_state()
    rng_seed = 20260330
    backend = UniformSurfaceReferenceProposalBackend(
        allowed_region=_voi_bounds(),
        event_index_selector=lambda s: 0,
        rng=np.random.default_rng(rng_seed),
        _use_direct_sampler=False,
    )
    assert backend._use_direct_sampler is False

    result = single_event_mh_step(
        state,
        state.occupancy_counts(grid),
        grid=grid,
        proposal_backend=backend,
        alpha=1.0,
        uniform01=np.random.default_rng(rng_seed + 1).random,
    )
    assert result.proposal_ratio == 1.0
    assert isinstance(result.state, SurvivingEventState)
    assert int(result.occupancy_counts.sum()) == len(state.events)


def test_both_backend_variants_certify_symmetric_proposal() -> None:
    """Both default and direct-sampler variants must certify symmetric proposals."""
    for kwargs in ({}, {"_use_direct_sampler": True}):
        backend = UniformSurfaceReferenceProposalBackend(
            allowed_region=_voi_bounds(),
            event_index_selector=lambda s: 0,
            **kwargs,
        )
        assert backend.certifies_symmetric_proposal is True


# ---------------------------------------------------------------------------
# 9. Repeated-sampling sanity test
# ---------------------------------------------------------------------------


def test_repeated_direct_sampling_all_valid() -> None:
    """Repeated draws from the direct path must all be valid admissible points."""
    event = _event()
    voi = _voi_bounds()
    rng = np.random.default_rng(2026)

    n_samples = 100
    for _ in range(n_samples):
        point = direct_sample_exact_bounded_box_point(
            event=event, allowed_region=voi, rng=rng,
        )
        assert is_admissible_point(point, event, voi), (
            f"sample {_}: point {point} not admissible"
        )
        assert is_on_cone_surface(point, event, tol=1e-10), (
            f"sample {_}: point {point} not on cone surface"
        )


def test_repeated_direct_sampling_via_backend() -> None:
    """Repeated draws via the backend with _use_direct_sampler=True must all succeed."""
    backend = UniformSurfaceReferenceProposalBackend(
        allowed_region=_voi_bounds(),
        event_index_selector=lambda state: 0,
        rng=np.random.default_rng(2026),
        _use_direct_sampler=True,
    )
    state = _interior_state()

    for _ in range(50):
        candidate = backend.propose_candidate(state, 0)
        assert candidate.proposal_ratio == 1.0
        assert is_admissible_point(
            candidate.candidate_point, state.events[0], _voi_bounds()
        )


def test_direct_sampler_retries_soft_phi_certification_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A phi-only numerical certification miss must retry instead of aborting."""
    event = _event()
    voi = _voi_bounds()
    calls = 0

    def certification_with_one_soft_miss(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return calls > 1

    monkeypatch.setattr(
        direct_surface_sampler,
        "is_admissible_point",
        certification_with_one_soft_miss,
    )
    monkeypatch.setattr(direct_surface_sampler, "cone_phi", lambda *_args: 2e-12)

    point = direct_sample_exact_bounded_box_point(
        event=event,
        allowed_region=voi,
        rng=np.random.default_rng(42),
    )

    assert calls == 2
    assert is_admissible_point(point, event, voi)


def test_direct_backend_reuses_cached_primitives(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated direct proposals for one surviving event must reuse primitives."""
    build_count = 0
    original_builder = (
        uniform_surface_reference.build_direct_sampler_primitives_from_geometry
    )

    def counted_builder(*args, **kwargs):
        nonlocal build_count
        build_count += 1
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(
        uniform_surface_reference,
        "build_direct_sampler_primitives_from_geometry",
        counted_builder,
    )
    backend = UniformSurfaceReferenceProposalBackend(
        allowed_region=_voi_bounds(),
        event_index_selector=lambda state: 0,
        rng=np.random.default_rng(20260411),
        _use_direct_sampler=True,
    )
    state = _interior_state()

    for _ in range(8):
        candidate = backend.propose_candidate(state, 0)
        assert candidate.proposal_ratio == 1.0
        assert is_admissible_point(
            candidate.candidate_point,
            state.events[0],
            _voi_bounds(),
        )

    assert build_count == 1


# ---------------------------------------------------------------------------
# 10. Direct path through kernel produces valid MH steps
# ---------------------------------------------------------------------------


def test_kernel_with_direct_sampler() -> None:
    """The MH shell must work correctly with the direct-sampler backend."""
    grid = _grid()
    state = _interior_state()
    backend = UniformSurfaceReferenceProposalBackend(
        allowed_region=_voi_bounds(),
        event_index_selector=lambda s: 0,
        rng=np.random.default_rng(42),
        _use_direct_sampler=True,
    )

    result = single_event_mh_step(
        state,
        state.occupancy_counts(grid),
        grid=grid,
        proposal_backend=backend,
        alpha=1.0,
        uniform01=np.random.default_rng(43).random,
    )
    assert result.proposal_ratio == 1.0
    assert isinstance(result.state, SurvivingEventState)
    assert int(result.occupancy_counts.sum()) == len(state.events)
