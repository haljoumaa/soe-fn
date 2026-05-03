"""Tests for exact azimuth-law primitives and numerical direct-sampler inversion.

Covers:
  - Finite nonnegative M_{k,j} for all segments
  - M_{k,j} = 0 on empty segments
  - Sum of M_{k,j} equals total mass and is consistent with reduced law
  - Cumulative masses are monotone with correct endpoints
  - C_{k,j}(left+) = 0 and C_{k,j}(right-) = M_{k,j} on visible segments
  - C_{k,j} strictly increases on visible segments
  - Numerical inverse CDF recovers the requested segment mass to tight tolerance
  - Numerical inverse stays inside the selected segment bracket
  - Numerical inverse is monotone and stable under tighter/looser tolerances
  - Local interval / sampled-point construction remains admissible downstream
  - Fail-closed behavior for empty / zero-mass / invalid-segment inverse calls
  - Density includes exact (s/2) factor
  - Synthetic edge-case regression keeps the repaired sampler path stable
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

from soe.contracts import EventObj
from soe.adapters.voi_voxel import VoiBounds
from soe.geometry.cone import build_cone_local_geometry
from soe.geometry.cone import cone_generator_direction, cone_surface_point, is_admissible_point
from soe.geometry.cone_box_surface import (
    build_cone_box_decomposition,
    evaluate_area_profile,
    evaluate_fixed_azimuth,
)
from soe.geometry.cone_box_azimuth_law import integrate_inv_g_sq
from soe.geometry.ray_box import open_box_ray_interval
from soe.geometry.sampled_surrogate import midpoint_azimuth_grid
import soe.soe.direct_surface_sampler as direct_surface_sampler_module
from soe.soe.direct_surface_sampler import (
    DirectSamplerNotReady,
    DirectSamplerPrimitives,
    build_direct_sampler_primitives,
    build_direct_sampler_primitives_from_geometry,
    direct_sample_area_weighted_azimuth_and_interval,
    direct_sample_exact_bounded_box_point,
    evaluate_segment_density,
    evaluate_segment_partial_cdf,
    invert_segment_partial_cdf,
)

_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(
    apex=(0.0, 0.0, 0.0),
    axis=(0.0, 0.0, 1.0),
    theta=math.pi / 4,
) -> EventObj:
    axis_arr = np.asarray(axis, dtype=float)
    axis_arr = axis_arr / np.linalg.norm(axis_arr)
    return EventObj(
        apex=np.asarray(apex, dtype=float),
        axis=axis_arr,
        theta=float(theta),
    )


def _voi(xmin, xmax, ymin, ymax, zmin, zmax) -> VoiBounds:
    return VoiBounds(
        xmin=float(xmin),
        xmax=float(xmax),
        ymin=float(ymin),
        ymax=float(ymax),
        zmin=float(zmin),
        zmax=float(zmax),
    )


def _build_primitives(event, voi) -> DirectSamplerPrimitives:
    cone_geom = build_cone_local_geometry(event)
    return build_direct_sampler_primitives_from_geometry(cone_geom, voi)


def _segment_arc_length(regime) -> float:
    xi_a = regime.xi_start
    xi_b = regime.xi_end
    return (xi_b - xi_a) if xi_b > xi_a else (xi_b + _TWO_PI - xi_a)


def _segment_point(regime, frac: float) -> float:
    xi = regime.xi_start + frac * _segment_arc_length(regime)
    if xi >= _TWO_PI:
        xi -= _TWO_PI
    return xi


def _segment_contains_xi(regime, xi: float, *, atol: float = 0.0) -> bool:
    xi_a = regime.xi_start
    xi_b = regime.xi_end
    if xi_b <= xi_a:
        return xi >= xi_a - atol or xi <= xi_b + atol
    return xi_a - atol <= xi <= xi_b + atol


def _segment_offset(regime, xi: float) -> float:
    offset = xi - regime.xi_start
    if regime.xi_end <= regime.xi_start and offset < 0.0:
        offset += _TWO_PI
    return offset


def _pick_positive_segment(
    prims: DirectSamplerPrimitives,
) -> tuple[int, object]:
    for j, regime in enumerate(prims.azimuth_law.regimes):
        if regime.nonempty and prims.segment_masses[j] > 1e-12:
            return j, regime
    raise AssertionError("no positive-mass visible segment found")


def _invert_with_bisection_settings(
    monkeypatch: pytest.MonkeyPatch,
    prims: DirectSamplerPrimitives,
    j: int,
    target_mass: float,
    *,
    abs_tol: float,
    max_iter: int,
) -> float:
    with monkeypatch.context() as ctx:
        ctx.setattr(direct_surface_sampler_module, "_BISECT_ABS_TOL", abs_tol)
        ctx.setattr(direct_surface_sampler_module, "_BISECT_MAX_ITER", max_iter)
        return invert_segment_partial_cdf(prims, j, target_mass)


def _pick_visible_segment(
    prims: DirectSamplerPrimitives,
    *,
    lower_is_apex: bool,
) -> tuple[int, object]:
    for j, regime in enumerate(prims.azimuth_law.regimes):
        if (
            regime.nonempty
            and regime.lower_is_apex is lower_is_apex
            and prims.segment_masses[j] > 1e-12
        ):
            return j, regime
    kind = "floor/face" if lower_is_apex else "face/face"
    raise AssertionError(f"no positive-mass {kind} segment found")


def _expected_exact_segment_mass(prims: DirectSamplerPrimitives, j: int) -> float:
    regime = prims.azimuth_law.regimes[j]
    wraps = regime.xi_end <= regime.xi_start
    upper_int = integrate_inv_g_sq(
        regime.xi_start,
        regime.xi_end,
        regime.a_upper,
        regime.b_upper,
        regime.c_upper,
        wraps=wraps,
    )

    if regime.lower_is_apex:
        return (prims.s / 2.0) * regime.tau_upper**2 * upper_int

    lower_int = integrate_inv_g_sq(
        regime.xi_start,
        regime.xi_end,
        regime.a_lower,
        regime.b_lower,
        regime.c_lower,
        wraps=wraps,
    )
    return (prims.s / 2.0) * (
        regime.tau_upper**2 * upper_int - regime.tau_lower**2 * lower_int
    )


def _expected_exact_segment_partial_cdf(
    prims: DirectSamplerPrimitives,
    j: int,
    xi: float,
) -> float:
    regime = prims.azimuth_law.regimes[j]
    sub_wraps = (regime.xi_end <= regime.xi_start) and (xi < regime.xi_start)
    upper_int = integrate_inv_g_sq(
        regime.xi_start,
        xi,
        regime.a_upper,
        regime.b_upper,
        regime.c_upper,
        wraps=sub_wraps,
    )

    if regime.lower_is_apex:
        return (prims.s / 2.0) * regime.tau_upper**2 * upper_int

    lower_int = integrate_inv_g_sq(
        regime.xi_start,
        xi,
        regime.a_lower,
        regime.b_lower,
        regime.c_lower,
        wraps=sub_wraps,
    )
    return (prims.s / 2.0) * (
        regime.tau_upper**2 * upper_int - regime.tau_lower**2 * lower_int
    )


def _strict_open_representative_point(
    event: EventObj,
    *,
    allowed_region: VoiBounds,
    azimuths: tuple[float, ...],
) -> np.ndarray:
    """Mirror the authoritative strict-open representative-point construction."""
    apex = tuple(float(component) for component in event.apex)
    voi_box = (
        (allowed_region.xmin, allowed_region.xmax),
        (allowed_region.ymin, allowed_region.ymax),
        (allowed_region.zmin, allowed_region.zmax),
    )
    for xi in azimuths:
        direction = tuple(
            float(component) for component in cone_generator_direction(event, xi)
        )
        interval = open_box_ray_interval(apex, direction, voi_box)
        if interval is None:
            continue
        ell_minus, ell_plus = interval
        point = cone_surface_point(event, ell=0.5 * (ell_minus + ell_plus), xi=xi)
        x, y, z = point
        if (
            allowed_region.xmin < x < allowed_region.xmax
            and allowed_region.ymin < y < allowed_region.ymax
            and allowed_region.zmin < z < allowed_region.zmax
            and is_admissible_point(point, event, allowed_region)
        ):
            return point
    raise AssertionError("failed to construct strict-open representative point")


def _synthetic_regression_case() -> tuple[EventObj, VoiBounds, np.ndarray]:
    """Return an inline edge-case event without depending on private datasets."""
    event = _event(
        apex=(-8.0, 2.0, -5.0),
        axis=(0.35, -0.55, 0.76),
        theta=0.62,
    )
    allowed_region = _voi(-20.0, 20.0, -15.0, 15.0, -20.0, 20.0)
    representative_point = _strict_open_representative_point(
        event,
        allowed_region=allowed_region,
        azimuths=midpoint_azimuth_grid(16),
    )
    return event, allowed_region, representative_point


# Standard test geometries (same as azimuth-law tests)
_EVENT_A = _event()
_VOI_A = _voi(-1, 1, -1, 1, 0.5, 3)

_EVENT_B = _event(apex=(0.5, -0.3, 0.0), axis=(1, 1, 2), theta=math.pi / 6)
_VOI_B = _voi(-2, 3, -2, 3, 0, 5)

_EVENT_C = _event(apex=(0.0, 0.0, 1.0), axis=(0, 0, 1), theta=math.pi / 8)
_VOI_C = _voi(-0.5, 0.5, -0.5, 0.5, 1, 4)

# Geometry where cone clearly misses box
_EVENT_MISS = _event(apex=(0, 0, 0), axis=(0, 0, -1), theta=math.pi / 6)
_VOI_MISS = _voi(-1, 1, -1, 1, 2, 4)

# Asymmetric geometry likely to have both empty and visible segments
_EVENT_ASYM = _event(apex=(0, 0, 0), axis=(1, 0, 1), theta=math.pi / 6)
_VOI_ASYM = _voi(2, 4, -0.5, 0.5, 2, 4)


# ===================================================================
# 1. Segment masses -- nonneg, finite, zero on empty
# ===================================================================


class TestSegmentMasses:
    @pytest.fixture(
        params=[
            ("A", _EVENT_A, _VOI_A),
            ("B", _EVENT_B, _VOI_B),
            ("C", _EVENT_C, _VOI_C),
        ],
        ids=["centered", "offcenter", "narrow"],
    )
    def prims(self, request):
        _, event, voi = request.param
        return _build_primitives(event, voi)

    def test_masses_nonneg_finite(self, prims):
        for j, m in enumerate(prims.segment_masses):
            assert math.isfinite(m), f"segment {j}: mass={m} not finite"
            assert m >= -1e-14, f"segment {j}: mass={m} < 0"

    def test_empty_segments_zero_mass(self, prims):
        for j, regime in enumerate(prims.azimuth_law.regimes):
            if not regime.nonempty:
                assert prims.segment_masses[j] == 0.0

    def test_total_equals_sum_of_masses(self, prims):
        s = sum(prims.segment_masses)
        assert math.isclose(prims.total_mass, s, rel_tol=1e-12)

    def test_total_consistent_with_reduced_law(self, prims):
        """Total mass = (s/2) * reduced azimuth law total mass."""
        expected = (prims.s / 2.0) * prims.azimuth_law.total_mass
        assert math.isclose(prims.total_mass, expected, rel_tol=1e-14)

    def test_positive_total(self, prims):
        assert prims.total_mass > 0.0


class TestVisibleSegmentExactAssembly:
    @pytest.mark.parametrize(
        ("event", "voi", "lower_is_apex"),
        [
            (_EVENT_B, _VOI_B, True),
            (_EVENT_A, _VOI_A, False),
        ],
        ids=["floor-face", "face-face"],
    )
    def test_segment_mass_matches_endpoint_formula(
        self,
        event,
        voi,
        lower_is_apex: bool,
    ) -> None:
        prims = _build_primitives(event, voi)
        j, _ = _pick_visible_segment(prims, lower_is_apex=lower_is_apex)
        expected = _expected_exact_segment_mass(prims, j)
        assert math.isclose(
            prims.segment_masses[j],
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

    @pytest.mark.parametrize(
        ("event", "voi", "lower_is_apex"),
        [
            (_EVENT_B, _VOI_B, True),
            (_EVENT_A, _VOI_A, False),
        ],
        ids=["floor-face", "face-face"],
    )
    def test_partial_cdf_endpoint_identities(
        self,
        event,
        voi,
        lower_is_apex: bool,
    ) -> None:
        prims = _build_primitives(event, voi)
        j, regime = _pick_visible_segment(prims, lower_is_apex=lower_is_apex)

        c_start = evaluate_segment_partial_cdf(prims, j, regime.xi_start)
        c_end = evaluate_segment_partial_cdf(prims, j, regime.xi_end)

        assert abs(c_start) < 1e-12
        assert math.isclose(
            c_end,
            prims.segment_masses[j],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        assert math.isclose(
            c_end,
            _expected_exact_segment_mass(prims, j),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

    @pytest.mark.parametrize(
        ("event", "voi", "lower_is_apex"),
        [
            (_EVENT_B, _VOI_B, True),
            (_EVENT_A, _VOI_A, False),
        ],
        ids=["floor-face", "face-face"],
    )
    def test_partial_cdf_matches_endpoint_formula_at_interior_points(
        self,
        event,
        voi,
        lower_is_apex: bool,
    ) -> None:
        prims = _build_primitives(event, voi)
        j, regime = _pick_visible_segment(prims, lower_is_apex=lower_is_apex)

        for frac in (0.2, 0.5, 0.8):
            xi = _segment_point(regime, frac)
            observed = evaluate_segment_partial_cdf(prims, j, xi)
            expected = _expected_exact_segment_partial_cdf(prims, j, xi)
            assert math.isclose(
                observed,
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )

    def test_cached_partial_cdf_context_matches_endpoint_formula(self) -> None:
        prims = _build_primitives(_EVENT_A, _VOI_A)
        j, regime = _pick_visible_segment(prims, lower_is_apex=False)
        context = direct_surface_sampler_module._build_segment_partial_cdf_context(
            s=prims.s,
            regime=regime,
        )
        evaluate_cached = (
            direct_surface_sampler_module._evaluate_segment_partial_cdf_from_context
        )

        for frac in (0.2, 0.5, 0.8):
            xi = _segment_point(regime, frac)
            cached = evaluate_cached(context, xi)
            expected = _expected_exact_segment_partial_cdf(prims, j, xi)
            assert math.isclose(cached, expected, rel_tol=1e-12, abs_tol=1e-12)

    @pytest.mark.parametrize(
        ("event", "voi", "lower_is_apex"),
        [
            (_EVENT_B, _VOI_B, True),
            (_EVENT_A, _VOI_A, False),
        ],
        ids=["floor-face", "face-face"],
    )
    def test_partial_cdf_derivative_matches_density(
        self,
        event,
        voi,
        lower_is_apex: bool,
    ) -> None:
        prims = _build_primitives(event, voi)
        j, regime = _pick_visible_segment(prims, lower_is_apex=lower_is_apex)
        arc_len = _segment_arc_length(regime)
        h = min(1e-6, arc_len * 1e-4)
        frac_h = h / arc_len

        for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
            xi_left = _segment_point(regime, frac - frac_h)
            xi_mid = _segment_point(regime, frac)
            xi_right = _segment_point(regime, frac + frac_h)
            deriv = (
                evaluate_segment_partial_cdf(prims, j, xi_right)
                - evaluate_segment_partial_cdf(prims, j, xi_left)
            ) / (2.0 * h)
            density = evaluate_segment_density(prims, j, xi_mid)
            assert math.isclose(
                deriv,
                density,
                rel_tol=2e-5,
                abs_tol=1e-8,
            )

    @pytest.mark.parametrize(
        ("event", "voi", "lower_is_apex"),
        [
            (_EVENT_B, _VOI_B, True),
            (_EVENT_A, _VOI_A, False),
        ],
        ids=["floor-face", "face-face"],
    )
    def test_partial_cdf_strictly_increases_on_visible_segment(
        self,
        event,
        voi,
        lower_is_apex: bool,
    ) -> None:
        prims = _build_primitives(event, voi)
        j, regime = _pick_visible_segment(prims, lower_is_apex=lower_is_apex)
        prev = 0.0
        for frac in (0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9):
            val = evaluate_segment_partial_cdf(prims, j, _segment_point(regime, frac))
            assert val > prev + 1e-12
            prev = val


# ===================================================================
# 2. Cumulative masses -- monotone, correct endpoints
# ===================================================================


class TestCumulativeMasses:
    @pytest.fixture(
        params=[
            ("A", _EVENT_A, _VOI_A),
            ("B", _EVENT_B, _VOI_B),
            ("C", _EVENT_C, _VOI_C),
        ],
        ids=["centered", "offcenter", "narrow"],
    )
    def prims(self, request):
        _, event, voi = request.param
        return _build_primitives(event, voi)

    def test_starts_at_zero(self, prims):
        assert prims.cumulative_masses[0] == 0.0

    def test_monotone(self, prims):
        for j in range(1, len(prims.cumulative_masses)):
            assert (
                prims.cumulative_masses[j]
                >= prims.cumulative_masses[j - 1] - 1e-14
            )

    def test_final_entry_plus_last_mass_equals_total(self, prims):
        last_cum = prims.cumulative_masses[-1]
        last_mass = prims.segment_masses[-1]
        assert math.isclose(
            last_cum + last_mass, prims.total_mass, rel_tol=1e-12
        )


# ===================================================================
# 3. Segment density -- includes (s/2) factor
# ===================================================================


class TestSegmentDensity:
    def test_density_includes_s_factor(self):
        prims = _build_primitives(_EVENT_A, _VOI_A)
        coeff = prims.azimuth_law.decomposition.coefficients

        for j, regime in enumerate(prims.azimuth_law.regimes):
            if not regime.nonempty:
                continue
            xi_a = regime.xi_start
            xi_b = regime.xi_end
            arc_len = (
                (xi_b - xi_a) if xi_b > xi_a else (xi_b + _TWO_PI - xi_a)
            )
            xi_mid = xi_a + arc_len / 2
            if xi_mid >= _TWO_PI:
                xi_mid -= _TWO_PI

            density = evaluate_segment_density(prims, j, xi_mid)
            reduced = evaluate_area_profile(coeff, xi_mid)
            expected = (prims.s / 2.0) * reduced
            assert math.isclose(density, expected, rel_tol=1e-14)

    def test_density_zero_on_empty(self):
        prims = _build_primitives(_EVENT_A, _VOI_A)
        for j, regime in enumerate(prims.azimuth_law.regimes):
            if regime.nonempty:
                continue
            density = evaluate_segment_density(
                prims, j, regime.xi_start + 0.01
            )
            assert density == 0.0

    def test_density_nonneg(self):
        for event, voi in [(_EVENT_A, _VOI_A), (_EVENT_B, _VOI_B)]:
            prims = _build_primitives(event, voi)
            for j, regime in enumerate(prims.azimuth_law.regimes):
                if not regime.nonempty:
                    continue
                xi_a = regime.xi_start
                xi_b = regime.xi_end
                arc_len = (
                    (xi_b - xi_a) if xi_b > xi_a else (xi_b + _TWO_PI - xi_a)
                )
                for k in range(1, 11):
                    xi = xi_a + (k / 11) * arc_len
                    if xi >= _TWO_PI:
                        xi -= _TWO_PI
                    d = evaluate_segment_density(prims, j, xi)
                    assert d >= -1e-14, f"density < 0 at xi={xi}"


# ===================================================================
# 4. Numerical agreement for total mass
# ===================================================================


class TestNumericalAgreement:
    def _numerical_total_mass(self, prims, n_quad=10000):
        """Numerical quadrature of (s/2) * A_reduced over all segments.

        Uses midpoint rule to avoid evaluating at breakpoints where
        the area profile may have integrable singularities.
        """
        coeff = prims.azimuth_law.decomposition.coefficients
        half_s = prims.s / 2.0
        total = 0.0
        for regime in prims.azimuth_law.regimes:
            if not regime.nonempty:
                continue
            xi_a = regime.xi_start
            xi_b = regime.xi_end
            arc_len = (
                (xi_b - xi_a) if xi_b > xi_a else (xi_b + _TWO_PI - xi_a)
            )
            seg_sum = 0.0
            for i in range(n_quad):
                frac = (i + 0.5) / n_quad
                xi = xi_a + frac * arc_len
                if xi >= _TWO_PI:
                    xi -= _TWO_PI
                seg_sum += evaluate_area_profile(coeff, xi)
            total += half_s * seg_sum * (arc_len / n_quad)
        return total

    @pytest.mark.parametrize(
        "event,voi",
        [(_EVENT_A, _VOI_A), (_EVENT_B, _VOI_B), (_EVENT_C, _VOI_C)],
        ids=["centered", "offcenter", "narrow"],
    )
    def test_total_mass_vs_quadrature(self, event, voi):
        prims = _build_primitives(event, voi)
        numerical = self._numerical_total_mass(prims)
        rel = abs(prims.total_mass - numerical) / prims.total_mass
        assert rel < 2e-4, (
            f"total: analytic={prims.total_mass:.8e}, "
            f"numerical={numerical:.8e}"
        )


# ===================================================================
# 5. Partial CDF -- endpoints and monotonicity
# ===================================================================


class TestPartialCDF:
    @pytest.fixture(
        params=[
            ("A", _EVENT_A, _VOI_A),
            ("B", _EVENT_B, _VOI_B),
        ],
        ids=["centered", "offcenter"],
    )
    def prims(self, request):
        _, event, voi = request.param
        return _build_primitives(event, voi)

    def test_cdf_at_left_endpoint_is_zero(self, prims):
        for j, regime in enumerate(prims.azimuth_law.regimes):
            if not regime.nonempty:
                continue
            val = evaluate_segment_partial_cdf(prims, j, regime.xi_start)
            assert abs(val) < 1e-11, f"segment {j}: cdf(start)={val}"

    def test_cdf_at_right_endpoint_equals_mass(self, prims):
        for j, regime in enumerate(prims.azimuth_law.regimes):
            if not regime.nonempty:
                continue
            m = prims.segment_masses[j]
            if m < 1e-14:
                continue
            val = evaluate_segment_partial_cdf(prims, j, regime.xi_end)
            assert math.isclose(val, m, rel_tol=1e-9), (
                f"segment {j}: cdf(end)={val}, mass={m}"
            )

    def test_cdf_strictly_increases_on_visible(self, prims):
        for j, regime in enumerate(prims.azimuth_law.regimes):
            if not regime.nonempty:
                continue
            m = prims.segment_masses[j]
            if m < 1e-14:
                continue
            xi_a = regime.xi_start
            xi_b = regime.xi_end
            arc_len = (
                (xi_b - xi_a) if xi_b > xi_a else (xi_b + _TWO_PI - xi_a)
            )

            n_probes = 20
            prev = 0.0
            for i in range(1, n_probes + 1):
                frac = i / (n_probes + 1)
                xi = xi_a + frac * arc_len
                if xi >= _TWO_PI:
                    xi -= _TWO_PI
                val = evaluate_segment_partial_cdf(prims, j, xi)
                assert val >= prev - 1e-12, (
                    f"segment {j}, probe {i}: cdf={val} < prev={prev}"
                )
                prev = val

    def test_cdf_zero_on_empty(self, prims):
        for j, regime in enumerate(prims.azimuth_law.regimes):
            if regime.nonempty:
                continue
            val = evaluate_segment_partial_cdf(
                prims, j, regime.xi_start + 0.01
            )
            assert val == 0.0


# ===================================================================
# 6. Numerical inverse CDF -- residual, bracket, and stability
# ===================================================================


class TestInverseCDF:
    @pytest.fixture(
        params=[
            ("A", _EVENT_A, _VOI_A),
            ("B", _EVENT_B, _VOI_B),
            ("C", _EVENT_C, _VOI_C),
        ],
        ids=["centered", "offcenter", "narrow"],
    )
    def prims(self, request):
        _, event, voi = request.param
        return _build_primitives(event, voi)

    def test_inverse_cdf_round_trip(self, prims):
        """Numerical inversion recovers the requested segment mass tightly."""
        for j, regime in enumerate(prims.azimuth_law.regimes):
            if not regime.nonempty:
                continue
            m = prims.segment_masses[j]
            if m < 1e-14:
                continue

            for frac in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
                target = frac * m
                xi_inv = invert_segment_partial_cdf(prims, j, target)

                assert _segment_contains_xi(regime, xi_inv, atol=1e-12), (
                    f"segment {j}: xi_inv={xi_inv} escaped its inversion bracket"
                )

                cdf_at_inv = evaluate_segment_partial_cdf(prims, j, xi_inv)
                residual = abs(cdf_at_inv - target)
                residual_bound = 3e-12 * max(1.0, m)
                assert residual <= residual_bound, (
                    f"segment {j}, frac={frac}: "
                    f"residual={residual:.10e} exceeds {residual_bound:.10e}"
                )

    def test_inverse_near_endpoints(self, prims):
        """Numerical inversion remains accurate near segment boundaries."""
        for j, regime in enumerate(prims.azimuth_law.regimes):
            if not regime.nonempty:
                continue
            m = prims.segment_masses[j]
            if m < 1e-14:
                continue

            for frac in [0.001, 0.999]:
                target = frac * m
                xi_inv = invert_segment_partial_cdf(prims, j, target)
                cdf_at_inv = evaluate_segment_partial_cdf(prims, j, xi_inv)
                residual = abs(cdf_at_inv - target)
                residual_bound = 3e-12 * max(1.0, m)
                assert residual <= residual_bound, (
                    f"segment {j}, frac={frac}: "
                    f"residual={residual:.10e} exceeds {residual_bound:.10e}"
                )

    def test_inverse_monotone(self, prims):
        """Increasing target masses yield nondecreasing inverted azimuths."""
        for j, regime in enumerate(prims.azimuth_law.regimes):
            if not regime.nonempty:
                continue
            m = prims.segment_masses[j]
            if m < 1e-14:
                continue

            fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
            xis = []
            for frac in fracs:
                target = frac * m
                xi_inv = invert_segment_partial_cdf(prims, j, target)
                xis.append(_segment_offset(regime, xi_inv))

            for i in range(1, len(xis)):
                assert xis[i] >= xis[i - 1] - 1e-10, (
                    f"segment {j}: non-monotone inverse at fracs "
                    f"{fracs[i-1]}, {fracs[i]}"
                )

    def test_inverse_stays_inside_segment_bracket(self, prims):
        """Returned azimuths stay inside the selected visible-segment bracket."""
        for j, regime in enumerate(prims.azimuth_law.regimes):
            if not regime.nonempty:
                continue
            m = prims.segment_masses[j]
            if m < 1e-14:
                continue

            for frac in [0.05, 0.2, 0.5, 0.8, 0.95]:
                xi_inv = invert_segment_partial_cdf(prims, j, frac * m)
                assert _segment_contains_xi(regime, xi_inv, atol=1e-12), (
                    f"segment {j}: xi_inv={xi_inv} not in "
                    f"[{regime.xi_start}, {regime.xi_end}]"
                )
                assert 0.0 <= _segment_offset(regime, xi_inv) <= (
                    _segment_arc_length(regime) + 1e-12
                )

    def test_inverse_reuses_cached_context_inside_bisection(
        self,
        prims,
        monkeypatch,
    ):
        """The inverse loop must not call the uncached public integral path."""
        j, regime = _pick_positive_segment(prims)
        target = 0.37 * prims.segment_masses[j]

        def fail_public_integral(*args, **kwargs):
            raise AssertionError("uncached integrate_inv_g_sq was called")

        with monkeypatch.context() as ctx:
            ctx.setattr(
                direct_surface_sampler_module,
                "integrate_inv_g_sq",
                fail_public_integral,
            )
            xi_inv = invert_segment_partial_cdf(prims, j, target)

        assert _segment_contains_xi(regime, xi_inv, atol=1e-12)
        cdf_at_inv = evaluate_segment_partial_cdf(prims, j, xi_inv)
        assert abs(cdf_at_inv - target) <= (
            3e-12 * max(1.0, prims.segment_masses[j])
        )

    def test_inverse_tolerance_sensitivity_is_controlled(self, prims, monkeypatch):
        """Default settings track a tighter solve and improve over a looser one."""
        j, regime = _pick_positive_segment(prims)
        m = prims.segment_masses[j]

        for frac in [0.2, 0.5, 0.8]:
            target = frac * m
            xi_default = invert_segment_partial_cdf(prims, j, target)
            xi_loose = _invert_with_bisection_settings(
                monkeypatch,
                prims,
                j,
                target,
                abs_tol=1e-8,
                max_iter=64,
            )
            xi_tight = _invert_with_bisection_settings(
                monkeypatch,
                prims,
                j,
                target,
                abs_tol=1e-14,
                max_iter=128,
            )

            default_residual = abs(
                evaluate_segment_partial_cdf(prims, j, xi_default) - target
            )
            loose_residual = abs(
                evaluate_segment_partial_cdf(prims, j, xi_loose) - target
            )
            tight_residual = abs(
                evaluate_segment_partial_cdf(prims, j, xi_tight) - target
            )

            assert default_residual <= 3e-12 * max(1.0, m)
            assert tight_residual <= default_residual + 1e-14
            assert default_residual <= loose_residual + 1e-14
            assert _segment_offset(regime, xi_default) >= 0.0
            assert _segment_offset(regime, xi_tight) >= 0.0
            assert abs(_segment_offset(regime, xi_default) - _segment_offset(regime, xi_tight)) <= 1e-10

    def test_inverse_residual_keeps_interval_and_point_stable(self, monkeypatch):
        """Current bisection settings keep downstream interval and point construction stable."""
        event = _EVENT_B
        allowed_region = _VOI_B
        prims = _build_primitives(event, allowed_region)
        j, _ = _pick_positive_segment(prims)
        coeff = prims.azimuth_law.decomposition.coefficients

        for frac in [0.2, 0.5, 0.8]:
            target = frac * prims.segment_masses[j]
            xi_default = invert_segment_partial_cdf(prims, j, target)
            xi_tight = _invert_with_bisection_settings(
                monkeypatch,
                prims,
                j,
                target,
                abs_tol=1e-14,
                max_iter=128,
            )

            iv_default = evaluate_fixed_azimuth(coeff, xi_default)
            iv_tight = evaluate_fixed_azimuth(coeff, xi_tight)
            assert iv_default.nonempty
            assert iv_tight.nonempty

            assert abs(iv_default.ell_minus - iv_tight.ell_minus) <= 1e-10
            assert abs(iv_default.ell_plus - iv_tight.ell_plus) <= 1e-10
            assert abs(iv_default.area_profile - iv_tight.area_profile) <= 1e-9

            u_rad = 0.37
            ell_default = math.sqrt(
                iv_default.ell_minus * iv_default.ell_minus
                + u_rad * iv_default.area_profile
            )
            ell_tight = math.sqrt(
                iv_tight.ell_minus * iv_tight.ell_minus
                + u_rad * iv_tight.area_profile
            )
            point_default = cone_surface_point(event, ell=ell_default, xi=xi_default)
            point_tight = cone_surface_point(event, ell=ell_tight, xi=xi_tight)

            assert is_admissible_point(point_default, event, allowed_region)
            assert is_admissible_point(point_tight, event, allowed_region)
            assert np.linalg.norm(point_default - point_tight) <= 1e-9


# ===================================================================
# 7. Fail-closed behavior
# ===================================================================


class TestFailClosed:
    def test_empty_segment_raises(self):
        """Inverting on an empty segment must fail closed."""
        # Use a geometry with definite empty segments (cone misses box)
        prims = _build_primitives(_EVENT_MISS, _VOI_MISS)
        assert prims.total_mass == 0.0
        for j, regime in enumerate(prims.azimuth_law.regimes):
            if not regime.nonempty:
                with pytest.raises(ValueError, match="empty segment"):
                    invert_segment_partial_cdf(prims, j, 0.1)
                return
        # Fallback: try standard geometry
        prims2 = _build_primitives(_EVENT_A, _VOI_A)
        for j, regime in enumerate(prims2.azimuth_law.regimes):
            if not regime.nonempty:
                with pytest.raises(ValueError, match="empty segment"):
                    invert_segment_partial_cdf(prims2, j, 0.1)
                return
        pytest.skip("No empty segment found in test geometries")

    def test_target_zero_raises(self):
        prims = _build_primitives(_EVENT_A, _VOI_A)
        for j, regime in enumerate(prims.azimuth_law.regimes):
            if not regime.nonempty:
                continue
            m = prims.segment_masses[j]
            if m < 1e-14:
                continue
            with pytest.raises(ValueError, match="outside"):
                invert_segment_partial_cdf(prims, j, 0.0)
            break

    def test_target_equals_mass_raises(self):
        prims = _build_primitives(_EVENT_A, _VOI_A)
        for j, regime in enumerate(prims.azimuth_law.regimes):
            if not regime.nonempty:
                continue
            m = prims.segment_masses[j]
            if m < 1e-14:
                continue
            with pytest.raises(ValueError, match="outside"):
                invert_segment_partial_cdf(prims, j, m)
            break

    def test_target_negative_raises(self):
        prims = _build_primitives(_EVENT_A, _VOI_A)
        for j, regime in enumerate(prims.azimuth_law.regimes):
            if not regime.nonempty:
                continue
            m = prims.segment_masses[j]
            if m < 1e-14:
                continue
            with pytest.raises(ValueError, match="outside"):
                invert_segment_partial_cdf(prims, j, -0.1)
            break

    def test_target_exceeds_mass_raises(self):
        prims = _build_primitives(_EVENT_A, _VOI_A)
        for j, regime in enumerate(prims.azimuth_law.regimes):
            if not regime.nonempty:
                continue
            m = prims.segment_masses[j]
            if m < 1e-14:
                continue
            with pytest.raises(ValueError, match="outside"):
                invert_segment_partial_cdf(prims, j, m + 0.1)
            break

    def test_degenerate_cone_raises(self):
        """s <= 0 must fail closed."""
        cone_geom = build_cone_local_geometry(_EVENT_A)
        decomp = build_cone_box_decomposition(cone_geom, _VOI_A)
        with pytest.raises(ValueError, match="positive"):
            build_direct_sampler_primitives(decomp, 0.0)
        with pytest.raises(ValueError, match="positive"):
            build_direct_sampler_primitives(decomp, -1.0)

    def test_cone_misses_box_zero_total(self):
        """Cone pointing away from box: total mass = 0, all masses = 0."""
        prims = _build_primitives(_EVENT_MISS, _VOI_MISS)
        assert prims.total_mass == 0.0
        for m in prims.segment_masses:
            assert m == 0.0


# ===================================================================
# 8. Default rejection-backend unchanged
# ===================================================================


class TestDirectSamplerNowWorks:
    def test_direct_sampler_produces_valid_triple(self):
        """The direct draw function must return a valid (xi, interval, ell2_width) triple."""
        event = EventObj(
            apex=np.asarray([0.0, 0.0, 0.0], dtype=float),
            axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
            theta=math.pi / 4.0,
        )
        voi = VoiBounds(xmin=-1, xmax=2, ymin=-1, ymax=1, zmin=0, zmax=2)

        xi, interval, ell2_width = direct_sample_area_weighted_azimuth_and_interval(
            event=event,
            allowed_region=voi,
            rng=np.random.default_rng(0),
        )
        assert 0.0 <= xi < _TWO_PI
        ell_minus, ell_plus = interval
        assert 0.0 <= ell_minus < ell_plus
        assert ell2_width > 0.0
        assert math.isclose(
            ell2_width,
            ell_plus * ell_plus - ell_minus * ell_minus,
            rel_tol=1e-12,
        )


# ===================================================================
# 9. Convenience builder
# ===================================================================


class TestConvenienceBuilder:
    def test_from_geometry_matches_manual(self):
        cone_geom = build_cone_local_geometry(_EVENT_A)
        decomp = build_cone_box_decomposition(cone_geom, _VOI_A)

        p1 = build_direct_sampler_primitives(decomp, cone_geom.s)
        p2 = build_direct_sampler_primitives_from_geometry(cone_geom, _VOI_A)

        assert math.isclose(p1.total_mass, p2.total_mass, rel_tol=1e-14)
        assert len(p1.segment_masses) == len(p2.segment_masses)
        for a, b in zip(p1.segment_masses, p2.segment_masses):
            assert math.isclose(a, b, rel_tol=1e-14)


class TestSyntheticEdgeRegression:
    def test_previously_negative_visible_segment_mass_is_now_positive(self):
        event, allowed_region, representative_point = _synthetic_regression_case()
        assert is_admissible_point(representative_point, event, allowed_region)

        prims = build_direct_sampler_primitives_from_geometry(
            build_cone_local_geometry(event),
            allowed_region,
        )

        assert prims.segment_masses[0] > 0.0
        assert all(m >= 0.0 for m in prims.segment_masses)

        visible_indices = [
            j for j, regime in enumerate(prims.azimuth_law.regimes) if regime.nonempty
        ]
        assert visible_indices
        assert all(prims.segment_masses[j] > 0.0 for j in visible_indices)

        for j in range(1, len(prims.cumulative_masses)):
            assert prims.cumulative_masses[j] >= prims.cumulative_masses[j - 1]

        assert prims.total_mass > 0.0

    def test_synthetic_event_direct_sampler_draws_remain_admissible(self):
        event, allowed_region, _ = _synthetic_regression_case()
        cone_geom = build_cone_local_geometry(event)
        decomp = build_cone_box_decomposition(cone_geom, allowed_region)
        rng = np.random.default_rng(20260408)

        for _ in range(16):
            xi, interval, ell2_width = direct_sample_area_weighted_azimuth_and_interval(
                event=event,
                allowed_region=allowed_region,
                rng=rng,
                _decomposition=decomp,
            )
            ell_minus, ell_plus = interval
            iv = evaluate_fixed_azimuth(decomp.coefficients, xi)

            assert 0.0 <= xi < _TWO_PI
            assert iv.nonempty
            assert ell2_width > 0.0
            assert math.isclose(iv.ell_minus, ell_minus, rel_tol=1e-12)
            assert math.isclose(iv.ell_plus, ell_plus, rel_tol=1e-12)
            assert math.isclose(iv.area_profile, ell2_width, rel_tol=1e-12)

            point = direct_sample_exact_bounded_box_point(
                event=event,
                allowed_region=allowed_region,
                rng=rng,
                _decomposition=decomp,
            )
            assert is_admissible_point(point, event, allowed_region)
