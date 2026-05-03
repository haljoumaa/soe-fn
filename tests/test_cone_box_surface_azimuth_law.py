"""Deterministic tests for exact non-live azimuth-law machinery."""

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
from soe.geometry.cone_box_surface import (
    build_cone_box_coefficients,
    build_cone_box_decomposition,
    build_decomposition,
    evaluate_area_profile,
    evaluate_fixed_azimuth,
)
from soe.geometry.cone_box_azimuth_law import (
    AzimuthLaw,
    SegmentRegime,
    build_azimuth_law,
    build_azimuth_law_from_geometry,
    compute_segment_mass,
    evaluate_primitive,
    integrate_inv_g_sq,
    partial_cdf,
    segment_area_profile,
    _count_half_tangent_crossings,
    _build_inv_g_sq_primitive_context,
    _integrate_inv_g_sq_from_context,
    _primitive_at_u,
    _trig_params,
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


def _build_law(event, voi):
    """Build an AzimuthLaw for the given event and VOI."""
    cone_geom = build_cone_local_geometry(event)
    return build_azimuth_law_from_geometry(cone_geom, voi)


def _numerical_segment_mass(law, j, n_quad=8000):
    """Numerical integration of A_j(xi) over segment j using trapezoidal rule."""
    regime = law.regimes[j]
    if not regime.nonempty:
        return 0.0
    xi_a = regime.xi_start
    xi_b = regime.xi_end
    if xi_b <= xi_a:
        arc_len = xi_b + _TWO_PI - xi_a
    else:
        arc_len = xi_b - xi_a
    total = 0.0
    coeff = law.decomposition.coefficients
    for i in range(n_quad + 1):
        frac = i / n_quad
        xi = xi_a + frac * arc_len
        if xi >= _TWO_PI:
            xi -= _TWO_PI
        a_val = evaluate_area_profile(coeff, xi)
        w = 0.5 if (i == 0 or i == n_quad) else 1.0
        total += w * a_val
    return total * (arc_len / n_quad)


def _numerical_inv_g_sq(
    xi_a: float,
    xi_b: float,
    a: float,
    b: float,
    c: float,
    *,
    wraps: bool = False,
    n_quad: int = 200000,
) -> float:
    """Midpoint-rule quadrature for ``1 / (a + b cos xi + c sin xi)^2``."""
    arc_len = (xi_b + _TWO_PI - xi_a) if wraps else (xi_b - xi_a)
    xis = xi_a + ((np.arange(n_quad, dtype=float) + 0.5) / n_quad) * arc_len
    xis %= _TWO_PI
    g = a + b * np.cos(xis) + c * np.sin(xis)
    return float(np.mean(1.0 / (g * g)) * arc_len)


# ---------------------------------------------------------------------------
# Standard test geometries
# ---------------------------------------------------------------------------

# Apex at origin, axis along +z, theta = pi/4, centered box
_EVENT_A = _event()
_VOI_A = _voi(-1, 1, -1, 1, 0.5, 3)

# Off-center apex, tilted axis, larger box
_EVENT_B = _event(apex=(0.5, -0.3, 0.0), axis=(1, 1, 2), theta=math.pi / 6)
_VOI_B = _voi(-2, 3, -2, 3, 0, 5)

# Apex inside box, narrow cone
_EVENT_C = _event(apex=(0.0, 0.0, 1.0), axis=(0, 0, 1), theta=math.pi / 8)
_VOI_C = _voi(-0.5, 0.5, -0.5, 0.5, 1, 4)


# ===================================================================
# 1. Primitive evaluator — known integrals
# ===================================================================


class TestEvaluatePrimitive:
    """Verify the exact primitive J against known closed-form integrals."""

    def test_constant_g(self):
        """g = a (constant): integral of 1/a^2 over [0, L] = L/a^2."""
        a = 3.0
        for L in [0.5, 1.0, math.pi, 2.0]:
            result = integrate_inv_g_sq(0.0, L, a, 0.0, 0.0)
            expected = L / (a * a)
            assert math.isclose(result, expected, rel_tol=1e-10), (
                f"L={L}: got {result}, expected {expected}"
            )

    def test_full_period_known_formula(self):
        """∫_0^{2pi} dxi / (a + b cos xi)^2 = 2*pi*a / (a^2 - b^2)^{3/2}."""
        cases = [
            (2.0, 1.0, 0.0),
            (3.0, 0.5, 0.0),
            (5.0, 2.0, 0.0),
            (2.0, 0.0, 1.0),  # sin term only
            (3.0, 1.0, 1.5),  # mixed
        ]
        for a, b, c in cases:
            R_sq = b * b + c * c
            Delta = a * a - R_sq
            assert Delta > 0, "Formula requires a > R"
            expected = 2.0 * math.pi * a / (Delta * math.sqrt(Delta))
            # Use the single-segment (0, 2pi) form:
            coeff_obj = type(
                "FakeCoeff",
                (),
                {
                    "alpha": (a, 0, 0),
                    "beta": (b, 0, 0),
                    "gamma": (c, 0, 0),
                    "tau": (0, 0, 0, 0, 0, 0),
                    "apex": (0, 0, 0),
                },
            )
            # Just use integrate_inv_g_sq over full period:
            result = integrate_inv_g_sq(0.0, _TWO_PI, a, b, c)
            assert math.isclose(result, expected, rel_tol=1e-9), (
                f"a={a}, b={b}, c={c}: got {result}, expected {expected}"
            )

    def test_half_period_symmetry(self):
        """For g = a + b cos xi, ∫_0^{pi} = ∫_{pi}^{2pi} (symmetry)."""
        a, b = 4.0, 1.5
        I1 = integrate_inv_g_sq(0.0, math.pi, a, b, 0.0)
        I2 = integrate_inv_g_sq(math.pi, _TWO_PI, a, b, 0.0)
        assert math.isclose(I1, I2, rel_tol=1e-10)

    def test_primitive_at_zero_is_zero(self):
        """J(0; a, b, 0) should be 0 (u = tan(0) = 0)."""
        for a, b in [(2.0, 1.0), (5.0, 3.0)]:
            J0 = evaluate_primitive(0.0, a, b, 0.0)
            assert abs(J0) < 1e-14, f"J(0)={J0}"


class TestDeltaPositiveRegression:
    """Focused regression coverage for the proved ``Delta > 0`` branch."""

    @pytest.mark.parametrize(
        ("a", "b", "c", "u_values"),
        [
            (2.5, 1.0, 0.5, (-1.5, -0.25, 0.0, 0.3, 2.0)),
            # A0 < 0: old atan2-based primitive had a false branch cut at u = 0.
            (-2.5, 1.0, 0.5, (-1.5, -0.25, 0.0, 0.3, 2.0)),
        ],
    )
    def test_delta_positive_primitive_derivative_matches_integrand(
        self,
        a: float,
        b: float,
        c: float,
        u_values: tuple[float, ...],
    ) -> None:
        R, phi, A0, B0, Delta = _trig_params(a, b, c)
        assert Delta > 0.0
        for u in u_values:
            h = 1e-7 if abs(u) < 0.1 else 1e-6
            deriv = (
                _primitive_at_u(u + h, a, R, A0, B0, Delta)
                - _primitive_at_u(u - h, a, R, A0, B0, Delta)
            ) / (2.0 * h)
            expected = 2.0 * (1.0 + u * u) / ((A0 + B0 * u * u) ** 2)
            assert math.isclose(deriv, expected, rel_tol=5e-7, abs_tol=5e-8), (
                f"a={a}, b={b}, c={c}, u={u}: deriv={deriv}, expected={expected}"
            )

    def test_delta_positive_integral_matches_quadrature_without_crossing(self) -> None:
        a, b, c = -2.5, 1.0, 0.5
        xi_a = 0.2
        xi_b = 1.3
        analytic = integrate_inv_g_sq(xi_a, xi_b, a, b, c)
        numerical = _numerical_inv_g_sq(xi_a, xi_b, a, b, c)
        assert math.isclose(analytic, numerical, rel_tol=2e-7, abs_tol=1e-8)

    def test_delta_positive_integral_matches_quadrature_with_crossing(self) -> None:
        # phi = 0 here, so this arc crosses the half-tangent point t = pi.
        a, b, c = -2.5, 1.0, 0.0
        xi_a = 2.0
        xi_b = 4.2
        analytic = integrate_inv_g_sq(xi_a, xi_b, a, b, c)
        numerical = _numerical_inv_g_sq(xi_a, xi_b, a, b, c)
        assert math.isclose(analytic, numerical, rel_tol=2e-7, abs_tol=1e-8)

    def test_delta_negative_branch_still_matches_quadrature(self) -> None:
        a, b, c = 0.5, 1.0, 0.0
        xi_a = 0.2
        xi_b = 1.0
        analytic = integrate_inv_g_sq(xi_a, xi_b, a, b, c)
        numerical = _numerical_inv_g_sq(xi_a, xi_b, a, b, c)
        assert math.isclose(analytic, numerical, rel_tol=2e-7, abs_tol=1e-8)

    def test_delta_zero_branch_still_matches_quadrature(self) -> None:
        a, b, c = 1.0, 1.0, 0.0
        xi_a = 0.2
        xi_b = 1.0
        analytic = integrate_inv_g_sq(xi_a, xi_b, a, b, c)
        numerical = _numerical_inv_g_sq(xi_a, xi_b, a, b, c)
        assert math.isclose(analytic, numerical, rel_tol=2e-7, abs_tol=1e-8)


# ===================================================================
# 2. Crossing count
# ===================================================================


class TestCrossingCount:
    def test_no_crossing(self):
        assert _count_half_tangent_crossings(0.0, 2.0) == 0

    def test_one_crossing(self):
        assert _count_half_tangent_crossings(0.0, _TWO_PI) == 1
        assert _count_half_tangent_crossings(2.0, 5.0) == 1  # pi ~ 3.14 in range

    def test_range_below_pi(self):
        assert _count_half_tangent_crossings(0.5, 3.0) == 0

    def test_wrapping_extra(self):
        # From near 2pi to near 0 (wrapping adds 2pi to t_b)
        # 5.0 to 7.28: pi ~ 3.14 not in range, 3pi ~ 9.42 not in range.
        # So 0 crossings.
        assert _count_half_tangent_crossings(5.0, 1.0 + _TWO_PI) == 0


# ===================================================================
# 3. integrate_inv_g_sq — wrapping
# ===================================================================


class TestIntegrateWrapping:
    def test_split_equals_wrap(self):
        """Non-wrapping split at 2pi/3 should match wrapping integral."""
        a, b, c = 3.0, 1.0, 0.5
        xi_a = 4.5
        xi_b = 1.0
        # Wrapping integral
        I_wrap = integrate_inv_g_sq(xi_a, xi_b, a, b, c, wraps=True)
        # Split: xi_a to 2pi, then 0 to xi_b
        I1 = integrate_inv_g_sq(xi_a, _TWO_PI, a, b, c)
        I2 = integrate_inv_g_sq(0.0, xi_b, a, b, c)
        assert math.isclose(I_wrap, I1 + I2, rel_tol=1e-10), (
            f"wrap={I_wrap}, split={I1 + I2}"
        )

    def test_full_circle_via_wrap(self):
        """Wrapping from xi to xi should give the full-period integral."""
        a = 2.5
        xi = 1.0
        I_wrap = integrate_inv_g_sq(xi, xi, a, 0.0, 0.0, wraps=True)
        expected = _TWO_PI / (a * a)
        assert math.isclose(I_wrap, expected, rel_tol=1e-10)


class TestCachedPrimitiveContext:
    @pytest.mark.parametrize(
        ("xi_a", "xi_b", "a", "b", "c", "wraps"),
        [
            (0.2, 1.3, -2.5, 1.0, 0.5, False),
            (2.0, 4.2, -2.5, 1.0, 0.0, False),
            (0.2, 1.0, 0.5, 1.0, 0.0, False),
            (0.2, 1.0, 1.0, 1.0, 0.0, False),
            (4.5, 1.0, 3.0, 1.0, 0.5, True),
        ],
        ids=[
            "delta-positive-no-crossing",
            "delta-positive-crossing",
            "delta-negative",
            "delta-zero",
            "wrap",
        ],
    )
    def test_cached_integral_matches_public_analytic_path(
        self,
        xi_a: float,
        xi_b: float,
        a: float,
        b: float,
        c: float,
        wraps: bool,
    ) -> None:
        context = _build_inv_g_sq_primitive_context(xi_a, a, b, c)
        cached = _integrate_inv_g_sq_from_context(xi_b, context, wraps=wraps)
        public = integrate_inv_g_sq(xi_a, xi_b, a, b, c, wraps=wraps)
        assert math.isclose(cached, public, rel_tol=0.0, abs_tol=1e-14)


# ===================================================================
# 4. Segment masses — basic properties
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
    def law(self, request):
        _, event, voi = request.param
        return _build_law(event, voi)

    def test_masses_nonneg(self, law):
        for j, m in enumerate(law.segment_masses):
            assert m >= -1e-14, f"segment {j}: mass={m} < 0"

    def test_total_equals_sum(self, law):
        s = sum(law.segment_masses)
        assert math.isclose(law.total_mass, s, rel_tol=1e-12), (
            f"total={law.total_mass}, sum={s}"
        )

    def test_cumulative_monotone(self, law):
        for j in range(1, len(law.cumulative_masses)):
            assert law.cumulative_masses[j] >= law.cumulative_masses[j - 1] - 1e-14

    def test_cumulative_starts_at_zero(self, law):
        assert law.cumulative_masses[0] == 0.0

    def test_empty_segments_zero_mass(self, law):
        for j, regime in enumerate(law.regimes):
            if not regime.nonempty:
                assert law.segment_masses[j] == 0.0

    def test_positive_total(self, law):
        """At least some cone-box intersection exists for these geometries."""
        assert law.total_mass > 0.0


# ===================================================================
# 5. Numerical agreement
# ===================================================================


class TestNumericalAgreement:
    @pytest.fixture(
        params=[
            ("A", _EVENT_A, _VOI_A),
            ("B", _EVENT_B, _VOI_B),
            ("C", _EVENT_C, _VOI_C),
        ],
        ids=["centered", "offcenter", "narrow"],
    )
    def law(self, request):
        _, event, voi = request.param
        return _build_law(event, voi)

    def test_segment_mass_vs_quadrature(self, law):
        """Each nonempty segment mass matches fine trapezoidal quadrature."""
        for j, regime in enumerate(law.regimes):
            if not regime.nonempty:
                continue
            analytic = law.segment_masses[j]
            numerical = _numerical_segment_mass(law, j, n_quad=10000)
            assert analytic > 0 or numerical < 1e-12
            if analytic > 1e-14:
                rel = abs(analytic - numerical) / analytic
                assert rel < 2e-4, (
                    f"segment {j}: analytic={analytic:.8e}, "
                    f"numerical={numerical:.8e}, rel={rel:.4e}"
                )

    def test_total_mass_vs_quadrature(self, law):
        total_num = sum(
            _numerical_segment_mass(law, j, n_quad=10000)
            for j in range(len(law.regimes))
        )
        rel = abs(law.total_mass - total_num) / law.total_mass
        assert rel < 2e-4, (
            f"total: analytic={law.total_mass:.8e}, numerical={total_num:.8e}"
        )


# ===================================================================
# 6. Partial CDF
# ===================================================================


class TestPartialCDF:
    @pytest.fixture(
        params=[
            ("A", _EVENT_A, _VOI_A),
            ("B", _EVENT_B, _VOI_B),
        ],
        ids=["centered", "offcenter"],
    )
    def law(self, request):
        _, event, voi = request.param
        return _build_law(event, voi)

    def test_cdf_at_start_is_zero(self, law):
        for j, regime in enumerate(law.regimes):
            if not regime.nonempty:
                continue
            val = partial_cdf(law, j, regime.xi_start)
            assert abs(val) < 1e-12, f"segment {j}: cdf(start)={val}"

    def test_cdf_at_end_equals_mass(self, law):
        for j, regime in enumerate(law.regimes):
            if not regime.nonempty:
                continue
            val = partial_cdf(law, j, regime.xi_end)
            m = law.segment_masses[j]
            if m > 1e-14:
                assert math.isclose(val, m, rel_tol=1e-9), (
                    f"segment {j}: cdf(end)={val}, mass={m}"
                )

    def test_cdf_monotone(self, law):
        """Partial CDF is monotone at uniformly spaced probe points."""
        for j, regime in enumerate(law.regimes):
            if not regime.nonempty:
                continue
            xi_a = regime.xi_start
            xi_b = regime.xi_end
            if xi_b <= xi_a:
                arc_len = xi_b + _TWO_PI - xi_a
            else:
                arc_len = xi_b - xi_a

            n_probes = 20
            prev = 0.0
            for i in range(1, n_probes + 1):
                frac = i / (n_probes + 1)
                xi = xi_a + frac * arc_len
                if xi >= _TWO_PI:
                    xi -= _TWO_PI
                val = partial_cdf(law, j, xi)
                assert val >= prev - 1e-12, (
                    f"segment {j}, probe {i}: cdf={val} < prev={prev}"
                )
                prev = val


# ===================================================================
# 7. Empty / degenerate cases
# ===================================================================


class TestEmptyCases:
    def test_all_empty_when_cone_misses_box(self):
        """Cone points away from box: all segments empty, total mass = 0."""
        event = _event(apex=(0, 0, 0), axis=(0, 0, -1), theta=math.pi / 6)
        voi = _voi(-1, 1, -1, 1, 2, 4)  # box above apex, cone points down
        law = _build_law(event, voi)
        assert law.total_mass == 0.0
        for m in law.segment_masses:
            assert m == 0.0

    def test_apex_inside_box_has_apex_lower(self):
        """When apex is inside box, some segments should have lower_is_apex=True."""
        event = _event(apex=(0, 0, 0.5))
        voi = _voi(-1, 1, -1, 1, 0, 2)
        law = _build_law(event, voi)
        has_apex = any(r.lower_is_apex and r.nonempty for r in law.regimes)
        assert has_apex, "Expected at least one apex-lower segment"


# ===================================================================
# 8. Grazing / breakpoint-adjacent — no broadening
# ===================================================================


class TestNoBroadening:
    def test_values_stable_near_breakpoints(self):
        """Area profile and partial CDF evaluated just inside segment
        boundaries should be consistent (no spurious broadening)."""
        law = _build_law(_EVENT_A, _VOI_A)
        eps = 1e-8
        for j, regime in enumerate(law.regimes):
            if not regime.nonempty:
                continue
            # Probe just inside the left endpoint
            xi_a = regime.xi_start
            xi_b = regime.xi_end
            if xi_b <= xi_a:
                arc_len = xi_b + _TWO_PI - xi_a
            else:
                arc_len = xi_b - xi_a
            if arc_len < 4 * eps:
                continue  # skip very short segments

            xi_left = xi_a + eps
            if xi_left >= _TWO_PI:
                xi_left -= _TWO_PI
            xi_right = xi_a + arc_len - eps
            if xi_right >= _TWO_PI:
                xi_right -= _TWO_PI

            # Area profile should be finite and nonneg
            a_left = segment_area_profile(law, j, xi_left)
            a_right = segment_area_profile(law, j, xi_right)
            assert math.isfinite(a_left) and a_left >= -1e-14
            assert math.isfinite(a_right) and a_right >= -1e-14

            # Partial CDF should be finite, nonneg, and monotone
            c_left = partial_cdf(law, j, xi_left)
            c_right = partial_cdf(law, j, xi_right)
            assert math.isfinite(c_left) and c_left >= -1e-12
            assert math.isfinite(c_right) and c_right >= c_left - 1e-12


# ===================================================================
# 9. Segment area profile convenience
# ===================================================================


class TestSegmentAreaProfile:
    def test_matches_substrate_evaluator(self):
        law = _build_law(_EVENT_A, _VOI_A)
        coeff = law.decomposition.coefficients
        for j, regime in enumerate(law.regimes):
            if not regime.nonempty:
                continue
            xi_a = regime.xi_start
            xi_b = regime.xi_end
            arc_len = (xi_b - xi_a) if xi_b > xi_a else (xi_b + _TWO_PI - xi_a)
            xi_mid = xi_a + arc_len / 2
            if xi_mid >= _TWO_PI:
                xi_mid -= _TWO_PI
            from_law = segment_area_profile(law, j, xi_mid)
            from_substrate = evaluate_area_profile(coeff, xi_mid)
            assert math.isclose(from_law, from_substrate, rel_tol=1e-14)


# ===================================================================
# 10. build_azimuth_law_from_geometry convenience
# ===================================================================


class TestConvenienceBuilder:
    def test_roundtrip(self):
        """build_azimuth_law_from_geometry produces the same law."""
        event = _EVENT_A
        voi = _VOI_A
        cone_geom = build_cone_local_geometry(event)
        law1 = build_azimuth_law_from_geometry(cone_geom, voi)
        decomp = build_cone_box_decomposition(cone_geom, voi)
        law2 = build_azimuth_law(decomp)
        assert math.isclose(law1.total_mass, law2.total_mass, rel_tol=1e-14)
        assert len(law1.segment_masses) == len(law2.segment_masses)
