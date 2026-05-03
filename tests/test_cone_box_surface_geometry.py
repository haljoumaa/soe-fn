"""Deterministic tests for exact cone-box surface geometry substrate."""

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
from soe.geometry.cone import (
    build_cone_local_geometry,
    cone_generator_direction_from_local_geometry,
)
from soe.geometry.ray_box import open_box_ray_interval
from soe.geometry.cone_box_surface import (
    FACE_NAMES,
    AzimuthSegment,
    ConeBoxCoefficients,
    ConeBoxSurfaceDecomposition,
    FixedAzimuthInterval,
    build_cone_box_coefficients,
    build_cone_box_decomposition,
    build_decomposition,
    compute_breakpoints,
    evaluate_area_profile,
    evaluate_fixed_azimuth,
    solve_linear_trig_zeros,
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


def _ray_box_reference(cone_geom, voi_bounds, xi):
    """Get (ell_minus, ell_plus) from the existing strict-open ray_box path."""
    direction = cone_generator_direction_from_local_geometry(cone_geom, xi)
    box_bounds = (
        (voi_bounds.xmin, voi_bounds.xmax),
        (voi_bounds.ymin, voi_bounds.ymax),
        (voi_bounds.zmin, voi_bounds.zmax),
    )
    return open_box_ray_interval(
        tuple(float(x) for x in cone_geom.apex),
        tuple(float(x) for x in direction),
        box_bounds,
    )


# ===================================================================
# 1. Trig solver
# ===================================================================


class TestSolveLinearTrigZeros:
    """Exact closed-form solver for A + B cos(xi) + C sin(xi) = 0."""

    def test_cos_zero(self):
        # cos(xi) = 0  =>  xi = pi/2, 3pi/2
        roots = solve_linear_trig_zeros(0.0, 1.0, 0.0)
        assert len(roots) == 2
        assert math.isclose(roots[0], math.pi / 2, abs_tol=1e-12)
        assert math.isclose(roots[1], 3 * math.pi / 2, abs_tol=1e-12)

    def test_sin_zero(self):
        # sin(xi) = 0  =>  xi = 0, pi
        roots = solve_linear_trig_zeros(0.0, 0.0, 1.0)
        assert len(roots) == 2
        assert math.isclose(roots[0], 0.0, abs_tol=1e-12)
        assert math.isclose(roots[1], math.pi, abs_tol=1e-12)

    def test_single_solution_at_zero(self):
        # 1 - cos(xi) = 0  =>  cos(xi) = 1  =>  xi = 0
        roots = solve_linear_trig_zeros(1.0, -1.0, 0.0)
        assert len(roots) == 1
        assert math.isclose(roots[0], 0.0, abs_tol=1e-12)

    def test_single_solution_at_pi(self):
        # 1 + cos(xi) = 0  =>  cos(xi) = -1  =>  xi = pi
        roots = solve_linear_trig_zeros(1.0, 1.0, 0.0)
        assert len(roots) == 1
        assert math.isclose(roots[0], math.pi, abs_tol=1e-12)

    def test_no_solution(self):
        # 2 + cos(xi) = 0  =>  cos(xi) = -2, impossible
        assert solve_linear_trig_zeros(2.0, 1.0, 0.0) == []

    def test_degenerate_all_zero(self):
        assert solve_linear_trig_zeros(0.0, 0.0, 0.0) == []

    def test_mixed_equation(self):
        # 1 + cos(xi) + sin(xi) = 0  =>  xi = pi, 3pi/2
        roots = solve_linear_trig_zeros(1.0, 1.0, 1.0)
        assert len(roots) == 2
        # Verify each root satisfies the equation
        for xi in roots:
            val = 1.0 + math.cos(xi) + math.sin(xi)
            assert abs(val) < 1e-10, f"xi={xi}, val={val}"

    def test_solutions_sorted(self):
        # Arbitrary equation: check sorted
        for A, B, C in [(0.3, -0.7, 0.5), (-0.1, 0.9, -0.4)]:
            roots = solve_linear_trig_zeros(A, B, C)
            assert roots == sorted(roots)

    def test_solutions_in_range(self):
        for A, B, C in [(0.0, 1.0, 0.0), (0.5, -0.3, 0.8), (0.0, 0.0, 1.0)]:
            for xi in solve_linear_trig_zeros(A, B, C):
                assert 0.0 <= xi < _TWO_PI

    def test_solutions_satisfy_equation(self):
        for A, B, C in [
            (0.3, -0.7, 0.5),
            (-0.1, 0.9, -0.4),
            (0.0, 1.0, 1.0),
            (1.0, -1.0, 0.0),
        ]:
            for xi in solve_linear_trig_zeros(A, B, C):
                val = A + B * math.cos(xi) + C * math.sin(xi)
                assert abs(val) < 1e-10, f"A={A}, B={B}, C={C}, xi={xi}, val={val}"


# ===================================================================
# 2. Coefficient construction
# ===================================================================


class TestConeBoxCoefficients:
    """Verify coefficient values for a known axis-aligned configuration."""

    def _setup(self):
        event = _event(apex=(0, 0, 0), axis=(0, 0, 1), theta=math.pi / 4)
        voi = _voi(-1, 1, -1, 1, 0, 2)
        geom = build_cone_local_geometry(event)
        coeff = build_cone_box_coefficients(geom, voi)
        return geom, coeff

    def test_alpha_beta_gamma(self):
        geom, coeff = self._setup()
        lam = geom.lambda_
        s = geom.s
        # u = (0,0,1), p = (1,0,0), q = (0,1,0)
        assert math.isclose(coeff.alpha[0], 0.0, abs_tol=1e-14)
        assert math.isclose(coeff.alpha[1], 0.0, abs_tol=1e-14)
        assert math.isclose(coeff.alpha[2], lam, abs_tol=1e-14)
        assert math.isclose(coeff.beta[0], s, abs_tol=1e-14)
        assert math.isclose(coeff.beta[1], 0.0, abs_tol=1e-14)
        assert math.isclose(coeff.beta[2], 0.0, abs_tol=1e-14)
        assert math.isclose(coeff.gamma[0], 0.0, abs_tol=1e-14)
        assert math.isclose(coeff.gamma[1], s, abs_tol=1e-14)
        assert math.isclose(coeff.gamma[2], 0.0, abs_tol=1e-14)

    def test_tau(self):
        _, coeff = self._setup()
        assert math.isclose(coeff.tau[0], -1.0, abs_tol=1e-14)  # x_lo
        assert math.isclose(coeff.tau[1], 1.0, abs_tol=1e-14)   # x_hi
        assert math.isclose(coeff.tau[2], -1.0, abs_tol=1e-14)  # y_lo
        assert math.isclose(coeff.tau[3], 1.0, abs_tol=1e-14)   # y_hi
        assert math.isclose(coeff.tau[4], 0.0, abs_tol=1e-14)   # z_lo
        assert math.isclose(coeff.tau[5], 2.0, abs_tol=1e-14)   # z_hi

    def test_w_component(self):
        geom, coeff = self._setup()
        s = geom.s
        lam = geom.lambda_
        # w(0) = (s, 0, lam)
        assert math.isclose(coeff.w_component(0, 0.0), s, abs_tol=1e-14)
        assert math.isclose(coeff.w_component(1, 0.0), 0.0, abs_tol=1e-14)
        assert math.isclose(coeff.w_component(2, 0.0), lam, abs_tol=1e-14)
        # w(pi/2) = (0, s, lam)
        assert math.isclose(coeff.w_component(0, math.pi / 2), 0.0, abs_tol=1e-14)
        assert math.isclose(coeff.w_component(1, math.pi / 2), s, abs_tol=1e-14)
        assert math.isclose(coeff.w_component(2, math.pi / 2), lam, abs_tol=1e-14)

    def test_rejects_degenerate_cone(self):
        # s = 0 when theta = 0 (impossible for EventObj) but test the guard
        event = _event(theta=math.pi / 4)
        geom = build_cone_local_geometry(event)
        from dataclasses import replace
        bad_geom = replace(geom, s=0.0)
        with pytest.raises(ValueError, match="degenerate"):
            build_cone_box_coefficients(bad_geom, _voi(-1, 1, -1, 1, 0, 2))


# ===================================================================
# 3. Breakpoint structure
# ===================================================================


class TestBreakpoints:
    """Breakpoint list: finite, sorted, normalized mod 2pi, deduplicated."""

    @pytest.fixture()
    def axis_aligned_coeff(self):
        event = _event(apex=(0, 0, 0), axis=(0, 0, 1), theta=math.pi / 4)
        voi = _voi(-1, 1, -1, 1, 0, 2)
        geom = build_cone_local_geometry(event)
        return build_cone_box_coefficients(geom, voi)

    @pytest.fixture()
    def off_axis_coeff(self):
        event = _event(
            apex=(0.5, -0.3, 0.1),
            axis=(1, 1, 1),
            theta=math.pi / 3,
        )
        voi = _voi(-2, 3, -2, 3, -1, 4)
        geom = build_cone_local_geometry(event)
        return build_cone_box_coefficients(geom, voi)

    def test_sorted_and_in_range(self, axis_aligned_coeff):
        bp = compute_breakpoints(axis_aligned_coeff)
        assert len(bp) > 0
        for xi in bp:
            assert 0.0 <= xi < _TWO_PI
        assert list(bp) == sorted(bp)

    def test_deduplicated(self, axis_aligned_coeff):
        bp = compute_breakpoints(axis_aligned_coeff)
        for i in range(len(bp) - 1):
            assert bp[i + 1] - bp[i] > 1e-13

    def test_off_axis_breakpoints_valid(self, off_axis_coeff):
        bp = compute_breakpoints(off_axis_coeff)
        assert len(bp) > 0
        for xi in bp:
            assert 0.0 <= xi < _TWO_PI
        assert list(bp) == sorted(bp)

    def test_axis_aligned_denominator_zeros_present(self, axis_aligned_coeff):
        """w_x = s*cos(xi) = 0 at pi/2, 3pi/2; w_y = s*sin(xi) = 0 at 0, pi."""
        bp = compute_breakpoints(axis_aligned_coeff)
        expected_denom = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
        for xi_exp in expected_denom:
            assert any(
                abs(xi_exp - xi) < 1e-10 or abs(xi_exp - xi - _TWO_PI) < 1e-10
                for xi in bp
            ), f"expected breakpoint near {xi_exp} not found in {bp}"


# ===================================================================
# 4. Fixed-azimuth interval evaluation
# ===================================================================


class TestFixedAzimuthInterval:
    """Exact ell^-/ell^+ and area profile for single azimuths."""

    @pytest.fixture()
    def apex_inside_setup(self):
        """Apex at origin, inside the box, axis along z, theta=pi/4."""
        event = _event(apex=(0, 0, 0), axis=(0, 0, 1), theta=math.pi / 4)
        voi = _voi(-1, 1, -1, 1, 0, 2)
        geom = build_cone_local_geometry(event)
        coeff = build_cone_box_coefficients(geom, voi)
        return geom, voi, coeff

    @pytest.fixture()
    def apex_outside_setup(self):
        """Apex at (3, 0, 1), outside box in x, axis along z, theta=pi/3."""
        event = _event(apex=(3, 0, 1), axis=(0, 0, 1), theta=math.pi / 3)
        voi = _voi(-1, 1, -1, 1, 0, 2)
        geom = build_cone_local_geometry(event)
        coeff = build_cone_box_coefficients(geom, voi)
        return geom, voi, coeff

    def test_apex_inside_all_nonempty(self, apex_inside_setup):
        """With apex inside box and cone opening upward, all azimuths hit."""
        _, _, coeff = apex_inside_setup
        for xi in [0.0, 0.3, 1.0, math.pi, 4.0, 5.5]:
            iv = evaluate_fixed_azimuth(coeff, xi)
            assert iv.nonempty, f"expected nonempty at xi={xi}"
            assert iv.area_profile > 0.0

    def test_apex_outside_some_empty(self, apex_outside_setup):
        """With apex far outside box, many azimuths should be empty."""
        _, _, coeff = apex_outside_setup
        # xi=0: ray goes in +x direction, away from box => empty
        iv_0 = evaluate_fixed_azimuth(coeff, 0.0)
        assert not iv_0.nonempty

    def test_area_profile_formula(self, apex_inside_setup):
        """A(xi) = ell_plus^2 - ell_minus^2."""
        _, _, coeff = apex_inside_setup
        for xi in [0.1, 0.7, 2.5, 4.9]:
            iv = evaluate_fixed_azimuth(coeff, xi)
            if iv.nonempty:
                expected = iv.ell_plus ** 2 - iv.ell_minus ** 2
                assert math.isclose(iv.area_profile, expected, rel_tol=1e-14)

    def test_ell_minus_positive(self, apex_inside_setup):
        """ell_minus >= 0 always."""
        _, _, coeff = apex_inside_setup
        for xi in np.linspace(0.01, _TWO_PI - 0.01, 20):
            iv = evaluate_fixed_azimuth(coeff, float(xi))
            if iv.nonempty:
                assert iv.ell_minus >= 0.0

    def test_ell_ordering(self, apex_inside_setup):
        """ell_minus < ell_plus when nonempty."""
        _, _, coeff = apex_inside_setup
        for xi in np.linspace(0.01, _TWO_PI - 0.01, 20):
            iv = evaluate_fixed_azimuth(coeff, float(xi))
            if iv.nonempty:
                assert iv.ell_minus < iv.ell_plus


# ===================================================================
# 5. Agreement with ray_box strict-open interval
# ===================================================================


class TestAgreementWithRayBox:
    """ell^-/ell^+ must agree with open_box_ray_interval at non-breakpoints."""

    @pytest.fixture()
    def setups(self):
        configs = [
            # apex inside, z-axis
            (_event(apex=(0, 0, 0.5), axis=(0, 0, 1), theta=math.pi / 4),
             _voi(-1, 1, -1, 1, 0, 2)),
            # apex inside, off-axis
            (_event(apex=(0.1, -0.2, 0.3), axis=(1, 2, 3), theta=math.pi / 3),
             _voi(-2, 2, -2, 2, -1, 3)),
            # apex on lower face
            (_event(apex=(0, 0, 0), axis=(0, 0, 1), theta=math.pi / 4),
             _voi(-1, 1, -1, 1, 0, 2)),
        ]
        result = []
        for event, voi in configs:
            geom = build_cone_local_geometry(event)
            coeff = build_cone_box_coefficients(geom, voi)
            bp = compute_breakpoints(coeff)
            result.append((geom, voi, coeff, bp))
        return result

    def _is_near_breakpoint(self, xi, breakpoints, tol=1e-8):
        for b in breakpoints:
            if abs(xi - b) < tol or abs(xi - b + _TWO_PI) < tol or abs(xi - b - _TWO_PI) < tol:
                return True
        return False

    def test_interval_values_match(self, setups):
        """At non-breakpoint azimuths, our ell^-/ell^+ match ray_box."""
        for geom, voi, coeff, bp in setups:
            for xi in np.linspace(0.05, _TWO_PI - 0.05, 40):
                xi = float(xi)
                if self._is_near_breakpoint(xi, bp):
                    continue
                iv = evaluate_fixed_azimuth(coeff, xi)
                ref = _ray_box_reference(geom, voi, xi)
                if ref is None:
                    assert not iv.nonempty, f"xi={xi}: ray_box empty but we say nonempty"
                else:
                    assert iv.nonempty, f"xi={xi}: ray_box nonempty but we say empty"
                    assert math.isclose(iv.ell_minus, ref[0], rel_tol=1e-12, abs_tol=1e-15), \
                        f"xi={xi}: ell_minus mismatch {iv.ell_minus} vs {ref[0]}"
                    assert math.isclose(iv.ell_plus, ref[1], rel_tol=1e-12, abs_tol=1e-15), \
                        f"xi={xi}: ell_plus mismatch {iv.ell_plus} vs {ref[1]}"


# ===================================================================
# 6. Segment decomposition
# ===================================================================


class TestDecomposition:
    """Full breakpoint/segment decomposition structure."""

    @pytest.fixture()
    def axis_aligned_decomp(self):
        event = _event(apex=(0, 0, 0), axis=(0, 0, 1), theta=math.pi / 4)
        voi = _voi(-1, 1, -1, 1, 0, 2)
        geom = build_cone_local_geometry(event)
        return build_cone_box_decomposition(geom, voi)

    @pytest.fixture()
    def off_axis_decomp(self):
        event = _event(apex=(0.5, -0.3, 0.1), axis=(1, 1, 1), theta=math.pi / 3)
        voi = _voi(-2, 3, -2, 3, -1, 4)
        geom = build_cone_local_geometry(event)
        return build_cone_box_decomposition(geom, voi)

    def test_segment_count_equals_breakpoint_count(self, axis_aligned_decomp):
        d = axis_aligned_decomp
        if len(d.breakpoints) == 0:
            assert len(d.segments) == 1
        else:
            assert len(d.segments) == len(d.breakpoints)

    def test_segments_cover_full_circle(self, axis_aligned_decomp):
        """Total arc length of all segments should be 2*pi."""
        d = axis_aligned_decomp
        total = 0.0
        for seg in d.segments:
            if seg.xi_end > seg.xi_start:
                total += seg.xi_end - seg.xi_start
            else:
                total += (_TWO_PI - seg.xi_start) + seg.xi_end
        assert math.isclose(total, _TWO_PI, abs_tol=1e-10)

    def test_segment_endpoints_are_breakpoints(self, axis_aligned_decomp):
        d = axis_aligned_decomp
        if not d.breakpoints:
            return
        bp_set = set(d.breakpoints)
        for seg in d.segments:
            assert seg.xi_start in bp_set
            assert seg.xi_end in bp_set

    def test_off_axis_valid(self, off_axis_decomp):
        d = off_axis_decomp
        assert len(d.segments) >= 1
        total = 0.0
        for seg in d.segments:
            if seg.xi_end > seg.xi_start:
                total += seg.xi_end - seg.xi_start
            else:
                total += (_TWO_PI - seg.xi_start) + seg.xi_end
        assert math.isclose(total, _TWO_PI, abs_tol=1e-10)


# ===================================================================
# 7. Regime stability within segments
# ===================================================================


class TestRegimeStability:
    """Active faces and nonemptiness must be constant within each segment."""

    @pytest.fixture()
    def decomps(self):
        configs = [
            (_event(apex=(0, 0, 0.5), axis=(0, 0, 1), theta=math.pi / 4),
             _voi(-1, 1, -1, 1, 0, 2)),
            (_event(apex=(0.5, -0.3, 0.1), axis=(1, 1, 1), theta=math.pi / 3),
             _voi(-2, 3, -2, 3, -1, 4)),
            (_event(apex=(0, 0, 0), axis=(0, 0, 1), theta=math.pi / 6),
             _voi(-3, 3, -3, 3, 0, 5)),
        ]
        result = []
        for event, voi in configs:
            geom = build_cone_local_geometry(event)
            coeff = build_cone_box_coefficients(geom, voi)
            decomp = build_decomposition(coeff)
            result.append((coeff, decomp))
        return result

    def test_faces_stable(self, decomps):
        """Sample 5 points per segment; active faces must all agree."""
        for coeff, decomp in decomps:
            for seg in decomp.segments:
                # Generate sample points strictly inside the segment arc
                if seg.xi_end > seg.xi_start:
                    arc = seg.xi_end - seg.xi_start
                    samples = [seg.xi_start + arc * f for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
                else:
                    arc = (_TWO_PI - seg.xi_start) + seg.xi_end
                    samples = []
                    for f in (0.1, 0.3, 0.5, 0.7, 0.9):
                        xi = seg.xi_start + arc * f
                        if xi >= _TWO_PI:
                            xi -= _TWO_PI
                        samples.append(xi)

                for xi in samples:
                    iv = evaluate_fixed_azimuth(coeff, xi)
                    assert iv.nonempty == seg.nonempty, (
                        f"nonempty mismatch at xi={xi} in segment "
                        f"[{seg.xi_start}, {seg.xi_end}]"
                    )
                    if seg.nonempty:
                        assert iv.lower_face == seg.lower_face, (
                            f"lower_face mismatch at xi={xi}: "
                            f"{iv.lower_face} vs {seg.lower_face}"
                        )
                        assert iv.upper_face == seg.upper_face, (
                            f"upper_face mismatch at xi={xi}: "
                            f"{iv.upper_face} vs {seg.upper_face}"
                        )

    def test_sign_pattern_stable(self, decomps):
        """Sign pattern of w_c must be constant within each segment."""
        for coeff, decomp in decomps:
            for seg in decomp.segments:
                if seg.xi_end > seg.xi_start:
                    arc = seg.xi_end - seg.xi_start
                    mid = seg.xi_start + 0.5 * arc
                else:
                    arc = (_TWO_PI - seg.xi_start) + seg.xi_end
                    mid = seg.xi_start + 0.5 * arc
                    if mid >= _TWO_PI:
                        mid -= _TWO_PI

                # Check a few points
                for frac in (0.2, 0.5, 0.8):
                    xi = seg.xi_start + arc * frac
                    if xi >= _TWO_PI:
                        xi -= _TWO_PI
                    sp = []
                    for c in range(3):
                        wc = coeff.w_component(c, xi)
                        if wc > 0:
                            sp.append(1)
                        elif wc < 0:
                            sp.append(-1)
                        else:
                            sp.append(0)
                    assert tuple(sp) == seg.sign_pattern, (
                        f"sign_pattern mismatch at xi={xi}"
                    )


# ===================================================================
# 8. Empty vs nonempty classification
# ===================================================================


class TestEmptyNonemptyClassification:
    """Correct empty/nonempty I_k(xi) determination."""

    def test_apex_inside_always_nonempty(self):
        """Apex strictly inside box => all azimuths nonempty."""
        event = _event(apex=(0, 0, 1), axis=(0, 0, 1), theta=math.pi / 4)
        voi = _voi(-2, 2, -2, 2, 0, 3)
        geom = build_cone_local_geometry(event)
        coeff = build_cone_box_coefficients(geom, voi)
        for xi in np.linspace(0, _TWO_PI, 50, endpoint=False):
            iv = evaluate_fixed_azimuth(coeff, float(xi))
            assert iv.nonempty, f"expected nonempty at xi={xi}"

    def test_apex_far_outside_mostly_empty(self):
        """Apex far from box => most azimuths empty."""
        event = _event(apex=(100, 0, 0), axis=(0, 0, 1), theta=math.pi / 4)
        voi = _voi(-1, 1, -1, 1, -1, 1)
        geom = build_cone_local_geometry(event)
        coeff = build_cone_box_coefficients(geom, voi)
        empty_count = 0
        for xi in np.linspace(0, _TWO_PI, 100, endpoint=False):
            iv = evaluate_fixed_azimuth(coeff, float(xi))
            if not iv.nonempty:
                empty_count += 1
        # Most azimuths should be empty when apex is far away
        assert empty_count > 50, f"only {empty_count}/100 empty"


# ===================================================================
# 9. Boundary / grazing cases fail closed
# ===================================================================


class TestBoundaryGrazingCases:
    """Boundary-only and grazing configurations must not broaden support."""

    def test_apex_on_upper_face_wc_zero(self):
        """Apex on upper face (excluded by half-open) with w_c=0 => empty."""
        # Apex at z = zmax (excluded by half-open [zmin, zmax))
        event = _event(apex=(0, 0, 2), axis=(0, 0, 1), theta=math.pi / 4)
        voi = _voi(-1, 1, -1, 1, 0, 2)
        geom = build_cone_local_geometry(event)
        coeff = build_cone_box_coefficients(geom, voi)
        # w_z = lambda_ > 0 for all xi, so the parallel-slab path is for
        # w_x=0 or w_y=0 only.  But apex is on z=zmax boundary:
        # tau_z_lo = 0 - 2 = -2 <= 0, tau_z_hi = 2 - 2 = 0.
        # z-slab: entry (w_z > 0) = tau_z_lo / w_z = -2/lam < 0,
        #         exit  = tau_z_hi / w_z = 0/lam = 0.
        # So ell_plus <= 0 => ell_minus < ell_plus fails => empty.
        for xi in [0.0, 1.0, 3.0, 5.0]:
            iv = evaluate_fixed_azimuth(coeff, xi)
            assert not iv.nonempty, f"expected empty at xi={xi}"

    def test_apex_on_lower_face_wc_zero_parallel(self):
        """Apex on lower face with w_c exactly 0: half-open includes lower."""
        # Apex at (0, 0, 0), box z in [0, 2).  At azimuths where w_z > 0 (always
        # for z-axis cone), the z-entry is ell = 0 and z-exit is ell = 2/lambda.
        # The interval (0, 2/lambda) is nonempty.
        event = _event(apex=(0, 0, 0), axis=(0, 0, 1), theta=math.pi / 4)
        voi = _voi(-1, 1, -1, 1, 0, 2)
        geom = build_cone_local_geometry(event)
        coeff = build_cone_box_coefficients(geom, voi)
        iv = evaluate_fixed_azimuth(coeff, 0.3)
        assert iv.nonempty

    def test_tangent_no_interior_intersection(self):
        """Ray tangent to box edge should produce empty interval."""
        # Apex at (1, 0, 0.5), axis along z, theta=pi/4
        # Box [0, 2) x [0, 2) x [0, 2).  At xi where the ray goes
        # exactly along the y=0 face, the y-slab entry = exit = 0,
        # and the interval should be empty (boundary-only).
        event = _event(apex=(1, 0, 0.5), axis=(0, 0, 1), theta=math.pi / 4)
        voi = _voi(0, 2, 0, 2, 0, 2)
        geom = build_cone_local_geometry(event)
        coeff = build_cone_box_coefficients(geom, voi)
        # Find azimuth where w_y = 0 (a breakpoint)
        bp = compute_breakpoints(coeff)
        # At w_y = 0 breakpoint with apex at y=0 (lower face):
        # tau_y_lo = 0, tau_y_hi = 2.  With w_y = 0, half-open check:
        # tau_lo <= 0 (0 <= 0 true) and tau_hi > 0 (2 > 0 true) => inside slab.
        # So the slab is satisfied and other coordinates determine the interval.
        # This is actually nonempty at the breakpoint itself.  The "tangent"
        # case only fails when the slab is missed.
        # Instead test with apex on the UPPER y face:
        event2 = _event(apex=(1, 2, 0.5), axis=(0, 0, 1), theta=math.pi / 4)
        coeff2 = build_cone_box_coefficients(
            build_cone_local_geometry(event2), voi
        )
        # tau_y_lo = 0 - 2 = -2, tau_y_hi = 2 - 2 = 0.
        # At w_y = 0: tau_lo <= 0 (true) and tau_hi > 0 (false, = 0) => empty.
        for bp_xi in compute_breakpoints(coeff2):
            wc_y = coeff2.w_component(1, bp_xi)
            if abs(wc_y) < 1e-10:
                iv = evaluate_fixed_azimuth(coeff2, bp_xi)
                assert not iv.nonempty, (
                    f"expected empty at w_y=0 breakpoint xi={bp_xi}"
                )


# ===================================================================
# 10. Deterministic perturbation stability
# ===================================================================


class TestPerturbationStability:
    """Small perturbations within a segment must not change the regime."""

    def test_perturbation_within_segment(self):
        event = _event(apex=(0.1, -0.2, 0.3), axis=(1, 2, 3), theta=math.pi / 3)
        voi = _voi(-2, 2, -2, 2, -1, 3)
        geom = build_cone_local_geometry(event)
        coeff = build_cone_box_coefficients(geom, voi)
        decomp = build_decomposition(coeff)

        for seg in decomp.segments:
            if seg.xi_end > seg.xi_start:
                arc = seg.xi_end - seg.xi_start
            else:
                arc = (_TWO_PI - seg.xi_start) + seg.xi_end

            # Skip very tiny segments where perturbation could cross
            if arc < 1e-6:
                continue

            mid = seg.xi_start + 0.5 * arc
            if mid >= _TWO_PI:
                mid -= _TWO_PI

            # Perturb by +-1e-8 (well within segment)
            for delta in [-1e-8, 0.0, 1e-8]:
                xi = mid + delta
                if xi >= _TWO_PI:
                    xi -= _TWO_PI
                if xi < 0:
                    xi += _TWO_PI
                iv = evaluate_fixed_azimuth(coeff, xi)
                assert iv.nonempty == seg.nonempty
                if seg.nonempty:
                    assert iv.lower_face == seg.lower_face
                    assert iv.upper_face == seg.upper_face


# ===================================================================
# 11. Endpoint openness
# ===================================================================


class TestEndpointOpenness:
    """Half-open box semantics correctly inherited by interval endpoints."""

    def test_lower_face_gives_closed_endpoint(self):
        """When ell_minus is determined by a lower-bound face, it is closed."""
        event = _event(apex=(-2, 0, 1), axis=(1, 0, 0), theta=math.pi / 6)
        voi = _voi(-1, 1, -1, 1, 0, 2)
        geom = build_cone_local_geometry(event)
        coeff = build_cone_box_coefficients(geom, voi)
        # At xi where the ray enters through x_lo face (even index = closed):
        # We need an azimuth where x is the entry face.
        # With axis along x, lambda_ ~ 0.866, s ~ 0.5
        # w_x(0) = lam * u_x + s * p_x  (need to know frame)
        # Just sample and find a nonempty one
        for xi in np.linspace(0, _TWO_PI, 100, endpoint=False):
            iv = evaluate_fixed_azimuth(coeff, float(xi))
            if iv.nonempty and iv.lower_face is not None:
                face_idx = iv.lower_face
                if face_idx % 2 == 0:  # lower face
                    assert iv.lower_closed, (
                        f"lower face {FACE_NAMES[face_idx]} should give closed endpoint"
                    )
                else:  # upper face
                    assert not iv.lower_closed, (
                        f"upper face {FACE_NAMES[face_idx]} should give open endpoint"
                    )
            if iv.nonempty and iv.upper_face is not None:
                face_idx = iv.upper_face
                if face_idx % 2 == 0:  # lower face
                    assert iv.upper_closed
                else:
                    assert not iv.upper_closed

    def test_ray_domain_gives_open_endpoint(self):
        """When ell_minus = 0 (ray domain), the endpoint is open."""
        event = _event(apex=(0, 0, 1), axis=(0, 0, 1), theta=math.pi / 4)
        voi = _voi(-2, 2, -2, 2, 0, 3)
        geom = build_cone_local_geometry(event)
        coeff = build_cone_box_coefficients(geom, voi)
        # Apex inside box: ell_minus = 0 for all azimuths
        iv = evaluate_fixed_azimuth(coeff, 0.5)
        assert iv.nonempty
        assert iv.lower_face is None
        assert iv.ell_minus == 0.0
        assert not iv.lower_closed  # ray domain is open


# ===================================================================
# 12. evaluate_area_profile convenience
# ===================================================================


class TestEvaluateAreaProfile:
    def test_matches_fixed_azimuth(self):
        event = _event(apex=(0, 0, 0.5), axis=(0, 0, 1), theta=math.pi / 4)
        voi = _voi(-1, 1, -1, 1, 0, 2)
        geom = build_cone_local_geometry(event)
        coeff = build_cone_box_coefficients(geom, voi)
        for xi in [0.1, 1.5, 3.7, 5.9]:
            assert evaluate_area_profile(coeff, xi) == evaluate_fixed_azimuth(coeff, xi).area_profile
