"""Exact non-live azimuth-law machinery for bounded VoiBounds.

Builds on the landed cone-box surface geometry substrate in
``cone_box_surface.py`` to provide:

- exact per-segment regime coefficient extraction,
- exact primitive of ``1/g(xi)^2`` for ``g(xi) = a + b cos xi + c sin xi``,
- exact segment masses via analytic primitive differences,
- cumulative segment masses,
- continuous per-segment partial-CDF evaluators.

This module does NOT implement sampling or inversion, and does NOT alter
the live proposal path in ``uniform_surface_reference.py``.

Mathematical reference
---------------------
For each denominator ``g(xi) = a + b cos xi + c sin xi``, define:

* ``R = sqrt(b^2 + c^2)``, ``phi = atan2(c, b)``
* ``t = xi - phi``, ``u = tan(t/2)``
* ``A0 = a + R``, ``B0 = a - R``, ``Delta = A0 * B0 = a^2 - b^2 - c^2``

The exact primitive ``J`` of ``1/g(xi)^2`` is given piecewise by Delta sign.
Segment masses use primitive differences; the partial CDF is the running
primitive difference from the segment left endpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from soe.geometry.cone_box_surface import (
    ConeBoxCoefficients,
    ConeBoxSurfaceDecomposition,
    AzimuthSegment,
    evaluate_fixed_azimuth,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi
_DELTA_TOL = 1e-24
_ATANH_BOUNDARY_TOL = 1e-12
_COS_HALF_T_TOL = 1e-14
_U_ASYMP_THRESHOLD = 1e13


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentRegime:
    """Per-segment immutable metadata for exact azimuth-law integration.

    For the active upper face:
        ``ell_plus(xi) = tau_upper / g_upper(xi)``
        ``g_upper(xi)  = a_upper + b_upper * cos(xi) + c_upper * sin(xi)``

    For the active lower face (when ``not lower_is_apex``):
        ``ell_minus(xi) = tau_lower / g_lower(xi)``

    When ``lower_is_apex``, ``ell_minus(xi) = 0``.
    """

    xi_start: float
    xi_end: float
    nonempty: bool
    lower_is_apex: bool
    tau_upper: float
    a_upper: float
    b_upper: float
    c_upper: float
    tau_lower: float
    a_lower: float
    b_lower: float
    c_lower: float


@dataclass(frozen=True, slots=True)
class AzimuthLaw:
    """Complete exact non-live azimuth-law data for one event and one box.

    Attributes
    ----------
    decomposition:
        The underlying surface geometry decomposition.
    regimes:
        Per-segment face-pair coefficients (one per segment).
    segment_masses:
        ``segment_masses[j]`` is ``M_j = integral of A_j(xi) over S_j``.
    cumulative_masses:
        ``cumulative_masses[j] = sum(segment_masses[0:j])``.
        In particular, ``cumulative_masses[0] = 0``.
    total_mass:
        Sum of all segment masses.
    """

    decomposition: ConeBoxSurfaceDecomposition
    regimes: tuple[SegmentRegime, ...]
    segment_masses: tuple[float, ...]
    cumulative_masses: tuple[float, ...]
    total_mass: float


# ---------------------------------------------------------------------------
# Trig parameter helpers
# ---------------------------------------------------------------------------


def _trig_params(
    a: float, b: float, c: float,
) -> tuple[float, float, float, float, float]:
    """Return ``(R, phi, A0, B0, Delta)`` for ``g(xi) = a + b cos xi + c sin xi``."""
    R = math.sqrt(b * b + c * c)
    phi = math.atan2(c, b) if R > 1e-30 else 0.0
    A0 = a + R
    B0 = a - R
    Delta = A0 * B0  # a**2 - b**2 - c**2
    return R, phi, A0, B0, Delta


# ---------------------------------------------------------------------------
# Exact primitive of 1/g(xi)^2
# ---------------------------------------------------------------------------


def _generalized_atanh(x: float) -> float:
    """``artanh(x)`` for ``|x| < 1``, ``acoth(x)`` for ``|x| > 1``."""
    ax = abs(x)
    if ax < 1.0 - _ATANH_BOUNDARY_TOL:
        return math.atanh(x)
    if ax > 1.0 + _ATANH_BOUNDARY_TOL:
        return 0.5 * math.log(abs((x + 1.0) / (x - 1.0)))
    raise ValueError(
        f"_generalized_atanh at domain boundary: |x|={ax:.15e}"
    )


def _primitive_at_u(
    u: float,
    a: float,
    R: float,
    A0: float,
    B0: float,
    Delta: float,
) -> float:
    """Evaluate the exact primitive J at half-tangent value *u*."""
    u_sq = u * u
    denom = A0 + B0 * u_sq

    if abs(Delta) > _DELTA_TOL:
        # Rational term  -(2R / Delta) * u / (A0 + B0 u^2)
        rational = -(2.0 * R / Delta) * u / denom if abs(denom) > 1e-30 else 0.0

        if Delta > 0.0:
            sqrt_D = math.sqrt(Delta)
            # For Delta > 0 the exact primitive is
            #   -(2R/Delta) * u/(A0 + B0 u^2)
            #   + (2a/(Delta*sqrt(Delta))) * atan((u*sqrt(Delta))/A0).
            # This must use atan, not atan2: when A0 < 0 the atan2 form
            # introduces a false branch cut at u = 0 even though the
            # integrand is regular there.
            atan_val = math.atan((u * sqrt_D) / A0)
            return (2.0 * a / (Delta * sqrt_D)) * atan_val + rational
        else:
            delta = math.sqrt(-Delta)
            x = delta * u / A0
            h = _generalized_atanh(x)
            return -(2.0 * a / (delta * delta * delta)) * h + rational
    else:
        # Delta = 0
        if abs(B0) <= abs(A0):
            # B0 ~ 0  =>  a ~ R:  J = (1/(2a^2))(u + u^3/3)
            inv_2a2 = 1.0 / (2.0 * a * a)
            return inv_2a2 * (u + u * u_sq / 3.0)
        else:
            # A0 ~ 0  =>  a ~ -R:  J = -(1/(2a^2))(1/u + 1/(3u^3))
            if abs(u) < 1e-15:
                raise ValueError("Primitive diverges: Delta=0, A0~0, u~0")
            inv_u = 1.0 / u
            inv_2a2 = 1.0 / (2.0 * a * a)
            return -inv_2a2 * (inv_u + inv_u * inv_u * inv_u / 3.0)


def _primitive_asymptotic(
    u: float,
    a: float,
    R: float,
    A0: float,
    B0: float,
    Delta: float,
) -> float:
    """Asymptotic expansion of J for ``|u| >> 1``."""
    sign_u = 1.0 if u > 0.0 else -1.0
    inv_u = 1.0 / u

    if abs(Delta) > _DELTA_TOL:
        corr = -2.0 * A0 / (Delta * B0) * inv_u if abs(B0) > 1e-30 else 0.0
        if Delta > 0.0:
            # The Delta > 0 primitive uses atan((u*sqrt(Delta))/A0), so the
            # large-|u| branch constant is sign(u) * |a| * pi / (Delta^(3/2)).
            return sign_u * abs(a) * math.pi / (Delta * math.sqrt(Delta)) + corr
        else:
            return corr
    else:
        if abs(B0) <= abs(A0):
            raise ValueError("Asymptotic diverges for Delta=0, B0~0, large u")
        return -(1.0 / (2.0 * a * a)) * inv_u


def _eval_J_at_t(
    t: float,
    a: float,
    R: float,
    A0: float,
    B0: float,
    Delta: float,
    *,
    left_limit: bool = True,
) -> float:
    """Evaluate J at a given ``t = xi - phi``.

    At the half-tangent singularity (``cos(t/2) ~ 0``), ``left_limit``
    selects the branch: ``True`` for the left limit (``u -> +inf``),
    ``False`` for the right limit (``u -> -inf``).
    """
    half_t = t * 0.5
    cos_ht = math.cos(half_t)
    if abs(cos_ht) < _COS_HALF_T_TOL:
        # At singularity: choose branch by left_limit
        u = _U_ASYMP_THRESHOLD if left_limit else -_U_ASYMP_THRESHOLD
        return _primitive_asymptotic(u, a, R, A0, B0, Delta)
    u = math.tan(half_t)
    if abs(u) > _U_ASYMP_THRESHOLD:
        return _primitive_asymptotic(u, a, R, A0, B0, Delta)
    return _primitive_at_u(u, a, R, A0, B0, Delta)


def evaluate_primitive(xi: float, a: float, b: float, c: float) -> float:
    """Evaluate the exact primitive ``J(xi; a, b, c)`` of ``1/g(xi)^2``.

    Public for testing.  Not part of the live proposal path.
    """
    R, phi, A0, B0, Delta = _trig_params(a, b, c)
    return _eval_J_at_t(xi - phi, a, R, A0, B0, Delta)


# ---------------------------------------------------------------------------
# Half-tangent singularity crossing count
# ---------------------------------------------------------------------------


def _count_half_tangent_crossings(t_a: float, t_b: float) -> int:
    """Count how many ``t = pi + 2k*pi`` fall in the open interval ``(t_a, t_b)``."""
    if t_b <= t_a:
        return 0
    lo = (t_a - math.pi) / _TWO_PI
    hi = (t_b - math.pi) / _TWO_PI
    k_min = math.floor(lo) + 1  # smallest integer > lo
    k_max = math.ceil(hi) - 1  # largest integer < hi
    if k_min > k_max:
        return 0
    return k_max - k_min + 1


@dataclass(frozen=True, slots=True)
class _InvGSqPrimitiveContext:
    """Fixed-left-end primitive context for repeated ``1/g(xi)^2`` integrals."""

    a: float
    R: float
    phi: float
    A0: float
    B0: float
    Delta: float
    t_start: float
    J_start: float
    start_on_half_tangent: bool


def _build_inv_g_sq_primitive_context(
    xi_start: float,
    a: float,
    b: float,
    c: float,
) -> _InvGSqPrimitiveContext:
    """Precompute constants for repeated integrals from fixed ``xi_start``."""
    R, phi, A0, B0, Delta = _trig_params(a, b, c)
    t_start = xi_start - phi
    return _InvGSqPrimitiveContext(
        a=a,
        R=R,
        phi=phi,
        A0=A0,
        B0=B0,
        Delta=Delta,
        t_start=t_start,
        J_start=_eval_J_at_t(t_start, a, R, A0, B0, Delta),
        start_on_half_tangent=abs(math.cos(t_start * 0.5)) < _COS_HALF_T_TOL,
    )


def _integrate_inv_g_sq_from_context(
    xi_b: float,
    context: _InvGSqPrimitiveContext,
    *,
    wraps: bool = False,
) -> float:
    """Exact ``1/g(xi)^2`` integral using fixed-left-end cached constants."""
    t_b = xi_b - context.phi
    t_b_eff = t_b + _TWO_PI if wraps else t_b

    J_b = _eval_J_at_t(
        t_b,
        context.a,
        context.R,
        context.A0,
        context.B0,
        context.Delta,
    )
    result = J_b - context.J_start

    # Crossing correction -- only needed when Delta > 0.
    if context.Delta > _DELTA_TOL and t_b_eff > context.t_start:
        n = _count_half_tangent_crossings(context.t_start, t_b_eff)

        # If the LEFT endpoint t_start falls exactly on a half-tangent
        # singularity, J_start is the left limit but the integral from the
        # endpoint rightward requires the right limit.
        if context.start_on_half_tangent:
            n += 1

        if n > 0:
            jump = (
                2.0
                * abs(context.a)
                * math.pi
                / (context.Delta * math.sqrt(context.Delta))
            )
            result += n * jump

    return result


# ---------------------------------------------------------------------------
# Definite integral of 1/g(xi)^2 over an arc
# ---------------------------------------------------------------------------


def integrate_inv_g_sq(
    xi_a: float,
    xi_b: float,
    a: float,
    b: float,
    c: float,
    *,
    wraps: bool = False,
) -> float:
    """Exact integral of ``1/(a + b cos xi + c sin xi)^2`` over a CCW arc.

    Parameters
    ----------
    xi_a, xi_b:
        Arc endpoints in ``[0, 2*pi)``.
    a, b, c:
        Coefficients of ``g(xi)``.
    wraps:
        ``True`` when the arc goes from *xi_a* counterclockwise through
        ``2*pi`` back to *xi_b* (i.e. ``xi_b <= xi_a``).

    Public for testing.  Not part of the live proposal path.
    """
    context = _build_inv_g_sq_primitive_context(xi_a, a, b, c)
    return _integrate_inv_g_sq_from_context(xi_b, context, wraps=wraps)


# ---------------------------------------------------------------------------
# Segment regime extraction
# ---------------------------------------------------------------------------


def _extract_regime(
    seg: AzimuthSegment,
    coeff: ConeBoxCoefficients,
) -> SegmentRegime:
    """Extract per-segment face coefficients for exact integration."""
    if not seg.nonempty:
        return SegmentRegime(
            xi_start=seg.xi_start,
            xi_end=seg.xi_end,
            nonempty=False,
            lower_is_apex=True,
            tau_upper=0.0,
            a_upper=0.0,
            b_upper=0.0,
            c_upper=0.0,
            tau_lower=0.0,
            a_lower=0.0,
            b_lower=0.0,
            c_lower=0.0,
        )

    f_up = seg.upper_face
    if f_up is None:
        raise ValueError("nonempty segment must have an upper face")
    c_up = f_up // 2

    tau_upper = coeff.tau[f_up]
    a_upper = coeff.alpha[c_up]
    b_upper = coeff.beta[c_up]
    c_upper = coeff.gamma[c_up]

    if seg.lower_face is None:
        return SegmentRegime(
            xi_start=seg.xi_start,
            xi_end=seg.xi_end,
            nonempty=True,
            lower_is_apex=True,
            tau_upper=tau_upper,
            a_upper=a_upper,
            b_upper=b_upper,
            c_upper=c_upper,
            tau_lower=0.0,
            a_lower=0.0,
            b_lower=0.0,
            c_lower=0.0,
        )

    f_lo = seg.lower_face
    c_lo = f_lo // 2
    return SegmentRegime(
        xi_start=seg.xi_start,
        xi_end=seg.xi_end,
        nonempty=True,
        lower_is_apex=False,
        tau_upper=tau_upper,
        a_upper=a_upper,
        b_upper=b_upper,
        c_upper=c_upper,
        tau_lower=coeff.tau[f_lo],
        a_lower=coeff.alpha[c_lo],
        b_lower=coeff.beta[c_lo],
        c_lower=coeff.gamma[c_lo],
    )


# ---------------------------------------------------------------------------
# Segment mass
# ---------------------------------------------------------------------------


def _segment_wraps(regime: SegmentRegime) -> bool:
    """True if the segment wraps around ``0 / 2*pi``."""
    return regime.xi_end <= regime.xi_start


def compute_segment_mass(regime: SegmentRegime) -> float:
    """Compute the exact mass ``M_j`` for one segment.

    ``M_j = integral_{S_j} A_j(xi) dxi``

    where ``A_j = tau_up^2 / g_up^2 - tau_lo^2 / g_lo^2``.
    """
    if not regime.nonempty:
        return 0.0

    wraps = _segment_wraps(regime)

    upper_int = integrate_inv_g_sq(
        regime.xi_start,
        regime.xi_end,
        regime.a_upper,
        regime.b_upper,
        regime.c_upper,
        wraps=wraps,
    )
    mass = regime.tau_upper**2 * upper_int

    if not regime.lower_is_apex:
        lower_int = integrate_inv_g_sq(
            regime.xi_start,
            regime.xi_end,
            regime.a_lower,
            regime.b_lower,
            regime.c_lower,
            wraps=wraps,
        )
        mass -= regime.tau_lower**2 * lower_int

    return mass


# ---------------------------------------------------------------------------
# Per-segment partial CDF
# ---------------------------------------------------------------------------


def partial_cdf(law: AzimuthLaw, j: int, xi: float) -> float:
    """Evaluate ``C_j(xi) = integral_{xi_start}^{xi} A_j(eta) deta``.

    *xi* must lie within segment *j* (between ``xi_start`` and ``xi_end``
    going counterclockwise).

    Returns ``0`` at ``xi = xi_start`` and ``segment_masses[j]`` at
    ``xi = xi_end``.
    """
    regime = law.regimes[j]
    if not regime.nonempty:
        return 0.0

    seg_wraps = _segment_wraps(regime)
    # sub-arc from xi_start to xi wraps iff the segment wraps AND xi is
    # in the post-wrap portion (xi < xi_start on the circle)
    sub_wraps = seg_wraps and (xi < regime.xi_start)

    upper_int = integrate_inv_g_sq(
        regime.xi_start,
        xi,
        regime.a_upper,
        regime.b_upper,
        regime.c_upper,
        wraps=sub_wraps,
    )
    cdf = regime.tau_upper**2 * upper_int

    if not regime.lower_is_apex:
        lower_int = integrate_inv_g_sq(
            regime.xi_start,
            xi,
            regime.a_lower,
            regime.b_lower,
            regime.c_lower,
            wraps=sub_wraps,
        )
        cdf -= regime.tau_lower**2 * lower_int

    return cdf


# ---------------------------------------------------------------------------
# Segment area profile (convenience)
# ---------------------------------------------------------------------------


def segment_area_profile(law: AzimuthLaw, j: int, xi: float) -> float:
    """Evaluate ``A_j(xi) = ell_plus(xi)^2 - ell_minus(xi)^2`` for segment *j*.

    Delegates to the geometry substrate evaluator.
    """
    return evaluate_fixed_azimuth(law.decomposition.coefficients, xi).area_profile


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_azimuth_law(
    decomposition: ConeBoxSurfaceDecomposition,
) -> AzimuthLaw:
    """Build the complete exact non-live azimuth-law from a surface decomposition.

    Computes per-segment regime coefficients, exact segment masses,
    cumulative masses, and total mass.
    """
    coeff = decomposition.coefficients
    regimes = tuple(
        _extract_regime(seg, coeff) for seg in decomposition.segments
    )
    masses = tuple(compute_segment_mass(r) for r in regimes)

    cum: list[float] = [0.0]
    for m in masses:
        cum.append(cum[-1] + m)
    total = cum[-1]
    cumulative = tuple(cum[:-1])

    return AzimuthLaw(
        decomposition=decomposition,
        regimes=regimes,
        segment_masses=masses,
        cumulative_masses=cumulative,
        total_mass=total,
    )


# ---------------------------------------------------------------------------
# Convenience: build from geometry + bounds
# ---------------------------------------------------------------------------


def build_azimuth_law_from_geometry(
    cone_geometry,  # ConeLocalGeometry
    voi_bounds,  # VoiBounds
) -> AzimuthLaw:
    """Build the complete azimuth law from cone geometry and VOI bounds."""
    from soe.geometry.cone_box_surface import build_cone_box_decomposition

    decomposition = build_cone_box_decomposition(cone_geometry, voi_bounds)
    return build_azimuth_law(decomposition)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AzimuthLaw",
    "SegmentRegime",
    "build_azimuth_law",
    "build_azimuth_law_from_geometry",
    "compute_segment_mass",
    "evaluate_primitive",
    "integrate_inv_g_sq",
    "partial_cdf",
    "segment_area_profile",
]
