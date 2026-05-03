"""Exact cone-box surface geometry substrate for  VoiBounds.

Provides the finite azimuth breakpoint decomposition and exact per-segment
radial interval / area-profile evaluators for one event and one bounded box.
This module does not alter the live proposal path.  It prepares the exact
geometric objects needed by the direct-sampler layer.

Mathematical reference
---------------------
For event *k* with cone local geometry ``(a, u, lambda_, s, frame=(p, q, u))``:

* Generator direction:
  ``w(xi) = lambda_ * u + s * (cos(xi) * p + sin(xi) * q)``
* Per-coordinate component:
  ``w_c(xi) = alpha_c + beta_c * cos(xi) + gamma_c * sin(xi)``
* Face radial distance:
  ``ell_f(xi) = tau_f / w_{c(f)}(xi)``
* Azimuth area profile:
  ``A(xi) = ell_plus(xi)**2 - ell_minus(xi)**2``
* Surface-area element in ``(ell, xi)`` coordinates:
  ``dmu = s * ell * dell * dxi``
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from soe.adapters.voi_voxel import VoiBounds
from soe.geometry.cone import ConeLocalGeometry



_TWO_PI = 2.0 * math.pi
_DEDUP_TOL = 1e-12
_TRIG_TOL = 1e-12

# Face index convention: 0=x_lo, 1=x_hi, 2=y_lo, 3=y_hi, 4=z_lo, 5=z_hi
FACE_NAMES: tuple[str, ...] = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
_FACE_IS_LOWER: tuple[bool, ...] = (True, False, True, False, True, False)




@dataclass(frozen=True, slots=True)
class ConeBoxCoefficients:
    """Immutable per-event per-box linear-trig coefficients.

    For coordinate axis *c* in ``{0(x), 1(y), 2(z)}``:

        ``w_c(xi) = alpha[c] + beta[c] * cos(xi) + gamma[c] * sin(xi)``

    Face offsets ``tau[f] = face_bound_f - apex_{c(f)}`` for *f* in 0..5.
    """

    alpha: tuple[float, float, float]
    beta: tuple[float, float, float]
    gamma: tuple[float, float, float]
    tau: tuple[float, float, float, float, float, float]
    apex: tuple[float, float, float]

    def w_component(self, c: int, xi: float) -> float:
        """Evaluate ``w_c(xi)`` for coordinate axis *c*."""
        return (
            self.alpha[c]
            + self.beta[c] * math.cos(xi)
            + self.gamma[c] * math.sin(xi)
        )


@dataclass(frozen=True, slots=True)
class FixedAzimuthInterval:
    """Exact radial interval and area profile at a single azimuth *xi*."""

    nonempty: bool
    ell_minus: float
    ell_plus: float
    area_profile: float
    lower_face: int | None  # face index 0-5, or None for ray_domain (ell>0)
    upper_face: int | None  # face index 0-5, or None if empty
    lower_closed: bool  # True iff ell_minus endpoint is closed
    upper_closed: bool  # True iff ell_plus endpoint is closed


@dataclass(frozen=True, slots=True)
class AzimuthSegment:
    """Regime metadata for one open azimuth arc between consecutive breakpoints.

    Within this arc the sign pattern of ``w_c``, the active lower and upper
    radial faces, and the emptiness/nonemptiness of ``I_k(xi)`` are all
    constant.
    """

    xi_start: float
    xi_end: float
    nonempty: bool
    sign_pattern: tuple[int, int, int]
    lower_face: int | None
    upper_face: int | None
    lower_closed: bool
    upper_closed: bool


@dataclass(frozen=True, slots=True)
class ConeBoxSurfaceDecomposition:
    """Complete finite azimuth breakpoint decomposition for one event + box.

    ``breakpoints`` are sorted, deduplicated, and in ``[0, 2*pi)``.
    ``segments`` partition ``[0, 2*pi)`` into open arcs between consecutive
    breakpoints (the last segment wraps around to the first breakpoint).
    """

    coefficients: ConeBoxCoefficients
    breakpoints: tuple[float, ...]
    segments: tuple[AzimuthSegment, ...]



def _normalize_angle(xi: float) -> float:
    """Map angle to ``[0, 2*pi)``."""
    xi = xi % _TWO_PI
    if xi < 0.0:
        xi += _TWO_PI
    return xi


def solve_linear_trig_zeros(A: float, B: float, C: float) -> list[float]:
    """Solve ``A + B cos(xi) + C sin(xi) = 0`` for ``xi in [0, 2*pi)``.

    Uses the exact closed-form ``R cos(xi - psi)`` decomposition:

        ``B cos(xi) + C sin(xi) = R cos(xi - psi)``

    where ``R = sqrt(B**2 + C**2)`` and ``psi = atan2(C, B)``.

    Returns a sorted list of solutions.  Returns ``[]`` if the equation is
    identically zero (degenerate) or has no real solutions.
    """
    A, B, C = float(A), float(B), float(C)

    R_sq = B * B + C * C
    if R_sq < _TRIG_TOL * _TRIG_TOL:
        # B ~ C ~ 0: equation reduces to A ~ 0 (degenerate) or no solution.
        return []

    R = math.sqrt(R_sq)
    ratio = -A / R  # need cos(xi - psi) = ratio

    if ratio < -(1.0 + _TRIG_TOL) or ratio > (1.0 + _TRIG_TOL):
        return []

    ratio = max(-1.0, min(1.0, ratio))
    psi = math.atan2(C, B)
    theta = math.acos(ratio)

    # Single-solution cases (tangent).
    if theta < _TRIG_TOL:
        return [_normalize_angle(psi)]
    if abs(theta - math.pi) < _TRIG_TOL:
        return [_normalize_angle(psi + math.pi)]

    xi1 = _normalize_angle(psi + theta)
    xi2 = _normalize_angle(psi - theta)

    solutions = sorted([xi1, xi2])

    # Deduplicate (including wrap-around near 0 / 2*pi).
    if abs(solutions[1] - solutions[0]) < _DEDUP_TOL:
        return [solutions[0]]
    if _TWO_PI - solutions[1] + solutions[0] < _DEDUP_TOL:
        return [solutions[0]]
    return solutions




def build_cone_box_coefficients(
    cone_geometry: ConeLocalGeometry,
    voi_bounds: VoiBounds,
) -> ConeBoxCoefficients:
    """Build exact linear-trig coefficients for one event + one bounded box.

    Parameters
    ----------
    cone_geometry:
        Immutable per-event cone invariants (apex, lambda\\_, s, frame).
    voi_bounds:
        The bounded half-open VOI box.
    """
    if not isinstance(cone_geometry, ConeLocalGeometry):
        raise TypeError("cone_geometry must be a ConeLocalGeometry")
    if not isinstance(voi_bounds, VoiBounds):
        raise TypeError("voi_bounds must be a VoiBounds")
    if cone_geometry.s <= 0.0:
        raise ValueError("degenerate cone (s=0) is not supported")

    frame = cone_geometry.frame
    lam = cone_geometry.lambda_
    s = cone_geometry.s

    alpha = (
        lam * float(frame.u[0]),
        lam * float(frame.u[1]),
        lam * float(frame.u[2]),
    )
    beta = (
        s * float(frame.p[0]),
        s * float(frame.p[1]),
        s * float(frame.p[2]),
    )
    gamma = (
        s * float(frame.q[0]),
        s * float(frame.q[1]),
        s * float(frame.q[2]),
    )

    ax = float(cone_geometry.apex[0])
    ay = float(cone_geometry.apex[1])
    az = float(cone_geometry.apex[2])

    tau = (
        float(voi_bounds.xmin) - ax,
        float(voi_bounds.xmax) - ax,
        float(voi_bounds.ymin) - ay,
        float(voi_bounds.ymax) - ay,
        float(voi_bounds.zmin) - az,
        float(voi_bounds.zmax) - az,
    )

    return ConeBoxCoefficients(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        tau=tau,
        apex=(ax, ay, az),
    )



def _collect_breakpoint_candidates(coeff: ConeBoxCoefficients) -> list[float]:
    """Collect all breakpoint candidates from denominator zeros and
    cross-axis face crossovers."""
    candidates: list[float] = []

    # 1. Denominator zeros: w_c(xi) = 0 for c in {0, 1, 2}.
    for c in range(3):
        candidates.extend(
            solve_linear_trig_zeros(coeff.alpha[c], coeff.beta[c], coeff.gamma[c])
        )

    # 2. Cross-axis face crossovers.
    #    For faces f1 on axis c1, f2 on axis c2 (c1 != c2):
    #       tau[f1] * w_{c2}(xi) = tau[f2] * w_{c1}(xi)
    #    Cross-multiplied: (tau1*alpha2 - tau2*alpha1) + ... cos + ... sin = 0
    for c1 in range(3):
        for c2 in range(c1 + 1, 3):
            for f1_off in (0, 1):
                f1 = 2 * c1 + f1_off
                tau1 = coeff.tau[f1]
                for f2_off in (0, 1):
                    f2 = 2 * c2 + f2_off
                    tau2 = coeff.tau[f2]
                    A_cross = tau1 * coeff.alpha[c2] - tau2 * coeff.alpha[c1]
                    B_cross = tau1 * coeff.beta[c2] - tau2 * coeff.beta[c1]
                    C_cross = tau1 * coeff.gamma[c2] - tau2 * coeff.gamma[c1]
                    candidates.extend(
                        solve_linear_trig_zeros(A_cross, B_cross, C_cross)
                    )

    return candidates


def _dedup_sorted_angles(candidates: list[float]) -> tuple[float, ...]:
    """Sort, deduplicate, and normalize breakpoint candidates to [0, 2*pi)."""
    candidates.sort()
    if not candidates:
        return ()

    deduped: list[float] = [candidates[0]]
    for xi in candidates[1:]:
        if xi - deduped[-1] > _DEDUP_TOL:
            deduped.append(xi)

    # Wrap-around: last entry near 2*pi may duplicate the first entry near 0.
    if (
        len(deduped) >= 2
        and (_TWO_PI - deduped[-1] + deduped[0]) <= _DEDUP_TOL
    ):
        deduped.pop()

    return tuple(deduped)


def compute_breakpoints(coeff: ConeBoxCoefficients) -> tuple[float, ...]:
    """Compute all exact azimuth breakpoints for the given coefficients.

    Breakpoints are returned sorted and deduplicated in ``[0, 2*pi)``.
    """
    return _dedup_sorted_angles(_collect_breakpoint_candidates(coeff))





def evaluate_fixed_azimuth(
    coeff: ConeBoxCoefficients,
    xi: float,
) -> FixedAzimuthInterval:
    """Compute exact ``ell^-(xi)``, ``ell^+(xi)``, and ``A(xi)``.

    Uses half-open VOI semantics for the ``w_c = 0`` parallel-slab case.
    For nonzero ``w_c``, face distances and openness are exact.
    """
    xi = float(xi)
    cos_xi = math.cos(xi)
    sin_xi = math.sin(xi)

    ell_minus = 0.0
    ell_plus = math.inf
    lower_face: int | None = None
    upper_face: int | None = None

    _empty = FixedAzimuthInterval(
        nonempty=False,
        ell_minus=0.0,
        ell_plus=0.0,
        area_profile=0.0,
        lower_face=None,
        upper_face=None,
        lower_closed=False,
        upper_closed=False,
    )

    for c in range(3):
        wc = coeff.alpha[c] + coeff.beta[c] * cos_xi + coeff.gamma[c] * sin_xi
        tau_lo = coeff.tau[2 * c]
        tau_hi = coeff.tau[2 * c + 1]

        if wc > 0.0:
            entry = tau_lo / wc
            exit_ = tau_hi / wc
            entry_face = 2 * c
            exit_face = 2 * c + 1
        elif wc < 0.0:
            entry = tau_hi / wc
            exit_ = tau_lo / wc
            entry_face = 2 * c + 1
            exit_face = 2 * c
        else:
            # w_c = 0: ray parallel to face pair.
            # Half-open containment: lo <= apex_c < hi  <=>  tau_lo <= 0 and tau_hi > 0.
            if tau_lo <= 0.0 and tau_hi > 0.0:
                continue  # apex in slab, no radial constraint from this coord
            return _empty

        if entry > ell_minus:
            ell_minus = entry
            lower_face = entry_face
        if exit_ < ell_plus:
            ell_plus = exit_
            upper_face = exit_face

    if not (ell_minus < ell_plus):
        return FixedAzimuthInterval(
            nonempty=False,
            ell_minus=ell_minus,
            ell_plus=ell_plus,
            area_profile=0.0,
            lower_face=lower_face,
            upper_face=upper_face,
            lower_closed=False,
            upper_closed=False,
        )

    if not math.isfinite(ell_plus):
        raise ValueError(
            "ell_plus is infinite for bounded box -- this should not occur"
        )

    # Endpoint openness from half-open box semantics.
    # Lower faces (even index) are closed; upper faces (odd index) are open.
    if lower_face is None:
        lower_closed = False  # ray_domain: ell > 0 is open
    else:
        lower_closed = _FACE_IS_LOWER[lower_face]

    if upper_face is None:
        upper_closed = False
    else:
        upper_closed = _FACE_IS_LOWER[upper_face]

    area = ell_plus * ell_plus - ell_minus * ell_minus

    return FixedAzimuthInterval(
        nonempty=True,
        ell_minus=ell_minus,
        ell_plus=ell_plus,
        area_profile=area,
        lower_face=lower_face,
        upper_face=upper_face,
        lower_closed=lower_closed,
        upper_closed=upper_closed,
    )


def evaluate_area_profile(coeff: ConeBoxCoefficients, xi: float) -> float:
    """Return ``A_k(xi) = ell^+(xi)**2 - ell^-(xi)**2``."""
    return evaluate_fixed_azimuth(coeff, xi).area_profile




def _sign_of(x: float) -> int:
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


def build_decomposition(
    coeff: ConeBoxCoefficients,
) -> ConeBoxSurfaceDecomposition:
    """Build the complete azimuth breakpoint / segment decomposition.

    Each segment is an open arc between consecutive breakpoints where the
    sign pattern, active faces, and emptiness are constant.
    """
    breakpoints = compute_breakpoints(coeff)
    n = len(breakpoints)

    def _classify(mid_xi: float) -> AzimuthSegment:
        iv = evaluate_fixed_azimuth(coeff, mid_xi)
        cos_m = math.cos(mid_xi)
        sin_m = math.sin(mid_xi)
        sp = tuple(
            _sign_of(
                coeff.alpha[c] + coeff.beta[c] * cos_m + coeff.gamma[c] * sin_m
            )
            for c in range(3)
        )
        return AzimuthSegment(
            xi_start=0.0,  # placeholder, overwritten below
            xi_end=0.0,
            nonempty=iv.nonempty,
            sign_pattern=sp,  # type: ignore[arg-type]
            lower_face=iv.lower_face,
            upper_face=iv.upper_face,
            lower_closed=iv.lower_closed,
            upper_closed=iv.upper_closed,
        )

    if n == 0:
        seg = _classify(math.pi)
        seg = AzimuthSegment(
            xi_start=0.0,
            xi_end=_TWO_PI,
            nonempty=seg.nonempty,
            sign_pattern=seg.sign_pattern,
            lower_face=seg.lower_face,
            upper_face=seg.upper_face,
            lower_closed=seg.lower_closed,
            upper_closed=seg.upper_closed,
        )
        return ConeBoxSurfaceDecomposition(
            coefficients=coeff,
            breakpoints=(),
            segments=(seg,),
        )

    segments: list[AzimuthSegment] = []

    for i in range(n):
        xi_start = breakpoints[i]
        xi_end = breakpoints[(i + 1) % n]

        # Midpoint of the arc (handling wrap-around for last segment).
        if xi_end > xi_start:
            mid_xi = 0.5 * (xi_start + xi_end)
        else:
            mid_xi = 0.5 * (xi_start + xi_end + _TWO_PI)
            if mid_xi >= _TWO_PI:
                mid_xi -= _TWO_PI

        tmp = _classify(mid_xi)
        segments.append(
            AzimuthSegment(
                xi_start=xi_start,
                xi_end=xi_end,
                nonempty=tmp.nonempty,
                sign_pattern=tmp.sign_pattern,
                lower_face=tmp.lower_face,
                upper_face=tmp.upper_face,
                lower_closed=tmp.lower_closed,
                upper_closed=tmp.upper_closed,
            )
        )

    return ConeBoxSurfaceDecomposition(
        coefficients=coeff,
        breakpoints=breakpoints,
        segments=tuple(segments),
    )




def build_cone_box_decomposition(
    cone_geometry: ConeLocalGeometry,
    voi_bounds: VoiBounds,
) -> ConeBoxSurfaceDecomposition:
    """Build the complete decomposition from cone geometry and VOI bounds."""
    coeff = build_cone_box_coefficients(cone_geometry, voi_bounds)
    return build_decomposition(coeff)



__all__ = [
    "FACE_NAMES",
    "AzimuthSegment",
    "ConeBoxCoefficients",
    "ConeBoxSurfaceDecomposition",
    "FixedAzimuthInterval",
    "build_cone_box_coefficients",
    "build_cone_box_decomposition",
    "build_decomposition",
    "compute_breakpoints",
    "evaluate_area_profile",
    "evaluate_fixed_azimuth",
    "solve_linear_trig_zeros",
]
