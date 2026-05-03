"""Exact azimuth-law primitives and finite-precision direct-draw sampler.

This module provides the exact segment-mass / partial-CDF primitives and a
finite-precision direct-draw realization for the narrow bounded-box
exact-surface proposal path.  It builds on the landed cone-box surface
decomposition (``soe.geometry.cone_box_surface``) and azimuth-law machinery
(``soe.geometry.cone_box_azimuth_law``).

Mathematically, the sampler is defined by the exact surface-area law.  In code, the azimuth inversion step is realized numerically
by a safeguarded bracketed solve of the exact per-segment partial CDF
``C_{k,j}``, using bisection as the conservative fallback step.  The mass
assembly, partial CDF evaluation, and conditional radial inversion remain
exact given the already-landed geometry primitives.

All densities and masses include the exact ``(s_k / 2)`` factor from the
cone surface-area element ``dmu = s * ell * dell * dxi``.

Primitives provided for one event *k* and one bounded box:

- ``A_{k,j}(xi)``: exact segmentwise azimuth surface-area density
- ``M_{k,j}``: exact segment mass
- ``S_{k,j}``: exact cumulative mass
- ``C_{k,j}(xi)``: exact per-segment partial CDF
- ``C^{-1}_{k,j}(u)``: finite-precision numerical inverse of the exact
  per-segment CDF (safeguarded bracketed solve with bisection fallback)

Direct-draw functions:

- ``direct_sample_area_weighted_azimuth_and_interval``: finite-precision
  azimuth sampler from the exact segmentwise law, returning
  ``(xi, (ell_minus, ell_plus), ell2_width)``.
- ``direct_sample_exact_bounded_box_point``: full finite-precision direct
  draw (numerical azimuth inversion + exact conditional radial + authoritative
  world mapping + support certification), returning a world-space ``(3,)``
  ndarray.

The direct-draw path is the default on the
``UniformSurfaceReferenceProposalBackend`` for the narrow supported
bounded-box scope.  The rejection-based backend remains available
as an explicit fallback via ``_use_direct_sampler=False``.  No proposal
law, target law, acceptance law, or kernel contract changed.
"""

from __future__ import annotations

import bisect
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from soe.adapters.voi_voxel import VoiBounds
from soe.contracts import EventObj
from soe.geometry.cone import (
    ConeLocalGeometry,
    build_cone_local_geometry,
    cone_phi,
    cone_surface_point_from_local_geometry,
    is_admissible_point,
)
from soe.geometry.cone_box_surface import (
    ConeBoxSurfaceDecomposition,
    build_cone_box_decomposition,
    evaluate_fixed_azimuth,
)
from soe.geometry.cone_box_azimuth_law import (
    AzimuthLaw,
    _build_inv_g_sq_primitive_context,
    _integrate_inv_g_sq_from_context,
    build_azimuth_law,
    integrate_inv_g_sq,
)
from soe.geometry.ray_box import Interval

if TYPE_CHECKING:
    from soe.soe.uniform_surface_reference import _ExactBoundedBoxProposalSupport

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi
# Bisection settings for numerical inversion of the exact segment partial CDF.
_BISECT_ABS_TOL = 1e-12
_BISECT_MAX_ITER = 64
_DIRECT_CERT_RETRY_MAX = 32
_DIRECT_NUMERICAL_PHI_SOFT_TOL = 1e-10

# ---------------------------------------------------------------------------
# Scaffold sentinel (preserved from earlier prompt)
# ---------------------------------------------------------------------------


class DirectSamplerNotReady(RuntimeError):
    """Raised when the direct-sampler draw path is invoked before implementation."""


class DirectSamplerError(RuntimeError):
    """Raised on internal direct-sampler failures."""


def _segment_wraps(xi_start: float, xi_end: float) -> bool:
    """True when the visible segment crosses ``0 / 2*pi``."""
    return xi_end <= xi_start


def _compute_exact_segment_mass(
    *,
    s: float,
    regime,
) -> float:
    """Assemble exact ``M_{k,j}`` from visible-segment primitive evaluations."""
    if not regime.nonempty:
        return 0.0

    half_s = s / 2.0
    wraps = _segment_wraps(regime.xi_start, regime.xi_end)
    upper_int = integrate_inv_g_sq(
        regime.xi_start,
        regime.xi_end,
        regime.a_upper,
        regime.b_upper,
        regime.c_upper,
        wraps=wraps,
    )

    if regime.lower_is_apex:
        # floor/face case:
        # M_{k,j} = (s_k / 2) * tau_{h+}^2 * I_{h+}(beta_- , beta_+)
        return half_s * regime.tau_upper**2 * upper_int

    lower_int = integrate_inv_g_sq(
        regime.xi_start,
        regime.xi_end,
        regime.a_lower,
        regime.b_lower,
        regime.c_lower,
        wraps=wraps,
    )
    # face/face case:
    # M_{k,j} = (s_k / 2) * [tau_{h+}^2 * I_{h+} - tau_{h-}^2 * I_{h-}]
    return half_s * (
        regime.tau_upper**2 * upper_int - regime.tau_lower**2 * lower_int
    )


@dataclass(frozen=True, slots=True)
class _SegmentPartialCDFContext:
    """Segment-fixed terms for repeated partial-CDF evaluations."""

    half_s: float
    xi_start: float
    segment_wraps: bool
    upper_weight: float
    upper_context: object
    lower_weight: float
    lower_context: object | None


def _build_segment_partial_cdf_context(
    *,
    s: float,
    regime,
) -> _SegmentPartialCDFContext:
    """Precompute segment constants for inversion-loop partial CDF calls."""
    lower_context = None
    lower_weight = 0.0
    if not regime.lower_is_apex:
        lower_context = _build_inv_g_sq_primitive_context(
            regime.xi_start,
            regime.a_lower,
            regime.b_lower,
            regime.c_lower,
        )
        lower_weight = regime.tau_lower**2

    return _SegmentPartialCDFContext(
        half_s=s / 2.0,
        xi_start=regime.xi_start,
        segment_wraps=_segment_wraps(regime.xi_start, regime.xi_end),
        upper_weight=regime.tau_upper**2,
        upper_context=_build_inv_g_sq_primitive_context(
            regime.xi_start,
            regime.a_upper,
            regime.b_upper,
            regime.c_upper,
        ),
        lower_weight=lower_weight,
        lower_context=lower_context,
    )


def _evaluate_segment_partial_cdf_from_context(
    context: _SegmentPartialCDFContext,
    xi: float,
) -> float:
    """Evaluate exact ``C_{k,j}(xi)`` from a fixed-segment cached context."""
    sub_wraps = context.segment_wraps and (xi < context.xi_start)
    upper_int = _integrate_inv_g_sq_from_context(
        xi,
        context.upper_context,
        wraps=sub_wraps,
    )

    if context.lower_context is None:
        return context.half_s * context.upper_weight * upper_int

    lower_int = _integrate_inv_g_sq_from_context(
        xi,
        context.lower_context,
        wraps=sub_wraps,
    )
    return context.half_s * (
        context.upper_weight * upper_int - context.lower_weight * lower_int
    )


def _segment_fraction_to_xi(xi_start: float, arc_len: float, frac: float) -> float:
    """Map a counterclockwise segment fraction back to normalized azimuth."""
    xi = xi_start + frac * arc_len
    if xi >= _TWO_PI:
        xi -= _TWO_PI
    return xi


def _safeguarded_secant_fraction(
    *,
    lo: float,
    hi: float,
    val_lo: float,
    val_hi: float,
    target_mass: float,
    arc_len: float,
    remaining_iterations: int,
) -> float | None:
    """Return a strictly bracketed secant fraction when it is safe to use.

    The returned trial is accepted only if it is finite, strictly inside the
    active bracket, and still leaves enough remaining iterations for pure
    bisection to hit the unchanged absolute azimuth tolerance even in the
    wider of the two possible child brackets.
    """
    denom = val_hi - val_lo
    if not (math.isfinite(denom) and denom > 0.0):
        return None

    rel = (target_mass - val_lo) / denom
    if not (math.isfinite(rel) and 0.0 < rel < 1.0):
        return None

    candidate = lo + rel * (hi - lo)
    if not (math.isfinite(candidate) and lo < candidate < hi):
        return None

    worst_child_width = max(candidate - lo, hi - candidate)
    terminal_width = math.ldexp(
        worst_child_width * arc_len,
        -max(0, remaining_iterations),
    )
    if terminal_width > _BISECT_ABS_TOL:
        return None

    return candidate


# ---------------------------------------------------------------------------
# Open-interval RNG helper
# ---------------------------------------------------------------------------


def _sample_open01(rng: np.random.Generator) -> float:
    """Return one draw in ``(0, 1)`` to keep sampled values off endpoints."""
    while True:
        draw = float(rng.random())
        if 0.0 < draw < 1.0:
            return draw


# ---------------------------------------------------------------------------
# Segment selection by exact cumulative mass
# ---------------------------------------------------------------------------


def _select_segment_by_mass(
    primitives: DirectSamplerPrimitives,
    u_mass: float,
) -> int:
    """Select the unique visible segment *j* such that ``S_j <= u < S_j + M_j``.

    Uses binary search on the cumulative mass table.  Fails closed if
    no positive-mass segment contains the target mass.
    """
    j = bisect.bisect_right(primitives.cumulative_masses, u_mass) - 1
    n_segs = len(primitives.segment_masses)
    if j < 0 or j >= n_segs:
        raise DirectSamplerError(
            "direct-sampler: segment selection out of range"
        )
    if primitives.segment_masses[j] <= 0.0:
        raise DirectSamplerError(
            "direct-sampler: segment selection landed on zero-mass segment"
        )
    return j


# ---------------------------------------------------------------------------
# Direct draw: exact masses/CDFs + numerical azimuth inversion (steps 3.1-3.3)
# ---------------------------------------------------------------------------


def direct_sample_area_weighted_azimuth_and_interval(
    *,
    event: EventObj,
    allowed_region: VoiBounds,
    rng: np.random.Generator,
    audit: Any | None = None,
    _proposal_support: _ExactBoundedBoxProposalSupport | None = None,
    _decomposition: ConeBoxSurfaceDecomposition | None = None,
    _primitives: DirectSamplerPrimitives | None = None,
) -> tuple[float, Interval, float]:
    """Finite-precision azimuth sampler for the bounded-box path.

    Performs the segmentwise azimuth sequence:

    1. Build exact primitives (segment masses, cumulative masses, total mass).
    2. Draw ``u_mass`` from ``(0, T_k)`` and select visible segment by mass.
    3. Numerically invert the exact segment partial CDF to get azimuth ``xi``.
    4. Evaluate exact radial interval at ``xi``.

    Returns ``(xi, (ell_minus, ell_plus), ell2_width)`` -- the same triple as
    the rejection-based ``_sample_area_weighted_azimuth_and_interval``.

    This is a finite-precision realization of the exact segmentwise azimuth
    law from the note: the visible-segment masses and per-segment CDF are
    exact, while the inverse ``C_{k,j}^{-1}`` is realized numerically by a
    safeguarded bracketed solve with bisection fallback.  Fails closed if
    total mass is non-positive, if the scope is unsupported, or if the
    returned azimuth yields an empty radial interval.
    """
    if not isinstance(allowed_region, VoiBounds):
        raise TypeError(
            "direct-sampler only supports bounded VoiBounds regions"
        )

    if _primitives is not None:
        if not isinstance(_primitives, DirectSamplerPrimitives):
            raise TypeError("_primitives must be a DirectSamplerPrimitives")
        primitives = _primitives
        decomposition = primitives.azimuth_law.decomposition
    else:
        # Build cone geometry
        if _proposal_support is not None:
            cone_geometry = _proposal_support.cone_geometry
        else:
            cone_geometry = build_cone_local_geometry(event)

        # Build decomposition
        if _decomposition is not None:
            decomposition = _decomposition
        else:
            decomposition = build_cone_box_decomposition(cone_geometry, allowed_region)

        # Build primitives (exact segment masses, cumulative masses, total mass)
        primitives = build_direct_sampler_primitives(decomposition, cone_geometry.s)

    # Step 3.1: Fail closed if total admissible surface mass is non-positive
    if primitives.total_mass <= 0.0:
        raise DirectSamplerError(
            "direct-sampler: zero total admissible surface mass "
            "(event cone has no visible surface in the bounded box)"
        )

    # Step 3.2: Draw u_mass from (0, T_k) and select visible segment by mass
    u_mass = _sample_open01(rng) * primitives.total_mass
    j = _select_segment_by_mass(primitives, u_mass)
    u_seg = u_mass - primitives.cumulative_masses[j]

    # Step 3.3: Numerically invert the exact segment partial CDF to get xi
    xi = invert_segment_partial_cdf(primitives, j, u_seg)

    # Evaluate exact radial interval at the drawn azimuth
    iv = evaluate_fixed_azimuth(decomposition.coefficients, xi)
    if not iv.nonempty:
        raise DirectSamplerError(
            "direct-sampler: azimuth drawn from positive-mass segment "
            "yielded empty radial interval"
        )

    ell_minus = iv.ell_minus
    ell_plus = iv.ell_plus
    ell2_width = iv.area_profile  # ell_plus^2 - ell_minus^2

    if ell2_width <= 0.0:
        raise DirectSamplerError(
            "direct-sampler: azimuth drawn from positive-mass segment "
            "yielded zero or negative area profile"
        )

    return xi, (ell_minus, ell_plus), ell2_width


# ---------------------------------------------------------------------------
# Full direct draw (steps 3.1-3.7)
# ---------------------------------------------------------------------------


def direct_sample_exact_bounded_box_point(
    *,
    event: EventObj,
    allowed_region: VoiBounds,
    rng: np.random.Generator,
    audit: Any | None = None,
    _proposal_support: _ExactBoundedBoxProposalSupport | None = None,
    _decomposition: ConeBoxSurfaceDecomposition | None = None,
    _primitives: DirectSamplerPrimitives | None = None,
) -> np.ndarray:
    """Finite-precision realization of the exact bounded-box direct sampler.

    Sequence:

    1. Exact segment-mass selection and numerical azimuth inversion of the
       exact segment CDF (3.1-3.3).
    2. Exact conditional radial draw from density proportional to ``ell`` on
       the admissible interval:
       ``ell = sqrt(ell_minus^2 + u * ell2_width)`` (step 3.4).
    3. Authoritative world-space mapping via ``Psi_k(ell, xi)`` (step 3.5).
    4. Support certification: sampled point must be admissible (step 3.6).

    Returns the sampled world-space point as a ``(3,)`` ndarray.

    The implemented target objects are the exact segmentwise masses, exact
    segmentwise CDFs, and exact conditional radial law from the note.  The
    live azimuth draw is their finite-precision numerical realization via a
    safeguarded bracketed solve of ``C_{k,j}``.  The surrounding proposal path
    continues to use the symmetric-proposal contract and therefore reports
    ``proposal_ratio = 1``.
    """
    for attempt in range(_DIRECT_CERT_RETRY_MAX):
        # Steps 3.1-3.3: exact mass selection + numerical azimuth inversion
        if audit is not None:
            t0 = time.perf_counter()
        xi, interval, ell2_width = direct_sample_area_weighted_azimuth_and_interval(
            event=event,
            allowed_region=allowed_region,
            rng=rng,
            audit=audit,
            _proposal_support=_proposal_support,
            _decomposition=_decomposition,
            _primitives=_primitives,
        )
        if audit is not None:
            audit.direct_sampler_azimuth_seconds += time.perf_counter() - t0
            audit.direct_sampler_azimuth_count += 1
        ell_minus, ell_plus = interval

        # Step 3.4: Exact conditional radial draw
        # f(ell | xi) = 2*ell / (ell_plus^2 - ell_minus^2) on [ell_minus, ell_plus]
        # CDF: F(ell) = (ell^2 - ell_minus^2) / (ell_plus^2 - ell_minus^2)
        # Inverse: ell = sqrt(ell_minus^2 + u_rad * (ell_plus^2 - ell_minus^2))
        if audit is not None:
            t0 = time.perf_counter()
        u_rad = _sample_open01(rng)
        ell_sq = (ell_minus * ell_minus) + u_rad * ell2_width
        ell = math.sqrt(ell_sq)
        if audit is not None:
            audit.direct_sampler_radial_seconds += time.perf_counter() - t0
            audit.direct_sampler_radial_count += 1

        # Step 3.5: Authoritative world-space mapping via Psi_k(ell, xi)
        if audit is not None:
            t0 = time.perf_counter()
        if _proposal_support is not None:
            cone_geometry = _proposal_support.cone_geometry
        else:
            cone_geometry = build_cone_local_geometry(event)

        point = cone_surface_point_from_local_geometry(cone_geometry, ell=ell, xi=xi)
        if audit is not None:
            audit.direct_sampler_point_seconds += time.perf_counter() - t0
            audit.direct_sampler_point_count += 1

        # Step 3.6: Exact support certification
        if audit is not None:
            t0 = time.perf_counter()

        if is_admissible_point(point, event, allowed_region):
            if audit is not None:
                audit.direct_sampler_certify_seconds += time.perf_counter() - t0
                audit.direct_sampler_certify_count += 1
            return point

        d = point - event.apex
        s_dot = float(np.dot(d, event.axis))
        phi = cone_phi(point, event)
        contains = allowed_region.contains(point)

        if audit is not None:
            audit.direct_sampler_certify_seconds += time.perf_counter() - t0
            audit.direct_sampler_certify_count += 1

        if contains and s_dot >= 0.0 and abs(phi) <= _DIRECT_NUMERICAL_PHI_SOFT_TOL:
            continue

        raise DirectSamplerError(
            "direct-sampler: sampled point failed admissibility certification\n"
            f"attempt={attempt}\n"
            f"xi={xi!r}\n"
            f"ell_minus={ell_minus!r}\n"
            f"ell_plus={ell_plus!r}\n"
            f"ell={ell!r}\n"
            f"point={point!r}\n"
            f"contains={contains!r}\n"
            f"phi={phi!r}\n"
            f"abs_phi={abs(phi)!r}\n"
            f"s_dot={s_dot!r}\n"
        )

    raise DirectSamplerError(
        "direct-sampler: repeated numerical admissibility failures "
        f"after {_DIRECT_CERT_RETRY_MAX} retries"
    )


# ---------------------------------------------------------------------------
# Primitives data type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DirectSamplerPrimitives:
    """Complete exact azimuth-law primitives for one event and one bounded box.

    All masses and densities include the exact ``(s_k / 2)`` factor from the
    surface-area element ``dmu = s * ell * dell * dxi``:

        ``A_{k,j}(xi) = (s / 2) * (ell_plus(xi)^2 - ell_minus(xi)^2)``

    Attributes
    ----------
    azimuth_law:
        The underlying reduced azimuth law (without the ``s/2`` factor).
    s:
        Cone parameter ``s_k = sqrt(1 - lambda_k^2)``.
    segment_masses:
        ``segment_masses[j] = M_{k,j}``, the exact integral of
        ``A_{k,j}(xi)`` over segment *j*.
    cumulative_masses:
        ``cumulative_masses[j] = S_{k,j} = sum(segment_masses[0:j])``.
        In particular, ``cumulative_masses[0] = 0``.
    total_mass:
        Total admissible surface area ``S_{k,m} = sum of all segment masses``.
    """

    azimuth_law: AzimuthLaw
    s: float
    segment_masses: tuple[float, ...]
    cumulative_masses: tuple[float, ...]
    total_mass: float


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_direct_sampler_primitives(
    decomposition: ConeBoxSurfaceDecomposition,
    s: float,
) -> DirectSamplerPrimitives:
    """Build complete exact azimuth-law primitives for one event and one box.

    Parameters
    ----------
    decomposition:
        The cone-box surface decomposition from ``cone_box_surface.py``.
    s:
        The cone parameter ``s_k = sqrt(1 - lambda_k^2)``.  Must be
        positive (nondegenerate cone).

    Raises
    ------
    ValueError:
        If ``s <= 0`` (degenerate cone not supported on the exact path).
    TypeError:
        If ``decomposition`` is not a ``ConeBoxSurfaceDecomposition``.
    """
    if not isinstance(decomposition, ConeBoxSurfaceDecomposition):
        raise TypeError("decomposition must be a ConeBoxSurfaceDecomposition")
    if s <= 0.0:
        raise ValueError("s must be positive (nondegenerate cone)")

    law = build_azimuth_law(decomposition)
    masses = tuple(
        _compute_exact_segment_mass(s=s, regime=regime) for regime in law.regimes
    )

    cum: list[float] = [0.0]
    for m in masses:
        cum.append(cum[-1] + m)
    total = cum[-1]
    cumulative = tuple(cum[:-1])

    return DirectSamplerPrimitives(
        azimuth_law=law,
        s=s,
        segment_masses=masses,
        cumulative_masses=cumulative,
        total_mass=total,
    )


def build_direct_sampler_primitives_from_geometry(
    cone_geometry: ConeLocalGeometry,
    voi_bounds: VoiBounds,
) -> DirectSamplerPrimitives:
    """Build complete primitives from cone geometry and VOI bounds."""
    decomposition = build_cone_box_decomposition(cone_geometry, voi_bounds)
    return build_direct_sampler_primitives(decomposition, cone_geometry.s)


# ---------------------------------------------------------------------------
# Segment density evaluator
# ---------------------------------------------------------------------------


def evaluate_segment_density(
    primitives: DirectSamplerPrimitives,
    j: int,
    xi: float,
) -> float:
    """Evaluate exact ``A_{k,j}(xi) = (s/2) * (ell_plus(xi)^2 - ell_minus(xi)^2)``.

    Returns ``0.0`` on empty segments.
    """
    regime = primitives.azimuth_law.regimes[j]
    if not regime.nonempty:
        return 0.0

    coeff = primitives.azimuth_law.decomposition.coefficients
    iv = evaluate_fixed_azimuth(coeff, xi)
    return (primitives.s / 2.0) * iv.area_profile


# ---------------------------------------------------------------------------
# Per-segment partial CDF evaluator
# ---------------------------------------------------------------------------


def evaluate_segment_partial_cdf(
    primitives: DirectSamplerPrimitives,
    j: int,
    xi: float,
) -> float:
    """Evaluate ``C_{k,j}(xi) = integral_{beta_{j-1}}^{xi} A_k(eta) deta``.

    Returns a value in ``[0, M_{k,j}]`` for ``xi`` within segment *j*.
    Returns ``0.0`` on empty segments.
    """
    regime = primitives.azimuth_law.regimes[j]
    if not regime.nonempty:
        return 0.0

    context = _build_segment_partial_cdf_context(
        s=primitives.s,
        regime=regime,
    )
    return _evaluate_segment_partial_cdf_from_context(context, xi)


# ---------------------------------------------------------------------------
# Per-segment inverse CDF (numerical realization)
# ---------------------------------------------------------------------------


def invert_segment_partial_cdf(
    primitives: DirectSamplerPrimitives,
    j: int,
    target_mass: float,
) -> float:
    """Approximate ``C_{k,j}^{-1}(target_mass)`` for segment *j*.

    Uses a deterministic bracket-preserving hybrid solve on the exact monotone
    partial CDF.  Each iteration tries a secant-style interpolation step from
    the active bracket only when that step is finite, strictly interior, and
    still leaves enough remaining iterations for pure bisection to hit the
    unchanged absolute tolerance.  Otherwise it immediately takes the current
    midpoint/bisection step.  The input CDF is exact; the returned azimuth is a
    finite-precision numerical approximation controlled by the module-level
    bisection tolerance and iteration cap.

    Parameters
    ----------
    primitives:
        The direct-sampler primitives object.
    j:
        Segment index.
    target_mass:
        Target mass value in ``(0, M_{k,j})``.

    Returns
    -------
    xi:
        An azimuth angle in the segment whose exact partial-CDF value matches
        ``target_mass`` up to the configured numerical tolerance.

    Raises
    ------
    ValueError:
        If the segment is empty, has zero or negative mass,
        or ``target_mass`` is outside ``(0, M_{k,j})``.
    """
    regime = primitives.azimuth_law.regimes[j]
    if not regime.nonempty:
        raise ValueError(f"Cannot invert CDF on empty segment {j}")

    M_j = primitives.segment_masses[j]
    if M_j <= 0.0:
        raise ValueError(
            f"Cannot invert CDF on zero-mass segment {j}: M={M_j}"
        )

    if not (0.0 < target_mass < M_j):
        raise ValueError(
            f"target_mass={target_mass} outside (0, {M_j}) for segment {j}"
        )

    xi_a = regime.xi_start
    xi_b = regime.xi_end
    wraps = xi_b <= xi_a
    arc_len = (xi_b + _TWO_PI - xi_a) if wraps else (xi_b - xi_a)
    cdf_context = _build_segment_partial_cdf_context(
        s=primitives.s,
        regime=regime,
    )

    lo = 0.0  # fraction of arc from xi_a
    hi = 1.0
    val_lo = 0.0
    val_hi = M_j

    for iteration in range(_BISECT_MAX_ITER):
        remaining_iterations = _BISECT_MAX_ITER - iteration - 1
        frac = _safeguarded_secant_fraction(
            lo=lo,
            hi=hi,
            val_lo=val_lo,
            val_hi=val_hi,
            target_mass=target_mass,
            arc_len=arc_len,
            remaining_iterations=remaining_iterations,
        )
        if frac is None:
            frac = 0.5 * (lo + hi)

        xi = _segment_fraction_to_xi(xi_a, arc_len, frac)

        val = _evaluate_segment_partial_cdf_from_context(cdf_context, xi)
        if not math.isfinite(val):
            raise DirectSamplerError(
                "direct-sampler: non-finite partial CDF during inversion"
            )

        if val < target_mass:
            lo = frac
            val_lo = val
        else:
            hi = frac
            val_hi = val

        if (hi - lo) * arc_len < _BISECT_ABS_TOL:
            break

    frac = 0.5 * (lo + hi)
    return _segment_fraction_to_xi(xi_a, arc_len, frac)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DirectSamplerError",
    "DirectSamplerNotReady",
    "DirectSamplerPrimitives",
    "build_direct_sampler_primitives",
    "build_direct_sampler_primitives_from_geometry",
    "direct_sample_area_weighted_azimuth_and_interval",
    "direct_sample_exact_bounded_box_point",
    "evaluate_segment_density",
    "evaluate_segment_partial_cdf",
    "invert_segment_partial_cdf",
]
