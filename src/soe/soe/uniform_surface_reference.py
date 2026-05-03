"""Exactness-first reference backend for uniform surface-area proposals on `X_k`.

The reference proposal is the law that is uniform with respect to admissible
surface measure on the selected event's admissible set `X_k`. This module implements only cases whose exactness can be justified
directly from the repo's already-authoritative contracts.

Current scope:
The landed repository now justifies one narrow exact backend: the bounded-box
`VoiBounds` case when the selected event already has a pointwise-admissible
representative point in the strict-open VOI interior.
That current point is used only as a positive-area support certificate. The
proposal law itself depends only on the selected event and the box geometry.
Non-VoiBounds inputs, boundary-only support certificates, and any broader `Q_k^U`
claim still fail explicitly rather than silently approximating the
proposal law with coordinate-uniform sampling, repair/clipping, or unverified
support broadening.
"""

from __future__ import annotations

import math
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from soe.adapters.voi_voxel import VoiBounds
from soe.contracts import EventObj
from soe.geometry.cone import (
    ConeLocalGeometry,
    build_cone_local_geometry,
    cone_lambda,
    cone_generator_direction_from_local_geometry,
    cone_surface_point_from_local_geometry,
    is_admissible_point,
)
from soe.geometry.ray_box import BoxBounds, Interval, open_box_ray_interval
from soe.soe.direct_surface_sampler import (
    DirectSamplerNotReady,
    DirectSamplerPrimitives,
    build_direct_sampler_primitives_from_geometry,
    direct_sample_area_weighted_azimuth_and_interval,
    direct_sample_exact_bounded_box_point,
)
from soe.soe.proposals import ProposalCandidate
from soe.soe.state import SurvivingEventState

EventIndexSelector = Callable[[SurvivingEventState], int]
EventKeyVec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class _AllowedRegionCacheKey:
    """Immutable cache key for the active bounded-box allowed region."""

    region_type: str
    bounds: tuple[float, float, float, float, float, float]
    grid_shape: tuple[int, int, int] | None


@dataclass(frozen=True, slots=True)
class _EventProposalSupportCacheKey:
    """Immutable cache key for one event under one fixed allowed region."""

    allowed_region: _AllowedRegionCacheKey
    apex: EventKeyVec3
    axis: EventKeyVec3
    theta: float
    species: str | None


@dataclass(frozen=True, slots=True)
class _ExactBoundedBoxSupportCertificate:
    """Immutable static support certificate for one event and fixed VOI box."""

    allowed_region: VoiBounds
    strict_open_box_bounds: BoxBounds


@dataclass(frozen=True, slots=True)
class _ExactBoundedBoxProposalSupport:
    """Cached fixed event/VOI data for the exact bounded-box proposal path."""

    cone_geometry: ConeLocalGeometry
    event_apex: EventKeyVec3
    max_corner_radius_squared: float
    support_certificate: _ExactBoundedBoxSupportCertificate


class UnsupportedUniformSurfaceProposal(RuntimeError):
    """Raised when the exact uniform-surface proposal law is not yet justified."""


def _as_allowed_region(value: object) -> VoiBounds:
    """Validate the authoritative allowed-region boundary object."""
    if not isinstance(value, VoiBounds):
        raise TypeError("allowed_region must be a VoiBounds")
    return value


def _as_state(value: SurvivingEventState) -> SurvivingEventState:
    """Validate the authoritative representative-point state carrier."""
    if not isinstance(value, SurvivingEventState):
        raise TypeError("state must be a SurvivingEventState")
    return value


def _as_event_index(value: object, *, n_events: int) -> int:
    """Validate the selected event index against the surviving-event pool."""
    if isinstance(value, bool):
        raise ValueError("selected event index must be an integer")
    try:
        event_index = operator.index(value)
    except TypeError as exc:
        raise ValueError("selected event index must be an integer") from exc
    if not 0 <= event_index < n_events:
        raise IndexError(f"selected event index out of range: {event_index}")
    return int(event_index)


def _as_event_selector(value: EventIndexSelector) -> EventIndexSelector:
    """Validate the injected event-selection callable."""
    if not callable(value):
        raise TypeError("event_index_selector must be callable")
    return value


def _as_rng(value: object) -> np.random.Generator:
    """Validate the local generator used by the exact backend."""
    if not isinstance(value, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return value


def _selected_event(state: SurvivingEventState, *, event_index: int) -> EventObj:
    """Return the selected canonical event after index validation."""
    index = _as_event_index(event_index, n_events=len(state.events))
    event = state.events[index]
    if not isinstance(event, EventObj):
        raise TypeError("state.events must contain only EventObj instances")
    return event


def _strict_open_box_bounds(allowed_region: VoiBounds) -> BoxBounds:
    """Return the VOI bounds as a strict-open box for exact area sampling."""
    if not isinstance(allowed_region, VoiBounds):
        raise TypeError("allowed_region must be a VoiBounds")
    return (
        (float(allowed_region.xmin), float(allowed_region.xmax)),
        (float(allowed_region.ymin), float(allowed_region.ymax)),
        (float(allowed_region.zmin), float(allowed_region.zmax)),
    )


def _allowed_region_cache_key(allowed_region: VoiBounds) -> _AllowedRegionCacheKey:
    """Return the immutable cache key for one fixed bounded-box region."""
    if not isinstance(allowed_region, VoiBounds):
        raise TypeError("allowed_region must be a VoiBounds")
    return _AllowedRegionCacheKey(
        region_type=type(allowed_region).__name__,
        bounds=(
            float(allowed_region.xmin),
            float(allowed_region.xmax),
            float(allowed_region.ymin),
            float(allowed_region.ymax),
            float(allowed_region.zmin),
            float(allowed_region.zmax),
        ),
        grid_shape=None
        if allowed_region._grid_shape is None
        else tuple(int(component) for component in allowed_region._grid_shape),
    )


def _event_vec3_key(vector: np.ndarray) -> EventKeyVec3:
    """Return an immutable `(x, y, z)` cache key for one finite event vector."""
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _event_proposal_support_cache_key(
    event: EventObj,
    *,
    allowed_region_key: _AllowedRegionCacheKey,
) -> _EventProposalSupportCacheKey:
    """Return the immutable cache key for one event and fixed allowed region."""
    return _EventProposalSupportCacheKey(
        allowed_region=allowed_region_key,
        apex=_event_vec3_key(event.apex),
        axis=_event_vec3_key(event.axis),
        theta=float(event.theta),
        species=event.species,
    )


def _is_strict_open_box_interior(point: np.ndarray, allowed_region: VoiBounds) -> bool:
    """Return whether the point lies strictly inside the VOI on every axis."""
    x, y, z = (float(point[0]), float(point[1]), float(point[2]))
    return (
        allowed_region.xmin < x < allowed_region.xmax
        and allowed_region.ymin < y < allowed_region.ymax
        and allowed_region.zmin < z < allowed_region.zmax
    )


def _max_corner_radius_squared(event: EventObj, allowed_region: VoiBounds) -> float:
    """Return a finite global upper bound for `ell^2` over the bounded box."""
    xmax = float(allowed_region.xmax)
    xmin = float(allowed_region.xmin)
    ymax = float(allowed_region.ymax)
    ymin = float(allowed_region.ymin)
    zmax = float(allowed_region.zmax)
    zmin = float(allowed_region.zmin)
    corners = (
        np.asarray((xmin, ymin, zmin), dtype=float),
        np.asarray((xmin, ymin, zmax), dtype=float),
        np.asarray((xmin, ymax, zmin), dtype=float),
        np.asarray((xmin, ymax, zmax), dtype=float),
        np.asarray((xmax, ymin, zmin), dtype=float),
        np.asarray((xmax, ymin, zmax), dtype=float),
        np.asarray((xmax, ymax, zmin), dtype=float),
        np.asarray((xmax, ymax, zmax), dtype=float),
    )
    return max(float(np.dot(corner - event.apex, corner - event.apex)) for corner in corners)


def _sample_open01(rng: np.random.Generator) -> float:
    """Return one draw in `(0, 1)` to keep sampled `ell` off open-interval edges."""
    while True:
        draw = float(rng.random())
        if 0.0 < draw < 1.0:
            return draw


def _sample_radius_from_radial_endpoints(
    *,
    ell_minus: float,
    ell_plus: float,
    rng: np.random.Generator,
) -> float:
    """Sample `ell` exactly from `ell^2 ~ Uniform(ell_minus^2, ell_plus^2)`."""
    ell_minus = float(ell_minus)
    ell_plus = float(ell_plus)
    if not 0.0 <= ell_minus < ell_plus < math.inf:
        raise UnsupportedUniformSurfaceProposal(
            "exact bounded-box uniform-surface proposals require a nondegenerate "
            "bounded radial interval"
        )

    ell2_width = (ell_plus * ell_plus) - (ell_minus * ell_minus)
    if not ell2_width > 0.0:
        raise UnsupportedUniformSurfaceProposal(
            "exact bounded-box uniform-surface proposals require a positive "
            "radial squared-width"
        )

    u = _sample_open01(rng)
    ell_sq = (ell_minus * ell_minus) + (u * ell2_width)
    return math.sqrt(ell_sq)


def _visible_interval_and_area_profile_for_azimuth(
    event: EventObj,
    allowed_region: VoiBounds,
    *,
    xi: float,
    _cone_geometry: ConeLocalGeometry | None = None,
    _box_bounds: BoxBounds | None = None,
    _event_apex: EventKeyVec3 | None = None,
) -> tuple[Interval, float] | tuple[None, float]:
    """Return the current exact-box fixed-`xi` visibility interval and area profile."""
    cone_geometry = (
        build_cone_local_geometry(event) if _cone_geometry is None else _cone_geometry
    )
    direction = cone_generator_direction_from_local_geometry(cone_geometry, xi)
    strict_open_box_bounds = (
        _box_bounds if _box_bounds is not None else _strict_open_box_bounds(allowed_region)
    )
    event_apex = _event_apex if _event_apex is not None else _event_vec3_key(event.apex)
    interval = open_box_ray_interval(
        event_apex,
        tuple(float(component) for component in direction),
        strict_open_box_bounds,
    )
    if interval is None:
        return None, 0.0

    ell_minus, ell_plus = interval
    if not math.isfinite(ell_plus):
        raise UnsupportedUniformSurfaceProposal(
            "exact bounded-box uniform-surface proposals require bounded finite "
            "ray intervals for every accepted azimuth"
        )
    ell2_width = (ell_plus * ell_plus) - (ell_minus * ell_minus)
    if not ell2_width > 0.0:
        return None, 0.0
    return interval, ell2_width


def _visible_ray_interval_for_azimuth(
    event: EventObj,
    allowed_region: VoiBounds,
    *,
    xi: float,
) -> Interval | None:
    """Return the exact visible open ray interval for one azimuth on the certified subset."""
    interval, _ = _visible_interval_and_area_profile_for_azimuth(
        event,
        allowed_region,
        xi=xi,
    )
    return interval


def _radial_endpoints_for_azimuth(
    event: EventObj,
    allowed_region: VoiBounds,
    *,
    xi: float,
) -> tuple[float, float] | None:
    """Return `(ell_minus, ell_plus)` for one visible azimuth on the certified subset."""
    interval = _visible_ray_interval_for_azimuth(event, allowed_region, xi=xi)
    if interval is None:
        return None
    ell_minus, ell_plus = interval
    return (ell_minus, ell_plus)


def _visible_area_profile_for_azimuth(
    event: EventObj,
    allowed_region: VoiBounds,
    *,
    xi: float,
) -> float:
    """Return `A(xi) = ell_plus^2 - ell_minus^2` for the current exact-box subset."""
    _, ell2_width = _visible_interval_and_area_profile_for_azimuth(
        event,
        allowed_region,
        xi=xi,
    )
    return ell2_width


def _sample_area_weighted_azimuth_and_interval(
    *,
    event: EventObj,
    allowed_region: VoiBounds,
    rng: np.random.Generator,
    audit: Any | None = None,
    _proposal_support: _ExactBoundedBoxProposalSupport | None = None,
) -> tuple[float, Interval, float]:
    """Sample `xi` exactly with density proportional to `A(xi)` on the box subset.

    The proposal uses rejection sampling from the uniform azimuth law on
    `[0, 2*pi)`, with envelope `R_max^2`, where `R_max` is the maximum distance
    from the event apex to any box corner. Since
    `A(xi) = ell_plus(xi)^2 - ell_minus(xi)^2 <= ell_plus(xi)^2 <= R_max^2`,
    the accepted azimuth law is exactly proportional to `A(xi)`.
    """
    proposal_support = _proposal_support
    if proposal_support is None:
        if audit is not None:
            t0 = time.perf_counter()
        max_radius_squared = _max_corner_radius_squared(event, allowed_region)
        if audit is not None:
            audit.max_corner_radius_seconds += time.perf_counter() - t0
            audit.max_corner_radius_count += 1
        cone_geometry = build_cone_local_geometry(event)
        strict_open_box_bounds = _strict_open_box_bounds(allowed_region)
        event_apex = _event_vec3_key(event.apex)
    else:
        max_radius_squared = proposal_support.max_corner_radius_squared
        cone_geometry = proposal_support.cone_geometry
        strict_open_box_bounds = proposal_support.support_certificate.strict_open_box_bounds
        event_apex = proposal_support.event_apex

    if not max_radius_squared > 0.0:
        raise UnsupportedUniformSurfaceProposal(
            "exact bounded-box uniform-surface proposals require a nondegenerate "
            "bounded VOI box"
        )

    rejection_iterations = 0
    if audit is not None:
        loop_t0 = time.perf_counter()

    while True:
        rejection_iterations += 1
        xi = 2.0 * math.pi * float(rng.random())

        # -- runtime-audit: time cone_generator_direction inside interval call --
        interval, ell2_width = _visible_interval_and_area_profile_for_azimuth(
            event,
            allowed_region,
            xi=xi,
            _cone_geometry=cone_geometry,
            _box_bounds=strict_open_box_bounds,
            _event_apex=event_apex,
        )
        if interval is None:
            continue
        if float(rng.random()) * max_radius_squared >= ell2_width:
            continue

        if audit is not None:
            audit.azimuth_rejection_loop_seconds += time.perf_counter() - loop_t0
            audit.azimuth_rejection_loop_count += 1
            audit.azimuth_rejection_iterations_total += rejection_iterations
        return xi, interval, ell2_width


def _build_exact_bounded_box_support_certificate(
    event: EventObj,
    allowed_region: VoiBounds,
) -> _ExactBoundedBoxSupportCertificate:
    """Build the static exact support certificate for the narrow bounded-box subset."""
    lambda_ = cone_lambda(event)
    if not 0.0 < lambda_ < 1.0:
        raise UnsupportedUniformSurfaceProposal(
            "exact bounded-box uniform-surface proposals require "
            "events with 0 < lambda < 1"
        )
    return _ExactBoundedBoxSupportCertificate(
        allowed_region=allowed_region,
        strict_open_box_bounds=_strict_open_box_bounds(allowed_region),
    )


def _require_current_point_exact_bounded_box_support(
    state: SurvivingEventState,
    *,
    event_index: int,
    event: EventObj,
    support_certificate: _ExactBoundedBoxSupportCertificate,
) -> VoiBounds:
    """Validate the current representative point against the cached support subset."""
    allowed_region = support_certificate.allowed_region
    current_point = state.representative_points[event_index]
    if not is_admissible_point(current_point, event, allowed_region):
        raise UnsupportedUniformSurfaceProposal(
            "exact bounded-box uniform-surface proposals require the current "
            "representative point to be pointwise admissible in the selected VOI"
        )
    if not _is_strict_open_box_interior(current_point, allowed_region):
        raise UnsupportedUniformSurfaceProposal(
            "exact bounded-box uniform-surface proposals are currently "
            "certified only when the current representative point lies in the "
            "strict-open VOI interior"
        )
    return allowed_region


def _sample_exact_bounded_box_point(
    *,
    event: EventObj,
    allowed_region: VoiBounds,
    rng: np.random.Generator,
    audit: Any | None = None,
    _proposal_support: _ExactBoundedBoxProposalSupport | None = None,
) -> np.ndarray:
    """Sample exactly from the uniform admissible-surface law on one box case.

    Under `Psi_e(ell, xi) = apex + ell * w_e(xi)`, the cone-surface Jacobian is
    proportional to `ell`. For fixed `xi`, the admissible ray/box intersection
    is the exact open interval returned by `open_box_ray_interval(...)`.
    Therefore the exact surface-area law is obtained by:

    1. sampling `xi` with density proportional to `ell_plus(xi)^2 - ell_minus(xi)^2`,
       done exactly here by rejection sampling against the uniform azimuth law
       with the global envelope `R_max^2`, where `R_max` bounds the distance
       from the event apex to the bounded box;
    2. sampling `ell` conditionally from density proportional to `ell` on the
       selected interval, i.e. `ell^2 ~ Uniform(ell_minus^2, ell_plus^2)`.
    """
    xi, interval, _ = _sample_area_weighted_azimuth_and_interval(
        event=event,
        allowed_region=allowed_region,
        rng=rng,
        audit=audit,
        _proposal_support=_proposal_support,
    )
    ell_minus, ell_plus = interval

    # -- runtime-audit: radius sampling --
    if audit is not None:
        t0 = time.perf_counter()
    ell = _sample_radius_from_radial_endpoints(
        ell_minus=ell_minus,
        ell_plus=ell_plus,
        rng=rng,
    )
    if audit is not None:
        audit.radius_sampling_seconds += time.perf_counter() - t0
        audit.radius_sampling_count += 1

    # -- runtime-audit: surface point construction --
    if audit is not None:
        t0 = time.perf_counter()
    if _proposal_support is None:
        point = cone_surface_point_from_local_geometry(
            build_cone_local_geometry(event),
            ell=ell,
            xi=xi,
        )
    else:
        point = cone_surface_point_from_local_geometry(
            _proposal_support.cone_geometry,
            ell=ell,
            xi=xi,
        )
    if audit is not None:
        audit.surface_point_seconds += time.perf_counter() - t0
        audit.surface_point_count += 1

    # -- runtime-audit: admissibility verification --
    if audit is not None:
        t0 = time.perf_counter()
    admissible = is_admissible_point(point, event, allowed_region)
    if audit is not None:
        audit.admissibility_verify_seconds += time.perf_counter() - t0
        audit.admissibility_verify_count += 1

    if not admissible:
        raise UnsupportedUniformSurfaceProposal(
            "internal exact bounded-box sampler produced a point outside "
            "the authoritative admissible set"
        )
    return point


@dataclass(slots=True)
class UniformSurfaceReferenceProposalBackend:
    """Reference proposal backend for the baseline uniform surface-area law.

    The current repository justifies one narrow exact subset only: bounded
    `VoiBounds` together with a current representative point that already
    certifies positive-area support from the strict-open box interior.
    Within that subset this backend samples from the exact uniform
    surface-measure law on the selected event's admissible set `X_k` and
    returns symmetric proposal metadata. All broader cases still fail
    explicitly.
    """

    certifies_symmetric_proposal: ClassVar[bool] = True

    allowed_region: VoiBounds
    event_index_selector: EventIndexSelector
    rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)
    # runtime-audit: optional timing accumulator, None by default
    _runtime_audit: Any | None = field(default=None, repr=False)
    # Backend selection for the exact bounded-box proposal path.
    # True (default): use the exact inverse-CDF direct-sampler backend.
    # False: use the rejection-based azimuth backend (available as fallback).
    # Outside the supported VoiBounds scope, both paths fail explicitly.
    _use_direct_sampler: bool = field(default=True, repr=False)
    _allowed_region_cache_key: _AllowedRegionCacheKey | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _event_proposal_support_cache: dict[
        _EventProposalSupportCacheKey, _ExactBoundedBoxProposalSupport
    ] = field(default_factory=dict, init=False, repr=False)
    _event_direct_sampler_primitives_cache: dict[
        tuple[int, _EventProposalSupportCacheKey], DirectSamplerPrimitives
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the narrow backend configuration."""
        object.__setattr__(self, "allowed_region", _as_allowed_region(self.allowed_region))
        object.__setattr__(
            self,
            "event_index_selector",
            _as_event_selector(self.event_index_selector),
        )
        object.__setattr__(self, "rng", _as_rng(self.rng))
        object.__setattr__(
            self,
            "_allowed_region_cache_key",
            _allowed_region_cache_key(self.allowed_region),
        )
        # runtime-audit: record provenance once at construction
        audit = self._runtime_audit
        if audit is not None:
            audit.proposal_backend_module = __name__
            audit.proposal_backend_class = type(self).__name__
            audit.allowed_region_type = type(self.allowed_region).__name__
            audit.backend_sampler_kind = (
                "direct" if self._use_direct_sampler else "rejection"
            )

    def _get_or_build_event_proposal_support(
        self,
        event: EventObj,
    ) -> _ExactBoundedBoxProposalSupport:
        """Return cached fixed event/VOI proposal-support data for the active run."""
        allowed_region_key = self._allowed_region_cache_key
        if allowed_region_key is None:
            support_certificate = _build_exact_bounded_box_support_certificate(
                event,
                self.allowed_region,
            )
            return _ExactBoundedBoxProposalSupport(
                cone_geometry=build_cone_local_geometry(event),
                event_apex=_event_vec3_key(event.apex),
                max_corner_radius_squared=_max_corner_radius_squared(
                    event,
                    support_certificate.allowed_region,
                ),
                support_certificate=support_certificate,
            )

        cache_key = _event_proposal_support_cache_key(
            event,
            allowed_region_key=allowed_region_key,
        )
        cached_support = self._event_proposal_support_cache.get(cache_key)
        audit = self._runtime_audit
        if cached_support is not None:
            if audit is not None:
                audit.cone_local_geometry_cache_hits += 1
                audit.max_corner_radius_cache_hits += 1
                audit.support_certificate_cache_hits += 1
            return cached_support

        if audit is not None:
            audit.cone_local_geometry_cache_misses += 1
            audit.max_corner_radius_cache_misses += 1
            audit.support_certificate_cache_misses += 1

        support_certificate = _build_exact_bounded_box_support_certificate(
            event,
            self.allowed_region,
        )
        cone_geometry = build_cone_local_geometry(event)
        if audit is not None:
            t0 = time.perf_counter()
        max_corner_radius_squared = _max_corner_radius_squared(
            event,
            support_certificate.allowed_region,
        )
        if audit is not None:
            audit.max_corner_radius_seconds += time.perf_counter() - t0
            audit.max_corner_radius_count += 1

        cached_support = _ExactBoundedBoxProposalSupport(
            cone_geometry=cone_geometry,
            event_apex=_event_vec3_key(event.apex),
            max_corner_radius_squared=max_corner_radius_squared,
            support_certificate=support_certificate,
        )
        self._event_proposal_support_cache[cache_key] = cached_support
        return cached_support

    def _get_or_build_direct_sampler_primitives(
        self,
        event: EventObj,
        *,
        event_index: int,
        proposal_support: _ExactBoundedBoxProposalSupport,
    ) -> DirectSamplerPrimitives:
        """Return cached direct-sampler primitives for one event/VOI pair."""
        allowed_region_key = self._allowed_region_cache_key
        if allowed_region_key is None:
            return build_direct_sampler_primitives_from_geometry(
                proposal_support.cone_geometry,
                proposal_support.support_certificate.allowed_region,
            )

        cache_key = (
            int(event_index),
            _event_proposal_support_cache_key(
                event,
                allowed_region_key=allowed_region_key,
            ),
        )
        cached_primitives = self._event_direct_sampler_primitives_cache.get(cache_key)
        if cached_primitives is not None:
            return cached_primitives

        primitives = build_direct_sampler_primitives_from_geometry(
            proposal_support.cone_geometry,
            proposal_support.support_certificate.allowed_region,
        )
        self._event_direct_sampler_primitives_cache[cache_key] = primitives
        return primitives

    def select_event_index(self, state: SurvivingEventState) -> int:
        """Delegate event selection to the injected selector."""
        audit = self._runtime_audit
        if audit is not None:
            t0 = time.perf_counter()
        validated_state = _as_state(state)
        result = _as_event_index(
            self.event_index_selector(validated_state),
            n_events=len(validated_state.events),
        )
        if audit is not None:
            audit.event_selection_seconds += time.perf_counter() - t0
            audit.event_selection_count += 1
        return result

    def propose_candidate(
        self,
        state: SurvivingEventState,
        event_index: int,
    ) -> ProposalCandidate:
        """Sample one exact uniform-surface proposal for the selected event.

        Unsupported cases fail explicitly. No coordinate-uniform approximation,
        clipping, repair, or feasibility correction is applied here.
        """
        audit = self._runtime_audit
        validated_state = _as_state(state)
        selected_index = _as_event_index(event_index, n_events=len(validated_state.events))
        event = _selected_event(validated_state, event_index=selected_index)

        # -- runtime-audit: support certificate validation --
        if audit is not None:
            t0 = time.perf_counter()
        proposal_support = self._get_or_build_event_proposal_support(event)
        allowed_region = _require_current_point_exact_bounded_box_support(
            validated_state,
            event_index=selected_index,
            event=event,
            support_certificate=proposal_support.support_certificate,
        )
        if audit is not None:
            audit.support_certificate_seconds += time.perf_counter() - t0
            audit.support_certificate_count += 1

        # -- guarded azimuth backend dispatch --
        # Default: exact inverse-CDF direct-sampler (live).
        # When _use_direct_sampler is False, the rejection-based path is
        # used as an explicit fallback/debug path.
        if self._use_direct_sampler:
            direct_sampler_primitives = self._get_or_build_direct_sampler_primitives(
                event,
                event_index=selected_index,
                proposal_support=proposal_support,
            )
            point = direct_sample_exact_bounded_box_point(
                event=event,
                allowed_region=allowed_region,
                rng=self.rng,
                audit=audit,
                _proposal_support=proposal_support,
                _primitives=direct_sampler_primitives,
            )
            return ProposalCandidate(candidate_point=point)

        point = _sample_exact_bounded_box_point(
            event=event,
            allowed_region=allowed_region,
            rng=self.rng,
            audit=audit,
            _proposal_support=proposal_support,
        )
        return ProposalCandidate(candidate_point=point)


__all__ = [
    "UniformSurfaceReferenceProposalBackend",
    "UnsupportedUniformSurfaceProposal",
]
