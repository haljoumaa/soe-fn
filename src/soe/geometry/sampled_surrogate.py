"""Deterministic sampled-surrogate usability certificates.

This module implements sampled predicates only. A `True` result is a one-sided
certificate that at least one sampled generator ray hits the configured
strict-open geometry target. Depending on the chosen helper, that target is
either at least one strict-open allowed voxel box interior or one strict-open
whole-box interior. A `False` result means only that the current sampled rule
did not certify the event; it is not an exact proof of unusability.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

from soe.contracts import EventObj

from .cone import (
    ConeLocalGeometry,
    build_cone_local_geometry,
)
from .ray_box import (
    BoxBounds,
    Interval,
    _as_box_bounds,
    _prevalidated_open_box_ray_interval,
    _prevalidated_ray_hits_open_box,
)


def _as_int(value: int, *, field_name: str) -> int:
    """Validate an integer parameter without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _as_allowed_boxes(allowed_boxes: object) -> tuple[BoxBounds, ...]:
    """Materialize the caller-supplied allowed boxes once."""
    return tuple(_as_box_bounds(box) for box in allowed_boxes)


@lru_cache(maxsize=None)
def midpoint_azimuth_grid(K: int) -> tuple[float, ...]:
    """Return the deterministic midpoint azimuth grid `xi_m = 2*pi*(m+0.5)/K`."""
    K = _as_int(K, field_name="K")
    if K < 3:
        raise ValueError("K must be >= 3")
    return tuple((2.0 * math.pi * (float(m) + 0.5)) / float(K) for m in range(K))


@lru_cache(maxsize=None)
def dyadic_azimuth_counts(K0: int, J_max: int) -> tuple[int, ...]:
    """Return the dyadic azimuth counts `(2**j) * K0` for `j = 0, ..., J_max`."""
    K0 = _as_int(K0, field_name="K0")
    J_max = _as_int(J_max, field_name="J_max")
    if K0 < 3:
        raise ValueError("K0 must be >= 3")
    if J_max < 0:
        raise ValueError("J_max must be >= 0")
    return tuple((2**j) * K0 for j in range(J_max + 1))


def _as_cone_geometry(
    value: ConeLocalGeometry | None,
    *,
    event: EventObj,
) -> ConeLocalGeometry:
    """Use one precomputed per-event cone bundle when the caller provides it."""
    if value is None:
        return build_cone_local_geometry(event)
    if not isinstance(value, ConeLocalGeometry):
        raise TypeError("_cone_geometry must be a ConeLocalGeometry or None")
    return value


@dataclass(frozen=True, slots=True)
class _SampledOpenBoxHit:
    """One sampled open-box hit reused across event-local preprocessing steps."""

    xi: float
    direction: tuple[float, float, float]
    interval: Interval


def _iter_direction_tuples_on_azimuths(
    *,
    azimuths: tuple[float, ...],
    cone_geometry: ConeLocalGeometry,
):
    """Yield sampled generator directions as plain float tuples."""
    frame = cone_geometry.frame
    p0, p1, p2 = (float(frame.p[0]), float(frame.p[1]), float(frame.p[2]))
    q0, q1, q2 = (float(frame.q[0]), float(frame.q[1]), float(frame.q[2]))
    u0, u1, u2 = (float(frame.u[0]), float(frame.u[1]), float(frame.u[2]))
    generator_scale = float(cone_geometry.s)
    lambda_u0 = float(cone_geometry.lambda_) * u0
    lambda_u1 = float(cone_geometry.lambda_) * u1
    lambda_u2 = float(cone_geometry.lambda_) * u2

    for xi in azimuths:
        cos_xi = math.cos(xi)
        sin_xi = math.sin(xi)
        yield xi, (
            lambda_u0 + generator_scale * ((cos_xi * p0) + (sin_xi * q0)),
            lambda_u1 + generator_scale * ((cos_xi * p1) + (sin_xi * q1)),
            lambda_u2 + generator_scale * ((cos_xi * p2) + (sin_xi * q2)),
        )


def _iter_sampled_full_box_open_hits(
    *,
    whole_box: object,
    azimuths: tuple[float, ...],
    cone_geometry: ConeLocalGeometry,
):
    """Yield sampled strict-open whole-box hit data in deterministic azimuth order."""
    bounds = _as_box_bounds(whole_box)
    ax, ay, az = (
        float(cone_geometry.apex[0]),
        float(cone_geometry.apex[1]),
        float(cone_geometry.apex[2]),
    )
    for xi, direction in _iter_direction_tuples_on_azimuths(
        azimuths=azimuths,
        cone_geometry=cone_geometry,
    ):
        interval = _prevalidated_open_box_ray_interval(
            ax,
            ay,
            az,
            direction[0],
            direction[1],
            direction[2],
            bounds,
        )
        if interval is not None:
            yield _SampledOpenBoxHit(
                xi=xi,
                direction=direction,
                interval=interval,
            )


def _first_sampled_full_box_open_hit(
    *,
    whole_box: object,
    azimuths: tuple[float, ...],
    cone_geometry: ConeLocalGeometry,
) -> _SampledOpenBoxHit | None:
    """Return the first sampled strict-open whole-box hit, if any."""
    for hit in _iter_sampled_full_box_open_hits(
        whole_box=whole_box,
        azimuths=azimuths,
        cone_geometry=cone_geometry,
    ):
        return hit
    return None


def _sampled_event_usable_on_azimuths(
    *,
    boxes: tuple[BoxBounds, ...],
    whole_box: BoxBounds | None,
    azimuths: tuple[float, ...],
    cone_geometry: ConeLocalGeometry,
) -> bool:
    """Evaluate one sampled predicate from precomputed geometry and azimuths."""
    ax, ay, az = (
        float(cone_geometry.apex[0]),
        float(cone_geometry.apex[1]),
        float(cone_geometry.apex[2]),
    )
    validated_whole_box = None if whole_box is None else _as_box_bounds(whole_box)
    for _, direction in _iter_direction_tuples_on_azimuths(
        azimuths=azimuths,
        cone_geometry=cone_geometry,
    ):
        dx, dy, dz = direction
        if validated_whole_box is not None:
            if _prevalidated_open_box_ray_interval(
                ax,
                ay,
                az,
                dx,
                dy,
                dz,
                validated_whole_box,
            ) is not None:
                return True
            continue
        for box in boxes:
            if _prevalidated_ray_hits_open_box(ax, ay, az, dx, dy, dz, box):
                return True
    return False


def sampled_event_usable(
    event: EventObj,
    allowed_boxes: object,
    *,
    K: int,
    _cone_geometry: ConeLocalGeometry | None = None,
) -> bool:
    """Return the sampled one-sided usability certificate for a single azimuth grid.

    The predicate is:
    there exists at least one allowed voxel box and at least one sampled
    midpoint azimuth whose canonical cone generator ray intersects that
    box's strict-open interior for some `ell > 0`.

    `False` here means only "not certified by this sampled rule".
    """
    if not isinstance(event, EventObj):
        raise TypeError("event must be an EventObj")
    azimuths = midpoint_azimuth_grid(K)
    boxes = _as_allowed_boxes(allowed_boxes)
    if not boxes:
        return False
    return _sampled_event_usable_on_azimuths(
        boxes=boxes,
        whole_box=None,
        azimuths=azimuths,
        cone_geometry=_as_cone_geometry(_cone_geometry, event=event),
    )


def adaptive_sampled_event_usable(
    event: EventObj,
    allowed_boxes: object,
    *,
    K0: int,
    J_max: int,
    _cone_geometry: ConeLocalGeometry | None = None,
) -> bool:
    """Return the OR of the sampled certificate across a dyadic azimuth schedule.

    This adaptive predicate remains one-sided only. `False` means only that no
    sampled grid in the requested dyadic schedule produced a certificate.
    """
    if not isinstance(event, EventObj):
        raise TypeError("event must be an EventObj")
    counts = dyadic_azimuth_counts(K0, J_max)
    boxes = _as_allowed_boxes(allowed_boxes)
    if not boxes:
        return False
    cone_geometry = _as_cone_geometry(_cone_geometry, event=event)

    for K in counts:
        if _sampled_event_usable_on_azimuths(
            boxes=boxes,
            whole_box=None,
            azimuths=midpoint_azimuth_grid(K),
            cone_geometry=cone_geometry,
        ):
            return True
    return False


def sampled_full_box_event_usable(
    event: EventObj,
    whole_box: object,
    *,
    K: int,
    _cone_geometry: ConeLocalGeometry | None = None,
) -> bool:
    """Return the sampled one-sided usability certificate for one open whole box.

    The predicate is:
    there exists at least one sampled midpoint azimuth whose canonical cone
    generator ray intersects the strict-open whole-box interior for some
    `ell > 0`.

    `False` here means only "not certified by this sampled rule".
    """
    if not isinstance(event, EventObj):
        raise TypeError("event must be an EventObj")
    azimuths = midpoint_azimuth_grid(K)
    return _sampled_event_usable_on_azimuths(
        boxes=(),
        whole_box=whole_box,
        azimuths=azimuths,
        cone_geometry=_as_cone_geometry(_cone_geometry, event=event),
    )


def adaptive_sampled_full_box_event_usable(
    event: EventObj,
    whole_box: object,
    *,
    K0: int,
    J_max: int,
    _cone_geometry: ConeLocalGeometry | None = None,
) -> bool:
    """Return the OR of the sampled full-box certificate across a dyadic schedule.

    This adaptive predicate remains one-sided only. `False` means only that no
    sampled grid in the requested dyadic schedule produced a certificate.
    """
    if not isinstance(event, EventObj):
        raise TypeError("event must be an EventObj")
    counts = dyadic_azimuth_counts(K0, J_max)
    cone_geometry = _as_cone_geometry(_cone_geometry, event=event)

    for K in counts:
        if _sampled_event_usable_on_azimuths(
            boxes=(),
            whole_box=whole_box,
            azimuths=midpoint_azimuth_grid(K),
            cone_geometry=cone_geometry,
        ):
            return True
    return False


__all__ = [
    "adaptive_sampled_full_box_event_usable",
    "adaptive_sampled_event_usable",
    "dyadic_azimuth_counts",
    "midpoint_azimuth_grid",
    "sampled_full_box_event_usable",
    "sampled_event_usable",
]
