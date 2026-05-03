"""Exact fixed-`xi` strict-open ray-box primitives.

These helpers operate on plain numeric inputs for the ray
`r(ell) = apex + ell * direction` and a strict-open axis-aligned box. They
implement only the exact open-slab rule for `ell > 0`: no closed-box fallback,
no half-open semantics, no tolerance expansion, and no upper-bound clamping.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import TypeAlias

Interval: TypeAlias = tuple[float, float]
Vec3: TypeAlias = tuple[float, float, float]
BoxBounds: TypeAlias = tuple[Interval, Interval, Interval]


@dataclass(frozen=True, slots=True)
class OpenBoxRayIntervalResult:
    """Structured exact fixed-direction strict-open box-interior interval data.

    `ell_minus` and `ell_plus` are populated only when every coordinate
    contributes a nonempty open slab interval. In boundary-only cases they may
    still satisfy `ell_minus >= ell_plus`, which is reported explicitly via
    `has_open_hit=False`.
    """

    x_interval: Interval | None
    y_interval: Interval | None
    z_interval: Interval | None
    ell_minus: float | None
    ell_plus: float | None
    has_open_hit: bool

    @property
    def interval(self) -> Interval | None:
        """Return the exact open hit interval, or `None` when it is empty."""
        if not self.has_open_hit:
            return None
        assert self.ell_minus is not None
        assert self.ell_plus is not None
        return (self.ell_minus, self.ell_plus)


def _as_finite_scalar(value: float, *, field_name: str) -> float:
    """Validate a finite scalar."""
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{field_name} must be finite")
    return scalar


def _as_vec3(value: object, *, field_name: str) -> Vec3:
    """Validate a finite numeric 3-vector."""
    entries = tuple(value)
    if len(entries) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 entries")
    return (
        _as_finite_scalar(entries[0], field_name=f"{field_name}[0]"),
        _as_finite_scalar(entries[1], field_name=f"{field_name}[1]"),
        _as_finite_scalar(entries[2], field_name=f"{field_name}[2]"),
    )


def _as_box_bounds(value: object) -> BoxBounds:
    """Validate three strict-open coordinate slabs."""
    entries = tuple(value)
    if len(entries) != 3:
        raise ValueError("bounds must contain exactly 3 coordinate intervals")

    bounds: list[Interval] = []
    for idx, interval in enumerate(entries):
        slab = tuple(interval)
        if len(slab) != 2:
            raise ValueError(f"bounds[{idx}] must contain exactly 2 entries")
        lower = _as_finite_scalar(slab[0], field_name=f"bounds[{idx}][0]")
        upper = _as_finite_scalar(slab[1], field_name=f"bounds[{idx}][1]")
        if not lower < upper:
            raise ValueError(f"bounds[{idx}] must satisfy lower < upper")
        bounds.append((lower, upper))
    return (bounds[0], bounds[1], bounds[2])


def _as_open_interval(interval: Interval, *, field_name: str) -> Interval:
    """Validate an open interval represented by `(lower, upper)`."""
    lower = float(interval[0])
    upper = float(interval[1])
    if math.isnan(lower) or math.isnan(upper):
        raise ValueError(f"{field_name} must not contain NaN")
    if math.isinf(lower):
        raise ValueError(f"{field_name} lower bound must be finite")
    if not (math.isfinite(upper) or math.isinf(upper)):
        raise ValueError(f"{field_name} upper bound must be finite or +inf")
    if not lower < upper:
        raise ValueError(f"{field_name} must satisfy lower < upper")
    return (lower, upper)


def open_slab_interval(lower: float, upper: float, apex_coord: float, direction_coord: float) -> Interval | None:
    """Return the exact open `ell` interval for one coordinate slab.

    The returned interval is the strict-open set of `ell` values for which
    `lower < apex_coord + ell * direction_coord < upper` holds.
    """
    lower = _as_finite_scalar(lower, field_name="lower")
    upper = _as_finite_scalar(upper, field_name="upper")
    apex_coord = _as_finite_scalar(apex_coord, field_name="apex_coord")
    direction_coord = _as_finite_scalar(direction_coord, field_name="direction_coord")

    if not lower < upper:
        raise ValueError("lower must be strictly less than upper")

    if direction_coord > 0.0:
        return ((lower - apex_coord) / direction_coord, (upper - apex_coord) / direction_coord)
    if direction_coord < 0.0:
        return ((upper - apex_coord) / direction_coord, (lower - apex_coord) / direction_coord)
    if lower < apex_coord < upper:
        return (0.0, math.inf)
    return None


def _open_slab_interval_prevalidated(
    lower: float,
    upper: float,
    apex_coord: float,
    direction_coord: float,
) -> Interval | None:
    """Return one open slab interval from already-validated finite scalars."""
    if direction_coord > 0.0:
        return ((lower - apex_coord) / direction_coord, (upper - apex_coord) / direction_coord)
    if direction_coord < 0.0:
        return ((upper - apex_coord) / direction_coord, (lower - apex_coord) / direction_coord)
    if lower < apex_coord < upper:
        return (0.0, math.inf)
    return None


def intersect_open_intervals(intervals: Iterable[Interval | None]) -> Interval | None:
    """Intersect open intervals together with the ray domain `(0, inf)`.

    The result is the exact open interval `(ell_minus, ell_plus)` for which all
    supplied coordinate constraints and the ray-domain constraint `ell > 0`
    hold simultaneously.
    """
    ell_minus = 0.0
    ell_plus = math.inf

    for idx, interval in enumerate(intervals):
        if interval is None:
            return None
        lower, upper = _as_open_interval(interval, field_name=f"intervals[{idx}]")
        ell_minus = max(ell_minus, lower)
        ell_plus = min(ell_plus, upper)

    if ell_minus < ell_plus:
        return (ell_minus, ell_plus)
    return None


def fixed_direction_open_box_radial_interval(
    apex: object,
    direction: object,
    bounds: object,
) -> OpenBoxRayIntervalResult:
    """Return exact strict-open whole-box radial data for one fixed direction."""
    ax, ay, az = _as_vec3(apex, field_name="apex")
    dx, dy, dz = _as_vec3(direction, field_name="direction")
    (lx, ux), (ly, uy), (lz, uz) = _as_box_bounds(bounds)

    x_interval = open_slab_interval(lx, ux, ax, dx)
    y_interval = open_slab_interval(ly, uy, ay, dy)
    z_interval = open_slab_interval(lz, uz, az, dz)

    if x_interval is None or y_interval is None or z_interval is None:
        return OpenBoxRayIntervalResult(
            x_interval=x_interval,
            y_interval=y_interval,
            z_interval=z_interval,
            ell_minus=None,
            ell_plus=None,
            has_open_hit=False,
        )

    ell_minus = max(0.0, x_interval[0], y_interval[0], z_interval[0])
    ell_plus = min(x_interval[1], y_interval[1], z_interval[1])
    return OpenBoxRayIntervalResult(
        x_interval=x_interval,
        y_interval=y_interval,
        z_interval=z_interval,
        ell_minus=ell_minus,
        ell_plus=ell_plus,
        has_open_hit=ell_minus < ell_plus,
    )


def _prevalidated_open_box_ray_interval(
    ax: float,
    ay: float,
    az: float,
    dx: float,
    dy: float,
    dz: float,
    bounds: BoxBounds,
) -> Interval | None:
    """Return the exact open-box interval from already-validated inputs."""
    (lx, ux), (ly, uy), (lz, uz) = bounds

    x_interval = _open_slab_interval_prevalidated(lx, ux, ax, dx)
    if x_interval is None:
        return None
    y_interval = _open_slab_interval_prevalidated(ly, uy, ay, dy)
    if y_interval is None:
        return None
    z_interval = _open_slab_interval_prevalidated(lz, uz, az, dz)
    if z_interval is None:
        return None

    ell_minus = max(0.0, x_interval[0], y_interval[0], z_interval[0])
    ell_plus = min(x_interval[1], y_interval[1], z_interval[1])
    if ell_minus < ell_plus:
        return (ell_minus, ell_plus)
    return None


def _prevalidated_ray_hits_open_box(
    ax: float,
    ay: float,
    az: float,
    dx: float,
    dy: float,
    dz: float,
    bounds: BoxBounds,
) -> bool:
    """Return whether validated inputs hit the strict-open box interior."""
    return _prevalidated_open_box_ray_interval(
        ax,
        ay,
        az,
        dx,
        dy,
        dz,
        bounds,
    ) is not None


def open_box_ray_interval(apex: object, direction: object, bounds: object) -> Interval | None:
    """Compatibility wrapper returning only the exact open hit interval."""
    return fixed_direction_open_box_radial_interval(apex, direction, bounds).interval


def ray_intersects_open_box(apex: object, direction: object, bounds: object) -> bool:
    """Return whether the ray hits the strict-open box interior for some `ell > 0`."""
    return fixed_direction_open_box_radial_interval(apex, direction, bounds).has_open_hit


__all__ = [
    "BoxBounds",
    "Interval",
    "OpenBoxRayIntervalResult",
    "Vec3",
    "fixed_direction_open_box_radial_interval",
    "intersect_open_intervals",
    "open_box_ray_interval",
    "open_slab_interval",
    "ray_intersects_open_box",
]
