"""Authoritative pointwise cone geometry for canonical `EventObj` values.

The authoritative geometry path in this module operates on canonical
`soe.contracts.core.EventObj` values only. Thin legacy compatibility shims
are kept at the bottom so older imports do not create a competing geometry
implementation elsewhere in the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from soe.adapters.voi_voxel import VoiBounds
from soe.contracts import EventObj

_FRAME_TOL = 1e-12
_SURFACE_TOL = 1e-12
_CANONICAL_BASIS = (
    np.asarray([1.0, 0.0, 0.0], dtype=float),
    np.asarray([0.0, 1.0, 0.0], dtype=float),
    np.asarray([0.0, 0.0, 1.0], dtype=float),
)


def _as_vec3(value: object, *, field_name: str) -> np.ndarray:
    """Validate a finite `(x, y, z)` vector."""
    vec = np.asarray(value, dtype=float)
    if vec.shape != (3,):
        raise ValueError(f"{field_name} must have shape (3,), got {vec.shape}")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{field_name} must be finite")
    return vec


def _as_event(event: EventObj) -> EventObj:
    """Validate that geometry operates on canonical core events."""
    if not isinstance(event, EventObj):
        raise TypeError("event must be an EventObj")
    return event


def _as_nonnegative_tol(value: float, *, field_name: str) -> float:
    """Validate a finite nonnegative tolerance."""
    tol = float(value)
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError(f"{field_name} must be finite and >= 0")
    return tol


def _as_positive_length(value: float, *, field_name: str) -> float:
    """Validate a finite strictly positive length parameter."""
    length = float(value)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(f"{field_name} must be finite and > 0")
    return length


def _point_from_event_apex(r: object, event: EventObj) -> tuple[np.ndarray, np.ndarray, float]:
    """Return `(r, d, ||d||)` for canonical event geometry."""
    point = _as_vec3(r, field_name="r")
    d = point - event.apex
    norm = float(np.linalg.norm(d))
    return point, d, norm


def _assert_not_apex(r: object, event: EventObj, *, func_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Reject evaluation exactly at the event apex."""
    point, d, norm = _point_from_event_apex(r, event)
    if norm <= 0.0:
        raise ValueError(f"{func_name} is undefined at the cone apex")
    return point, d, norm


def _allowed_region_contains(point: np.ndarray, allowed_region: VoiBounds) -> bool:
    """Check membership in the allowed region boundary object."""
    return allowed_region.contains(point)


@dataclass(frozen=True, slots=True)
class ConeFrame:
    """Deterministic orthonormal right-handed cone-local frame `(p, q, u)`."""

    p: np.ndarray
    q: np.ndarray
    u: np.ndarray

    def __post_init__(self) -> None:
        """Validate the orthonormal frame contract."""
        p = _as_vec3(self.p, field_name="p")
        q = _as_vec3(self.q, field_name="q")
        u = _as_vec3(self.u, field_name="u")

        for name, vec in (("p", p), ("q", q), ("u", u)):
            if not math.isclose(float(np.linalg.norm(vec)), 1.0, rel_tol=0.0, abs_tol=_FRAME_TOL):
                raise ValueError(f"{name} must be unit-length")

        if not math.isclose(float(np.dot(p, q)), 0.0, rel_tol=0.0, abs_tol=_FRAME_TOL):
            raise ValueError("p and q must be orthogonal")
        if not math.isclose(float(np.dot(p, u)), 0.0, rel_tol=0.0, abs_tol=_FRAME_TOL):
            raise ValueError("p and u must be orthogonal")
        if not math.isclose(float(np.dot(q, u)), 0.0, rel_tol=0.0, abs_tol=_FRAME_TOL):
            raise ValueError("q and u must be orthogonal")

        orientation = float(np.dot(np.cross(p, q), u))
        if not math.isclose(orientation, 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("frame must be right-handed")

        object.__setattr__(self, "p", p)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "u", u)


@dataclass(frozen=True, slots=True)
class ConeLocalGeometry:
    """Immutable per-event cone invariants reused across hot geometry loops."""

    apex: np.ndarray
    lambda_: float
    s: float
    frame: ConeFrame

    def __post_init__(self) -> None:
        """Validate the cached invariant bundle."""
        apex = _as_vec3(self.apex, field_name="apex")
        lambda_ = float(self.lambda_)
        s = float(self.s)
        if not math.isfinite(lambda_):
            raise ValueError("lambda_ must be finite")
        if not math.isfinite(s) or s < 0.0:
            raise ValueError("s must be finite and >= 0")
        if not isinstance(self.frame, ConeFrame):
            raise TypeError("frame must be a ConeFrame")

        object.__setattr__(self, "apex", apex)
        object.__setattr__(self, "lambda_", lambda_)
        object.__setattr__(self, "s", s)


def cone_lambda(event: EventObj) -> float:
    """Return the canonical cone parameter `lambda = cos(theta)`."""
    event = _as_event(event)
    return float(math.cos(event.theta))


def cone_phi(r: object, event: EventObj) -> float:
    """Return `((r - a)·u)/||r - a|| - lambda` for a canonical event.

    The apex is excluded explicitly: `r = a` is invalid.
    """
    event = _as_event(event)
    _, d, norm = _assert_not_apex(r, event, func_name="cone_phi")
    s = float(np.dot(d, event.axis))
    return (s / norm) - cone_lambda(event)


def is_on_cone_surface(r: object, event: EventObj, tol: float = _SURFACE_TOL) -> bool:
    """Return whether `r` lies on the canonical cone surface within `tol`.

    The apex is excluded explicitly.
    """
    tol = _as_nonnegative_tol(tol, field_name="tol")
    return abs(cone_phi(r, event)) <= tol


def is_one_sided_feasible(r: object, event: EventObj) -> bool:
    """Return the canonical one-sided half-space predicate `dot(r-a, u) >= 0`.

    This is the half-space check only; apex exclusion is handled separately by
    `cone_phi`, `is_on_cone_surface`, and `is_admissible_point`.
    """
    event = _as_event(event)
    _, d, _ = _point_from_event_apex(r, event)
    return float(np.dot(d, event.axis)) >= 0.0


def is_admissible_point(
    r: object,
    event: EventObj,
    allowed_region: VoiBounds,
    tol: float = _SURFACE_TOL,
) -> bool:
    """Return pointwise admissibility of a point in the allowed region.

    A point is admissible iff it is inside the allowed region, is not the apex,
    lies on the cone surface, and satisfies the one-sided half-space predicate.
    """
    event = _as_event(event)
    tol = _as_nonnegative_tol(tol, field_name="tol")
    point, d, norm = _assert_not_apex(r, event, func_name="is_admissible_point")
    if not _allowed_region_contains(point, allowed_region):
        return False
    s = float(np.dot(d, event.axis))
    phi = (s / norm) - cone_lambda(event)
    return abs(phi) <= tol and s >= 0.0


def build_cone_frame(event: EventObj) -> ConeFrame:
    """Build the deterministic cone-local frame `(p, q, u)`.

    The ordered fallback basis is `(e_x, e_y, e_z)`. The chosen fallback axis
    `g_e` minimizes `|g·u|`, with ties broken by the fixed order
    `e_x ≺ e_y ≺ e_z`.
    """
    event = _as_event(event)
    u = _as_vec3(event.axis, field_name="event.axis")

    scores = [abs(float(np.dot(g, u))) for g in _CANONICAL_BASIS]
    g = _CANONICAL_BASIS[int(np.argmin(scores))]
    projection = g - float(np.dot(g, u)) * u
    projection_norm = float(np.linalg.norm(projection))
    if projection_norm <= 0.0:
        raise ValueError("cannot construct cone frame from axis")

    p = projection / projection_norm
    q = np.cross(u, p)
    q_norm = float(np.linalg.norm(q))
    if q_norm <= 0.0:
        raise ValueError("cannot construct cone frame orthogonal complement")
    q = q / q_norm
    return ConeFrame(p=p, q=q, u=u)


def build_cone_local_geometry(event: EventObj) -> ConeLocalGeometry:
    """Build the immutable per-event cone invariants used by hot-path scans."""
    event = _as_event(event)
    lambda_ = float(math.cos(event.theta))
    return ConeLocalGeometry(
        apex=event.apex,
        lambda_=lambda_,
        s=math.sqrt(max(0.0, 1.0 - (lambda_ * lambda_))),
        frame=build_cone_frame(event),
    )


def cone_generator_direction_from_local_geometry(
    cone_geometry: ConeLocalGeometry,
    xi: float,
) -> np.ndarray:
    """Return the canonical unit generator direction from cached invariants."""
    if not isinstance(cone_geometry, ConeLocalGeometry):
        raise TypeError("cone_geometry must be a ConeLocalGeometry")
    xi = float(xi)
    if not math.isfinite(xi):
        raise ValueError("xi must be finite")

    frame = cone_geometry.frame
    direction = (
        (cone_geometry.lambda_ * frame.u)
        + cone_geometry.s * ((math.cos(xi) * frame.p) + (math.sin(xi) * frame.q))
    )
    return np.asarray(direction, dtype=float)


def cone_generator_direction(event: EventObj, xi: float) -> np.ndarray:
    """Return the canonical unit generator direction `w_e(xi)` for azimuth `xi`."""
    return cone_generator_direction_from_local_geometry(
        build_cone_local_geometry(event),
        xi,
    )


def cone_surface_point_from_local_geometry(
    cone_geometry: ConeLocalGeometry,
    ell: float,
    xi: float,
) -> np.ndarray:
    """Return `Psi_e(ell, xi)` from cached per-event cone invariants."""
    ell = _as_positive_length(ell, field_name="ell")
    direction = cone_generator_direction_from_local_geometry(cone_geometry, xi)
    return np.asarray(cone_geometry.apex + (ell * direction), dtype=float)


def cone_surface_point(event: EventObj, ell: float, xi: float) -> np.ndarray:
    """Return `Psi_e(ell, xi) = apex + ell * w_e(xi)` for `ell > 0`.

    The generator direction is
    `w_e(xi) = lambda*u + sqrt(1-lambda^2)*(cos(xi)*p + sin(xi)*q)`.
    """
    return cone_surface_point_from_local_geometry(
        build_cone_local_geometry(event),
        ell,
        xi,
    )


# Legacy compatibility shims -------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConeEventRaw:
    """Legacy raw-event shim; authoritative canonicalization lives at ingestion."""

    apex_xyz_cm: np.ndarray
    axis_xyz_raw: np.ndarray
    theta_rad_raw: float


@dataclass(frozen=True, slots=True)
class ConeEvent:
    """Legacy canonical-event shim backed by the canonical `EventObj` contract."""

    apex_xyz_cm: np.ndarray
    axis_xyz: np.ndarray
    theta_rad: float
    species: str | None = None

    def __post_init__(self) -> None:
        event = EventObj(
            apex=self.apex_xyz_cm,
            axis=self.axis_xyz,
            theta=self.theta_rad,
            species=self.species,
        )
        object.__setattr__(self, "apex_xyz_cm", event.apex)
        object.__setattr__(self, "axis_xyz", event.axis)
        object.__setattr__(self, "theta_rad", event.theta)
        object.__setattr__(self, "species", event.species)

    @property
    def lambda_(self) -> float:
        """Compatibility view of `lambda = cos(theta)`."""
        return cone_lambda(self.to_event_obj())

    def to_event_obj(self) -> EventObj:
        """Return the authoritative canonical core event object."""
        return EventObj(
            apex=self.apex_xyz_cm,
            axis=self.axis_xyz,
            theta=self.theta_rad,
            species=self.species,
        )


def canonicalize_cone_event(raw: ConeEventRaw) -> ConeEvent:
    """Legacy raw canonicalization shim.

    Canonicalization now belongs at the adapter ingestion boundary. This helper
    remains only so older imports do not define a second geometry path.
    """
    apex = _as_vec3(raw.apex_xyz_cm, field_name="apex_xyz_cm")
    axis_raw = _as_vec3(raw.axis_xyz_raw, field_name="axis_xyz_raw")
    axis_norm = float(np.linalg.norm(axis_raw))
    if not math.isfinite(axis_norm) or axis_norm <= 0.0:
        raise ValueError("axis_xyz_raw must have nonzero finite norm")

    theta = float(raw.theta_rad_raw)
    if not math.isfinite(theta) or not (0.0 < theta < math.pi):
        raise ValueError("theta_rad_raw must be finite and in (0, pi)")

    axis_unit = axis_raw / axis_norm
    lambda_raw = float(math.cos(theta))
    if lambda_raw < 0.0:
        axis_unit = -axis_unit
        theta = math.pi - theta

    return ConeEvent(apex_xyz_cm=apex, axis_xyz=axis_unit, theta_rad=theta)


def phi_cone(event: ConeEvent, r_xyz_cm: object) -> float:
    """Compatibility wrapper for the old `(event, r)` cone-phi signature."""
    return cone_phi(r_xyz_cm, event.to_event_obj())


def cone_implicit_sq(event: ConeEvent, r_xyz_cm: object) -> float:
    """Compatibility helper `s^2 - ||d||^2 * lambda^2` for older tests."""
    event_obj = event.to_event_obj()
    _, d, _ = _point_from_event_apex(r_xyz_cm, event_obj)
    s = float(np.dot(d, event_obj.axis))
    dd = float(np.dot(d, d))
    lambda_ = cone_lambda(event_obj)
    return s * s - dd * (lambda_ * lambda_)


def is_one_nappe(event: ConeEvent, r_xyz_cm: object) -> bool:
    """Compatibility wrapper for the old one-nappe predicate name."""
    return is_one_sided_feasible(r_xyz_cm, event.to_event_obj())


__all__ = [
    "ConeEvent",
    "ConeEventRaw",
    "ConeFrame",
    "ConeLocalGeometry",
    "build_cone_frame",
    "build_cone_local_geometry",
    "canonicalize_cone_event",
    "cone_implicit_sq",
    "cone_lambda",
    "cone_phi",
    "cone_generator_direction_from_local_geometry",
    "cone_surface_point",
    "cone_surface_point_from_local_geometry",
    "is_admissible_point",
    "is_on_cone_surface",
    "is_one_nappe",
    "is_one_sided_feasible",
    "phi_cone",
]
