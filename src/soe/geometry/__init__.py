"""Authoritative geometry package exports for canonical `EventObj` values."""

from .cone import (
    ConeFrame,
    build_cone_frame,
    cone_lambda,
    cone_phi,
    cone_surface_point,
    is_admissible_point,
    is_on_cone_surface,
    is_one_sided_feasible,
)

__all__ = [
    "ConeFrame",
    "cone_lambda",
    "cone_phi",
    "is_on_cone_surface",
    "is_one_sided_feasible",
    "is_admissible_point",
    "build_cone_frame",
    "cone_surface_point",
]
