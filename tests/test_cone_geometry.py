"""Focused geometry tests for canonical cone events."""

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

from soe.adapters.voi_voxel import VoiBounds, VoxelGrid
from soe.contracts import EventObj, VOIConfig
from soe.geometry.cone import (
    build_cone_frame,
    cone_generator_direction,
    cone_lambda,
    cone_phi,
    cone_surface_point,
    is_admissible_point,
    is_on_cone_surface,
    is_one_sided_feasible,
)
from soe.geometry.ray_box import ray_intersects_open_box


def _event(
    *,
    apex: tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    theta: float = math.pi / 4.0,
) -> EventObj:
    """Build a canonical event for tests."""
    axis_arr = np.asarray(axis, dtype=float)
    axis_arr = axis_arr / np.linalg.norm(axis_arr)
    return EventObj(apex=np.asarray(apex, dtype=float), axis=axis_arr, theta=theta)


def test_cone_lambda_matches_cos_theta() -> None:
    """`cone_lambda(event)` should equal `cos(theta)` for canonical events."""
    event = _event(theta=math.pi / 3.0)
    assert math.isclose(cone_lambda(event), math.cos(event.theta), abs_tol=1e-12)


def test_cone_phi_is_zero_for_known_surface_point() -> None:
    """Known canonical surface points should satisfy the implicit equation."""
    event = _event(theta=math.pi / 4.0)
    point = cone_surface_point(event, ell=2.0, xi=0.0)
    assert abs(cone_phi(point, event)) < 1e-12


def test_apex_is_excluded_from_surface_phi_and_admissibility() -> None:
    """Apex evaluation should be rejected explicitly."""
    event = _event()
    voi = VoiBounds(xmin=-1.0, xmax=2.0, ymin=-1.0, ymax=2.0, zmin=-1.0, zmax=2.0)

    with pytest.raises(ValueError):
        cone_phi(event.apex, event)
    with pytest.raises(ValueError):
        is_on_cone_surface(event.apex, event, tol=1e-12)
    with pytest.raises(ValueError):
        is_admissible_point(event.apex, event, voi, tol=1e-12)


def test_one_sided_feasibility_is_the_half_space_check() -> None:
    """One-sided feasibility should follow `dot(r-a, u) >= 0` exactly."""
    event = _event(theta=math.pi / 4.0)
    assert is_one_sided_feasible(np.asarray([1.0, 0.0, 1.0]), event) is True
    assert is_one_sided_feasible(np.asarray([1.0, 0.0, -1.0]), event) is False


def test_admissibility_is_gated_by_voi() -> None:
    """Pointwise admissibility should respect the bounded VOI region."""
    event = _event(theta=math.pi / 4.0)
    point = np.asarray([1.0, 0.0, 1.0])
    config = VOIConfig(
        bounds=np.asarray(
            [
                [0.0, 4.0],
                [-1.0, 1.0],
                [0.0, 4.0],
            ],
            dtype=float,
        ),
        grid_shape=(4, 2, 4),
    )
    voi = VoiBounds.from_config(config)

    assert is_admissible_point(point, event, voi, tol=1e-12) is True
    assert is_admissible_point(np.asarray([1.0, 0.0, -1.0]), event, voi, tol=1e-12) is False


def test_build_cone_frame_uses_deterministic_tie_break() -> None:
    """Equal fallback scores should choose `e_x` before `e_y` and `e_z`."""
    event = _event(axis=(1.0, 1.0, 1.0), theta=math.pi / 3.0)
    frame = build_cone_frame(event)

    u = event.axis
    ex = np.asarray([1.0, 0.0, 0.0], dtype=float)
    expected_p = ex - float(np.dot(ex, u)) * u
    expected_p = expected_p / np.linalg.norm(expected_p)
    assert np.allclose(frame.p, expected_p, atol=1e-12)


def test_build_cone_frame_is_orthonormal_and_right_handed() -> None:
    """The constructed frame should be orthonormal with positive orientation."""
    event = _event(axis=(1.0, 2.0, 3.0), theta=math.pi / 3.0)
    frame = build_cone_frame(event)

    assert np.isclose(np.linalg.norm(frame.p), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(frame.q), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(frame.u), 1.0, atol=1e-12)
    assert np.isclose(np.dot(frame.p, frame.q), 0.0, atol=1e-12)
    assert np.isclose(np.dot(frame.p, frame.u), 0.0, atol=1e-12)
    assert np.isclose(np.dot(frame.q, frame.u), 0.0, atol=1e-12)
    assert np.isclose(np.dot(np.cross(frame.p, frame.q), frame.u), 1.0, atol=1e-12)


def test_cone_surface_point_lies_on_the_canonical_half_cone() -> None:
    """`Psi_e(ell, xi)` should land on the on-surface one-sided half-cone."""
    event = _event(theta=math.pi / 3.0)
    point = cone_surface_point(event, ell=3.5, xi=1.1)

    assert np.linalg.norm(point - event.apex) > 0.0
    assert is_on_cone_surface(point, event, tol=1e-12) is True
    assert is_one_sided_feasible(point, event) is True


def test_cone_generator_direction_matches_surface_point_direction() -> None:
    """The generator helper should agree with the normalized surface-point direction."""
    event = _event(theta=math.pi / 4.0)
    xi = math.pi / 6.0
    direction = cone_generator_direction(event, xi)
    point = cone_surface_point(event, ell=2.0, xi=xi)
    expected = (point - event.apex) / np.linalg.norm(point - event.apex)

    assert np.isclose(np.linalg.norm(direction), 1.0, atol=1e-12)
    assert np.allclose(direction, expected, atol=1e-12)


def test_canonical_cone_generator_direction_can_drive_open_box_query() -> None:
    """A canonical cone generator direction should work as plain ray-box input."""
    event = _event(theta=math.pi / 4.0)
    direction = tuple(cone_generator_direction(event, 0.0).tolist())
    bounds = ((0.6, 0.8), (-0.1, 0.1), (0.6, 0.8))

    assert ray_intersects_open_box(tuple(event.apex.tolist()), direction, bounds) is True
