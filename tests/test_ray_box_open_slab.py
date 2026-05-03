"""Exact tests for the strict-open fixed-`xi` ray-box primitive."""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from soe.geometry.ray_box import (
    fixed_direction_open_box_radial_interval,
    intersect_open_intervals,
    open_box_ray_interval,
    open_slab_interval,
    ray_intersects_open_box,
)


def test_positive_direction_component_gives_expected_open_interval() -> None:
    """A positive direction should preserve the slab ordering."""
    interval = open_slab_interval(2.0, 5.0, -1.0, 2.0)
    assert interval == (1.5, 3.0)


def test_negative_direction_component_gives_reversed_but_correct_interval() -> None:
    """A negative direction should reverse the formula while keeping `lower < upper`."""
    interval = open_slab_interval(2.0, 5.0, 8.0, -2.0)
    assert interval == (1.5, 3.0)


def test_zero_direction_inside_slab_yields_open_positive_ray_interval() -> None:
    """A stationary coordinate strictly inside its slab should contribute `(0, inf)`."""
    interval = open_slab_interval(0.0, 1.0, 0.25, 0.0)
    assert interval == (0.0, math.inf)


def test_zero_direction_on_boundary_or_outside_slab_is_empty() -> None:
    """Strict-open slab semantics reject boundary and exterior stationary coordinates."""
    assert open_slab_interval(0.0, 1.0, 0.0, 0.0) is None
    assert open_slab_interval(0.0, 1.0, 1.0, 0.0) is None
    assert open_slab_interval(0.0, 1.0, -0.25, 0.0) is None


def test_fixed_direction_whole_box_result_reports_a_clear_strict_open_hit() -> None:
    """A visible open-box interior interval should expose exact structured endpoints."""
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    result = fixed_direction_open_box_radial_interval(
        (-1.0, 0.5, 0.5),
        (1.0, 0.0, 0.0),
        bounds,
    )

    assert result.x_interval == (1.0, 2.0)
    assert result.y_interval == (0.0, math.inf)
    assert result.z_interval == (0.0, math.inf)
    assert result.ell_minus == 1.0
    assert result.ell_plus == 2.0
    assert result.has_open_hit is True
    assert result.interval == (1.0, 2.0)


def test_fixed_direction_whole_box_result_is_empty_when_a_stationary_axis_is_outside() -> None:
    """A zero-direction coordinate outside its open slab must make the whole box empty."""
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    result = fixed_direction_open_box_radial_interval(
        (-1.0, -0.25, 0.5),
        (1.0, 0.0, 0.0),
        bounds,
    )

    assert result.x_interval == (1.0, 2.0)
    assert result.y_interval is None
    assert result.z_interval == (0.0, math.inf)
    assert result.ell_minus is None
    assert result.ell_plus is None
    assert result.has_open_hit is False
    assert result.interval is None


def test_fixed_direction_whole_box_result_rejects_boundary_only_contact() -> None:
    """Equality `ell_minus == ell_plus` must stay an empty strict-open interval."""
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    result = fixed_direction_open_box_radial_interval(
        (-1.0, 0.0, 0.5),
        (1.0, 1.0, 0.0),
        bounds,
    )

    assert result.x_interval == (1.0, 2.0)
    assert result.y_interval == (0.0, 1.0)
    assert result.z_interval == (0.0, math.inf)
    assert result.ell_minus == 1.0
    assert result.ell_plus == 1.0
    assert result.has_open_hit is False
    assert result.interval is None


def test_fixed_direction_whole_box_result_handles_negative_direction_exactly() -> None:
    """The whole-box helper should preserve the exact reversed ordering for `d_c < 0`."""
    bounds = ((0.0, 2.0), (0.0, 1.0), (0.0, 1.0))
    result = fixed_direction_open_box_radial_interval(
        (3.0, 0.5, 0.5),
        (-2.0, 0.0, 0.0),
        bounds,
    )

    assert result.x_interval == (0.5, 1.5)
    assert result.y_interval == (0.0, math.inf)
    assert result.z_interval == (0.0, math.inf)
    assert result.ell_minus == 0.5
    assert result.ell_plus == 1.5
    assert result.has_open_hit is True
    assert result.interval == (0.5, 1.5)


def test_fixed_direction_whole_box_result_keeps_stationary_interior_axes_open() -> None:
    """A stationary interior axis should contribute `(0, inf)` in the structured result."""
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    result = fixed_direction_open_box_radial_interval(
        (0.25, -1.0, 0.75),
        (0.0, 1.0, 0.0),
        bounds,
    )

    assert result.x_interval == (0.0, math.inf)
    assert result.y_interval == (1.0, 2.0)
    assert result.z_interval == (0.0, math.inf)
    assert result.ell_minus == 1.0
    assert result.ell_plus == 2.0
    assert result.has_open_hit is True
    assert result.interval == (1.0, 2.0)


def test_fixed_direction_whole_box_result_truncates_negative_lower_end_at_zero() -> None:
    """The combined interval should apply the exact `ell > 0` truncation via `max(0, ...)`."""
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    result = fixed_direction_open_box_radial_interval(
        (0.5, 0.5, 0.5),
        (1.0, 0.0, 0.0),
        bounds,
    )

    assert result.x_interval == (-0.5, 0.5)
    assert result.y_interval == (0.0, math.inf)
    assert result.z_interval == (0.0, math.inf)
    assert result.ell_minus == 0.0
    assert result.ell_plus == 0.5
    assert result.has_open_hit is True
    assert result.interval == (0.0, 0.5)


def test_interior_hit_returns_true() -> None:
    """A ray that enters the strict-open interior should be accepted."""
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    assert ray_intersects_open_box((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0), bounds) is True
    assert open_box_ray_interval((-1.0, 0.5, 0.5), (1.0, 0.0, 0.0), bounds) == (1.0, 2.0)


def test_boundary_only_touch_returns_false() -> None:
    """Touching the box only on its boundary should not count as an interior hit."""
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    interval = open_box_ray_interval((-1.0, 0.0, 0.5), (1.0, 1.0, 0.0), bounds)

    assert interval is None
    assert ray_intersects_open_box((-1.0, 0.0, 0.5), (1.0, 1.0, 0.0), bounds) is False


def test_hit_requiring_positive_ell_is_handled_exactly() -> None:
    """An apex already inside the box still needs some `ell > 0`, not just `ell = 0`."""
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    interval = open_box_ray_interval((0.5, 0.5, 0.5), (1.0, 0.0, 0.0), bounds)

    assert interval == (0.0, 0.5)
    assert ray_intersects_open_box((0.5, 0.5, 0.5), (1.0, 0.0, 0.0), bounds) is True


def test_boundary_apex_with_immediate_inward_motion_opens_a_positive_interval() -> None:
    """A boundary apex is allowed when the ray enters the strict-open interior for `ell > 0`."""
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    interval = open_box_ray_interval((0.0, 0.5, 0.5), (1.0, 0.0, 0.0), bounds)

    assert interval == (0.0, 1.0)
    assert ray_intersects_open_box((0.0, 0.5, 0.5), (1.0, 0.0, 0.0), bounds) is True


def test_open_upper_face_exclusion_is_exact() -> None:
    """Moving along an excluded upper face should remain outside the open interior."""
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    interval = open_box_ray_interval((-1.0, 0.5, 1.0), (1.0, 0.0, 0.0), bounds)

    assert interval is None
    assert ray_intersects_open_box((-1.0, 0.5, 1.0), (1.0, 0.0, 0.0), bounds) is False


def test_no_tolerance_based_inclusion_at_upper_boundary() -> None:
    """A point even one representable step beyond the upper face must stay excluded."""
    bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    just_above_upper = math.nextafter(1.0, math.inf)

    interval = open_box_ray_interval((just_above_upper, 0.5, 0.5), (0.0, 1.0, 0.0), bounds)

    assert interval is None
    assert ray_intersects_open_box((just_above_upper, 0.5, 0.5), (0.0, 1.0, 0.0), bounds) is False


def test_combining_coordinate_intervals_uses_strict_open_intersection() -> None:
    """The combined interval should be nonempty exactly when `ell_minus < ell_plus`."""
    interval = intersect_open_intervals(((1.0, 3.0), (-2.0, 5.0), (0.0, math.inf)))
    empty = intersect_open_intervals(((1.0, 3.0), (3.0, 4.0), (0.0, math.inf)))

    assert interval == (1.0, 3.0)
    assert empty is None
