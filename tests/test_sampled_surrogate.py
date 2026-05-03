"""Focused deterministic tests for the sampled-surrogate geometry layer."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import soe.geometry.sampled_surrogate as sampled_surrogate_module
from soe.contracts import EventObj
from soe.geometry.cone import build_cone_local_geometry
from soe.geometry.sampled_surrogate import (
    _first_sampled_full_box_open_hit,
    adaptive_sampled_full_box_event_usable,
    adaptive_sampled_event_usable,
    dyadic_azimuth_counts,
    midpoint_azimuth_grid,
    sampled_full_box_event_usable,
    sampled_event_usable,
)


def _event(
    *,
    apex: tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    theta: float = math.pi / 4.0,
) -> EventObj:
    axis_arr = np.asarray(axis, dtype=float)
    axis_arr = axis_arr / np.linalg.norm(axis_arr)
    return EventObj(apex=np.asarray(apex, dtype=float), axis=axis_arr, theta=theta)


def test_midpoint_azimuth_grid_returns_expected_values() -> None:
    """The midpoint grid should use the exact deterministic half-cell offset."""
    grid = midpoint_azimuth_grid(4)

    assert grid == (
        math.pi / 4.0,
        3.0 * math.pi / 4.0,
        5.0 * math.pi / 4.0,
        7.0 * math.pi / 4.0,
    )


def test_dyadic_azimuth_counts_return_expected_schedule() -> None:
    """The dyadic schedule should double counts at each refinement level."""
    assert dyadic_azimuth_counts(3, 3) == (3, 6, 12, 24)


def test_sampled_event_usable_returns_true_for_sampled_hit() -> None:
    """A box aligned to a sampled midpoint direction should be certified."""
    event = _event()
    box = (
        (0.95, 1.05),
        (0.95, 1.05),
        (1.36, 1.46),
    )

    assert sampled_event_usable(event, (box,), K=4) is True


def test_adaptive_refinement_can_certify_when_coarse_grid_cannot() -> None:
    """Refinement should succeed when only a finer midpoint direction hits."""
    event = _event()
    box = (
        (1.28656, 1.32656),
        (0.52120, 0.56120),
        (1.39421, 1.43421),
    )

    assert sampled_event_usable(event, (box,), K=4) is False
    assert adaptive_sampled_event_usable(event, (box,), K0=4, J_max=1) is True


def test_sampled_false_means_only_not_certified_by_this_rule() -> None:
    """A missed sample should be described only as a missing sampled certificate."""
    event = _event()
    box = (
        (1.39421, 1.43421),
        (-0.02, 0.02),
        (1.39421, 1.43421),
    )

    assert sampled_event_usable(event, (box,), K=4) is False


def test_sampled_full_box_event_usable_returns_true_for_sampled_hit() -> None:
    """A whole-box open interior aligned to a sampled midpoint should be certified."""
    event = _event()
    whole_box = (
        (0.95, 1.05),
        (0.95, 1.05),
        (1.36, 1.46),
    )

    assert sampled_full_box_event_usable(event, whole_box, K=4) is True


def test_adaptive_sampled_full_box_can_certify_when_coarse_grid_cannot() -> None:
    """Dyadic refinement should work for the full-box sampled predicate too."""
    event = _event()
    whole_box = (
        (1.28656, 1.32656),
        (0.52120, 0.56120),
        (1.39421, 1.43421),
    )

    assert sampled_full_box_event_usable(event, whole_box, K=4) is False
    assert adaptive_sampled_full_box_event_usable(event, whole_box, K0=4, J_max=1) is True


def test_sampled_full_box_false_means_only_not_certified_by_this_rule() -> None:
    """A missed whole-box sample is still only a missing sampled certificate."""
    event = _event()
    whole_box = (
        (1.39421, 1.43421),
        (-0.02, 0.02),
        (1.39421, 1.43421),
    )

    assert sampled_full_box_event_usable(event, whole_box, K=4) is False


def test_sampled_full_box_uses_precomputed_cone_geometry_when_provided(monkeypatch) -> None:
    """The sampled full-box hot path should reuse one precomputed event-local frame."""
    event = _event()
    whole_box = (
        (0.95, 1.05),
        (0.95, 1.05),
        (1.36, 1.46),
    )
    cone_geometry = build_cone_local_geometry(event)

    def _fail_rebuild(*args, **kwargs):  # pragma: no cover - failure path only
        del args, kwargs
        raise AssertionError("unexpected cone-geometry rebuild")

    monkeypatch.setattr(sampled_surrogate_module, "build_cone_local_geometry", _fail_rebuild)

    assert sampled_full_box_event_usable(
        event,
        whole_box,
        K=4,
        _cone_geometry=cone_geometry,
    ) is True


def test_first_sampled_full_box_open_hit_matches_public_certificate() -> None:
    """The internal first-hit helper must agree with the public sampled certificate."""
    event = _event()
    whole_box = (
        (0.95, 1.05),
        (0.95, 1.05),
        (1.36, 1.46),
    )
    cone_geometry = build_cone_local_geometry(event)
    azimuths = midpoint_azimuth_grid(4)

    assert (
        _first_sampled_full_box_open_hit(
            whole_box=whole_box,
            azimuths=azimuths,
            cone_geometry=cone_geometry,
        )
        is not None
    ) is sampled_full_box_event_usable(
        event,
        whole_box,
        K=4,
        _cone_geometry=cone_geometry,
    )
