"""Focused tests for explicit event usability and minimal surviving-event state."""

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
from soe.geometry.cone import cone_surface_point
from soe.soe.state import (
    SampledSurvivalConfig,
    USABILITY_DECISION_APPROXIMATION,
    USABILITY_DECISION_APPROXIMATION_FULL_BOX,
    USABILITY_IMPLEMENTATION_NOTE,
    SurvivingEventState,
    check_event_usability,
)


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


def _voi_config() -> VOIConfig:
    """Build a common VOI contract for state tests."""
    return VOIConfig(
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


def _voi_bounds() -> VoiBounds:
    """Build the bounded VOI region."""
    return VoiBounds.from_config(_voi_config())


def _box_bounds(
    box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
) -> np.ndarray:
    """Materialize xyz box bounds in `[[xmin, xmax], ...]` order."""
    return np.asarray(
        [
            [box[0][0], box[0][1]],
            [box[1][0], box[1][1]],
            [box[2][0], box[2][1]],
        ],
        dtype=float,
    )


def _gridded_voi_from_box(
    box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    *,
    grid_shape: tuple[int, int, int] = (1, 1, 1),
) -> VoiBounds:
    """Build a grid-aware VOI boundary object from xyz box bounds."""
    return VoiBounds.from_config(
        VOIConfig(
            bounds=_box_bounds(box),
            grid_shape=grid_shape,
        )
    )


def _bare_voi_from_box(
    box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
) -> VoiBounds:
    """Build a bare ungridded VOI boundary object from xyz box bounds."""
    return VoiBounds(
        xmin=box[0][0],
        xmax=box[0][1],
        ymin=box[1][0],
        ymax=box[1][1],
        zmin=box[2][0],
        zmax=box[2][1],
    )



def test_both_usability_gates_are_explicitly_labeled_as_sampled_approximations() -> None:
    """Both ray-cell and full-box usability gates must carry the approximation label."""
    event = _event()
    box = (
        (0.95, 1.05),
        (0.95, 1.05),
        (1.36, 1.46),
    )

    ray_cell = check_event_usability(
        event,
        _gridded_voi_from_box(box),
        survival_config=SampledSurvivalConfig.fixed(K=4),
    )
    assert ray_cell.usable is True
    assert ray_cell.is_exact is False
    assert ray_cell.decision_kind == USABILITY_DECISION_APPROXIMATION
    assert ray_cell.decision_metadata["gate_mode"] == "ray_cell"
    assert "Approximation" in USABILITY_IMPLEMENTATION_NOTE
    assert "sampled ray-cell interior-hit surrogate" in USABILITY_IMPLEMENTATION_NOTE

    full_box = check_event_usability(
        event,
        _bare_voi_from_box(box),
        survival_config=SampledSurvivalConfig.fixed_full_box(K=4),
    )
    assert full_box.usable is True
    assert full_box.is_exact is False
    assert full_box.decision_kind == USABILITY_DECISION_APPROXIMATION_FULL_BOX
    assert full_box.decision_metadata["gate_mode"] == "full_box"
    assert "strict-open whole-box interior" in full_box.reason


def test_sampled_surrogate_controls_retain_reject_with_fixed_vs_adaptive_modes() -> None:
    """Fixed and adaptive sampled schedules should drive survival decisions."""
    event = _event()
    box = (
        (1.28656, 1.32656),
        (0.52120, 0.56120),
        (1.39421, 1.43421),
    )

    fixed = check_event_usability(
        event,
        _gridded_voi_from_box(box),
        survival_config=SampledSurvivalConfig.fixed(K=4),
    )
    adaptive = check_event_usability(
        event,
        _gridded_voi_from_box(box),
        survival_config=SampledSurvivalConfig.adaptive(K0=4, J_max=1),
    )

    assert fixed.usable is False
    assert adaptive.usable is True
    assert fixed.is_exact is False
    assert adaptive.is_exact is False
    assert fixed.decision_kind == USABILITY_DECISION_APPROXIMATION
    assert adaptive.decision_kind == USABILITY_DECISION_APPROXIMATION
    assert fixed.decision_metadata["K"] == 4
    assert adaptive.decision_metadata["K0"] == 4
    assert adaptive.decision_metadata["J_max"] == 1
    assert "not an exact proof of unusability" in fixed.reason


def test_sampled_full_box_gate_selection_routes_over_bare_ungridded_voi_bounds() -> None:
    """Choosing the full-box mode should route away from the ray-cell grid requirement."""
    event = _event()
    box = (
        (0.95, 1.05),
        (0.95, 1.05),
        (1.36, 1.46),
    )
    allowed_region = _bare_voi_from_box(box)

    with pytest.raises(
        ValueError,
        match="sampled ray-cell survival requires grid-aware voxel-cell semantics",
    ):
        check_event_usability(
            event,
            allowed_region,
            survival_config=SampledSurvivalConfig.fixed(K=4),
        )

    decision = check_event_usability(
        event,
        allowed_region,
        survival_config=SampledSurvivalConfig.fixed_full_box(K=4),
    )

    assert decision.usable is True
    assert decision.decision_kind == USABILITY_DECISION_APPROXIMATION_FULL_BOX



def test_sampled_ray_cell_survival_rejects_bare_ungridded_voi_bounds() -> None:
    """The ray-cell gate must not silently degrade to one whole open VOI box."""
    box = (
        (0.0, 0.1),
        (0.9, 1.1),
        (-0.1, 0.1),
    )
    event = _event(axis=(0.0, 1.0, 1.0), theta=math.pi / 4.0)

    with pytest.raises(
        ValueError,
        match="sampled ray-cell survival requires grid-aware voxel-cell semantics",
    ):
        check_event_usability(
            event,
            _bare_voi_from_box(box),
            survival_config=SampledSurvivalConfig.fixed(K=6),
        )


def test_state_rejects_mismatched_representative_point_counts() -> None:
    """State must require exactly one representative point per surviving event."""
    event = _event()
    point = cone_surface_point(event, ell=math.sqrt(2.0), xi=0.0)

    with pytest.raises(ValueError):
        SurvivingEventState.from_surviving_events(
            [event],
            _voi_bounds(),
            [point, point],
        )


def test_state_rejects_point_not_on_event_feasible_surface() -> None:
    """Representative points are validated separately after sampled survival."""
    event = _event()
    box = (
        (0.95, 1.05),
        (0.95, 1.05),
        (1.36, 1.46),
    )
    off_surface = np.asarray([1.0, 0.0, 2.0], dtype=float)

    with pytest.raises(ValueError, match="representative point is not pointwise admissible"):
        SurvivingEventState.from_surviving_events(
            [event],
            _gridded_voi_from_box(box),
            [off_surface],
            survival_config=SampledSurvivalConfig.fixed(K=4),
        )


def test_state_rejects_events_when_sampled_gate_does_not_certify_survival() -> None:
    """An admissible representative point must not override sampled rejection."""
    event = _event()
    box = (
        (1.28656, 1.32656),
        (0.52120, 0.56120),
        (1.39421, 1.43421),
    )
    representative_point = cone_surface_point(event, ell=2.0, xi=math.pi / 8.0)

    with pytest.raises(ValueError, match="did not survive sampled preprocessing"):
        SurvivingEventState.from_surviving_events(
            [event],
            _gridded_voi_from_box(box),
            [representative_point],
            survival_config=SampledSurvivalConfig.fixed(K=4),
        )


def test_state_accepts_exactly_one_admissible_point_per_surviving_event() -> None:
    """Minimal state should accept one admissible representative point per event."""
    allowed_region = _gridded_voi_from_box(
        (
            (0.0, 3.0),
            (0.0, 2.0),
            (0.0, 3.0),
        )
    )
    event_a = _event(theta=math.pi / 4.0)
    event_b = _event(apex=(0.25, 0.0, 0.0), theta=math.pi / 4.0)
    point_a = cone_surface_point(event_a, ell=2.0, xi=math.pi / 4.0)
    point_b = cone_surface_point(event_b, ell=2.0, xi=math.pi / 4.0)

    state = SurvivingEventState.from_surviving_events(
        [event_a, event_b],
        allowed_region,
        [point_a, point_b],
        survival_config=SampledSurvivalConfig.fixed(K=4),
    )

    assert len(state.events) == 2
    assert len(state.representative_points) == 2
    assert np.allclose(state.representative_points[0], point_a)
    assert np.allclose(state.representative_points[1], point_b)


def test_state_accepts_representative_points_after_full_box_survival() -> None:
    """The minimal state contract should work unchanged under the sampled full-box gate."""
    allowed_region = _bare_voi_from_box(
        (
            (0.0, 3.0),
            (0.0, 2.0),
            (0.0, 3.0),
        )
    )
    event = _event(theta=math.pi / 4.0)
    point = cone_surface_point(event, ell=2.0, xi=math.pi / 4.0)

    state = SurvivingEventState.from_surviving_events(
        [event],
        allowed_region,
        [point],
        survival_config=SampledSurvivalConfig.fixed_full_box(K=4),
    )

    assert len(state.events) == 1
    assert len(state.representative_points) == 1
    assert np.allclose(state.representative_points[0], point)
