"""Focused tests for local target/acceptance algebra."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from soe.soe.target import (
    baseline_alpha1_acceptance,
    mh_acceptance_probability,
    single_event_target_ratio,
)


def test_same_voxel_target_ratio_is_one() -> None:
    """A same-voxel move must leave the adopted local target unchanged."""
    assert single_event_target_ratio(4, 4, alpha=2.5, same_voxel=True) == pytest.approx(
        1.0
    )


def test_same_voxel_general_acceptance_is_one_for_unit_proposal_ratio() -> None:
    """A same-voxel move must accept with probability one under unit proposals."""
    target_ratio = single_event_target_ratio(3, 3, alpha=0.5, same_voxel=True)
    assert mh_acceptance_probability(target_ratio, proposal_ratio=1.0) == pytest.approx(
        1.0
    )


def test_cross_voxel_target_ratio_matches_adopted_alpha_surrogate() -> None:
    """Cross-voxel moves should follow the adopted occupancy-only surrogate rule."""
    assert single_event_target_ratio(
        old_count=4,
        new_count=2,
        alpha=0.5,
        same_voxel=False,
    ) == pytest.approx((2.0 + 0.5) / (4.0 - 1.0 + 0.5))


def test_baseline_alpha1_acceptance_matches_closed_form() -> None:
    """The baseline symmetric rule must match `min(1, (new_count + 1) / old_count)`."""
    assert baseline_alpha1_acceptance(
        old_count=4,
        new_count=2,
        same_voxel=False,
    ) == pytest.approx(min(1.0, (2.0 + 1.0) / 4.0))


def test_baseline_regression_denominator_uses_old_count_not_old_count_minus_one() -> None:
    """The landed baseline keeps `old_count` in the denominator."""
    actual = baseline_alpha1_acceptance(old_count=3, new_count=1, same_voxel=False)
    wrong_denominator = min(1.0, (1.0 + 1.0) / (3.0 - 1.0))

    assert actual == pytest.approx(2.0 / 3.0)
    assert wrong_denominator == pytest.approx(1.0)
    assert actual != pytest.approx(wrong_denominator)


def test_general_acceptance_composes_with_asymmetric_proposal_ratio() -> None:
    """The MH helper should multiply target and proposal ratios before truncation."""
    target_ratio = single_event_target_ratio(
        old_count=5,
        new_count=2,
        alpha=2.0,
        same_voxel=False,
    )

    assert mh_acceptance_probability(
        target_ratio,
        proposal_ratio=0.25,
    ) == pytest.approx(min(1.0, target_ratio * 0.25))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"old_count": -1, "new_count": 0, "alpha": 1.0, "same_voxel": False},
            "old_count must be a nonnegative integer",
        ),
        (
            {"old_count": 1, "new_count": -1, "alpha": 1.0, "same_voxel": False},
            "new_count must be a nonnegative integer",
        ),
        (
            {"old_count": 1, "new_count": 0, "alpha": 0.0, "same_voxel": False},
            "alpha must be a positive finite real",
        ),
        (
            {"old_count": 0, "new_count": 2, "alpha": 1.5, "same_voxel": False},
            "cross-voxel moves require old_count >= 1",
        ),
    ],
)
def test_single_event_target_ratio_rejects_invalid_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Invalid counts or alpha should fail fast."""
    with pytest.raises(ValueError, match=match):
        single_event_target_ratio(**kwargs)
