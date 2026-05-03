"""Local target and acceptance algebra for the MH reconstruction kernel.

This module implements only the adopted occupancy-only surrogate target used
for local single-event ratio calculations. It is intentionally independent of
geometry, proposal generation, RNG, and state mutation.
"""

from __future__ import annotations

import math
import operator
from numbers import Real


def _as_nonnegative_count(value: object, *, field_name: str) -> int:
    """Validate one occupancy count input."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a nonnegative integer")
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a nonnegative integer") from exc
    if count < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return int(count)


def _as_positive_alpha(value: object) -> float:
    """Validate the positive occupancy-surrogate parameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("alpha must be a positive finite real")
    alpha = float(value)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be a positive finite real")
    return alpha


def _as_nonnegative_ratio(value: object, *, field_name: str) -> float:
    """Validate one nonnegative Metropolis-Hastings ratio factor."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a nonnegative finite real")
    ratio = float(value)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError(f"{field_name} must be a nonnegative finite real")
    return ratio


def single_event_target_ratio(
    old_count: object,
    new_count: object,
    *,
    alpha: object,
    same_voxel: bool,
) -> float:
    """Return the adopted occupancy-only surrogate target ratio for one move."""
    validated_old_count = _as_nonnegative_count(old_count, field_name="old_count")
    validated_new_count = _as_nonnegative_count(new_count, field_name="new_count")
    validated_alpha = _as_positive_alpha(alpha)

    if not isinstance(same_voxel, bool):
        raise ValueError("same_voxel must be a bool")
    if same_voxel:
        return 1.0
    if validated_old_count < 1:
        raise ValueError("cross-voxel moves require old_count >= 1")

    return (validated_new_count + validated_alpha) / (
        validated_old_count - 1 + validated_alpha
    )


def mh_acceptance_probability(
    target_ratio: object,
    *,
    proposal_ratio: object = 1.0,
) -> float:
    """Return `min(1, r_target * r_proposal)` for a local MH decision."""
    validated_target_ratio = _as_nonnegative_ratio(
        target_ratio,
        field_name="target_ratio",
    )
    validated_proposal_ratio = _as_nonnegative_ratio(
        proposal_ratio,
        field_name="proposal_ratio",
    )
    return min(1.0, validated_target_ratio * validated_proposal_ratio)


def baseline_alpha1_acceptance(
    old_count: object,
    new_count: object,
    *,
    same_voxel: bool,
) -> float:
    """Return the symmetric `alpha = 1` OE/SOE baseline acceptance rule."""
    validated_old_count = _as_nonnegative_count(old_count, field_name="old_count")
    validated_new_count = _as_nonnegative_count(new_count, field_name="new_count")

    if not isinstance(same_voxel, bool):
        raise ValueError("same_voxel must be a bool")
    if same_voxel:
        return 1.0
    if validated_old_count < 1:
        raise ValueError("cross-voxel moves require old_count >= 1")

    return min(1.0, (validated_new_count + 1.0) / float(validated_old_count))


__all__ = [
    "baseline_alpha1_acceptance",
    "mh_acceptance_probability",
    "single_event_target_ratio",
]
