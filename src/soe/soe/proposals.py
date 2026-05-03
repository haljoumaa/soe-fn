"""Reconstruction proposal interface and proposal-ratio contract.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

import numpy as np

from soe.soe.state import SurvivingEventState


def _as_point(value: object, *, field_name: str) -> np.ndarray:
    """Validate one finite representative-point proposal."""
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{field_name} must have shape (3,), got {point.shape}")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{field_name} must be finite")
    return point


def _as_nonnegative_ratio(value: object, *, field_name: str) -> float:
    """Validate one nonnegative proposal-ratio factor."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a nonnegative finite real")
    ratio = float(value)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError(f"{field_name} must be a nonnegative finite real")
    return ratio


def _as_finite_log_density(value: object, *, field_name: str) -> float:
    """Validate one finite log proposal density."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite real")
    log_density = float(value)
    if not math.isfinite(log_density):
        raise ValueError(f"{field_name} must be a finite real")
    return log_density


@dataclass(frozen=True, slots=True, init=False)
class ProposalCandidate:
    """One proposed representative-point replacement plus MH-ratio metadata.

    Symmetric proposals may omit all proposal-density metadata, in which case
    the implied Metropolis-Hastings proposal ratio is `1`. The omission is
    still preserved explicitly via `proposal_ratio_was_omitted` so the kernel
    can distinguish it from an explicit `proposal_ratio=1.0`. Non-symmetric
    proposals must supply either a direct `proposal_ratio` or both
    `log_q_forward` and `log_q_reverse`.
    """

    candidate_point: np.ndarray
    proposal_ratio: float
    proposal_ratio_was_omitted: bool
    log_q_forward: float | None
    log_q_reverse: float | None

    def __init__(
        self,
        candidate_point: object,
        *,
        proposal_ratio: object | None = None,
        log_q_forward: object | None = None,
        log_q_reverse: object | None = None,
    ) -> None:
        """Normalize proposal metadata into a resolved MH proposal ratio."""
        point = _as_point(candidate_point, field_name="candidate_point")

        if proposal_ratio is not None and (
            log_q_forward is not None or log_q_reverse is not None
        ):
            raise ValueError(
                "specify either proposal_ratio or both log_q_forward/log_q_reverse"
            )

        if proposal_ratio is None and log_q_forward is None and log_q_reverse is None:
            resolved_ratio = 1.0
            ratio_was_omitted = True
            forward_log_density = None
            reverse_log_density = None
        elif proposal_ratio is not None:
            resolved_ratio = _as_nonnegative_ratio(
                proposal_ratio,
                field_name="proposal_ratio",
            )
            ratio_was_omitted = False
            forward_log_density = None
            reverse_log_density = None
        else:
            if log_q_forward is None or log_q_reverse is None:
                raise ValueError(
                    "log_q_forward and log_q_reverse must be provided together"
                )
            forward_log_density = _as_finite_log_density(
                log_q_forward,
                field_name="log_q_forward",
            )
            reverse_log_density = _as_finite_log_density(
                log_q_reverse,
                field_name="log_q_reverse",
            )
            try:
                resolved_ratio = math.exp(reverse_log_density - forward_log_density)
            except OverflowError as exc:
                raise ValueError(
                    "log_q_forward and log_q_reverse must imply a finite proposal_ratio"
                ) from exc
            if not math.isfinite(resolved_ratio):
                raise ValueError(
                    "log_q_forward and log_q_reverse must imply a finite proposal_ratio"
                )
            ratio_was_omitted = False

        object.__setattr__(self, "candidate_point", point)
        object.__setattr__(self, "proposal_ratio", resolved_ratio)
        object.__setattr__(self, "proposal_ratio_was_omitted", ratio_was_omitted)
        object.__setattr__(self, "log_q_forward", forward_log_density)
        object.__setattr__(self, "log_q_reverse", reverse_log_density)

    @classmethod
    def from_log_proposal_densities(
        cls,
        candidate_point: object,
        *,
        log_q_forward: object,
        log_q_reverse: object,
    ) -> "ProposalCandidate":
        """Build a candidate from paired forward/reverse log proposal densities."""
        return cls(
            candidate_point,
            log_q_forward=log_q_forward,
            log_q_reverse=log_q_reverse,
        )


class SingleEventProposalBackend(Protocol):
    """Contract for sampling one candidate for one selected event.

    Proposal backends are expected to sample from an event-local proposal law
    over the admissible set. This interface does not implement that law; it
    only standardizes the event-selection and candidate-return shape needed by
    the generic MH shell, including support for non-symmetric proposal ratios.
    Backends that omit proposal-ratio metadata must explicitly certify that
    their proposal law is symmetric on the backend's certified support.
    """

    certifies_symmetric_proposal: bool

    def select_event_index(self, state: SurvivingEventState) -> int:
        """Choose exactly one event index for the current state."""

    def propose_candidate(
        self,
        state: SurvivingEventState,
        event_index: int,
    ) -> ProposalCandidate:
        """Return one candidate replacement point for the chosen event."""


__all__ = [
    "ProposalCandidate",
    "SingleEventProposalBackend",
]
