"""Generic one-step single-event MH kernel shell.

This module implements only a local occupancy-target Metropolis-Hastings shell
around the authoritative representative-point state. It stays independent of
continuous proposal geometry: callers supply a backend that picks one event
index and one candidate replacement point.
"""

from __future__ import annotations

import math
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real

import numpy as np

from soe.adapters.voi_voxel import VoxelGrid
from soe.soe.chain_health import ChainHealthCollector
from soe.soe.occupancy import _apply_local_delta_to_owned_counts
from soe.soe.proposals import ProposalCandidate, SingleEventProposalBackend
from soe.soe.state import SurvivingEventState, _ChainMutableSurvivingEventState
from soe.soe.target import mh_acceptance_probability, single_event_target_ratio

VoxelIndex = tuple[int, int, int]


def _as_grid(value: object) -> VoxelGrid:
    """Validate the explicit grid carrier used by the local kernel shell."""
    if isinstance(value, VoxelGrid):
        return value
    raise TypeError("grid must be a VoxelGrid")


def _as_event_index(value: object, *, n_events: int) -> int:
    """Validate the selected event index for the current state."""
    if isinstance(value, bool):
        raise ValueError("selected event index must be an integer")
    try:
        event_index = operator.index(value)
    except TypeError as exc:
        raise ValueError("selected event index must be an integer") from exc
    if not 0 <= event_index < n_events:
        raise IndexError(f"selected event index out of range: {event_index}")
    return int(event_index)


def _as_uniform01(value: object) -> float:
    """Validate one acceptance-threshold draw."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("uniform01() must return a finite real in [0, 1]")
    draw = float(value)
    if not math.isfinite(draw) or not 0.0 <= draw <= 1.0:
        raise ValueError("uniform01() must return a finite real in [0, 1]")
    return draw


def _as_counts(value: object, *, grid: VoxelGrid) -> np.ndarray:
    """Validate the current full-grid occupancy histogram without mutating it."""
    counts = _as_hot_loop_counts(value, grid=grid)
    if np.any(counts < 0):
        raise ValueError("counts must be nonnegative")
    return counts


def _as_hot_loop_counts(value: object, *, grid: VoxelGrid) -> np.ndarray:
    """Validate only the hot-loop shape/dtype counts contract."""
    counts = np.asarray(value)
    if counts.shape != grid.grid_shape:
        raise ValueError(f"counts must have shape {grid.grid_shape}, got {counts.shape}")
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("counts must be integer-valued")
    return counts


def _backend_certifies_symmetric_proposal(
    proposal_backend: SingleEventProposalBackend,
) -> bool:
    """Return whether the backend explicitly certifies symmetric proposals."""
    certified = getattr(proposal_backend, "certifies_symmetric_proposal", False)
    if not isinstance(certified, bool):
        raise TypeError("proposal backend certifies_symmetric_proposal must be a bool")
    return certified


def _updated_state_with_candidate(
    state: SurvivingEventState,
    *,
    event_index: int,
    candidate_point: np.ndarray,
    trusted: bool = False,
) -> SurvivingEventState:
    """Return a new authoritative representative-point state with one replacement."""
    if trusted:
     
        return state._trusted_replace_representative_point(
            event_index=event_index,
            candidate_point=candidate_point,
        )
    updated_points = list(state.representative_points)
    updated_points[event_index] = candidate_point
    return SurvivingEventState(events=state.events, representative_points=tuple(updated_points))


def _apply_single_event_delta_prevalidated_counts(
    counts: np.ndarray,
    *,
    old_voxel: VoxelIndex,
    new_voxel: VoxelIndex,
    copy: bool,
) -> np.ndarray:
    """Apply the exact local occupancy delta to prevalidated counts."""
    if copy:
        updated_counts = np.array(counts, dtype=np.int64, copy=True)
    else:
        updated_counts = counts
        if not updated_counts.flags.writeable:
            raise ValueError("prevalidated hot-loop counts buffer must be writeable")
    return _apply_local_delta_to_owned_counts(
        updated_counts,
        old_voxel_index=old_voxel,
        new_voxel_index=new_voxel,
    )


@dataclass(frozen=True, slots=True)
class SingleEventMHStepResult:
    """Result of one local single-event MH step."""

    selected_event_index: int
    accepted: bool
    same_voxel: bool
    old_voxel: VoxelIndex
    new_voxel: VoxelIndex
    acceptance_probability: float
    state: SurvivingEventState
    occupancy_counts: np.ndarray
    target_ratio: float
    proposal_ratio: float


def _single_event_mh_step_core(
    state: SurvivingEventState,
    current_counts: np.ndarray,
    *,
    grid: VoxelGrid,
    proposal_backend: SingleEventProposalBackend,
    alpha: object,
    uniform01: Callable[[], object],
    chain_health_collector: ChainHealthCollector | None = None,
    reuse_current_counts_on_accept: bool = False,
    trusted_state_update_on_accept: bool = False,
) -> SingleEventMHStepResult:
    """Run one generic single-event MH step over validated counts."""
    
    audit = getattr(proposal_backend, '_runtime_audit', None)

    event_index = _as_event_index(
        proposal_backend.select_event_index(state),
        n_events=len(state.events),
    )
    proposal = proposal_backend.propose_candidate(state, event_index)
    if not isinstance(proposal, ProposalCandidate):
        raise TypeError("proposal backend must return a ProposalCandidate")
    if proposal.proposal_ratio_was_omitted:
        if not _backend_certifies_symmetric_proposal(proposal_backend):
            raise ValueError(
                "proposal backend omitted proposal-ratio metadata without certifying symmetry"
            )
        proposal_ratio = 1.0
    else:
        proposal_ratio = proposal.proposal_ratio

   
    if audit is not None:
        t0 = time.perf_counter()
    old_voxel, new_voxel = state.updated_voxel_index_for_event(
        event_index,
        proposal.candidate_point,
        grid,
    )
    if audit is not None:
        audit.voxel_lookup_seconds += time.perf_counter() - t0
        audit.voxel_lookup_count += 1

    same_voxel = old_voxel == new_voxel
    old_count = int(current_counts[old_voxel])
    new_count = int(current_counts[new_voxel])

  
    if audit is not None:
        t0 = time.perf_counter()
    target_ratio = single_event_target_ratio(
        old_count,
        new_count,
        alpha=alpha,
        same_voxel=same_voxel,
    )
    acceptance_probability = mh_acceptance_probability(
        target_ratio,
        proposal_ratio=proposal_ratio,
    )
    if audit is not None:
        audit.target_ratio_seconds += time.perf_counter() - t0
        audit.target_ratio_count += 1

    if audit is not None:
        t0 = time.perf_counter()
    if acceptance_probability >= 1.0:
        accepted = True
    else:
        accepted = _as_uniform01(uniform01()) < acceptance_probability
    if audit is not None:
        audit.acceptance_decision_seconds += time.perf_counter() - t0
        audit.acceptance_decision_count += 1
        audit.total_step_count += 1
    if chain_health_collector is not None:
        chain_health_collector.record_step(
            same_voxel=same_voxel,
            old_count=old_count,
            new_count=new_count,
            accepted=accepted,
            proposal_ratio_was_omitted=proposal.proposal_ratio_was_omitted,
        )

    if not accepted:
        return SingleEventMHStepResult(
            selected_event_index=event_index,
            accepted=False,
            same_voxel=same_voxel,
            old_voxel=old_voxel,
            new_voxel=new_voxel,
            acceptance_probability=acceptance_probability,
            state=state,
            occupancy_counts=current_counts,
            target_ratio=target_ratio,
            proposal_ratio=proposal_ratio,
        )

    
    if audit is not None:
        t0 = time.perf_counter()
    
    updated_counts = _apply_single_event_delta_prevalidated_counts(
        current_counts,
        old_voxel=old_voxel,
        new_voxel=new_voxel,
        copy=not reuse_current_counts_on_accept,
    )
    updated_state = _updated_state_with_candidate(
        state,
        event_index=event_index,
        candidate_point=proposal.candidate_point,
        trusted=trusted_state_update_on_accept,
    )
    if audit is not None:
        audit.state_commit_seconds += time.perf_counter() - t0
        audit.state_commit_count += 1
        audit.accepted_step_count += 1

    return SingleEventMHStepResult(
        selected_event_index=event_index,
        accepted=True,
        same_voxel=same_voxel,
        old_voxel=old_voxel,
        new_voxel=new_voxel,
        acceptance_probability=acceptance_probability,
        state=updated_state,
        occupancy_counts=updated_counts,
        target_ratio=target_ratio,
        proposal_ratio=proposal_ratio,
    )


def single_event_mh_step(
    state: SurvivingEventState,
    counts: object,
    *,
    grid: VoxelGrid,
    proposal_backend: SingleEventProposalBackend,
    alpha: object,
    uniform01: Callable[[], object],
) -> SingleEventMHStepResult:
    """Run one generic single-event MH step over the representative-point state."""
    if not isinstance(state, SurvivingEventState):
        raise TypeError("state must be a SurvivingEventState")

    grid = _as_grid(grid)
    current_counts = _as_counts(counts, grid=grid)
    return _single_event_mh_step_core(
        state,
        current_counts,
        grid=grid,
        proposal_backend=proposal_backend,
        alpha=alpha,
        uniform01=uniform01,
    )


def _single_event_mh_step_prevalidated_counts(
    state: SurvivingEventState,
    counts: object,
    *,
    grid: VoxelGrid,
    proposal_backend: SingleEventProposalBackend,
    alpha: object,
    uniform01: Callable[[], object],
    chain_health_collector: ChainHealthCollector | None = None,
    chain_owns_counts_buffer: bool = False,
) -> SingleEventMHStepResult:
    """Run one MH step assuming counts were already nonnegativity-validated."""
    if not isinstance(state, SurvivingEventState):
        raise TypeError("state must be a SurvivingEventState")

    grid = _as_grid(grid)
    current_counts = _as_hot_loop_counts(counts, grid=grid)
    return _single_event_mh_step_core(
        state,
        current_counts,
        grid=grid,
        proposal_backend=proposal_backend,
        alpha=alpha,
        uniform01=uniform01,
        chain_health_collector=chain_health_collector,
        reuse_current_counts_on_accept=chain_owns_counts_buffer,
        trusted_state_update_on_accept=True,
    )


def _single_event_mh_step_mutable_chain_state(
    chain_state: _ChainMutableSurvivingEventState,
    counts: object,
    *,
    grid: VoxelGrid,
    proposal_backend: SingleEventProposalBackend,
    alpha: object,
    uniform01: Callable[[], object],
    chain_health_collector: ChainHealthCollector | None = None,
    chain_owns_counts_buffer: bool = False,
    snapshot_state_for_result: bool = False,
) -> SingleEventMHStepResult:
    """Run one MH step against the chain-internal mutable state."""
    if not isinstance(chain_state, _ChainMutableSurvivingEventState):
        raise TypeError("chain_state must be a _ChainMutableSurvivingEventState")

    grid = _as_grid(grid)
    current_counts = _as_hot_loop_counts(counts, grid=grid)
    state_view = chain_state.state_view()

    audit = getattr(proposal_backend, '_runtime_audit', None)

    event_index = _as_event_index(
        proposal_backend.select_event_index(state_view),
        n_events=len(chain_state.events),
    )
    proposal = proposal_backend.propose_candidate(state_view, event_index)
    if not isinstance(proposal, ProposalCandidate):
        raise TypeError("proposal backend must return a ProposalCandidate")
    if proposal.proposal_ratio_was_omitted:
        if not _backend_certifies_symmetric_proposal(proposal_backend):
            raise ValueError(
                "proposal backend omitted proposal-ratio metadata without certifying symmetry"
            )
        proposal_ratio = 1.0
    else:
        proposal_ratio = proposal.proposal_ratio


    if audit is not None:
        t0 = time.perf_counter()
    old_voxel, new_voxel = chain_state.updated_voxel_index_for_event(
        event_index,
        proposal.candidate_point,
        grid,
    )
    if audit is not None:
        audit.voxel_lookup_seconds += time.perf_counter() - t0
        audit.voxel_lookup_count += 1

    same_voxel = old_voxel == new_voxel
    old_count = int(current_counts[old_voxel])
    new_count = int(current_counts[new_voxel])


    if audit is not None:
        t0 = time.perf_counter()
    target_ratio = single_event_target_ratio(
        old_count,
        new_count,
        alpha=alpha,
        same_voxel=same_voxel,
    )
    acceptance_probability = mh_acceptance_probability(
        target_ratio,
        proposal_ratio=proposal_ratio,
    )
    if audit is not None:
        audit.target_ratio_seconds += time.perf_counter() - t0
        audit.target_ratio_count += 1


    if audit is not None:
        t0 = time.perf_counter()
    if acceptance_probability >= 1.0:
        accepted = True
    else:
        accepted = _as_uniform01(uniform01()) < acceptance_probability
    if audit is not None:
        audit.acceptance_decision_seconds += time.perf_counter() - t0
        audit.acceptance_decision_count += 1
        audit.total_step_count += 1
    if chain_health_collector is not None:
        chain_health_collector.record_step(
            same_voxel=same_voxel,
            old_count=old_count,
            new_count=new_count,
            accepted=accepted,
            proposal_ratio_was_omitted=proposal.proposal_ratio_was_omitted,
        )

    result_state = (
        chain_state.snapshot_state() if snapshot_state_for_result else state_view
    )
    if not accepted:
        return SingleEventMHStepResult(
            selected_event_index=event_index,
            accepted=False,
            same_voxel=same_voxel,
            old_voxel=old_voxel,
            new_voxel=new_voxel,
            acceptance_probability=acceptance_probability,
            state=result_state,
            occupancy_counts=current_counts,
            target_ratio=target_ratio,
            proposal_ratio=proposal_ratio,
        )

    
    if audit is not None:
        t0 = time.perf_counter()
    updated_counts = _apply_single_event_delta_prevalidated_counts(
        current_counts,
        old_voxel=old_voxel,
        new_voxel=new_voxel,
        copy=not chain_owns_counts_buffer,
    )
    chain_state.accept_candidate(
        event_index=event_index,
        candidate_point=proposal.candidate_point,
        new_voxel_index=new_voxel,
    )
    if snapshot_state_for_result:
        result_state = chain_state.snapshot_state()
    else:
        result_state = chain_state.state_view()
    if audit is not None:
        audit.state_commit_seconds += time.perf_counter() - t0
        audit.state_commit_count += 1
        audit.accepted_step_count += 1

    return SingleEventMHStepResult(
        selected_event_index=event_index,
        accepted=True,
        same_voxel=same_voxel,
        old_voxel=old_voxel,
        new_voxel=new_voxel,
        acceptance_probability=acceptance_probability,
        state=result_state,
        occupancy_counts=updated_counts,
        target_ratio=target_ratio,
        proposal_ratio=proposal_ratio,
    )


__all__ = [
    "ProposalCandidate",
    "SingleEventMHStepResult",
    "SingleEventProposalBackend",
    "single_event_mh_step",
]
