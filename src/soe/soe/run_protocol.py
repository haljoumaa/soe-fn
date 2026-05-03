"""Finite-run protocol layer on top of the generic one-step MH shell."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, replace
import time

import numpy as np

from soe.adapters.voi_voxel import VoxelGrid
from soe.soe.chain_health import ChainHealthCollector
from soe.soe.kernel import (
    SingleEventMHStepResult,
    _single_event_mh_step_mutable_chain_state,
)
from soe.soe.proposals import SingleEventProposalBackend
from soe.soe.state import SurvivingEventState, _ChainMutableSurvivingEventState

RetainedStateCallback = Callable[[SurvivingEventState, np.ndarray, int], None]


def _as_grid(value: object) -> VoxelGrid:
    """Validate the explicit grid carrier used by the run protocol."""
    if isinstance(value, VoxelGrid):
        return value
    raise TypeError("grid must be a VoxelGrid")


def _as_nonnegative_int(value: object, *, field_name: str) -> int:
    """Validate one nonnegative integer protocol parameter."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a nonnegative integer")
    try:
        int_value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a nonnegative integer") from exc
    if int_value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return int(int_value)


def _as_positive_int(value: object, *, field_name: str) -> int:
    """Validate one positive integer protocol parameter."""
    int_value = _as_nonnegative_int(value, field_name=field_name)
    if int_value < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return int_value


def _as_bool(value: object, *, field_name: str) -> bool:
    """Validate one boolean protocol flag."""
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool")
    return value


def _as_chain_owned_counts(value: object, *, grid: VoxelGrid) -> np.ndarray:
    """Validate initial counts and copy them into the chain-owned mutable buffer."""
    counts = np.asarray(value)
    if counts.shape != grid.grid_shape:
        raise ValueError(f"initial_counts must have shape {grid.grid_shape}, got {counts.shape}")
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("initial_counts must be integer-valued")
    chain_counts = np.array(counts, dtype=np.int64, copy=True)
    if np.any(chain_counts < 0):
        raise ValueError("initial_counts must be nonnegative")
    return chain_counts


def _require_matching_initial_counts(
    state: SurvivingEventState,
    counts: np.ndarray,
    *,
    grid: VoxelGrid,
) -> None:
    """Reject runs whose supplied counts disagree with the authoritative state."""
    expected_counts = state.occupancy_counts(grid)
    if not np.array_equal(counts, expected_counts):
        raise ValueError(
            "initial_counts must match the authoritative occupancy image of initial_state"
        )


def _snapshot_counts(counts: np.ndarray, *, read_only: bool) -> np.ndarray:
    """Detach one stable counts snapshot from the live chain-owned buffer."""
    snapshot = np.array(counts, dtype=np.int64, copy=True)
    if read_only:
        snapshot.setflags(write=False)
    return snapshot


def _snapshot_step_result(step_result: SingleEventMHStepResult) -> SingleEventMHStepResult:
    """Detach the per-step counts snapshot from later chain mutations."""
    return replace(
        step_result,
        occupancy_counts=_snapshot_counts(step_result.occupancy_counts, read_only=False),
    )


@dataclass(frozen=True, slots=True)
class FiniteRunConfig:
    """Explicit finite-run protocol in post-step units only.

    Convention:
    Only post-step states are eligible for retention. Burn-in excludes the
    first `burn_in_steps` post-step states. After that, retention keeps every
    `thin_every`-th eligible post-step state, using 1-based counting on the
    eligible post-step sequence.
    """

    num_steps: int
    burn_in_steps: int = 0
    thin_every: int = 1
    capture_terminal_state: bool = False
    capture_step_results: bool = False

    def __post_init__(self) -> None:
        """Validate the finite-run protocol parameters."""
        object.__setattr__(
            self,
            "num_steps",
            _as_nonnegative_int(self.num_steps, field_name="num_steps"),
        )
        object.__setattr__(
            self,
            "burn_in_steps",
            _as_nonnegative_int(self.burn_in_steps, field_name="burn_in_steps"),
        )
        object.__setattr__(
            self,
            "thin_every",
            _as_positive_int(self.thin_every, field_name="thin_every"),
        )
        object.__setattr__(
            self,
            "capture_terminal_state",
            _as_bool(self.capture_terminal_state, field_name="capture_terminal_state"),
        )
        object.__setattr__(
            self,
            "capture_step_results",
            _as_bool(self.capture_step_results, field_name="capture_step_results"),
        )

    def retain_post_step(self, step_index: int) -> bool:
        """Return whether this 1-based post-step index is retained."""
        validated_step_index = _as_positive_int(step_index, field_name="step_index")
        if validated_step_index <= self.burn_in_steps:
            return False
        eligible_index = validated_step_index - self.burn_in_steps
        return eligible_index % self.thin_every == 0


@dataclass(frozen=True, slots=True)
class ChainRunResult:
    """Finite-run result over retained post-step representative-point states.

    Full per-step kernel results are debug-only and remain empty unless
    `FiniteRunConfig.capture_step_results` is explicitly enabled.
    """

    final_state: SurvivingEventState
    final_occupancy_counts: np.ndarray
    retained_states: tuple[SurvivingEventState, ...]
    retained_step_indices: tuple[int, ...]
    accepted_step_count: int
    total_steps: int
    step_results: tuple[SingleEventMHStepResult, ...] = ()
    terminal_state: SurvivingEventState | None = None
    terminal_occupancy_counts: np.ndarray | None = None
    terminal_step_index: int | None = None
    chain_core_runtime_seconds: float | None = None


def run_chain(
    initial_state: SurvivingEventState,
    initial_counts: object,
    *,
    grid: VoxelGrid,
    proposal_backend: SingleEventProposalBackend,
    alpha: object,
    uniform01: Callable[[], object],
    run_config: FiniteRunConfig,
    on_retained: RetainedStateCallback | None = None,
    chain_health_collector: ChainHealthCollector | None = None,
) -> ChainRunResult:
    """Run a finite MH chain and retain post-step states under the given protocol.

    `on_retained`, when provided, receives the retained post-step state, a
    read-only snapshot of its post-step occupancy counts, and the retained step
    index. This is an implementation hook for low-memory retained-image
    consumers; it does not affect retention semantics or stored results.
    """
    if not isinstance(initial_state, SurvivingEventState):
        raise TypeError("initial_state must be a SurvivingEventState")
    if not isinstance(run_config, FiniteRunConfig):
        raise TypeError("run_config must be a FiniteRunConfig")
    if on_retained is not None and not callable(on_retained):
        raise TypeError("on_retained must be callable or None")
    if chain_health_collector is not None and not isinstance(
        chain_health_collector,
        ChainHealthCollector,
    ):
        raise TypeError("chain_health_collector must be a ChainHealthCollector or None")

    grid = _as_grid(grid)
    # Fast-path ownership: the chain owns and mutates `current_counts`; retained/reporting
    # consumers and debug snapshots must get detached copies, not aliased writable views.
    current_counts = _as_chain_owned_counts(initial_counts, grid=grid)
    _require_matching_initial_counts(initial_state, current_counts, grid=grid)
    current_chain_state = _ChainMutableSurvivingEventState.from_state(
        initial_state,
        grid=grid,
    )

    retained_states: list[SurvivingEventState] = []
    retained_step_indices: list[int] = []
    accepted_step_count = 0
    chain_core_runtime_seconds = 0.0
    step_results: list[SingleEventMHStepResult] | None = (
        [] if run_config.capture_step_results else None
    )

    for step_index in range(1, run_config.num_steps + 1):
        step_core_start = time.perf_counter()
        step_result = _single_event_mh_step_mutable_chain_state(
            current_chain_state,
            current_counts,
            grid=grid,
            proposal_backend=proposal_backend,
            alpha=alpha,
            uniform01=uniform01,
            chain_health_collector=chain_health_collector,
            chain_owns_counts_buffer=True,
            snapshot_state_for_result=run_config.capture_step_results,
        )
        if step_result.accepted:
            accepted_step_count += 1
        if step_results is not None:
            step_results.append(_snapshot_step_result(step_result))
        current_counts = step_result.occupancy_counts

        retain_post_step = run_config.retain_post_step(step_index)
        if retain_post_step:
            retained_states.append(current_chain_state.snapshot_state())
            retained_step_indices.append(step_index)
        chain_core_runtime_seconds += time.perf_counter() - step_core_start
        if retain_post_step:
            if on_retained is not None:
                retained_counts = _snapshot_counts(current_counts, read_only=True)
                on_retained(retained_states[-1], retained_counts, step_index)

    final_state = (
        initial_state
        if run_config.num_steps == 0
        else current_chain_state.snapshot_state()
    )
    terminal_state: SurvivingEventState | None = None
    terminal_occupancy_counts: np.ndarray | None = None
    terminal_step_index: int | None = None
    if run_config.capture_terminal_state:
        terminal_state = final_state
        terminal_occupancy_counts = current_counts
        terminal_step_index = run_config.num_steps

    return ChainRunResult(
        final_state=final_state,
        final_occupancy_counts=current_counts,
        retained_states=tuple(retained_states),
        retained_step_indices=tuple(retained_step_indices),
        accepted_step_count=accepted_step_count,
        total_steps=run_config.num_steps,
        step_results=tuple(step_results) if step_results is not None else (),
        terminal_state=terminal_state,
        terminal_occupancy_counts=terminal_occupancy_counts,
        terminal_step_index=terminal_step_index,
        chain_core_runtime_seconds=chain_core_runtime_seconds,
    )


__all__ = [
    "ChainRunResult",
    "FiniteRunConfig",
    "run_chain",
]
