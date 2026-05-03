"""Focused tests for the finite run protocol."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from soe.adapters.voi_voxel import VoxelGrid
from soe.contracts import EventObj, VOIConfig
from soe.soe.kernel import ProposalCandidate
from soe.soe.run_protocol import FiniteRunConfig, run_chain
from soe.soe.state import SurvivingEventState


def _event() -> EventObj:
    """Build a placeholder canonical event for toy run-protocol tests."""
    return EventObj(
        apex=np.asarray([0.0, 0.0, 0.0], dtype=float),
        axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
        theta=math.pi / 4.0,
    )


@dataclass(frozen=True, slots=True)
class ToyRunModel:
    """One-event finite-support toy model for protocol tests."""

    grid: VoxelGrid
    supports: tuple[np.ndarray, ...]
    event: EventObj

    @classmethod
    def build(cls) -> "ToyRunModel":
        """Build a simple three-voxel support set with unit occupancy everywhere."""
        grid = VoxelGrid.from_config(
            VOIConfig(
                bounds=np.asarray(
                    [
                        [0.0, 3.0],
                        [0.0, 1.0],
                        [0.0, 1.0],
                    ],
                    dtype=float,
                ),
                grid_shape=(3, 1, 1),
            )
        )
        supports = (
            np.asarray((0.25, 0.50, 0.50), dtype=float),
            np.asarray((1.25, 0.50, 0.50), dtype=float),
            np.asarray((2.25, 0.50, 0.50), dtype=float),
        )
        return cls(grid=grid, supports=supports, event=_event())

    def initial_state(self) -> SurvivingEventState:
        """Return the canonical initial state at support index `0`."""
        return self.state_from_key(0)

    def state_from_key(self, key: int) -> SurvivingEventState:
        """Materialize the authoritative state for one support index."""
        return SurvivingEventState(
            events=(self.event,),
            representative_points=(self.supports[key],),
        )

    def key_from_state(self, state: SurvivingEventState) -> int:
        """Recover the current support index from one authoritative state."""
        point = state.representative_points[0]
        for support_index, support_point in enumerate(self.supports):
            if np.array_equal(point, support_point):
                return support_index
        raise ValueError("state point is outside the toy finite support")


@dataclass(slots=True)
class CyclingBackend:
    """Deterministic backend that cycles through the toy support in order."""

    model: ToyRunModel
    certifies_symmetric_proposal: ClassVar[bool] = True

    def select_event_index(self, state: SurvivingEventState) -> int:
        """Always select the single event in the toy model."""
        del state
        return 0

    def propose_candidate(
        self,
        state: SurvivingEventState,
        event_index: int,
    ) -> ProposalCandidate:
        """Move to the next support point cyclically."""
        del event_index
        current_key = self.model.key_from_state(state)
        next_key = (current_key + 1) % len(self.model.supports)
        return ProposalCandidate(candidate_point=self.model.supports[next_key])


@dataclass(slots=True)
class SeededJumpBackend:
    """Seed-controlled finite-support backend for a small deterministic toy run."""

    model: ToyRunModel
    rng: np.random.Generator
    certifies_symmetric_proposal: ClassVar[bool] = True

    def select_event_index(self, state: SurvivingEventState) -> int:
        """Always select the single event in the toy model."""
        del state
        return 0

    def propose_candidate(
        self,
        state: SurvivingEventState,
        event_index: int,
    ) -> ProposalCandidate:
        """Jump to one of the other support points using the fixed local RNG."""
        del event_index
        current_key = self.model.key_from_state(state)
        candidate_keys = [key for key in range(len(self.model.supports)) if key != current_key]
        choice = int(self.rng.integers(len(candidate_keys)))
        return ProposalCandidate(candidate_point=self.model.supports[candidate_keys[choice]])


def _state_keys(states: tuple[SurvivingEventState, ...], *, model: ToyRunModel) -> tuple[int, ...]:
    """Convert retained toy states to support-index keys for exact assertions."""
    return tuple(model.key_from_state(state) for state in states)


def _expected_seeded_path(
    *,
    num_steps: int,
    initial_key: int,
    seed: int,
    num_supports: int,
) -> tuple[int, ...]:
    """Replicate the test-local seeded-jump rule to compute the exact support path."""
    rng = np.random.default_rng(seed)
    current_key = initial_key
    path: list[int] = []

    for _ in range(num_steps):
        candidate_keys = [key for key in range(num_supports) if key != current_key]
        current_key = candidate_keys[int(rng.integers(len(candidate_keys)))]
        path.append(current_key)

    return tuple(path)


def test_no_post_step_states_are_retained_before_burn_in() -> None:
    """Burn-in should exclude the first post-step states from retention entirely."""
    model = ToyRunModel.build()
    initial_state = model.initial_state()
    result = run_chain(
        initial_state,
        initial_state.occupancy_counts(model.grid),
        grid=model.grid,
        proposal_backend=CyclingBackend(model=model),
        alpha=1.0,
        uniform01=lambda: 0.5,
        run_config=FiniteRunConfig(num_steps=2, burn_in_steps=3, thin_every=1),
    )

    assert result.retained_states == ()
    assert result.retained_step_indices == ()
    assert result.accepted_step_count == 2
    assert result.total_steps == 2
    assert result.step_results == ()
    assert model.key_from_state(result.final_state) == 2


def test_debug_step_history_capture_is_opt_in() -> None:
    """Per-step kernel results should be retained only when explicitly requested."""
    model = ToyRunModel.build()
    initial_state = model.initial_state()
    result = run_chain(
        initial_state,
        initial_state.occupancy_counts(model.grid),
        grid=model.grid,
        proposal_backend=CyclingBackend(model=model),
        alpha=1.0,
        uniform01=lambda: 0.5,
        run_config=FiniteRunConfig(
            num_steps=2,
            burn_in_steps=3,
            thin_every=1,
            capture_step_results=True,
        ),
    )

    assert result.accepted_step_count == 2
    assert result.total_steps == 2
    assert len(result.step_results) == 2
    assert all(step.accepted for step in result.step_results)


def test_thinning_indices_are_explicit_and_correct() -> None:
    """Thinning should keep every `thin_every`-th eligible post-step state only."""
    model = ToyRunModel.build()
    initial_state = model.initial_state()
    result = run_chain(
        initial_state,
        initial_state.occupancy_counts(model.grid),
        grid=model.grid,
        proposal_backend=CyclingBackend(model=model),
        alpha=1.0,
        uniform01=lambda: 0.5,
        run_config=FiniteRunConfig(num_steps=6, burn_in_steps=1, thin_every=2),
    )

    assert result.retained_step_indices == (3, 5)
    assert _state_keys(result.retained_states, model=model) == (0, 2)
    assert result.accepted_step_count == result.total_steps == 6


def test_retention_callback_receives_retained_post_step_counts_only() -> None:
    """The low-memory retention hook should see the same post-step states as retained storage."""
    model = ToyRunModel.build()
    initial_state = model.initial_state()
    observed_steps: list[int] = []
    observed_keys: list[int] = []
    observed_counts: list[np.ndarray] = []
    observed_writeable: list[bool] = []

    def on_retained(state: SurvivingEventState, counts: np.ndarray, step_index: int) -> None:
        observed_steps.append(step_index)
        observed_keys.append(model.key_from_state(state))
        observed_counts.append(counts.copy())
        observed_writeable.append(bool(counts.flags.writeable))

    result = run_chain(
        initial_state,
        initial_state.occupancy_counts(model.grid),
        grid=model.grid,
        proposal_backend=CyclingBackend(model=model),
        alpha=1.0,
        uniform01=lambda: 0.5,
        run_config=FiniteRunConfig(num_steps=6, burn_in_steps=1, thin_every=2),
        on_retained=on_retained,
    )

    assert tuple(observed_steps) == result.retained_step_indices
    assert tuple(observed_keys) == _state_keys(result.retained_states, model=model)
    assert observed_writeable == [False, False]
    for observed_count, retained_state in zip(observed_counts, result.retained_states):
        np.testing.assert_array_equal(observed_count, retained_state.occupancy_counts(model.grid))


def test_run_chain_remains_backward_compatible_when_retention_callback_is_omitted() -> None:
    """Omitting the retention callback must not change retained-state semantics."""
    model = ToyRunModel.build()
    initial_state = model.initial_state()

    result_without_callback = run_chain(
        initial_state,
        initial_state.occupancy_counts(model.grid),
        grid=model.grid,
        proposal_backend=CyclingBackend(model=model),
        alpha=1.0,
        uniform01=lambda: 0.5,
        run_config=FiniteRunConfig(num_steps=6, burn_in_steps=1, thin_every=2),
    )
    result_with_callback = run_chain(
        initial_state,
        initial_state.occupancy_counts(model.grid),
        grid=model.grid,
        proposal_backend=CyclingBackend(model=model),
        alpha=1.0,
        uniform01=lambda: 0.5,
        run_config=FiniteRunConfig(num_steps=6, burn_in_steps=1, thin_every=2),
        on_retained=lambda _state, _counts, _step_index: None,
    )

    assert result_without_callback.retained_step_indices == result_with_callback.retained_step_indices
    assert _state_keys(result_without_callback.retained_states, model=model) == _state_keys(
        result_with_callback.retained_states,
        model=model,
    )
    assert model.key_from_state(result_without_callback.final_state) == model.key_from_state(
        result_with_callback.final_state
    )
    assert result_without_callback.chain_core_runtime_seconds is not None
    assert result_without_callback.chain_core_runtime_seconds >= 0.0


def test_terminal_state_capture_is_separate_from_retained_state_collection() -> None:
    """Terminal capture should not alter the retained-state indexing convention."""
    model = ToyRunModel.build()
    initial_state = model.initial_state()
    result = run_chain(
        initial_state,
        initial_state.occupancy_counts(model.grid),
        grid=model.grid,
        proposal_backend=CyclingBackend(model=model),
        alpha=1.0,
        uniform01=lambda: 0.5,
        run_config=FiniteRunConfig(
            num_steps=5,
            burn_in_steps=2,
            thin_every=2,
            capture_terminal_state=True,
        ),
    )

    assert result.retained_step_indices == (4,)
    assert _state_keys(result.retained_states, model=model) == (1,)
    assert result.accepted_step_count == result.total_steps == 5
    assert result.terminal_step_index == 5
    assert result.terminal_state is result.final_state
    assert result.terminal_occupancy_counts is result.final_occupancy_counts
    assert model.key_from_state(result.terminal_state) == 2


def test_retained_states_follow_the_post_step_not_initial_state_convention() -> None:
    """The first retained state should be the first post-step state, not the initial state."""
    model = ToyRunModel.build()
    initial_state = model.initial_state()
    result = run_chain(
        initial_state,
        initial_state.occupancy_counts(model.grid),
        grid=model.grid,
        proposal_backend=CyclingBackend(model=model),
        alpha=1.0,
        uniform01=lambda: 0.5,
        run_config=FiniteRunConfig(num_steps=1, burn_in_steps=0, thin_every=1),
    )

    assert model.key_from_state(initial_state) == 0
    assert result.retained_step_indices == (1,)
    assert _state_keys(result.retained_states, model=model) == (1,)
    assert result.accepted_step_count == result.total_steps == 1


def test_small_seeded_toy_run_is_deterministic_under_the_protocol() -> None:
    """A finite-support toy run should be exactly reproducible under a fixed seed."""
    model = ToyRunModel.build()
    initial_state = model.initial_state()
    seed = 20260313
    result = run_chain(
        initial_state,
        initial_state.occupancy_counts(model.grid),
        grid=model.grid,
        proposal_backend=SeededJumpBackend(
            model=model,
            rng=np.random.default_rng(seed),
        ),
        alpha=1.0,
        uniform01=np.random.default_rng(seed + 1).random,
        run_config=FiniteRunConfig(num_steps=5, burn_in_steps=1, thin_every=2),
    )

    expected_path = _expected_seeded_path(
        num_steps=5,
        initial_key=0,
        seed=seed,
        num_supports=len(model.supports),
    )

    assert _state_keys(result.retained_states, model=model) == (
        expected_path[2],
        expected_path[4],
    )
    assert result.retained_step_indices == (3, 5)
    assert result.accepted_step_count == result.total_steps == 5
    assert model.key_from_state(result.final_state) == expected_path[-1]


def test_run_chain_accepted_moves_use_internal_mutable_state(monkeypatch) -> None:
    """Accepted hot-path moves should not rebuild the old immutable state tuple."""
    model = ToyRunModel.build()
    initial_state = model.initial_state()
    old_replace_calls = 0
    old_voxel_lookup_calls = 0

    original_replace = SurvivingEventState._trusted_replace_representative_point
    original_voxel_lookup = SurvivingEventState.updated_voxel_index_for_event

    def counted_replace(self, *args, **kwargs):
        nonlocal old_replace_calls
        old_replace_calls += 1
        return original_replace(self, *args, **kwargs)

    def counted_voxel_lookup(self, *args, **kwargs):
        nonlocal old_voxel_lookup_calls
        old_voxel_lookup_calls += 1
        return original_voxel_lookup(self, *args, **kwargs)

    monkeypatch.setattr(
        SurvivingEventState,
        "_trusted_replace_representative_point",
        counted_replace,
    )
    monkeypatch.setattr(
        SurvivingEventState,
        "updated_voxel_index_for_event",
        counted_voxel_lookup,
    )

    result = run_chain(
        initial_state,
        initial_state.occupancy_counts(model.grid),
        grid=model.grid,
        proposal_backend=CyclingBackend(model=model),
        alpha=1.0,
        uniform01=lambda: 0.5,
        run_config=FiniteRunConfig(
            num_steps=4,
            burn_in_steps=0,
            thin_every=2,
            capture_terminal_state=True,
            capture_step_results=True,
        ),
    )

    assert result.accepted_step_count == 4
    assert old_replace_calls == 0
    assert old_voxel_lookup_calls == 0
    assert model.key_from_state(result.final_state) == 1
    assert _state_keys(result.retained_states, model=model) == (2, 1)
    assert result.terminal_state is result.final_state
    np.testing.assert_array_equal(
        result.final_occupancy_counts,
        result.final_state.occupancy_counts(model.grid),
    )
