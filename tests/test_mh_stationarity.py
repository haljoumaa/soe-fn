"""Finite-state validation for the MH kernel shell.

These tests stay entirely on tiny discrete toy state spaces. The symmetric
stationarity check is empirical, so it uses a fixed seed and a conservative
tolerance rather than a brittle exact-frequency assertion.
"""

from __future__ import annotations

import itertools
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from soe.adapters.voi_voxel import VoxelGrid
from soe.contracts import EventObj, VOIConfig
from soe.soe.kernel import ProposalCandidate, single_event_mh_step
from soe.soe.state import SurvivingEventState

StateKey = tuple[int, ...]


def _event() -> EventObj:
    """Build a placeholder canonical event for toy MH validation."""
    return EventObj(
        apex=np.asarray([0.0, 0.0, 0.0], dtype=float),
        axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
        theta=math.pi / 4.0,
    )


@dataclass(frozen=True, slots=True)
class ToyFiniteStateModel:
    """Tiny enumerable representative-point state space for toy MH checks."""

    grid: VoxelGrid
    supports_by_event: tuple[tuple[np.ndarray, ...], ...]
    events: tuple[EventObj, ...]

    @classmethod
    def symmetric_two_event_model(cls) -> "ToyFiniteStateModel":
        """Build a two-event, two-support system over two voxels."""
        grid = VoxelGrid.from_config(
            VOIConfig(
                bounds=np.asarray(
                    [
                        [0.0, 2.0],
                        [0.0, 1.0],
                        [0.0, 1.0],
                    ],
                    dtype=float,
                ),
                grid_shape=(2, 1, 1),
            )
        )
        supports_by_event = (
            (
                np.asarray((0.25, 0.50, 0.50), dtype=float),
                np.asarray((1.25, 0.50, 0.50), dtype=float),
            ),
            (
                np.asarray((0.75, 0.50, 0.50), dtype=float),
                np.asarray((1.75, 0.50, 0.50), dtype=float),
            ),
        )
        events = tuple(_event() for _ in supports_by_event)
        return cls(grid=grid, supports_by_event=supports_by_event, events=events)

    @classmethod
    def asymmetric_one_event_model(cls) -> "ToyFiniteStateModel":
        """Build a one-event, three-support system over three voxels."""
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
        supports_by_event = (
            (
                np.asarray((0.25, 0.50, 0.50), dtype=float),
                np.asarray((1.25, 0.50, 0.50), dtype=float),
                np.asarray((2.25, 0.50, 0.50), dtype=float),
            ),
        )
        return cls(grid=grid, supports_by_event=supports_by_event, events=(_event(),))

    def state_from_key(self, key: StateKey) -> SurvivingEventState:
        """Materialize one representative-point state from support indices."""
        representative_points = tuple(
            self.supports_by_event[event_index][support_index]
            for event_index, support_index in enumerate(key)
        )
        return SurvivingEventState(
            events=self.events,
            representative_points=representative_points,
        )

    def key_from_state(self, state: SurvivingEventState) -> StateKey:
        """Recover support indices for one state generated from this toy model."""
        indices: list[int] = []
        for event_index, point in enumerate(state.representative_points):
            supports = self.supports_by_event[event_index]
            for support_index, support_point in enumerate(supports):
                if np.array_equal(point, support_point):
                    indices.append(support_index)
                    break
            else:
                raise ValueError(
                    f"state point for event {event_index} is outside the toy support"
                )
        return tuple(indices)

    def enumerate_keys(self) -> tuple[StateKey, ...]:
        """Return the full finite state space as support-index tuples."""
        return tuple(
            itertools.product(
                *(range(len(supports)) for supports in self.supports_by_event)
            )
        )


@dataclass(slots=True)
class SymmetricToggleBackend:
    """Uniform event selection with a deterministic support toggle per event."""

    model: ToyFiniteStateModel
    rng: np.random.Generator

    def select_event_index(self, state: SurvivingEventState) -> int:
        """Select exactly one event uniformly."""
        del state
        return int(self.rng.integers(len(self.model.events)))

    def propose_candidate(
        self,
        state: SurvivingEventState,
        event_index: int,
    ) -> ProposalCandidate:
        """Toggle the selected event to its other support point."""
        current_support = self.model.key_from_state(state)[event_index]
        candidate_support = 1 - current_support
        return ProposalCandidate(
            candidate_point=self.model.supports_by_event[event_index][candidate_support],
            proposal_ratio=1.0,
        )


@dataclass(frozen=True, slots=True)
class DeterministicProposalBackend:
    """Deterministic backend used to inspect one specific state transition."""

    proposal: ProposalCandidate

    def select_event_index(self, state: SurvivingEventState) -> int:
        """Always choose the only event in the asymmetric toy model."""
        del state
        return 0

    def propose_candidate(
        self,
        state: SurvivingEventState,
        event_index: int,
    ) -> ProposalCandidate:
        """Return the pre-configured candidate for the inspected edge."""
        del state
        del event_index
        return self.proposal


def _occupancy_target_weight(state: SurvivingEventState, *, grid: VoxelGrid, alpha: float) -> float:
    """Return the adopted occupancy-only surrogate mass up to normalization."""
    counts = state.occupancy_counts(grid)
    return float(np.prod([math.gamma(int(count) + alpha) for count in counts.flat]))


def _exact_target_distribution(
    model: ToyFiniteStateModel,
    *,
    alpha: float,
) -> dict[StateKey, float]:
    """Enumerate the exact normalized target frequencies on the toy state space."""
    weights = {
        key: _occupancy_target_weight(model.state_from_key(key), grid=model.grid, alpha=alpha)
        for key in model.enumerate_keys()
    }
    normalizer = sum(weights.values())
    return {key: value / normalizer for key, value in weights.items()}


def _simulate_chain(
    model: ToyFiniteStateModel,
    *,
    initial_key: StateKey,
    backend: SymmetricToggleBackend,
    alpha: float,
    seed: int,
    num_steps: int,
) -> dict[StateKey, float]:
    """Run the generic one-step kernel and return empirical state frequencies."""
    rng = np.random.default_rng(seed)
    state = model.state_from_key(initial_key)
    counts = state.occupancy_counts(model.grid)
    visits: Counter[StateKey] = Counter()

    for _ in range(num_steps):
        result = single_event_mh_step(
            state,
            counts,
            grid=model.grid,
            proposal_backend=backend,
            alpha=alpha,
            uniform01=rng.random,
        )
        state = result.state
        counts = result.occupancy_counts
        visits[model.key_from_state(state)] += 1

    return {key: visits[key] / float(num_steps) for key in model.enumerate_keys()}


def _transition_matrix_from_kernel(
    model: ToyFiniteStateModel,
    proposal_probabilities: dict[int, dict[int, float]],
    *,
    alpha: float,
    include_proposal_ratio: bool,
) -> np.ndarray:
    """Build the exact one-step state transition matrix from the landed kernel."""
    keys = model.enumerate_keys()
    transition = np.zeros((len(keys), len(keys)), dtype=float)

    for source_index, key in enumerate(keys):
        state = model.state_from_key(key)
        counts = state.occupancy_counts(model.grid)
        current_support = key[0]

        for target_support, proposal_probability in proposal_probabilities[current_support].items():
            reverse_probability = proposal_probabilities[target_support][current_support]
            proposal_ratio = (
                reverse_probability / proposal_probability
                if include_proposal_ratio
                else 1.0
            )
            backend = DeterministicProposalBackend(
                proposal=ProposalCandidate(
                    candidate_point=model.supports_by_event[0][target_support],
                    proposal_ratio=proposal_ratio,
                )
            )
            result = single_event_mh_step(
                state,
                counts,
                grid=model.grid,
                proposal_backend=backend,
                alpha=alpha,
                uniform01=lambda: 0.0,
            )
            target_index = keys.index((target_support,))
            transition[source_index, target_index] = (
                proposal_probability * result.acceptance_probability
            )

        transition[source_index, source_index] = 1.0 - float(
            transition[source_index].sum()
        )

    return transition


def test_symmetric_baseline_empirical_frequencies_match_exact_target() -> None:
    """A long fixed-seed toy chain should match the exact finite-state target approximately."""
    model = ToyFiniteStateModel.symmetric_two_event_model()
    alpha = 1.0
    seed = 20260313
    num_steps = 60000
    backend = SymmetricToggleBackend(model=model, rng=np.random.default_rng(seed + 1))

    exact = _exact_target_distribution(model, alpha=alpha)
    empirical = _simulate_chain(
        model,
        initial_key=(0, 0),
        backend=backend,
        alpha=alpha,
        seed=seed,
        num_steps=num_steps,
    )

    exact_vector = np.asarray([exact[key] for key in model.enumerate_keys()], dtype=float)
    empirical_vector = np.asarray(
        [empirical[key] for key in model.enumerate_keys()],
        dtype=float,
    )

    np.testing.assert_allclose(empirical_vector.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        empirical_vector,
        exact_vector,
        atol=0.03,
    )


def test_asymmetric_proposal_ratio_restores_balance_and_omitting_it_breaks_it() -> None:
    """Proposal-ratio handling should restore the intended finite-state balance."""
    model = ToyFiniteStateModel.asymmetric_one_event_model()
    alpha = 1.0
    proposal_probabilities = {
        0: {1: 0.95, 2: 0.05},
        1: {0: 0.01, 2: 0.99},
        2: {0: 0.80, 1: 0.20},
    }
    target = np.asarray(
        [
            _exact_target_distribution(model, alpha=alpha)[key]
            for key in model.enumerate_keys()
        ],
        dtype=float,
    )
    correct_transition = _transition_matrix_from_kernel(
        model,
        proposal_probabilities,
        alpha=alpha,
        include_proposal_ratio=True,
    )
    wrong_transition = _transition_matrix_from_kernel(
        model,
        proposal_probabilities,
        alpha=alpha,
        include_proposal_ratio=False,
    )

    np.testing.assert_allclose(correct_transition.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(target @ correct_transition, target, atol=1e-12)

    keys = model.enumerate_keys()
    detailed_balance_residuals = []
    for source_index, source_key in enumerate(keys):
        for target_index, target_key in enumerate(keys):
            if source_index == target_index:
                continue
            detailed_balance_residuals.append(
                abs(
                    target[source_index] * correct_transition[source_index, target_index]
                    - target[target_index] * correct_transition[target_index, source_index]
                )
            )
    assert max(detailed_balance_residuals) < 1e-12

    assert np.max(np.abs(target @ wrong_transition - target)) > 0.05
