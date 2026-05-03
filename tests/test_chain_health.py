"""Focused tests for live-path chain-health aggregation and schema."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from soe.adapters.voi_voxel import VoxelGrid
from soe.contracts import EventObj, VOIConfig
from soe.soe.chain_health import ChainHealthCollector
from soe.soe.kernel import ProposalCandidate
from soe.soe.run_protocol import FiniteRunConfig, run_chain
from soe.soe.state import SurvivingEventState


def _grid() -> VoxelGrid:
    return VoxelGrid.from_config(
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


def _event() -> EventObj:
    return EventObj(
        apex=np.asarray([0.0, 0.0, 0.0], dtype=float),
        axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
        theta=math.pi / 4.0,
    )


def _initial_state() -> SurvivingEventState:
    points = (
        np.asarray((0.25, 0.50, 0.50), dtype=float),
        np.asarray((0.75, 0.50, 0.50), dtype=float),
        np.asarray((1.25, 0.50, 0.50), dtype=float),
    )
    events = tuple(_event() for _ in points)
    return SurvivingEventState(events=events, representative_points=points)


@dataclass(slots=True)
class ScriptedBackend:
    """Deterministic backend for exercising chain-health accounting."""

    steps: tuple[tuple[int, np.ndarray], ...]
    certifies_symmetric_proposal: ClassVar[bool] = True
    _cursor: int = 0

    def select_event_index(self, state: SurvivingEventState) -> int:
        del state
        return self.steps[self._cursor][0]

    def propose_candidate(
        self,
        state: SurvivingEventState,
        event_index: int,
    ) -> ProposalCandidate:
        del state
        expected_event_index, candidate_point = self.steps[self._cursor]
        assert event_index == expected_event_index
        self._cursor += 1
        return ProposalCandidate(candidate_point=candidate_point)


def test_chain_health_collector_tracks_live_run_pair_counts_and_rates() -> None:
    grid = _grid()
    initial_state = _initial_state()
    collector = ChainHealthCollector()
    backend = ScriptedBackend(
        steps=(
            (0, np.asarray((0.40, 0.50, 0.50), dtype=float)),
            (2, np.asarray((2.25, 0.50, 0.50), dtype=float)),
            (1, np.asarray((1.50, 0.50, 0.50), dtype=float)),
            (1, np.asarray((1.50, 0.50, 0.50), dtype=float)),
        )
    )
    draws = iter((0.90, 0.10, 0.90, 0.10))

    result = run_chain(
        initial_state,
        initial_state.occupancy_counts(grid),
        grid=grid,
        proposal_backend=backend,
        alpha=1.0,
        uniform01=lambda: next(draws),
        run_config=FiniteRunConfig(
            num_steps=4,
            burn_in_steps=1,
            thin_every=2,
            capture_terminal_state=True,
        ),
        chain_health_collector=collector,
    )

    assert result.total_steps == 4
    assert result.accepted_step_count == 3
    np.testing.assert_array_equal(
        result.final_occupancy_counts,
        np.asarray([[[1]], [[1]], [[1]]], dtype=np.int64),
    )

    report = collector.build_report(
        run_id="scripted-chain",
        artifact_dir=REPO_ROOT / "tests" / "artifacts",
        total_attempted_steps=result.total_steps,
        burn_in_steps=1,
        thin_every=2,
        surviving_event_count=len(initial_state.events),
        total_voxel_count=int(np.prod(np.asarray(grid.grid_shape, dtype=np.int64))),
        grid_shape=grid.grid_shape,
        alpha=1.0,
        terminal_nonzero_voxel_count=int(np.count_nonzero(result.final_occupancy_counts)),
    )
    payload = report.to_payload()

    assert list(payload.keys()) == [
        "schema_name",
        "schema_version",
        "metadata",
        "rates",
        "histograms",
        "pair_table",
        "occupancy_support",
    ]
    assert list(payload["metadata"].keys()) == [
        "run_id",
        "artifact_dir",
        "total_attempted_steps",
        "burn_in_steps",
        "thin_every",
        "surviving_event_count",
        "total_voxel_count",
        "grid_shape",
        "alpha",
        "proposal_ratio_metadata_omitted_step_count",
        "proposal_ratio_treated_as_one_step_count",
        "proposal_ratio_was_omitted_for_all_steps",
        "proposal_ratio_treated_as_one_for_all_steps",
    ]

    assert report.metadata.run_id == "scripted-chain"
    assert report.metadata.total_attempted_steps == 4
    assert report.metadata.burn_in_steps == 1
    assert report.metadata.thin_every == 2
    assert report.metadata.surviving_event_count == 3
    assert report.metadata.total_voxel_count == 3
    assert report.metadata.grid_shape == (3, 1, 1)
    assert report.metadata.alpha == 1.0
    assert report.metadata.proposal_ratio_metadata_omitted_step_count == 4
    assert report.metadata.proposal_ratio_treated_as_one_step_count == 4
    assert report.metadata.proposal_ratio_was_omitted_for_all_steps is True
    assert report.metadata.proposal_ratio_treated_as_one_for_all_steps is True

    assert report.rates.overall_acceptance_rate == pytest.approx(0.75)
    assert report.rates.same_voxel_proposal_rate == pytest.approx(0.25)
    assert report.rates.cross_voxel_proposal_rate == pytest.approx(0.75)
    assert report.rates.same_voxel_acceptance_rate == pytest.approx(1.0)
    assert report.rates.cross_voxel_acceptance_rate == pytest.approx(2.0 / 3.0)
    assert report.rates.fraction_new_count_eq_0 == pytest.approx(0.75)
    assert report.rates.fraction_old_count_eq_1 == pytest.approx(0.25)
    assert report.rates.fraction_cross_voxel_old1_new0 == pytest.approx(1.0 / 3.0)
    assert report.rates.fraction_accepted_cross_voxel_old1_new0 == pytest.approx(0.5)

    assert report.histograms.old_count == {"1": 1, "2": 3}
    assert report.histograms.new_count == {"0": 3, "2": 1}
    assert report.histograms.cross_voxel_old_count == {"1": 1, "2": 2}
    assert report.histograms.cross_voxel_new_count == {"0": 3}

    assert report.pair_table["1"]["0"].proposal_count == 1
    assert report.pair_table["1"]["0"].accepted_count == 1
    assert report.pair_table["1"]["0"].rejected_count == 0
    assert report.pair_table["1"]["0"].empirical_acceptance_rate == pytest.approx(1.0)
    assert report.pair_table["2"]["0"].proposal_count == 2
    assert report.pair_table["2"]["0"].accepted_count == 1
    assert report.pair_table["2"]["0"].rejected_count == 1
    assert report.pair_table["2"]["0"].empirical_acceptance_rate == pytest.approx(0.5)
    assert report.pair_table["2"]["2"].proposal_count == 1
    assert report.pair_table["2"]["2"].accepted_count == 1
    assert report.pair_table["2"]["2"].rejected_count == 0
    assert report.pair_table["2"]["2"].empirical_acceptance_rate == pytest.approx(1.0)

    assert report.occupancy_support.terminal_nonzero_voxel_count == 3
    assert report.occupancy_support.retained_mean_nonzero_voxel_count is None
