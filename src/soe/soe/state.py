"""Event usability preprocessing and minimal surviving-event state.

Approximation / Implementation choice:
The exact decision criterion is whether the feasible surface `Sigma_e` has
strictly positive surface measure in the actual allowed region. This repository does not yet implement that exact
positive-measure decision procedure. The current preprocessing survival gate
therefore remains an explicit approximation: it uses deterministic sampled
surrogates that check either sampled ray-cell interior hits or sampled full-box
open-box hits, depending on the configured gate mode. Those sampled
retain/reject decisions are not the exact mathematical criterion or the final
cone-surface admissibility logic.

Representative-point selection remains a separate post-survival step, and the
single representative point carried by the minimal state is validated
pointwise rather than used as the survival decision itself.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from soe.adapters.voi_voxel import VoiBounds, VoxelGrid
from soe.adapters.voi_voxel.voxel import materialize_allowed_boxes
from soe.contracts import EventObj
from soe.geometry.cone import ConeLocalGeometry, is_admissible_point
from soe.geometry.ray_box import BoxBounds
from soe.geometry.sampled_surrogate import (
    adaptive_sampled_full_box_event_usable,
    adaptive_sampled_event_usable,
    sampled_full_box_event_usable,
    sampled_event_usable,
)
from soe.soe.occupancy import (
    apply_single_event_delta as apply_single_event_occupancy_delta,
    build_occupancy_counts,
    count_density_view,
    points_to_voxel_indices,
    updated_voxel_index_for_event as updated_occupancy_voxel_index_for_event,
)

VoxelIndex = tuple[int, int, int]
SAMPLED_SURVIVAL_GATE_RAY_CELL = "ray_cell"
SAMPLED_SURVIVAL_GATE_FULL_BOX = "full_box"
USABILITY_DECISION_APPROXIMATION = "approximation:sampled_ray_cell_interior_hit"
USABILITY_DECISION_APPROXIMATION_FULL_BOX = "approximation:sampled_full_box_open_box_hit"
USABILITY_IMPLEMENTATION_NOTE = (
    "Approximation / Implementation choice: exact strictly-positive surface-"
    "measure usability for Sigma_e is not implemented; the current "
    "preprocessing survival gate uses deterministic sampled surrogates only. "
    "Depending on the configured gate mode, the approximation checks either a "
    "sampled ray-cell interior-hit surrogate or a sampled full-box open-box-hit "
    "surrogate. Positive results are sampled certificates, negative results are "
    "not exact proofs of unusability, and representative-point selection remains "
    "a separate post-survival step."
)


def _as_allowed_region(value: object) -> VoiBounds:
    """Validate the current allowed-region boundary object."""
    if not isinstance(value, VoiBounds):
        raise TypeError("allowed_region must be a VoiBounds")
    return value


def _as_event(event: EventObj) -> EventObj:
    """Validate a canonical event object."""
    if not isinstance(event, EventObj):
        raise TypeError("event must be an EventObj")
    return event


def _as_int(value: object, *, field_name: str) -> int:
    """Validate an integer parameter without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} must be an integer")
    return int(value)


def _as_gate_mode(value: object, *, field_name: str) -> str:
    """Validate the sampled-survival gate selector."""
    gate_mode = str(value)
    if gate_mode not in (SAMPLED_SURVIVAL_GATE_RAY_CELL, SAMPLED_SURVIVAL_GATE_FULL_BOX):
        raise ValueError(
            f"{field_name} must be '{SAMPLED_SURVIVAL_GATE_RAY_CELL}' or "
            f"'{SAMPLED_SURVIVAL_GATE_FULL_BOX}'"
        )
    return gate_mode


def _as_point(value: object, *, field_name: str) -> np.ndarray:
    """Validate a finite representative point in `(x, y, z)` order."""
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{field_name} must have shape (3,), got {point.shape}")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{field_name} must be finite")
    return point


def _iter_allowed_boxes(allowed_region: VoiBounds) -> tuple[
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...
]:
    """Materialize voxel-cell boxes used by the sampled ray-cell gate."""
    if allowed_region._grid_shape is None:
        raise ValueError(
            "sampled ray-cell survival requires grid-aware voxel-cell semantics; "
            "bare VoiBounds without grid metadata are not allowed"
        )
    return materialize_allowed_boxes(allowed_region)


def _whole_voi_open_box(allowed_region: VoiBounds) -> BoxBounds:
    """Return the strict-open whole VOI box for the sampled full-box gate."""
    return (
        (float(allowed_region.xmin), float(allowed_region.xmax)),
        (float(allowed_region.ymin), float(allowed_region.ymax)),
        (float(allowed_region.zmin), float(allowed_region.zmax)),
    )


@dataclass(frozen=True, slots=True)
class SampledSurvivalConfig:
    """Local configuration for the sampled preprocessing survival gate.

    Defaults stay intentionally narrow here: the standard path is
    `SampledSurvivalConfig.fixed(K=8)` on the sampled ray-cell gate. Adaptive
    mode is available only when the caller explicitly requests the dyadic
    `(K0, J_max)` schedule. The sampled full-box gate is a separate explicit
    approximation option for ungridded `VoiBounds`.
    """

    mode: str
    K: int | None = None
    K0: int | None = None
    J_max: int | None = None
    gate_mode: str = SAMPLED_SURVIVAL_GATE_RAY_CELL

    @classmethod
    def fixed(
        cls,
        *,
        K: int = 8,
        gate_mode: str = SAMPLED_SURVIVAL_GATE_RAY_CELL,
    ) -> "SampledSurvivalConfig":
        """Return the default fixed midpoint-grid sampled-survival mode."""
        return cls(mode="fixed", K=K, gate_mode=gate_mode)

    @classmethod
    def adaptive(
        cls,
        *,
        K0: int,
        J_max: int,
        gate_mode: str = SAMPLED_SURVIVAL_GATE_RAY_CELL,
    ) -> "SampledSurvivalConfig":
        """Return the dyadic adaptive sampled-survival mode."""
        return cls(mode="adaptive", K0=K0, J_max=J_max, gate_mode=gate_mode)

    @classmethod
    def fixed_full_box(cls, *, K: int = 8) -> "SampledSurvivalConfig":
        """Return the fixed midpoint-grid sampled full-box gate."""
        return cls.fixed(K=K, gate_mode=SAMPLED_SURVIVAL_GATE_FULL_BOX)

    @classmethod
    def adaptive_full_box(cls, *, K0: int, J_max: int) -> "SampledSurvivalConfig":
        """Return the dyadic sampled full-box gate."""
        return cls.adaptive(
            K0=K0,
            J_max=J_max,
            gate_mode=SAMPLED_SURVIVAL_GATE_FULL_BOX,
        )

    def __post_init__(self) -> None:
        """Validate the narrow sampled-survival configuration contract."""
        mode = str(self.mode)
        gate_mode = _as_gate_mode(self.gate_mode, field_name="gate_mode")
        if mode == "fixed":
            if self.K is None:
                raise ValueError("fixed sampled-survival mode requires K")
            K = _as_int(self.K, field_name="K")
            if K < 3:
                raise ValueError("K must be >= 3")
            if self.K0 is not None or self.J_max is not None:
                raise ValueError("fixed sampled-survival mode accepts only K")
            object.__setattr__(self, "mode", mode)
            object.__setattr__(self, "K", K)
            object.__setattr__(self, "K0", None)
            object.__setattr__(self, "J_max", None)
            object.__setattr__(self, "gate_mode", gate_mode)
            return

        if mode != "adaptive":
            raise ValueError("mode must be 'fixed' or 'adaptive'")
        if self.K0 is None or self.J_max is None:
            raise ValueError("adaptive sampled-survival mode requires K0 and J_max")
        K0 = _as_int(self.K0, field_name="K0")
        J_max = _as_int(self.J_max, field_name="J_max")
        if K0 < 3:
            raise ValueError("K0 must be >= 3")
        if J_max < 0:
            raise ValueError("J_max must be >= 0")
        if self.K is not None:
            raise ValueError("adaptive sampled-survival mode accepts only K0 and J_max")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "K", None)
        object.__setattr__(self, "K0", K0)
        object.__setattr__(self, "J_max", J_max)
        object.__setattr__(self, "gate_mode", gate_mode)

    @property
    def decision_kind(self) -> str:
        """Return the explicit approximation label for the chosen gate."""
        if self.gate_mode == SAMPLED_SURVIVAL_GATE_RAY_CELL:
            return USABILITY_DECISION_APPROXIMATION
        return USABILITY_DECISION_APPROXIMATION_FULL_BOX

    def decision_metadata(self) -> Mapping[str, object]:
        """Return auditable metadata for the current sampled decision."""
        metadata: dict[str, object] = {
            "gate_mode": self.gate_mode,
            "sample_mode": self.mode,
        }
        if self.mode == "fixed":
            assert self.K is not None
            metadata["K"] = self.K
        else:
            assert self.K0 is not None
            assert self.J_max is not None
            metadata["K0"] = self.K0
            metadata["J_max"] = self.J_max
        return MappingProxyType(metadata)


def _as_survival_config(value: SampledSurvivalConfig | None) -> SampledSurvivalConfig:
    """Use the default fixed sampled-survival mode unless the caller overrides it."""
    if value is None:
        return SampledSurvivalConfig.fixed(K=8)
    if not isinstance(value, SampledSurvivalConfig):
        raise TypeError("survival_config must be a SampledSurvivalConfig")
    return value


def _validate_representative_point(
    event: EventObj,
    allowed_region: VoiBounds,
    representative_point: object,
    *,
    tol: float,
) -> np.ndarray:
    """Validate the post-survival representative point for one surviving event."""
    point = _as_point(representative_point, field_name="representative_point")
    try:
        admissible = is_admissible_point(point, event, allowed_region, tol=tol)
    except ValueError as exc:
        raise ValueError(
            "representative point is invalid under the pointwise admissibility check: "
            f"{exc}"
        ) from exc
    if not admissible:
        raise ValueError(
            "representative point is not pointwise admissible in the current allowed region"
        )
    return point


@dataclass(frozen=True, slots=True)
class EventUsabilityDecision:
    """Explicit event-into-surviving-set decision.

    `is_exact=False` is intentional in the current codebase: the decision is a
    sampled-surrogate approximation rather than the exact positive-surface-
    measure criterion from the formulation.
    """

    usable: bool
    decision_kind: str
    is_exact: bool
    reason: str
    decision_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the local audit metadata."""
        object.__setattr__(
            self,
            "decision_metadata",
            MappingProxyType(dict(self.decision_metadata)),
        )


def check_event_usability(
    event: EventObj,
    allowed_region: VoiBounds,
    *,
    survival_config: SampledSurvivalConfig | None = None,
    _cone_geometry: ConeLocalGeometry | None = None,
) -> EventUsabilityDecision:
    """Decide whether an event enters the surviving set `E`.

    Approximation / Implementation choice:
    This function does not implement the exact `surface-measure > 0` decision
    for `Sigma_e`. Instead, it applies the deterministic sampled surrogate
    already implemented under `soe.geometry.sampled_surrogate`.
    `True` means only that the sampled gate certified at least one sampled hit
    of the configured target geometry. `False` means only that the sampled gate
    did not certify the event under the chosen schedule; it is not an exact
    proof of unusability. The sampled ray-cell gate is defined over voxel-cell
    interiors and therefore requires grid-aware `VoiBounds.from_config(...)`
    values. The sampled full-box gate is a separate approximate option for
    ungridded `VoiBounds` whole-box interiors only.
    """
    event = _as_event(event)
    allowed_region = _as_allowed_region(allowed_region)
    survival_config = _as_survival_config(survival_config)
    metadata = survival_config.decision_metadata()

    if survival_config.gate_mode == SAMPLED_SURVIVAL_GATE_RAY_CELL:
        boxes = _iter_allowed_boxes(allowed_region)
        if survival_config.mode == "fixed":
            assert survival_config.K is not None
            usable = sampled_event_usable(
                event,
                boxes,
                K=survival_config.K,
                _cone_geometry=_cone_geometry,
            )
        else:
            assert survival_config.K0 is not None
            assert survival_config.J_max is not None
            usable = adaptive_sampled_event_usable(
                event,
                boxes,
                K0=survival_config.K0,
                J_max=survival_config.J_max,
                _cone_geometry=_cone_geometry,
            )

        if not boxes:
            reason = (
                "Sampled ray-cell preprocessing gate had no allowed box interiors to test; "
                "this approximate rejection is not an exact proof of unusability."
            )
        elif usable:
            reason = (
                "Sampled ray-cell preprocessing gate certified at least one sampled "
                "generator ray hitting an allowed box interior."
            )
        else:
            reason = (
                "Sampled ray-cell preprocessing gate did not certify any sampled "
                "generator ray hitting an allowed box interior under the configured "
                "schedule; this approximate rejection is not an exact proof of "
                "unusability."
            )
    else:
        whole_box = _whole_voi_open_box(allowed_region)
        if survival_config.mode == "fixed":
            assert survival_config.K is not None
            usable = sampled_full_box_event_usable(
                event,
                whole_box,
                K=survival_config.K,
                _cone_geometry=_cone_geometry,
            )
        else:
            assert survival_config.K0 is not None
            assert survival_config.J_max is not None
            usable = adaptive_sampled_full_box_event_usable(
                event,
                whole_box,
                K0=survival_config.K0,
                J_max=survival_config.J_max,
                _cone_geometry=_cone_geometry,
            )

        if usable:
            reason = (
                "Sampled full-box preprocessing gate certified at least one sampled "
                "generator ray hitting the strict-open whole-box interior."
            )
        else:
            reason = (
                "Sampled full-box preprocessing gate did not certify any sampled "
                "generator ray hitting the strict-open whole-box interior under the "
                "configured schedule; this approximate rejection is not an exact "
                "proof of unusability."
            )

    return EventUsabilityDecision(
        usable=usable,
        decision_kind=survival_config.decision_kind,
        is_exact=False,
        reason=reason,
        decision_metadata=metadata,
    )


@dataclass(frozen=True, slots=True)
class SurvivingEventState:
    """Minimal surviving-event state contract.

    The state carries only the surviving canonical event pool `E` and exactly
    one representative point per surviving event. No sampler history, proposal
    state, acceptance counters, image accumulation, or diagnostics payloads are
    stored here. Occupancy counts, density, and local count deltas remain
    deterministic derived views over this continuous representative-point state.
    """

    events: tuple[EventObj, ...]
    representative_points: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        """Normalize events and representative points."""
        events = tuple(self.events)
        representative_points = tuple(
            _as_point(point, field_name="representative_point")
            for point in self.representative_points
        )
        if any(not isinstance(event, EventObj) for event in events):
            raise TypeError("events must contain only EventObj instances")
        if len(events) != len(representative_points):
            raise ValueError(
                "representative_points must contain exactly one point per surviving event"
            )

        object.__setattr__(self, "events", events)
        object.__setattr__(self, "representative_points", representative_points)

    @classmethod
    def _from_prevalidated_parts(
        cls,
        *,
        events: tuple[EventObj, ...],
        representative_points: tuple[np.ndarray, ...],
    ) -> "SurvivingEventState":
        """Build one state from already-normalized internal parts."""
        state = object.__new__(cls)
        object.__setattr__(state, "events", events)
        object.__setattr__(state, "representative_points", representative_points)
        return state

    def _trusted_replace_representative_point(
        self,
        *,
        event_index: object,
        candidate_point: object,
    ) -> "SurvivingEventState":
        """Replace one point using the already-valid state as the invariant source."""
        point_index = _as_int(event_index, field_name="event_index")
        if not 0 <= point_index < len(self.events):
            raise IndexError(f"event_index out of range: {point_index}")
        point = _as_point(candidate_point, field_name="candidate_point")
        updated_points = (
            self.representative_points[:point_index]
            + (point,)
            + self.representative_points[point_index + 1 :]
        )
        return type(self)._from_prevalidated_parts(
            events=self.events,
            representative_points=updated_points,
        )

    def voxel_indices(self, grid: VoxelGrid) -> tuple[tuple[int, int, int], ...]:
        """Return deterministic voxel assignments for representative points."""
        return points_to_voxel_indices(self.representative_points, grid)

    def occupancy_counts(self, grid: VoxelGrid) -> np.ndarray:
        """Return the full-grid occupancy histogram `n(X)` for this state."""
        return build_occupancy_counts(self.voxel_indices(grid), grid)

    def updated_voxel_index_for_event(
        self,
        event_index: int,
        candidate_point: object,
        grid: VoxelGrid,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Return the exact old/new voxel indices for one point replacement."""
        return updated_occupancy_voxel_index_for_event(
            self.representative_points,
            event_index,
            candidate_point,
            grid,
        )

    def apply_single_event_delta(
        self,
        counts: object,
        *,
        event_index: int,
        candidate_point: object,
        grid: VoxelGrid,
    ) -> np.ndarray:
        """Apply the exact local occupancy-count delta for one event replacement."""
        return apply_single_event_occupancy_delta(
            counts,
            self.representative_points,
            event_index,
            candidate_point,
            grid,
        )

    def canonical_image(self, grid: VoxelGrid) -> np.ndarray:
        """Return the canonical per-state image `D(X) = n(X)`."""
        return self.occupancy_counts(grid)

    def count_density_view(self, grid: VoxelGrid) -> np.ndarray:
        """Return the derived count-density view `rho_v(X) = n_v(X) / |B_v|`."""
        return count_density_view(self.occupancy_counts(grid), grid)

    @classmethod
    def from_surviving_events(
        cls,
        events: object,
        allowed_region: VoiBounds,
        representative_points: object,
        *,
        survival_config: SampledSurvivalConfig | None = None,
        tol: float = 1e-12,
    ) -> "SurvivingEventState":
        """Build the minimal state from already-supplied surviving-event points.

        Each supplied event must first survive the sampled preprocessing gate
        under the current allowed region and chosen sampled gate mode.
        Representative-point selection remains separate: after survival, each
        event must still have exactly one supplied representative point on its
        feasible surface.
        """
        allowed_region = _as_allowed_region(allowed_region)
        survival_config = _as_survival_config(survival_config)
        events_tuple = tuple(events)
        points_tuple = tuple(representative_points)

        if len(events_tuple) != len(points_tuple):
            raise ValueError(
                "representative_points must contain exactly one point per surviving event"
            )

        validated_points: list[np.ndarray] = []
        for idx, (event, point) in enumerate(zip(events_tuple, points_tuple)):
            decision = check_event_usability(event, allowed_region, survival_config=survival_config)
            if not decision.usable:
                raise ValueError(
                    f"event {idx} did not survive sampled preprocessing: {decision.reason}"
                )
            validated_points.append(
                _validate_representative_point(
                    event,
                    allowed_region,
                    point,
                    tol=tol,
                )
            )

        return cls(events=tuple(events_tuple), representative_points=tuple(validated_points))

    @classmethod
    def _from_already_admitted_events(
        cls,
        events: object,
        allowed_region: VoiBounds,
        representative_points: object,
        *,
        tol: float = 1e-12,
    ) -> "SurvivingEventState":
        """Trusted internal handoff for events already admitted by one preprocessing pass."""
        allowed_region = _as_allowed_region(allowed_region)
        events_tuple = tuple(events)
        points_tuple = tuple(representative_points)

        if len(events_tuple) != len(points_tuple):
            raise ValueError(
                "representative_points must contain exactly one point per surviving event"
            )

        validated_events: list[EventObj] = []
        validated_points: list[np.ndarray] = []
        for event, point in zip(events_tuple, points_tuple):
            validated_event = _as_event(event)
            validated_events.append(validated_event)
            validated_points.append(
                _validate_representative_point(
                    validated_event,
                    allowed_region,
                    point,
                    tol=tol,
                )
            )

        return cls._from_prevalidated_parts(
            events=tuple(validated_events),
            representative_points=tuple(validated_points),
        )


class _RepresentativePointSequence(Sequence[np.ndarray]):
    """Live sequence view over chain-internal representative points."""

    __slots__ = ("_points",)

    def __init__(self, points: list[np.ndarray]) -> None:
        self._points = points

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self._points)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self._points[index])
        return self._points[index]


class _ChainMutableSurvivingEventState:
    """Chain-internal mutable state with cached representative-point voxels.

    This is intentionally private to the finite-run hot path. Public results
    are still emitted as ordinary immutable `SurvivingEventState` snapshots.
    """

    __slots__ = (
        "events",
        "_representative_points",
        "_state_view",
        "_voxel_indices",
    )

    def __init__(
        self,
        *,
        events: tuple[EventObj, ...],
        representative_points: list[np.ndarray],
        voxel_indices: list[VoxelIndex],
    ) -> None:
        if len(events) != len(representative_points):
            raise ValueError(
                "representative_points must contain exactly one point per surviving event"
            )
        if len(voxel_indices) != len(representative_points):
            raise ValueError(
                "voxel_indices must contain exactly one index per surviving event"
            )
        self.events = events
        self._representative_points = representative_points
        self._voxel_indices = voxel_indices
        self._state_view = SurvivingEventState._from_prevalidated_parts(
            events=events,
            representative_points=_RepresentativePointSequence(
                self._representative_points
            ),
        )

    @classmethod
    def from_state(
        cls,
        state: SurvivingEventState,
        *,
        grid: VoxelGrid,
    ) -> "_ChainMutableSurvivingEventState":
        """Build the chain-internal state from one authoritative snapshot."""
        if not isinstance(state, SurvivingEventState):
            raise TypeError("state must be a SurvivingEventState")
        representative_points = [
            _as_point(point, field_name="representative_point")
            for point in state.representative_points
        ]
        voxel_indices = list(points_to_voxel_indices(representative_points, grid))
        return cls(
            events=state.events,
            representative_points=representative_points,
            voxel_indices=voxel_indices,
        )

    def state_view(self) -> SurvivingEventState:
        """Return a live `SurvivingEventState` view for proposal backends."""
        return self._state_view

    def snapshot_state(self) -> SurvivingEventState:
        """Materialize a stable public state snapshot."""
        return SurvivingEventState._from_prevalidated_parts(
            events=self.events,
            representative_points=tuple(self._representative_points),
        )

    def updated_voxel_index_for_event(
        self,
        event_index: object,
        candidate_point: object,
        grid: VoxelGrid,
    ) -> tuple[VoxelIndex, VoxelIndex]:
        """Return cached old voxel and freshly voxelised candidate voxel."""
        point_index = _as_int(event_index, field_name="event_index")
        if not 0 <= point_index < len(self._representative_points):
            raise IndexError(f"event_index out of range: {point_index}")
        candidate = _as_point(candidate_point, field_name="candidate_point")
        if not isinstance(grid, VoxelGrid):
            raise TypeError("grid must be a VoxelGrid")
        new_voxel_index = grid.world_to_index(candidate)
        if new_voxel_index is None:
            raise ValueError(
                "candidate representative point does not voxelise through the frozen "
                f"VOI world-to-index map for event {point_index}: {candidate.tolist()}"
            )
        return self._voxel_indices[point_index], new_voxel_index

    def accept_candidate(
        self,
        *,
        event_index: object,
        candidate_point: object,
        new_voxel_index: VoxelIndex,
    ) -> None:
        """Commit one accepted representative-point and voxel-index update."""
        point_index = _as_int(event_index, field_name="event_index")
        if not 0 <= point_index < len(self._representative_points):
            raise IndexError(f"event_index out of range: {point_index}")
        self._representative_points[point_index] = _as_point(
            candidate_point,
            field_name="candidate_point",
        )
        self._voxel_indices[point_index] = (
            int(new_voxel_index[0]),
            int(new_voxel_index[1]),
            int(new_voxel_index[2]),
        )


__all__ = [
    "EventUsabilityDecision",
    "SampledSurvivalConfig",
    "SAMPLED_SURVIVAL_GATE_FULL_BOX",
    "SAMPLED_SURVIVAL_GATE_RAY_CELL",
    "SurvivingEventState",
    "USABILITY_DECISION_APPROXIMATION",
    "USABILITY_DECISION_APPROXIMATION_FULL_BOX",
    "USABILITY_IMPLEMENTATION_NOTE",
    "check_event_usability",
]
