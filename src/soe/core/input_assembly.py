"""Authoritative reconstruction input-assembly.

This module is the single narrow assembly path for reconstruction inputs:

- HDF5 adapter -> canonical `EventObj[]`
- External VOI boundary -> `VOIConfig`
- Assembly -> `ReconstructionInput(events, voi)`

No raw HDF5 objects, `/meta` geometry, state, representative points, usability
decisions, diagnostics payloads, or provenance are part of the returned handoff
object.
"""

from __future__ import annotations

from pathlib import Path

from soe.adapters.hdf5_ngimager import load_canonical_events
from soe.contracts import (
    EventFilterConfig,
    EventObj,
    ReconstructionInput,
    RegistrationTransform,
    VOIConfig,
)


def _canonical_event_tuple(events: object) -> tuple[EventObj, ...]:
    """Normalize the authoritative canonical event sequence."""
    canonical_events = tuple(events)
    if any(not isinstance(event, EventObj) for event in canonical_events):
        raise TypeError(
            "events must contain only canonical EventObj instances; raw adapter-side "
            "records are not accepted by the reconstruction input handoff"
        )
    return canonical_events


def assemble_reconstruction_input(
    events: object,
    voi: VOIConfig,
) -> ReconstructionInput:
    """Assemble the authoritative `ReconstructionInput` object.

    This function consumes already-canonical `EventObj` values together with an
    externally supplied `VOIConfig` and returns only
    `ReconstructionInput(events, voi)`.
    """
    if not isinstance(voi, VOIConfig):
        raise TypeError("voi must be a VOIConfig")
    return ReconstructionInput(events=_canonical_event_tuple(events), voi=voi)


def assemble_reconstruction_input_from_hdf5(
    path: str | Path,
    *,
    registration: RegistrationTransform,
    filter_config: EventFilterConfig,
    voi: VOIConfig,
) -> ReconstructionInput:
    """Compose canonical HDF5 event loading with the authoritative handoff.

    The HDF5 adapter path used here yields canonical `EventObj[]` from `/cones`
    only; reconstruction VOI still comes exclusively from the external
    `VOIConfig`. Registration must remain explicit on this authoritative path:
    callers pass a `RegistrationTransform` even when the intended transform is
    the identity. The authoritative event-validity rule remains fixed inside
    adapter canonical ingestion as `0 < lambda_e < 1`; `filter_config` does
    not redefine that rule on this handoff.
    """
    if not isinstance(registration, RegistrationTransform):
        raise TypeError(
            "registration must be a RegistrationTransform; pass explicit identity "
            "registration when no frame change is needed"
        )
    if not isinstance(filter_config, EventFilterConfig):
        raise TypeError("filter_config must be an EventFilterConfig")
    if not isinstance(voi, VOIConfig):
        raise TypeError("voi must be a VOIConfig")

    canonical_events = load_canonical_events(
        path,
        registration=registration,
        filter_config=filter_config,
    )
    return assemble_reconstruction_input(canonical_events, voi)


__all__ = [
    "assemble_reconstruction_input",
    "assemble_reconstruction_input_from_hdf5",
]
