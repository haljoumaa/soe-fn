"""Focused tests for the authoritative core input-assembly path."""

from __future__ import annotations

from dataclasses import fields
import importlib
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from soe.adapters.hdf5_ngimager import RawEventRecord
from soe.contracts import (
    EventFilterConfig,
    EventObj,
    ReconstructionInput,
    RegistrationTransform,
    VOIConfig,
)
from soe.core import assemble_reconstruction_input, assemble_reconstruction_input_from_hdf5
from soe.core import input_assembly as input_assembly_module


def _event() -> EventObj:
    """Build a canonical event for handoff tests."""
    return EventObj(
        apex=np.asarray([0.0, 0.0, 0.0], dtype=float),
        axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
        theta=math.pi / 4.0,
    )


def _voi() -> VOIConfig:
    """Build an externally supplied VOI for handoff tests."""
    return VOIConfig(
        bounds=np.asarray(
            [
                [0.0, 4.0],
                [-1.0, 1.0],
                [0.0, 4.0],
            ],
            dtype=float,
        ),
        grid_shape=(4, 2, 4),
    )


def test_assembly_accepts_canonical_events_and_voi_config() -> None:
    """Canonical events plus external VOI should assemble into ReconstructionInput."""
    event = _event()
    voi = _voi()

    assembled = assemble_reconstruction_input([event], voi)

    assert isinstance(assembled, ReconstructionInput)
    assert assembled.events == (event,)
    assert assembled.voi is voi


def test_assembly_returns_only_the_frozen_handoff_contract() -> None:
    """The handoff object should remain `ReconstructionInput(events, voi)` only."""
    assembled = assemble_reconstruction_input([_event()], _voi())
    assert [field.name for field in fields(type(assembled))] == ["events", "voi"]


def test_assembly_rejects_raw_adapter_side_event_records() -> None:
    """Raw adapter-side records should be refused at the authoritative handoff."""
    raw_event = RawEventRecord(
        apex_raw=np.asarray([0.0, 0.0, 0.0], dtype=float),
        axis_raw=np.asarray([0.0, 0.0, 1.0], dtype=float),
        theta_raw=math.pi / 4.0,
        cone_row_index=0,
    )

    with pytest.raises(TypeError, match="canonical EventObj"):
        assemble_reconstruction_input([raw_event], _voi())


def test_hdf5_helper_does_not_depend_on_meta_geometry(tmp_path: Path) -> None:
    """The authoritative helper should use `/cones` only and ignore `lambda_min`."""
    path = tmp_path / "core_no_meta.h5"
    with h5py.File(path, "w") as handle:
        cones = handle.create_group("cones")
        cones.create_dataset("apex_xyz_cm", data=np.asarray([[0.0, 0.0, 0.0]], dtype=float))
        cones.create_dataset("axis_xyz", data=np.asarray([[0.0, 0.0, 1.0]], dtype=float))
        cones.create_dataset("theta_rad", data=np.asarray([math.pi / 4.0], dtype=float))

    assembled = assemble_reconstruction_input_from_hdf5(
        path,
        registration=RegistrationTransform(Q=np.eye(3), t=np.zeros(3)),
        filter_config=EventFilterConfig(lambda_min=0.9),
        voi=_voi(),
    )

    assert isinstance(assembled, ReconstructionInput)
    assert len(assembled.events) == 1


def test_hdf5_helper_requires_explicit_registration_even_for_identity() -> None:
    """The authoritative HDF5 handoff must reject implicit no-registration use."""
    with pytest.raises(TypeError, match="pass explicit identity registration"):
        assemble_reconstruction_input_from_hdf5(
            "unused.h5",
            registration=None,  # type: ignore[arg-type]
            filter_config=EventFilterConfig(),
            voi=_voi(),
        )


def test_hdf5_helper_passes_explicit_identity_registration_to_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit identity registration should still be forwarded through the handoff."""
    identity = RegistrationTransform(Q=np.eye(3), t=np.zeros(3))
    filter_config = EventFilterConfig()
    captured: dict[str, object] = {}

    def _fake_load(
        path: str | Path,
        *,
        registration: RegistrationTransform,
        filter_config: EventFilterConfig,
    ) -> list[EventObj]:
        captured["path"] = path
        captured["registration"] = registration
        captured["filter_config"] = filter_config
        return [_event()]

    monkeypatch.setattr(input_assembly_module, "load_canonical_events", _fake_load)

    assembled = assemble_reconstruction_input_from_hdf5(
        "unused.h5",
        registration=identity,
        filter_config=filter_config,
        voi=_voi(),
    )

    assert len(assembled.events) == 1
    assert np.allclose(assembled.events[0].apex, _event().apex)
    assert captured["registration"] is identity
    assert captured["filter_config"] is filter_config


def test_input_assembly_import_path_is_stable_and_narrow() -> None:
    """The authoritative import path should be the dedicated `soe.core` module."""
    module = importlib.import_module("soe.core.input_assembly")
    assert hasattr(module, "assemble_reconstruction_input")
    assert hasattr(module, "assemble_reconstruction_input_from_hdf5")
