"""Focused tests for HDF5 event ingestion."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from soe.adapters.hdf5_ngimager import load_canonical_events
from soe.contracts import EventFilterConfig, RegistrationTransform


def _write_ngimager_fixture(path: Path) -> None:
    """Create a small `/cones` fixture for canonical-event ingestion tests."""
    with h5py.File(path, "w") as handle:
        cones = handle.create_group("cones")
        cones.create_dataset(
            "apex_xyz_cm",
            data=np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [2.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
        )
        cones.create_dataset(
            "axis_xyz",
            data=np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
        )
        cones.create_dataset(
            "theta_rad",
            data=np.asarray(
                [math.pi / 3.0, 2.0 * math.pi / 3.0, math.pi / 2.0],
                dtype=np.float64,
            ),
        )
        cones.create_dataset("species", data=np.asarray([0, 1, 1], dtype=np.uint8))
        cones.create_dataset(
            "species_labels",
            data=np.asarray(["neutron", "gamma"], dtype=h5py.string_dtype("utf-8")),
        )
        cones.create_dataset("event_index", data=np.asarray([10, 11, 12], dtype=np.int32))
        cones.create_dataset("cone_id", data=np.asarray([100, 101, 102], dtype=np.int32))


def test_ingestion_applies_registration_and_frozen_baseline_guard(
    tmp_path: Path,
) -> None:
    """Authoritative ingestion should ignore run-level `lambda_min` and keep `0 < lambda_e < 1`."""
    fixture_path = tmp_path / "ngimager_ingestion.h5"
    _write_ngimager_fixture(fixture_path)

    registration = RegistrationTransform(
        Q=np.asarray(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        t=np.asarray([10.0, 20.0, 30.0]),
    )
    events = load_canonical_events(
        fixture_path,
        registration=registration,
        filter_config=EventFilterConfig(lambda_min=0.9),
    )

    assert len(events) == 2

    first, second = events
    assert np.allclose(first.apex, np.asarray([10.0, 21.0, 30.0]))
    assert np.allclose(first.axis, np.asarray([0.0, 1.0, 0.0]))
    assert math.isclose(first.theta, math.pi / 3.0, abs_tol=1e-12)
    assert first.species == "neutron"

    assert np.allclose(second.apex, np.asarray([10.0, 20.0, 31.0]))
    assert np.allclose(second.axis, np.asarray([0.0, 0.0, -1.0]))
    assert math.isclose(second.theta, math.pi / 3.0, abs_tol=1e-12)
    assert second.species == "gamma"


def test_ingestion_rejects_lambda_endpoints(tmp_path: Path) -> None:
    """Canonical ingestion should reject `lambda_e <= 0` and `lambda_e >= 1`."""
    fixture_path = tmp_path / "lambda_endpoints.h5"
    with h5py.File(fixture_path, "w") as handle:
        cones = handle.create_group("cones")
        cones.create_dataset(
            "apex_xyz_cm",
            data=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
        )
        cones.create_dataset(
            "axis_xyz",
            data=np.asarray(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        )
        cones.create_dataset(
            "theta_rad",
            data=np.asarray(
                [math.pi / 3.0, math.pi / 2.0, 1e-8],
                dtype=np.float64,
            ),
        )

    events = load_canonical_events(
        fixture_path,
        registration=RegistrationTransform(Q=np.eye(3), t=np.zeros(3)),
        filter_config=EventFilterConfig(lambda_min=0.999999),
    )

    assert len(events) == 1
    assert math.isclose(events[0].theta, math.pi / 3.0, abs_tol=1e-12)


def test_ingestion_preserves_unlabeled_species_deterministically(
    tmp_path: Path,
) -> None:
    """Numeric species without labels should be preserved as deterministic raw strings."""
    fixture_path = tmp_path / "species_raw.h5"
    with h5py.File(fixture_path, "w") as handle:
        cones = handle.create_group("cones")
        cones.create_dataset("apex_xyz_cm", data=np.asarray([[0.0, 0.0, 0.0]]))
        cones.create_dataset("axis_xyz", data=np.asarray([[0.0, 0.0, 1.0]]))
        cones.create_dataset("theta_rad", data=np.asarray([math.pi / 3.0]))
        cones.create_dataset("species", data=np.asarray([7], dtype=np.int32))

    events = load_canonical_events(
        fixture_path,
        registration=RegistrationTransform(Q=np.eye(3), t=np.zeros(3)),
        filter_config=EventFilterConfig(),
    )

    assert len(events) == 1
    assert events[0].species == "raw:7"
