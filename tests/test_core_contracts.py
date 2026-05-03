"""Contract-shape tests for the frozen core reconstruction interfaces."""

from __future__ import annotations

from dataclasses import fields
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from soe.contracts import (
    EventFilterConfig,
    EventObj,
    ReconstructionInput,
    RegistrationTransform,
    VOIConfig,
)


def test_core_contract_field_names_are_narrow() -> None:
    """Freeze the minimal core contract field surface."""
    assert [field.name for field in fields(VOIConfig)] == ["bounds", "grid_shape"]
    assert [field.name for field in fields(EventObj)] == ["apex", "axis", "theta", "species"]
    assert [field.name for field in fields(ReconstructionInput)] == ["events", "voi"]
    assert [field.name for field in fields(RegistrationTransform)] == ["Q", "t"]
    assert [field.name for field in fields(EventFilterConfig)] == ["lambda_min", "species"]


def test_core_contracts_import_and_validate_shape() -> None:
    """Construct the frozen contracts with valid minimal data."""
    voi = VOIConfig(
        bounds=np.array([[0.0, 4.0], [1.0, 5.0], [2.0, 6.0]]),
        grid_shape=(4, 4, 4),
    )
    event = EventObj(
        apex=np.array([0.0, 0.0, 0.0]),
        axis=np.array([0.0, 0.0, 1.0]),
        theta=np.pi / 4.0,
        species="gamma",
    )
    recon = ReconstructionInput(events=[event], voi=voi)
    transform = RegistrationTransform(Q=np.eye(3), t=np.zeros(3))
    filt = EventFilterConfig(lambda_min=0.5, species=("gamma",))

    assert recon.events == (event,)
    assert recon.voi is voi
    assert transform.Q.shape == (3, 3)
    assert transform.t.shape == (3,)
    assert filt.species == ("gamma",)


def test_core_contract_source_locks_strict_baseline_rule() -> None:
    """The contract docs should freeze the authoritative `0 < lambda_e < 1` rule."""
    source = (REPO_ROOT / "src/soe/contracts/core.py").read_text(encoding="utf-8")

    assert "0 < lambda_e < 1" in source
    assert "not defined by this config" in source


def test_event_filter_config_defaults_to_no_species_filter() -> None:
    """Default filter config should keep species filtering optional."""
    filt = EventFilterConfig()

    assert filt.species is None


def test_event_obj_excludes_noncanonical_fields() -> None:
    """EventObj must not accept ids or provenance fields."""
    with pytest.raises(TypeError):
        EventObj(
            apex=np.array([0.0, 0.0, 0.0]),
            axis=np.array([0.0, 0.0, 1.0]),
            theta=np.pi / 4.0,
            event_id=7,
        )


@pytest.mark.parametrize("lambda_min", [0.0, -1.0, 1.0, 1.1])
def test_event_filter_config_validates_auxiliary_lambda_min(lambda_min: float) -> None:
    """Auxiliary compatibility thresholds should still stay in the open interval."""
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        EventFilterConfig(lambda_min=lambda_min)


def test_registration_transform_rejects_reflections() -> None:
    """Registration stays on same-handed `SO(3)` transforms only."""
    with pytest.raises(ValueError, match="same-handed SO\\(3\\) only"):
        RegistrationTransform(
            Q=np.diag([-1.0, 1.0, 1.0]),
            t=np.zeros(3),
        )
