"""Enforce canonical length units and no implicit VOI/grid scaling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def test_length_unit_and_voi_voxel_passthrough() -> None:
    """Canonical length unit is cm and spacing is derived from extents/counts."""
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from soe.adapters.voi_voxel import VoiBounds, VoxelGrid
    from soe.contracts.units_frames import EPS_BOUNDARY, LENGTH_UNIT

    assert LENGTH_UNIT == "cm"
    assert EPS_BOUNDARY == 1e-12

    voi = VoiBounds(
        xmin=0.0,
        xmax=10.0,
        ymin=1.0,
        ymax=11.0,
        zmin=2.0,
        zmax=12.0,
    )
    grid = VoxelGrid(nx=10, ny=5, nz=2, voi=voi)

    assert voi.xmin == 0.0
    assert voi.xmax == 10.0
    assert grid.dx == (voi.xmax - voi.xmin) / grid.nx
    assert grid.dy == (voi.ymax - voi.ymin) / grid.ny
    assert grid.dz == (voi.zmax - voi.zmin) / grid.nz

    exact_voi = VoiBounds(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        zmin=0.0,
        zmax=1.0,
    )
    assert exact_voi.contains(np.array([1.0, 0.5, 0.5])) is False
