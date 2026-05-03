"""VOI and voxelisation bookkeeping semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


def _build_grid():
    from soe.adapters.voi_voxel import VoiBounds, VoxelGrid
    from soe.contracts import VOIConfig

    config = VOIConfig(
        bounds=np.array(
            [
                [0.0, 4.0],
                [10.0, 18.0],
                [-3.0, 3.0],
            ]
        ),
        grid_shape=(4, 4, 3),
    )
    voi = VoiBounds.from_config(config)
    grid = VoxelGrid.from_config(config)
    assert np.allclose(voi.bounds, config.bounds)
    return grid


def test_half_open_boundaries() -> None:
    """Lower bounds are inclusive and upper bounds are exclusive."""
    from soe.adapters.voi_voxel import world_to_index

    grid = _build_grid()

    assert world_to_index(grid, np.array([0.0, 10.0, -3.0])) == (0, 0, 0)
    assert world_to_index(grid, np.array([4.0, 10.0, -3.0])) is None
    assert world_to_index(grid, np.array([0.0, 18.0, -3.0])) is None
    assert world_to_index(grid, np.array([0.0, 10.0, 3.0])) is None

    idx = world_to_index(grid, np.array([grid.voi.xmax - 1e-9, 10.0, -3.0]))
    assert idx == (grid.nx - 1, 0, 0)
    assert grid.in_voi(np.array([grid.voi.xmax - 1e-9, 17.999999, 2.999999])) is True


def test_exact_tiling_and_no_upper_bound_clamping() -> None:
    """Grid spacing, edges, and volume are derived exactly from VOI and counts."""
    from soe.adapters.voi_voxel import world_to_index

    grid = _build_grid()

    assert grid.grid_shape == (4, 4, 3)
    assert np.isclose(grid.dx, 1.0)
    assert np.isclose(grid.dy, 2.0)
    assert np.isclose(grid.dz, 2.0)
    assert np.isclose(grid.voxel_volume, 4.0)
    assert np.allclose(grid.x_edges, np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert np.allclose(grid.y_edges, np.array([10.0, 12.0, 14.0, 16.0, 18.0]))
    assert np.allclose(grid.z_edges, np.array([-3.0, -1.0, 1.0, 3.0]))

    assert world_to_index(grid, np.array([4.0, 17.0, 0.0])) is None
    assert world_to_index(grid, np.array([3.999999999, 17.999999999, 2.999999999])) == (
        3,
        3,
        2,
    )


def test_round_trip_index_center_index() -> None:
    """Center mapping round-trips deterministically for selected indices."""
    from soe.adapters.voi_voxel import index_to_center, world_to_index

    grid = _build_grid()
    indices = [(0, 0, 0), (1, 2, 1), (grid.nx - 1, grid.ny - 1, grid.nz - 1)]

    for idx in indices:
        center = index_to_center(grid, idx)
        assert world_to_index(grid, center) == idx


def test_ordering_is_i_j_k_equals_x_y_z() -> None:
    """Index tuple ordering is deterministic: x->i, y->j, z->k."""
    from soe.adapters.voi_voxel import world_to_index

    grid = _build_grid()
    base = np.array([0.1, 10.1, -2.9])

    idx_base = world_to_index(grid, base)
    idx_x = world_to_index(grid, base + np.array([1.0, 0.0, 0.0]))
    idx_y = world_to_index(grid, base + np.array([0.0, 2.0, 0.0]))
    idx_z = world_to_index(grid, base + np.array([0.0, 0.0, 2.0]))

    assert idx_base == (0, 0, 0)
    assert idx_x == (1, 0, 0)
    assert idx_y == (0, 1, 0)
    assert idx_z == (0, 0, 1)
