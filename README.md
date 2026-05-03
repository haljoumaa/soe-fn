# novo-soe

SOE reconstruction-core implementation.

This repository implements the bounded-VOI reconstruction path for SOE event
data. A reconstruction starts from an explicit TOML config, reads canonical
NGImager events through the HDF5 ingestion boundary, combines them with an
externally supplied VOI/grid contract, filters event geometry into a
`ReconstructionInput`, initializes SOE state inside the bounded VOI, runs the
Metropolis-Hastings reconstruction chain, and persists the reconstruction
bundle.

The public surface covers:

- Core contracts for `VOIConfig`, `EventObj`, `ReconstructionInput`,
  registration, and event filtering.
- Read-only NGImager HDF5 event ingestion.
- VOI/voxel primitives, cone-event geometry, ray/box geometry, and sampled
  event filtering used by reconstruction.
- SOE state initialization, proposals, target evaluation, MH kernel, run
  protocol, occupancy, retained-state estimators, and chain health.
- Reconstruction orchestration and persisted outputs: `retained_mean`,
  `terminal_state`, `metadata`, admitted-event provenance, and run sidecars.

Run a reconstruction with:

```bash
python3 orchestration/run_reconstruction.py --config orchestration/configs/toy_cases/gamma_point_source/reconstruction.toml
```

Example configs live under `orchestration/configs/` and reference the included
public example inputs under `data/toy_cases/` and `data/water_phantom/`.
Outputs are written to ignored local artifact directories by default.
