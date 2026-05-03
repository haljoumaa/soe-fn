# SOE-FN

SOE-FN is a reconstruction-core implementation for stochastic origin ensemble
(SOE) reconstruction on bounded volume-of-interest (VOI) event data.

A reconstruction starts from an explicit TOML configuration, reads canonical
NGImager events through the HDF5 ingestion boundary, combines them with an
externally supplied VOI/grid contract, filters event geometry into a
`ReconstructionInput`, initializes SOE state inside the bounded VOI, runs the
Metropolis-Hastings reconstruction chain, and writes the reconstruction bundle.

## Scope

The repository covers the reconstruction chain up to persisted SOE outputs:

- core contracts for `VOIConfig`, `EventObj`, `ReconstructionInput`,
  registration, and event filtering;
- read-only NGImager HDF5 event ingestion;
- VOI and voxel primitives;
- cone-event geometry, ray/box geometry, and sampled event filtering;
- SOE state initialization, proposal generation, target evaluation,
  Metropolis-Hastings updates, run protocol, occupancy estimation,
  retained-state estimators, and chain-health diagnostics;
- reconstruction orchestration from TOML configuration files;
- persisted reconstruction outputs, including `retained_mean`,
  `terminal_state`, run `metadata`, admitted-event provenance, and sidecar
  files.

## Running a reconstruction

Run a reconstruction with:

```bash
python3 orchestration/run_reconstruction.py --config <path/to/reconstruction.toml>
````

Example configurations are stored under:

```text
orchestration/configs/
```

The example configurations reference input data under:

```text
data/toy_cases/
data/water_phantom/
```

Outputs are written to ignored local artifact directories by default (but you can change that if you wish).
