# SOE Architecture

This repository is organized around the SOE reconstruction core: explicit input
configuration, read-only event ingestion, bounded-VOI reconstruction contracts,
event geometry, SOE state initialization, Metropolis-Hastings reconstruction,
and persisted reconstruction outputs.

## Dependency Graph

- `orchestration/run_reconstruction.py` -> `src/soe/analysis/reconstruction_run.py`.
- `src/soe/analysis/reconstruction_run.py` -> contracts, core input assembly,
  VOI/voxel adapters, event geometry, and live SOE chain modules.
- `src/soe/core/input_assembly.py` -> canonical HDF5 event ingestion plus
  explicit external `VOIConfig`.
- `src/soe/adapters/hdf5_ngimager/reader.py` -> read-only HDF5 event records.

## Reconstruction Flow

1. `orchestration/run_reconstruction.py` loads a reconstruction TOML and
   resolves paths relative to that config.
2. `soe.core.assemble_reconstruction_input_from_hdf5(...)` reads canonical
   events and combines them with the configured bounded `VOIConfig`.
3. Event filtering applies the configured validity/species rules and produces
   `ReconstructionInput(events, voi)`.
4. Reconstruction-side preprocessing admits cone events that intersect the
   bounded VOI and creates one representative point per surviving event.
5. The SOE chain runs proposal, target, kernel, occupancy, and retained-state
   estimator logic over the configured grid.
6. The artifact writer persists `retained_mean`, `terminal_state`, `metadata`,
   admitted-event provenance, reconstruction run metrics, and chain-health
   sidecars.

## Responsibilities

- `contracts`: VOI, event-geometry, registration, filtering, and
  reconstruction-input contracts.
- `core`: reconstruction input assembly from canonical events and explicit VOI.
- `adapters/hdf5_ngimager`: NGImager HDF5 read-only ingestion.
- `adapters/voi_voxel`: VOI and voxel grid primitives.
- `geometry`: cone-event geometry, ray/box geometry, sampled surrogate checks,
  and exact bounded-box surface geometry used by reconstruction.
- `soe`: live state, proposal, target, kernel, run protocol, occupancy,
  estimators, and chain-health machinery.
- `analysis/reconstruction_*`: reconstruction run orchestration, artifact
  sidecars, metadata, run metrics, and persisted bundle helpers.

## Hard Boundary Rules

- `src/soe/adapters/hdf5_ngimager/reader.py` is the HDF5 ingestion boundary.
- Every `h5py.File(...)` mode in source must be literal `"r"`.
- HDF5 writing APIs are forbidden in source.
- Raw inputs are immutable.
- Generated artifacts should be written to ignored local artifact directories,
  not committed.
