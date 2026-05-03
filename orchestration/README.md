# Orchestration

The orchestration surface runs the SOE reconstruction core from a TOML config
and writes a persisted reconstruction bundle.

## Reconstruction

Entry point:

```bash
python3 orchestration/run_reconstruction.py --config path/to/reconstruction.toml
```

Admission-only mode reuses the same reconstruction TOML and the same live
`_build_initial_state(...)` admission boundary, then stops before MH:

```bash
python3 orchestration/run_reconstruction.py --config path/to/reconstruction.toml --admission-only
```

Recommended convention: keep reconstruction TOMLs under
`orchestration/configs/`. Relative paths inside a TOML are resolved from the
TOML file's directory.

The config supplies the read-only event input, the bounded VOI/grid contract,
optional registration and event filtering, preprocessing controls, deterministic
seeds, and finite MH run settings. The runner assembles
`ReconstructionInput(events, voi)`, initializes SOE state inside the bounded
VOI, runs the reconstruction chain, and writes the output bundle under
`reconstruction.artifacts_dir`.

Required config sections: `[input]`, `[reconstruction]`, `[run]`.

Required keys:

- `input.hdf5_path`
- `reconstruction.bounds_cm`
- `reconstruction.grid_shape`
- `run.num_steps`

Optional sections and parser-read keys:

- `[registration]`: `Q`, `t`
- `[filter]`: `lambda_min`, `species`
- `[survival]`: `mode`, `gate_mode`, `K`, `K0`, `J_max`
- `[seeds]`: `event_index_selector`, `proposal_backend`,
  `acceptance_uniform`

Optional keys with current defaults:

- `reconstruction.artifacts_dir="artifacts"`
- `reconstruction.retained_mean_filename="retained_mean.npy"`
- `reconstruction.terminal_state_filename="terminal_state.npy"`
- `reconstruction.surviving_event_cap` omitted
- `run.burn_in_steps=0`
- `run.thin_every=1`
- `run.capture_terminal_state=true`
- `run.capture_step_results=false`

Current narrow run support is bounded-VOI reconstruction with fixed sampled
full-box preprocessing. For the active reconstruction entrypoint, keep
`[survival]` on `mode="fixed"`, `gate_mode="full_box"`, and `K >= 3`.

Example reconstruction configs:

- `orchestration/configs/toy_cases/gamma_point_source/reconstruction.toml`
  provides a small gamma point-source style bounded-VOI run.
- `orchestration/configs/toy_cases/neutron_line_source/reconstruction.toml`
  provides a neutron line-source style bounded-VOI run.
- The 100 MeV water-phantom config provides a water-phantom style bounded-VOI
  run.
- The 150 MeV water-phantom config provides a water-phantom style bounded-VOI
  run.

The referenced public example inputs are checked in under `data/toy_cases/`
and `data/water_phantom/`. Artifact paths point under
`local_artifacts/reconstruction/`, which is ignored.

Full reconstruction writes:

- `retained_mean.npy`
- `terminal_state.npy`
- `reconstruction_metadata.json`
- `reconstruction_diagnostics.json`
- `chain_health.json`

Admission-only mode writes:

- `admitted_event_ids.json`
- `reconstruction_metadata.json`

`reconstruction_metadata.json` is the artifact-interpretation sidecar:
artifact filenames, reconstruction geometry, and retained-state provenance.
`reconstruction_diagnostics.json` records reconstruction-only run metrics.
`chain_health.json` is the aggregate MH-dynamics sidecar for the live
reconstruction path.
