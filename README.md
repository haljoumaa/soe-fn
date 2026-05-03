# SOE-FN

SOE-FN is a Python implementation of Stochastic Origin Ensemble (SOE) reconstruction
from fast-neutron cone-event data. It takes double-scatter neutron events, runs a
Metropolis–Hastings chain on a volume of interest, and produces a voxelised
occupancy image.

## What this does

SOE reconstruction places one representative point per event on its
cone surface, voxelises the ensemble into an occupancy image, and evolves the
configuration using Metropolis–Hastings proposals evaluated against the current
occupancy field.

This repository implements that reconstruction chain: reading cone-event data from
HDF5, filtering events against a rectangular VOI, initialising the ensemble state,
running the MH sampler, and writing the result — retained-mean occupancy, terminal
state, metadata, and event provenance.

Everything is driven by a single TOML config file.

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r requirements-dev.txt
```

Run a reconstruction:

```bash
python3 orchestration/run_reconstruction.py \
  --config orchestration/configs/toy_cases/gamma_point_source/reconstruction.toml
```

Example configs live under `orchestration/configs/`. They reference the public
example data in `data/toy_cases/` and `data/water_phantom/`. Outputs go to local
directories that are git-ignored by default.

## Public data

The `data/` directory contains small example inputs used by the configs and tests.
See `data/README.md` for the layout.

## Development

```bash
python3 -m pytest -q        # tests
python3 -m ruff check .     # linting
```

## License

MIT. See `LICENSE`.
