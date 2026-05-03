# Public Data Inputs

This directory contains public reconstruction-input examples for exercising the
SOE reconstruction core.

The HDF5 files are read-only NGImager-style event inputs used by the example
TOML configs under `orchestration/configs/`. Reconstruction runs should write
outputs to ignored local artifact directories such as
`local_artifacts/reconstruction/`.

Public folders:

- `toy_cases/`
- `water_phantom/`

Public input files:

- `toy_cases/gamma_point_source/imaging_data_records_usrdef_out.h5`
- `toy_cases/neutron_line_source/usrdef_out.h5`
- `water_phantom/dataset_neutrons_100MeV/usrdef_out.h5`
- `water_phantom/dataset_neutrons_100MeV/usrdef_out_bootstrap_50000.h5`
- `water_phantom/dataset_neutrons_105MeV/usrdef_out.h5`
- `water_phantom/dataset_neutrons_110MeV/usrdef_out.h5`
- `water_phantom/dataset_neutrons_115MeV/usrdef_out.h5`
- `water_phantom/dataset_neutrons_120MeV/usrdef_out.h5`
- `water_phantom/dataset_neutrons_125MeV/usrdef_out.h5`
- `water_phantom/dataset_neutrons_135MeV/usrdef_out.h5`
- `water_phantom/dataset_neutrons_150MeV/usrdef_out.h5`
