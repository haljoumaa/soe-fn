"""Read-only NGImager HDF5 loader utilities for `/meta` and `/cones`.
"""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np

from soe.contracts import EventFilterConfig, EventObj, RegistrationTransform

from .types import Cones, MetaGeometry, RawEventRecord


class Hdf5SchemaError(ValueError):
    """Raised when expected NGImager HDF5 structure is missing or malformed."""


_AXIS_NORM_MIN = 1e-12
_RAW_AXIS_CONVENTION = "second interaction -> first interaction"


def _passes_authoritative_baseline_validity(
    event: EventObj, lambda_canonical: float
) -> bool:
    """Apply the authoritative validity rule: strict nondegeneracy `0 < lambda_e < 1` on the canonical event."""
    if not (0.0 < lambda_canonical < 1.0):
        return False
    # `cos(pi/2)` is not represented as an exact zero in binary floating point,
    # so reject the equivalent canonical endpoint explicitly as well.
    return event.theta < (math.pi / 2.0)


def _to_serializable(value: object) -> object:
    """Convert common NumPy/HDF5 attribute values to JSON-friendly Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    if isinstance(value, np.ndarray):
        as_python = value.tolist()
        if isinstance(as_python, list):
            return [_to_serializable(item) for item in as_python]
        return _to_serializable(as_python)
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value


def _extract_plane(attrs: dict[str, object]) -> dict[str, list[float]]:
    """Collect plane.* numeric vector-style attributes into plane metadata."""
    plane: dict[str, list[float]] = {}
    for key, value in attrs.items():
        if not key.startswith("plane."):
            continue
        name = key.split(".", maxsplit=1)[1]

        values: list[float] = []
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values = [float(value)]
        elif isinstance(value, list) and value and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
        ):
            values = [float(item) for item in value]

        if values:
            plane[name] = values
    return plane


def _extract_grid(attrs: dict[str, object]) -> dict[str, float | int]:
    """Collect grid.* scalar numeric attributes into grid metadata."""
    grid: dict[str, float | int] = {}
    for key, value in attrs.items():
        if not key.startswith("grid."):
            continue
        name = key.split(".", maxsplit=1)[1]
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            grid[name] = value
        elif isinstance(value, float):
            grid[name] = value
    return grid


def _read_cones_dataset(
    cones_group: h5py.Group, canonical_name: str, aliases: tuple[str, ...]
) -> np.ndarray:
    """Read a required /cones dataset, allowing compatibility aliases if needed."""
    for name in aliases:
        dataset = cones_group.get(name)
        if dataset is not None:
            return np.asarray(dataset[()])
    raise Hdf5SchemaError(f"missing dataset /cones/{canonical_name}")


def _read_optional_cones_dataset(
    cones_group: h5py.Group, name: str, *, expected_rows: int | None = None
) -> np.ndarray | None:
    """Read an optional `/cones/*` dataset with an optional row-count check."""
    dataset = cones_group.get(name)
    if dataset is None:
        return None
    values = np.asarray(dataset[()])
    if expected_rows is not None:
        if values.ndim != 1 or values.shape[0] != expected_rows:
            raise Hdf5SchemaError(f"/cones/{name} must have shape (N,); got {values.shape}")
    return values


def _decode_string(value: object) -> str:
    """Decode a scalar string-like HDF5 value into text."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_label_table(values: np.ndarray | None) -> tuple[str, ...] | None:
    """Decode a one-dimensional label table into Python strings."""
    if values is None:
        return None
    flat = np.asarray(values).reshape(-1)
    return tuple(_decode_string(item) for item in flat)


def _read_optional_scalar_text_dataset(
    handle: h5py.File, dataset_path: str
) -> str | None:
    """Read one optional scalar string dataset from the HDF5 handle."""
    dataset = handle.get(dataset_path)
    if dataset is None:
        return None
    if not isinstance(dataset, h5py.Dataset):
        raise Hdf5SchemaError(f"/{dataset_path} is not a dataset")

    value = dataset[()]
    if np.asarray(value).ndim != 0:
        raise Hdf5SchemaError(f"/{dataset_path} must be a scalar string-like dataset")
    return _decode_string(value)


def _species_from_raw_value(
    raw_value: object, *, labels: tuple[str, ...] | None
) -> str:
    """Resolve a raw species value to a deterministic adapter-side string."""
    if isinstance(raw_value, (bytes, str)):
        return _decode_string(raw_value)

    scalar = _to_serializable(raw_value)
    if labels is not None and isinstance(scalar, (int, np.integer)):
        code = int(scalar)
        if 0 <= code < len(labels):
            return labels[code]

    return f"raw:{scalar}"


def _build_raw_event_records(cones_group: h5py.Group) -> list[RawEventRecord]:
    """Read `/cones/*` arrays and build raw adapter-side event records."""
    apex_xyz = _read_cones_dataset(cones_group, "apex_xyz", ("apex_xyz", "apex_xyz_cm"))
    axis_xyz = _read_cones_dataset(cones_group, "axis_xyz", ("axis_xyz",))
    theta = _read_cones_dataset(cones_group, "theta", ("theta", "theta_rad"))

    if apex_xyz.ndim != 2 or apex_xyz.shape[1] != 3:
        raise Hdf5SchemaError(f"/cones/apex_xyz must have shape (N,3); got {apex_xyz.shape}")
    if axis_xyz.ndim != 2 or axis_xyz.shape[1] != 3:
        raise Hdf5SchemaError(f"/cones/axis_xyz must have shape (N,3); got {axis_xyz.shape}")
    if theta.ndim != 1:
        raise Hdf5SchemaError(f"/cones/theta must have shape (N,); got {theta.shape}")

    n_rows = apex_xyz.shape[0]
    if axis_xyz.shape[0] != n_rows:
        raise Hdf5SchemaError(
            "row count mismatch: /cones/axis_xyz and /cones/apex_xyz must share N"
        )
    if theta.shape[0] != n_rows:
        raise Hdf5SchemaError(
            "row count mismatch: /cones/theta and /cones/apex_xyz must share N"
        )

    cone_id_values = _read_optional_cones_dataset(
        cones_group, "cone_id", expected_rows=n_rows
    )
    event_index_values = _read_optional_cones_dataset(
        cones_group, "event_index", expected_rows=n_rows
    )
    species_values = _read_optional_cones_dataset(
        cones_group, "species", expected_rows=n_rows
    )
    species_labels = _decode_label_table(
        _read_optional_cones_dataset(cones_group, "species_labels")
    )

    records: list[RawEventRecord] = []
    for idx in range(n_rows):
        species = None
        if species_values is not None:
            species = _species_from_raw_value(
                species_values[idx], labels=species_labels
            )

        cone_id = None if cone_id_values is None else int(cone_id_values[idx])
        event_index = (
            None if event_index_values is None else int(event_index_values[idx])
        )
        records.append(
            RawEventRecord(
                apex_raw=np.asarray(apex_xyz[idx], dtype=float),
                axis_raw=np.asarray(axis_xyz[idx], dtype=float),
                theta_raw=float(theta[idx]),
                species=species,
                cone_row_index=idx,
                event_index=event_index,
                cone_id=cone_id,
            )
        )
    return records


def _load_raw_event_records(path: str | Path) -> list[RawEventRecord]:
    """Load raw `/cones` rows inside the adapter boundary."""
    with h5py.File(str(path), "r") as handle:
        if "cones" not in handle:
            raise Hdf5SchemaError("missing group /cones")

        cones_group = handle.get("cones")
        if not isinstance(cones_group, h5py.Group):
            raise Hdf5SchemaError("/cones is not a group")

        return _build_raw_event_records(cones_group)


def _canonicalize_raw_event(
    record: RawEventRecord, registration: RegistrationTransform
) -> tuple[EventObj, float]:
    """Apply explicit registration and one-nappe canonicalization to a raw event."""
    apex_raw = np.asarray(record.apex_raw, dtype=float)
    axis_raw = np.asarray(record.axis_raw, dtype=float)
    theta_raw = float(record.theta_raw)

    if apex_raw.shape != (3,):
        raise Hdf5SchemaError(
            f"row {record.cone_row_index}: apex must have shape (3,), got {apex_raw.shape}"
        )
    if axis_raw.shape != (3,):
        raise Hdf5SchemaError(
            f"row {record.cone_row_index}: axis must have shape (3,), got {axis_raw.shape}"
        )
    if not np.isfinite(apex_raw).all():
        raise Hdf5SchemaError(f"row {record.cone_row_index}: apex contains non-finite values")
    if not np.isfinite(axis_raw).all():
        raise Hdf5SchemaError(f"row {record.cone_row_index}: axis contains non-finite values")
    if not math.isfinite(theta_raw):
        raise Hdf5SchemaError(f"row {record.cone_row_index}: theta is not finite")

    axis_norm = float(np.linalg.norm(axis_raw))
    if axis_norm <= _AXIS_NORM_MIN:
        raise Hdf5SchemaError(
            f"row {record.cone_row_index}: raw axis norm must exceed {_AXIS_NORM_MIN}"
        )
    if not (0.0 < theta_raw < math.pi):
        raise Hdf5SchemaError(f"row {record.cone_row_index}: theta must be in (0, pi)")

    
    apex_registered = (registration.Q @ apex_raw) + registration.t
    axis_registered = registration.Q @ axis_raw
    axis_registered_norm = float(np.linalg.norm(axis_registered))
    if axis_registered_norm <= _AXIS_NORM_MIN:
        raise Hdf5SchemaError(
            f"row {record.cone_row_index}: registered axis norm must exceed {_AXIS_NORM_MIN}"
        )

    axis_unit = axis_registered / axis_registered_norm
    theta_canonical = theta_raw
    lambda_raw = float(math.cos(theta_raw))
    if lambda_raw < 0.0:
        axis_unit = -axis_unit
        theta_canonical = math.pi - theta_raw

    lambda_canonical = float(math.cos(theta_canonical))
    if not (0.0 < theta_canonical <= (math.pi / 2.0)):
        raise Hdf5SchemaError(
            f"row {record.cone_row_index}: canonical theta must be in (0, pi/2]"
        )
    if not (-1e-12 <= lambda_canonical <= 1.0 + 1e-12):
        raise Hdf5SchemaError(
            f"row {record.cone_row_index}: canonical lambda must be in [0, 1]"
        )

    event = EventObj(
        apex=apex_registered,
        axis=axis_unit,
        theta=theta_canonical,
        species=record.species,
    )
    return event, lambda_canonical


def load_meta_and_cones(path: str | Path) -> tuple[MetaGeometry, Cones]:
    """Load /meta imaging-plane metadata and minimal raw /cones arrays."""
    with h5py.File(str(path), "r") as handle:
        if "meta" not in handle:
            raise Hdf5SchemaError("missing group /meta")
        if "cones" not in handle:
            raise Hdf5SchemaError("missing group /cones")

        meta_group = handle.get("meta")
        cones_group = handle.get("cones")
        if not isinstance(meta_group, h5py.Group):
            raise Hdf5SchemaError("/meta is not a group")
        if not isinstance(cones_group, h5py.Group):
            raise Hdf5SchemaError("/cones is not a group")

        meta_attrs: dict[str, object] = {}
        for key in meta_group.attrs.keys():
            meta_attrs[key] = _to_serializable(meta_group.attrs.get(key))
        meta = MetaGeometry(
            attrs=meta_attrs,
            plane=_extract_plane(meta_attrs),
            grid=_extract_grid(meta_attrs),
        )

        apex_xyz = _read_cones_dataset(
            cones_group, "apex_xyz", ("apex_xyz", "apex_xyz_cm")
        )
        axis_xyz = _read_cones_dataset(cones_group, "axis_xyz", ("axis_xyz",))
        theta = _read_cones_dataset(cones_group, "theta", ("theta", "theta_rad"))

        if apex_xyz.ndim != 2 or apex_xyz.shape[1] != 3:
            raise Hdf5SchemaError(
                f"/cones/apex_xyz must have shape (N,3); got {apex_xyz.shape}"
            )
        if axis_xyz.ndim != 2 or axis_xyz.shape[1] != 3:
            raise Hdf5SchemaError(
                f"/cones/axis_xyz must have shape (N,3); got {axis_xyz.shape}"
            )
        if theta.ndim != 1:
            raise Hdf5SchemaError(f"/cones/theta must have shape (N,); got {theta.shape}")

        n_rows = apex_xyz.shape[0]
        if axis_xyz.shape[0] != n_rows:
            raise Hdf5SchemaError(
                "row count mismatch: /cones/axis_xyz and /cones/apex_xyz must share N"
            )
        if theta.shape[0] != n_rows:
            raise Hdf5SchemaError(
                "row count mismatch: /cones/theta and /cones/apex_xyz must share N"
            )

        cone_id_dataset = cones_group.get("cone_id")
        if cone_id_dataset is None:
            cone_id = np.arange(n_rows, dtype=np.int64)
        else:
            cone_id = np.asarray(cone_id_dataset[()])
            if cone_id.ndim != 1 or cone_id.shape[0] != n_rows:
                raise Hdf5SchemaError(
                    f"/cones/cone_id must have shape (N,); got {cone_id.shape}"
                )

        return meta, Cones(
            cone_id=cone_id,
            apex_xyz=apex_xyz,
            axis_xyz=axis_xyz,
            theta=theta,
        )


def load_image_context_asset(
    path: str | Path, *, dataset_path: str = "images/summed/all"
) -> dict[str, object] | None:
    """Load one optional bundled 2D image context asset plus lightweight metadata.
    """
    with h5py.File(str(path), "r") as handle:
        dataset = handle.get(dataset_path)
        if dataset is None:
            return None
        if not isinstance(dataset, h5py.Dataset):
            raise Hdf5SchemaError(f"/{dataset_path} is not a dataset")

        image = np.asarray(dataset[()], dtype=float)
        if image.ndim != 2:
            raise Hdf5SchemaError(
                f"/{dataset_path} must have shape (H, W) for a 2D context image; got {image.shape}"
            )

        meta_attrs: dict[str, object] = {}
        meta_group = handle.get("meta")
        if meta_group is not None:
            if not isinstance(meta_group, h5py.Group):
                raise Hdf5SchemaError("/meta is not a group")
            for key in meta_group.attrs.keys():
                meta_attrs[key] = _to_serializable(meta_group.attrs.get(key))

        software = handle.attrs.get("software")
        software_text = None if software is None else _decode_string(_to_serializable(software))

        return {
            "dataset_path": dataset_path,
            "image": image,
            "plot_label": meta_attrs.get("run_plot_label"),
            "source_description": _read_optional_scalar_text_dataset(
                handle, "meta/run_meta/source"
            ),
            "meta_readme": _read_optional_scalar_text_dataset(handle, "meta/README"),
            "software": software_text,
            "plane": _extract_plane(meta_attrs),
            "grid": _extract_grid(meta_attrs),
        }


def load_canonical_events(
    path: str | Path,
    *,
    registration: RegistrationTransform,
    filter_config: EventFilterConfig,
) -> list[EventObj]:
    raw_records = _load_raw_event_records(path)

    surviving_events: list[EventObj] = []
    allowed_species = (
        set(filter_config.species) if filter_config.species is not None else None
    )
    for record in raw_records:
        event, lambda_canonical = _canonicalize_raw_event(record, registration)
        if not _passes_authoritative_baseline_validity(event, lambda_canonical):
            continue
        if allowed_species is not None and event.species not in allowed_species:
            continue
        surviving_events.append(event)

    return surviving_events
