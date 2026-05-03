"""Online chain-health aggregation and persistent JSON sidecar helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
import operator
from pathlib import Path

CHAIN_HEALTH_FILENAME = "chain_health.json"
CHAIN_HEALTH_SCHEMA_NAME = "soe_chain_health"
CHAIN_HEALTH_SCHEMA_VERSION = 1


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _as_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _as_optional_name(value: object | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _as_name(value, field_name=field_name)


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool")
    return value


def _as_optional_bool(value: object | None, *, field_name: str) -> bool | None:
    if value is None:
        return None
    return _as_bool(value, field_name=field_name)


def _as_nonnegative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a nonnegative integer")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a nonnegative integer") from exc
    if normalized < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return int(normalized)


def _as_optional_nonnegative_int(value: object | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_nonnegative_int(value, field_name=field_name)


def _as_positive_int(value: object, *, field_name: str) -> int:
    normalized = _as_nonnegative_int(value, field_name=field_name)
    if normalized < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return normalized


def _as_optional_rate(value: object | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    rate = float(value)
    if not math.isfinite(rate) or not (0.0 <= rate <= 1.0):
        raise ValueError(f"{field_name} must be a finite rate in [0, 1]")
    return rate


def _as_positive_float(value: object, *, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite float")
    return number


def _as_grid_shape(value: object, *, field_name: str) -> tuple[int, int, int]:
    try:
        grid_shape = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must contain 3 positive integers") from exc
    if len(grid_shape) != 3:
        raise ValueError(f"{field_name} must contain 3 entries")
    normalized: list[int] = []
    for axis_index, entry in enumerate(grid_shape):
        if isinstance(entry, bool):
            raise ValueError(f"{field_name}[{axis_index}] must be an integer")
        try:
            component = operator.index(entry)
        except TypeError as exc:
            raise ValueError(f"{field_name}[{axis_index}] must be an integer") from exc
        if component <= 0:
            raise ValueError(f"{field_name}[{axis_index}] must be positive")
        normalized.append(int(component))
    return (normalized[0], normalized[1], normalized[2])


def _as_nonnegative_int_key(value: object, *, field_name: str) -> str:
    key = _as_name(value, field_name=field_name)
    try:
        normalized = _as_nonnegative_int(int(key), field_name=field_name)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a canonical nonnegative integer string") from exc
    if key != str(normalized):
        raise ValueError(f"{field_name} must be a canonical nonnegative integer string")
    return key


def _as_histogram_payload(
    value: object,
    *,
    field_name: str,
) -> dict[str, int]:
    mapping = _as_mapping(value, field_name=field_name)
    normalized: list[tuple[int, str, int]] = []
    for raw_key, raw_count in mapping.items():
        key = _as_nonnegative_int_key(raw_key, field_name=f"{field_name}.key")
        count = _as_nonnegative_int(raw_count, field_name=f"{field_name}[{key}]")
        normalized.append((int(key), key, count))
    normalized.sort(key=lambda item: item[0])
    return {key: count for _, key, count in normalized}


def _as_schema_name(value: object) -> str:
    schema_name = _as_name(value, field_name="schema_name")
    if schema_name != CHAIN_HEALTH_SCHEMA_NAME:
        raise ValueError(
            f"schema_name must be {CHAIN_HEALTH_SCHEMA_NAME!r}, got {schema_name!r}"
        )
    return schema_name


def _as_schema_version(value: object) -> int:
    schema_version = _as_nonnegative_int(value, field_name="schema_version")
    if schema_version != CHAIN_HEALTH_SCHEMA_VERSION:
        raise ValueError(
            "schema_version must be "
            f"{CHAIN_HEALTH_SCHEMA_VERSION}, got {schema_version}"
        )
    return schema_version


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / float(denominator)


def _int_histogram_payload(counts: Mapping[int, int]) -> dict[str, int]:
    return {
        str(key): int(counts[key])
        for key in sorted(counts)
    }


def _chain_health_path(path: Path) -> Path:
    candidate = Path(path)
    if candidate.name == CHAIN_HEALTH_FILENAME:
        return candidate
    return candidate / CHAIN_HEALTH_FILENAME


@dataclass(frozen=True, slots=True)
class ChainHealthMetadata:
    """Run metadata for one persisted chain-health report."""

    total_attempted_steps: int
    burn_in_steps: int
    thin_every: int
    alpha: float
    run_id: str | None = None
    artifact_dir: str | None = None
    surviving_event_count: int | None = None
    total_voxel_count: int | None = None
    grid_shape: tuple[int, int, int] | None = None
    proposal_ratio_metadata_omitted_step_count: int | None = None
    proposal_ratio_treated_as_one_step_count: int | None = None
    proposal_ratio_was_omitted_for_all_steps: bool | None = None
    proposal_ratio_treated_as_one_for_all_steps: bool | None = None

    def __post_init__(self) -> None:
        total_attempted_steps = _as_nonnegative_int(
            self.total_attempted_steps,
            field_name="metadata.total_attempted_steps",
        )
        burn_in_steps = _as_nonnegative_int(
            self.burn_in_steps,
            field_name="metadata.burn_in_steps",
        )
        thin_every = _as_positive_int(self.thin_every, field_name="metadata.thin_every")
        alpha = _as_positive_float(self.alpha, field_name="metadata.alpha")
        run_id = _as_optional_name(self.run_id, field_name="metadata.run_id")
        artifact_dir = _as_optional_name(self.artifact_dir, field_name="metadata.artifact_dir")
        surviving_event_count = _as_optional_nonnegative_int(
            self.surviving_event_count,
            field_name="metadata.surviving_event_count",
        )
        total_voxel_count = _as_optional_nonnegative_int(
            self.total_voxel_count,
            field_name="metadata.total_voxel_count",
        )
        grid_shape = None
        if self.grid_shape is not None:
            grid_shape = _as_grid_shape(self.grid_shape, field_name="metadata.grid_shape")
        proposal_ratio_metadata_omitted_step_count = _as_optional_nonnegative_int(
            self.proposal_ratio_metadata_omitted_step_count,
            field_name="metadata.proposal_ratio_metadata_omitted_step_count",
        )
        proposal_ratio_treated_as_one_step_count = _as_optional_nonnegative_int(
            self.proposal_ratio_treated_as_one_step_count,
            field_name="metadata.proposal_ratio_treated_as_one_step_count",
        )
        proposal_ratio_was_omitted_for_all_steps = _as_optional_bool(
            self.proposal_ratio_was_omitted_for_all_steps,
            field_name="metadata.proposal_ratio_was_omitted_for_all_steps",
        )
        proposal_ratio_treated_as_one_for_all_steps = _as_optional_bool(
            self.proposal_ratio_treated_as_one_for_all_steps,
            field_name="metadata.proposal_ratio_treated_as_one_for_all_steps",
        )

        if burn_in_steps > total_attempted_steps:
            raise ValueError(
                "metadata.burn_in_steps must be <= metadata.total_attempted_steps"
            )
        if total_voxel_count is not None and grid_shape is not None:
            expected_total_voxel_count = grid_shape[0] * grid_shape[1] * grid_shape[2]
            if total_voxel_count != expected_total_voxel_count:
                raise ValueError(
                    "metadata.total_voxel_count must equal the product of metadata.grid_shape"
                )
        if proposal_ratio_metadata_omitted_step_count is not None:
            if proposal_ratio_metadata_omitted_step_count > total_attempted_steps:
                raise ValueError(
                    "metadata.proposal_ratio_metadata_omitted_step_count must be <= "
                    "metadata.total_attempted_steps"
                )
        if proposal_ratio_treated_as_one_step_count is not None:
            if proposal_ratio_treated_as_one_step_count > total_attempted_steps:
                raise ValueError(
                    "metadata.proposal_ratio_treated_as_one_step_count must be <= "
                    "metadata.total_attempted_steps"
                )
        if total_attempted_steps == 0:
            if proposal_ratio_was_omitted_for_all_steps is not None:
                raise ValueError(
                    "metadata.proposal_ratio_was_omitted_for_all_steps must be null "
                    "when metadata.total_attempted_steps is 0"
                )
            if proposal_ratio_treated_as_one_for_all_steps is not None:
                raise ValueError(
                    "metadata.proposal_ratio_treated_as_one_for_all_steps must be null "
                    "when metadata.total_attempted_steps is 0"
                )

        object.__setattr__(self, "total_attempted_steps", total_attempted_steps)
        object.__setattr__(self, "burn_in_steps", burn_in_steps)
        object.__setattr__(self, "thin_every", thin_every)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "artifact_dir", artifact_dir)
        object.__setattr__(self, "surviving_event_count", surviving_event_count)
        object.__setattr__(self, "total_voxel_count", total_voxel_count)
        object.__setattr__(self, "grid_shape", grid_shape)
        object.__setattr__(
            self,
            "proposal_ratio_metadata_omitted_step_count",
            proposal_ratio_metadata_omitted_step_count,
        )
        object.__setattr__(
            self,
            "proposal_ratio_treated_as_one_step_count",
            proposal_ratio_treated_as_one_step_count,
        )
        object.__setattr__(
            self,
            "proposal_ratio_was_omitted_for_all_steps",
            proposal_ratio_was_omitted_for_all_steps,
        )
        object.__setattr__(
            self,
            "proposal_ratio_treated_as_one_for_all_steps",
            proposal_ratio_treated_as_one_for_all_steps,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "artifact_dir": self.artifact_dir,
            "total_attempted_steps": self.total_attempted_steps,
            "burn_in_steps": self.burn_in_steps,
            "thin_every": self.thin_every,
            "surviving_event_count": self.surviving_event_count,
            "total_voxel_count": self.total_voxel_count,
            "grid_shape": None if self.grid_shape is None else list(self.grid_shape),
            "alpha": self.alpha,
            "proposal_ratio_metadata_omitted_step_count": self.proposal_ratio_metadata_omitted_step_count,
            "proposal_ratio_treated_as_one_step_count": self.proposal_ratio_treated_as_one_step_count,
            "proposal_ratio_was_omitted_for_all_steps": self.proposal_ratio_was_omitted_for_all_steps,
            "proposal_ratio_treated_as_one_for_all_steps": self.proposal_ratio_treated_as_one_for_all_steps,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ChainHealthMetadata":
        metadata = _as_mapping(payload, field_name="metadata")
        return cls(
            run_id=metadata.get("run_id"),
            artifact_dir=metadata.get("artifact_dir"),
            total_attempted_steps=metadata.get("total_attempted_steps"),
            burn_in_steps=metadata.get("burn_in_steps"),
            thin_every=metadata.get("thin_every"),
            surviving_event_count=metadata.get("surviving_event_count"),
            total_voxel_count=metadata.get("total_voxel_count"),
            grid_shape=metadata.get("grid_shape"),
            alpha=metadata.get("alpha"),
            proposal_ratio_metadata_omitted_step_count=metadata.get(
                "proposal_ratio_metadata_omitted_step_count"
            ),
            proposal_ratio_treated_as_one_step_count=metadata.get(
                "proposal_ratio_treated_as_one_step_count"
            ),
            proposal_ratio_was_omitted_for_all_steps=metadata.get(
                "proposal_ratio_was_omitted_for_all_steps"
            ),
            proposal_ratio_treated_as_one_for_all_steps=metadata.get(
                "proposal_ratio_treated_as_one_for_all_steps"
            ),
        )


@dataclass(frozen=True, slots=True)
class ChainHealthRates:
    """Aggregate proposal and acceptance fractions."""

    overall_acceptance_rate: float | None
    same_voxel_proposal_rate: float | None
    cross_voxel_proposal_rate: float | None
    same_voxel_acceptance_rate: float | None
    cross_voxel_acceptance_rate: float | None
    fraction_new_count_eq_0: float | None
    fraction_old_count_eq_1: float | None
    fraction_cross_voxel_old1_new0: float | None
    fraction_accepted_cross_voxel_old1_new0: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "overall_acceptance_rate",
            _as_optional_rate(
                self.overall_acceptance_rate,
                field_name="rates.overall_acceptance_rate",
            ),
        )
        object.__setattr__(
            self,
            "same_voxel_proposal_rate",
            _as_optional_rate(
                self.same_voxel_proposal_rate,
                field_name="rates.same_voxel_proposal_rate",
            ),
        )
        object.__setattr__(
            self,
            "cross_voxel_proposal_rate",
            _as_optional_rate(
                self.cross_voxel_proposal_rate,
                field_name="rates.cross_voxel_proposal_rate",
            ),
        )
        object.__setattr__(
            self,
            "same_voxel_acceptance_rate",
            _as_optional_rate(
                self.same_voxel_acceptance_rate,
                field_name="rates.same_voxel_acceptance_rate",
            ),
        )
        object.__setattr__(
            self,
            "cross_voxel_acceptance_rate",
            _as_optional_rate(
                self.cross_voxel_acceptance_rate,
                field_name="rates.cross_voxel_acceptance_rate",
            ),
        )
        object.__setattr__(
            self,
            "fraction_new_count_eq_0",
            _as_optional_rate(
                self.fraction_new_count_eq_0,
                field_name="rates.fraction_new_count_eq_0",
            ),
        )
        object.__setattr__(
            self,
            "fraction_old_count_eq_1",
            _as_optional_rate(
                self.fraction_old_count_eq_1,
                field_name="rates.fraction_old_count_eq_1",
            ),
        )
        object.__setattr__(
            self,
            "fraction_cross_voxel_old1_new0",
            _as_optional_rate(
                self.fraction_cross_voxel_old1_new0,
                field_name="rates.fraction_cross_voxel_old1_new0",
            ),
        )
        object.__setattr__(
            self,
            "fraction_accepted_cross_voxel_old1_new0",
            _as_optional_rate(
                self.fraction_accepted_cross_voxel_old1_new0,
                field_name="rates.fraction_accepted_cross_voxel_old1_new0",
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "overall_acceptance_rate": self.overall_acceptance_rate,
            "same_voxel_proposal_rate": self.same_voxel_proposal_rate,
            "cross_voxel_proposal_rate": self.cross_voxel_proposal_rate,
            "same_voxel_acceptance_rate": self.same_voxel_acceptance_rate,
            "cross_voxel_acceptance_rate": self.cross_voxel_acceptance_rate,
            "fraction_new_count_eq_0": self.fraction_new_count_eq_0,
            "fraction_old_count_eq_1": self.fraction_old_count_eq_1,
            "fraction_cross_voxel_old1_new0": self.fraction_cross_voxel_old1_new0,
            "fraction_accepted_cross_voxel_old1_new0": self.fraction_accepted_cross_voxel_old1_new0,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ChainHealthRates":
        rates = _as_mapping(payload, field_name="rates")
        return cls(
            overall_acceptance_rate=rates.get("overall_acceptance_rate"),
            same_voxel_proposal_rate=rates.get("same_voxel_proposal_rate"),
            cross_voxel_proposal_rate=rates.get("cross_voxel_proposal_rate"),
            same_voxel_acceptance_rate=rates.get("same_voxel_acceptance_rate"),
            cross_voxel_acceptance_rate=rates.get("cross_voxel_acceptance_rate"),
            fraction_new_count_eq_0=rates.get("fraction_new_count_eq_0"),
            fraction_old_count_eq_1=rates.get("fraction_old_count_eq_1"),
            fraction_cross_voxel_old1_new0=rates.get("fraction_cross_voxel_old1_new0"),
            fraction_accepted_cross_voxel_old1_new0=rates.get(
                "fraction_accepted_cross_voxel_old1_new0"
            ),
        )


@dataclass(frozen=True, slots=True)
class ChainHealthHistograms:
    """Sparse histograms keyed by canonical nonnegative integer strings."""

    old_count: dict[str, int]
    new_count: dict[str, int]
    cross_voxel_old_count: dict[str, int]
    cross_voxel_new_count: dict[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "old_count",
            _as_histogram_payload(self.old_count, field_name="histograms.old_count"),
        )
        object.__setattr__(
            self,
            "new_count",
            _as_histogram_payload(self.new_count, field_name="histograms.new_count"),
        )
        object.__setattr__(
            self,
            "cross_voxel_old_count",
            _as_histogram_payload(
                self.cross_voxel_old_count,
                field_name="histograms.cross_voxel_old_count",
            ),
        )
        object.__setattr__(
            self,
            "cross_voxel_new_count",
            _as_histogram_payload(
                self.cross_voxel_new_count,
                field_name="histograms.cross_voxel_new_count",
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "old_count": self.old_count,
            "new_count": self.new_count,
            "cross_voxel_old_count": self.cross_voxel_old_count,
            "cross_voxel_new_count": self.cross_voxel_new_count,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ChainHealthHistograms":
        histograms = _as_mapping(payload, field_name="histograms")
        return cls(
            old_count=histograms.get("old_count", {}),
            new_count=histograms.get("new_count", {}),
            cross_voxel_old_count=histograms.get("cross_voxel_old_count", {}),
            cross_voxel_new_count=histograms.get("cross_voxel_new_count", {}),
        )


@dataclass(frozen=True, slots=True)
class ChainHealthPairStats:
    """Aggregate counts for one `(old_count, new_count)` pair."""

    proposal_count: int
    accepted_count: int
    rejected_count: int
    empirical_acceptance_rate: float | None

    def __post_init__(self) -> None:
        proposal_count = _as_nonnegative_int(
            self.proposal_count,
            field_name="pair_stats.proposal_count",
        )
        accepted_count = _as_nonnegative_int(
            self.accepted_count,
            field_name="pair_stats.accepted_count",
        )
        rejected_count = _as_nonnegative_int(
            self.rejected_count,
            field_name="pair_stats.rejected_count",
        )
        empirical_acceptance_rate = _as_optional_rate(
            self.empirical_acceptance_rate,
            field_name="pair_stats.empirical_acceptance_rate",
        )
        if accepted_count > proposal_count:
            raise ValueError("pair_stats.accepted_count must be <= pair_stats.proposal_count")
        if rejected_count != proposal_count - accepted_count:
            raise ValueError(
                "pair_stats.rejected_count must equal "
                "pair_stats.proposal_count - pair_stats.accepted_count"
            )
        if proposal_count == 0:
            if empirical_acceptance_rate is not None:
                raise ValueError(
                    "pair_stats.empirical_acceptance_rate must be null when "
                    "pair_stats.proposal_count is 0"
                )
        else:
            derived_rate = accepted_count / float(proposal_count)
            if empirical_acceptance_rate is None or not math.isclose(
                empirical_acceptance_rate,
                derived_rate,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "pair_stats.empirical_acceptance_rate must equal "
                    "pair_stats.accepted_count / pair_stats.proposal_count"
                )
        object.__setattr__(self, "proposal_count", proposal_count)
        object.__setattr__(self, "accepted_count", accepted_count)
        object.__setattr__(self, "rejected_count", rejected_count)
        object.__setattr__(
            self,
            "empirical_acceptance_rate",
            empirical_acceptance_rate,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "proposal_count": self.proposal_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "empirical_acceptance_rate": self.empirical_acceptance_rate,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ChainHealthPairStats":
        stats = _as_mapping(payload, field_name="pair_stats")
        return cls(
            proposal_count=stats.get("proposal_count"),
            accepted_count=stats.get("accepted_count"),
            rejected_count=stats.get("rejected_count"),
            empirical_acceptance_rate=stats.get("empirical_acceptance_rate"),
        )


def _pair_table_payload(
    pair_table: Mapping[str, Mapping[str, ChainHealthPairStats]],
) -> dict[str, dict[str, object]]:
    normalized_old: list[tuple[int, str, dict[str, object]]] = []
    for raw_old_key, new_mapping in pair_table.items():
        old_key = _as_nonnegative_int_key(raw_old_key, field_name="pair_table.old_count")
        nested_mapping = _as_mapping(new_mapping, field_name=f"pair_table[{old_key}]")
        normalized_new: list[tuple[int, str, dict[str, object]]] = []
        for raw_new_key, stats in nested_mapping.items():
            new_key = _as_nonnegative_int_key(
                raw_new_key,
                field_name=f"pair_table[{old_key}].new_count",
            )
            if not isinstance(stats, ChainHealthPairStats):
                raise TypeError(
                    f"pair_table[{old_key}][{new_key}] must be a ChainHealthPairStats"
                )
            normalized_new.append((int(new_key), new_key, stats.to_payload()))
        normalized_new.sort(key=lambda item: item[0])
        normalized_old.append(
            (
                int(old_key),
                old_key,
                {new_key: payload for _, new_key, payload in normalized_new},
            )
        )
    normalized_old.sort(key=lambda item: item[0])
    return {old_key: payload for _, old_key, payload in normalized_old}


def _pair_table_from_payload(value: object) -> dict[str, dict[str, ChainHealthPairStats]]:
    mapping = _as_mapping(value, field_name="pair_table")
    normalized_old: list[tuple[int, str, dict[str, ChainHealthPairStats]]] = []
    for raw_old_key, raw_new_mapping in mapping.items():
        old_key = _as_nonnegative_int_key(raw_old_key, field_name="pair_table.old_count")
        new_mapping = _as_mapping(raw_new_mapping, field_name=f"pair_table[{old_key}]")
        normalized_new: list[tuple[int, str, ChainHealthPairStats]] = []
        for raw_new_key, raw_stats in new_mapping.items():
            new_key = _as_nonnegative_int_key(
                raw_new_key,
                field_name=f"pair_table[{old_key}].new_count",
            )
            normalized_new.append((int(new_key), new_key, ChainHealthPairStats.from_payload(raw_stats)))
        normalized_new.sort(key=lambda item: item[0])
        normalized_old.append(
            (
                int(old_key),
                old_key,
                {new_key: stats for _, new_key, stats in normalized_new},
            )
        )
    normalized_old.sort(key=lambda item: item[0])
    return {old_key: payload for _, old_key, payload in normalized_old}


@dataclass(frozen=True, slots=True)
class ChainHealthOccupancySupport:
    """Lightweight support summaries copied at run end."""

    terminal_nonzero_voxel_count: int | None
    retained_mean_nonzero_voxel_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terminal_nonzero_voxel_count",
            _as_optional_nonnegative_int(
                self.terminal_nonzero_voxel_count,
                field_name="occupancy_support.terminal_nonzero_voxel_count",
            ),
        )
        object.__setattr__(
            self,
            "retained_mean_nonzero_voxel_count",
            _as_optional_nonnegative_int(
                self.retained_mean_nonzero_voxel_count,
                field_name="occupancy_support.retained_mean_nonzero_voxel_count",
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "terminal_nonzero_voxel_count": self.terminal_nonzero_voxel_count,
            "retained_mean_nonzero_voxel_count": self.retained_mean_nonzero_voxel_count,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ChainHealthOccupancySupport":
        support = _as_mapping(payload, field_name="occupancy_support")
        return cls(
            terminal_nonzero_voxel_count=support.get("terminal_nonzero_voxel_count"),
            retained_mean_nonzero_voxel_count=support.get("retained_mean_nonzero_voxel_count"),
        )


@dataclass(frozen=True, slots=True)
class ChainHealthReport:
    """Validated persisted chain-health payload."""

    metadata: ChainHealthMetadata
    rates: ChainHealthRates
    histograms: ChainHealthHistograms
    pair_table: dict[str, dict[str, ChainHealthPairStats]]
    occupancy_support: ChainHealthOccupancySupport
    schema_name: str = CHAIN_HEALTH_SCHEMA_NAME
    schema_version: int = CHAIN_HEALTH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_name", _as_schema_name(self.schema_name))
        object.__setattr__(self, "schema_version", _as_schema_version(self.schema_version))
        if not isinstance(self.metadata, ChainHealthMetadata):
            raise TypeError("metadata must be a ChainHealthMetadata")
        if not isinstance(self.rates, ChainHealthRates):
            raise TypeError("rates must be a ChainHealthRates")
        if not isinstance(self.histograms, ChainHealthHistograms):
            raise TypeError("histograms must be a ChainHealthHistograms")
        object.__setattr__(self, "pair_table", _pair_table_from_payload(_pair_table_payload(self.pair_table)))
        if not isinstance(self.occupancy_support, ChainHealthOccupancySupport):
            raise TypeError("occupancy_support must be a ChainHealthOccupancySupport")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "metadata": self.metadata.to_payload(),
            "rates": self.rates.to_payload(),
            "histograms": self.histograms.to_payload(),
            "pair_table": _pair_table_payload(self.pair_table),
            "occupancy_support": self.occupancy_support.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "ChainHealthReport":
        mapping = _as_mapping(payload, field_name="chain_health")
        return cls(
            schema_name=mapping.get("schema_name"),
            schema_version=mapping.get("schema_version"),
            metadata=ChainHealthMetadata.from_payload(mapping.get("metadata")),
            rates=ChainHealthRates.from_payload(mapping.get("rates")),
            histograms=ChainHealthHistograms.from_payload(mapping.get("histograms")),
            pair_table=_pair_table_from_payload(mapping.get("pair_table", {})),
            occupancy_support=ChainHealthOccupancySupport.from_payload(
                mapping.get("occupancy_support", {})
            ),
        )


@dataclass(slots=True)
class _PairAccumulator:
    proposal_count: int = 0
    accepted_count: int = 0

    def observe(self, *, accepted: bool) -> None:
        self.proposal_count += 1
        if accepted:
            self.accepted_count += 1

    def to_stats(self) -> ChainHealthPairStats:
        rejected_count = self.proposal_count - self.accepted_count
        return ChainHealthPairStats(
            proposal_count=self.proposal_count,
            accepted_count=self.accepted_count,
            rejected_count=rejected_count,
            empirical_acceptance_rate=_safe_rate(self.accepted_count, self.proposal_count),
        )


@dataclass(slots=True)
class ChainHealthCollector:
    """Online aggregate collector fed from the live MH step path."""

    total_attempted_steps: int = 0
    accepted_step_count: int = 0
    same_voxel_proposal_count: int = 0
    same_voxel_accepted_count: int = 0
    cross_voxel_proposal_count: int = 0
    cross_voxel_accepted_count: int = 0
    new_count_zero_count: int = 0
    old_count_one_count: int = 0
    cross_voxel_old1_new0_count: int = 0
    accepted_cross_voxel_old1_new0_count: int = 0
    proposal_ratio_metadata_omitted_step_count: int = 0
    proposal_ratio_treated_as_one_step_count: int = 0
    _old_count_histogram: dict[int, int] = field(default_factory=dict)
    _new_count_histogram: dict[int, int] = field(default_factory=dict)
    _cross_voxel_old_count_histogram: dict[int, int] = field(default_factory=dict)
    _cross_voxel_new_count_histogram: dict[int, int] = field(default_factory=dict)
    _pair_table: dict[tuple[int, int], _PairAccumulator] = field(default_factory=dict)

    def record_step(
        self,
        *,
        same_voxel: object,
        old_count: object,
        new_count: object,
        accepted: object,
        proposal_ratio_was_omitted: object = False,
    ) -> None:
        same_voxel_bool = _as_bool(same_voxel, field_name="same_voxel")
        accepted_bool = _as_bool(accepted, field_name="accepted")
        proposal_ratio_was_omitted_bool = _as_bool(
            proposal_ratio_was_omitted,
            field_name="proposal_ratio_was_omitted",
        )
        old_count_int = _as_nonnegative_int(old_count, field_name="old_count")
        new_count_int = _as_nonnegative_int(new_count, field_name="new_count")

        if same_voxel_bool and old_count_int != new_count_int:
            raise ValueError("same-voxel proposals require old_count == new_count")
        if not same_voxel_bool and old_count_int < 1:
            raise ValueError("cross-voxel proposals require old_count >= 1")

        self.total_attempted_steps += 1
        if accepted_bool:
            self.accepted_step_count += 1
        if same_voxel_bool:
            self.same_voxel_proposal_count += 1
            if accepted_bool:
                self.same_voxel_accepted_count += 1
        else:
            self.cross_voxel_proposal_count += 1
            if accepted_bool:
                self.cross_voxel_accepted_count += 1
        if old_count_int == 1:
            self.old_count_one_count += 1
        if new_count_int == 0:
            self.new_count_zero_count += 1
        if not same_voxel_bool and old_count_int == 1 and new_count_int == 0:
            self.cross_voxel_old1_new0_count += 1
            if accepted_bool:
                self.accepted_cross_voxel_old1_new0_count += 1
        if proposal_ratio_was_omitted_bool:
            self.proposal_ratio_metadata_omitted_step_count += 1
            self.proposal_ratio_treated_as_one_step_count += 1

        self._old_count_histogram[old_count_int] = self._old_count_histogram.get(old_count_int, 0) + 1
        self._new_count_histogram[new_count_int] = self._new_count_histogram.get(new_count_int, 0) + 1
        if not same_voxel_bool:
            self._cross_voxel_old_count_histogram[old_count_int] = (
                self._cross_voxel_old_count_histogram.get(old_count_int, 0) + 1
            )
            self._cross_voxel_new_count_histogram[new_count_int] = (
                self._cross_voxel_new_count_histogram.get(new_count_int, 0) + 1
            )

        key = (old_count_int, new_count_int)
        pair_accumulator = self._pair_table.get(key)
        if pair_accumulator is None:
            pair_accumulator = _PairAccumulator()
            self._pair_table[key] = pair_accumulator
        pair_accumulator.observe(accepted=accepted_bool)

    def build_report(
        self,
        *,
        burn_in_steps: object,
        thin_every: object,
        alpha: object,
        run_id: object | None = None,
        artifact_dir: Path | str | None = None,
        surviving_event_count: object | None = None,
        total_voxel_count: object | None = None,
        grid_shape: object | None = None,
        terminal_nonzero_voxel_count: object | None = None,
        retained_mean_nonzero_voxel_count: object | None = None,
        total_attempted_steps: object | None = None,
    ) -> ChainHealthReport:
        total_attempted_steps_value = self.total_attempted_steps
        if total_attempted_steps is not None:
            expected_total_attempted_steps = _as_nonnegative_int(
                total_attempted_steps,
                field_name="metadata.total_attempted_steps",
            )
            if expected_total_attempted_steps != total_attempted_steps_value:
                raise ValueError(
                    "metadata.total_attempted_steps must match the collector-observed "
                    "number of attempted steps"
                )
        artifact_dir_str: str | None
        if artifact_dir is None:
            artifact_dir_str = None
        else:
            artifact_dir_str = str(Path(artifact_dir).resolve())

        proposal_ratio_was_omitted_for_all_steps = (
            None
            if total_attempted_steps_value == 0
            else self.proposal_ratio_metadata_omitted_step_count == total_attempted_steps_value
        )
        proposal_ratio_treated_as_one_for_all_steps = (
            None
            if total_attempted_steps_value == 0
            else self.proposal_ratio_treated_as_one_step_count == total_attempted_steps_value
        )

        rates = ChainHealthRates(
            overall_acceptance_rate=_safe_rate(
                self.accepted_step_count,
                total_attempted_steps_value,
            ),
            same_voxel_proposal_rate=_safe_rate(
                self.same_voxel_proposal_count,
                total_attempted_steps_value,
            ),
            cross_voxel_proposal_rate=_safe_rate(
                self.cross_voxel_proposal_count,
                total_attempted_steps_value,
            ),
            same_voxel_acceptance_rate=_safe_rate(
                self.same_voxel_accepted_count,
                self.same_voxel_proposal_count,
            ),
            cross_voxel_acceptance_rate=_safe_rate(
                self.cross_voxel_accepted_count,
                self.cross_voxel_proposal_count,
            ),
            fraction_new_count_eq_0=_safe_rate(
                self.new_count_zero_count,
                total_attempted_steps_value,
            ),
            fraction_old_count_eq_1=_safe_rate(
                self.old_count_one_count,
                total_attempted_steps_value,
            ),
            fraction_cross_voxel_old1_new0=_safe_rate(
                self.cross_voxel_old1_new0_count,
                self.cross_voxel_proposal_count,
            ),
            fraction_accepted_cross_voxel_old1_new0=_safe_rate(
                self.accepted_cross_voxel_old1_new0_count,
                self.cross_voxel_accepted_count,
            ),
        )
        histograms = ChainHealthHistograms(
            old_count=_int_histogram_payload(self._old_count_histogram),
            new_count=_int_histogram_payload(self._new_count_histogram),
            cross_voxel_old_count=_int_histogram_payload(self._cross_voxel_old_count_histogram),
            cross_voxel_new_count=_int_histogram_payload(self._cross_voxel_new_count_histogram),
        )

        nested_pair_table: dict[str, dict[str, ChainHealthPairStats]] = {}
        for (old_count_key, new_count_key), stats in sorted(self._pair_table.items()):
            old_key = str(old_count_key)
            new_key = str(new_count_key)
            if old_key not in nested_pair_table:
                nested_pair_table[old_key] = {}
            nested_pair_table[old_key][new_key] = stats.to_stats()

        return ChainHealthReport(
            metadata=ChainHealthMetadata(
                run_id=None if run_id is None else str(run_id),
                artifact_dir=artifact_dir_str,
                total_attempted_steps=total_attempted_steps_value,
                burn_in_steps=burn_in_steps,
                thin_every=thin_every,
                surviving_event_count=surviving_event_count,
                total_voxel_count=total_voxel_count,
                grid_shape=grid_shape,
                alpha=alpha,
                proposal_ratio_metadata_omitted_step_count=self.proposal_ratio_metadata_omitted_step_count,
                proposal_ratio_treated_as_one_step_count=self.proposal_ratio_treated_as_one_step_count,
                proposal_ratio_was_omitted_for_all_steps=proposal_ratio_was_omitted_for_all_steps,
                proposal_ratio_treated_as_one_for_all_steps=proposal_ratio_treated_as_one_for_all_steps,
            ),
            rates=rates,
            histograms=histograms,
            pair_table=nested_pair_table,
            occupancy_support=ChainHealthOccupancySupport(
                terminal_nonzero_voxel_count=terminal_nonzero_voxel_count,
                retained_mean_nonzero_voxel_count=retained_mean_nonzero_voxel_count,
            ),
        )


def validate_chain_health_payload(payload: object) -> ChainHealthReport:
    """Validate one in-memory chain-health payload."""
    return ChainHealthReport.from_payload(payload)


def write_chain_health_sidecar(
    artifacts_dir: Path,
    chain_health: ChainHealthReport,
) -> ChainHealthReport:
    """Write the chain-health JSON next to other reconstruction artifacts."""
    if not isinstance(chain_health, ChainHealthReport):
        raise TypeError("chain_health must be a ChainHealthReport")
    chain_health_path = _chain_health_path(Path(artifacts_dir).resolve())
    chain_health_path.parent.mkdir(parents=True, exist_ok=True)
    chain_health_path.write_text(
        json.dumps(chain_health.to_payload(), indent=2) + "\n",
        encoding="utf-8",
    )
    return chain_health


def load_chain_health(path: Path) -> ChainHealthReport:
    """Load and validate one chain-health sidecar."""
    chain_health_path = _chain_health_path(Path(path).resolve())
    if not chain_health_path.is_file():
        raise FileNotFoundError(f"Missing chain-health sidecar: {chain_health_path}")
    payload = json.loads(chain_health_path.read_text(encoding="utf-8"))
    return validate_chain_health_payload(payload)


__all__ = [
    "CHAIN_HEALTH_FILENAME",
    "CHAIN_HEALTH_SCHEMA_NAME",
    "CHAIN_HEALTH_SCHEMA_VERSION",
    "ChainHealthCollector",
    "ChainHealthHistograms",
    "ChainHealthMetadata",
    "ChainHealthOccupancySupport",
    "ChainHealthPairStats",
    "ChainHealthRates",
    "ChainHealthReport",
    "load_chain_health",
    "validate_chain_health_payload",
    "write_chain_health_sidecar",
]
