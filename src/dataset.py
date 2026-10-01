from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_INT32_MIN = np.iinfo(np.int32).min


@dataclass(slots=True)
class SequenceRecord:
    sizes: np.ndarray
    iats: np.ndarray
    environment: str
    category: str
    group: str
    flow_id: str
    flow_group: str | None = None
    start_offset: float = 0.0
    flow_start_timestamp: float = 0.0


@dataclass(slots=True)
class SequenceDataset:
    """Fixed-width packet sequences with factorized capture metadata."""

    sizes: np.ndarray
    iats: np.ndarray
    packet_counts: np.ndarray
    labels: np.ndarray
    environment_ids: np.ndarray
    group_ids: np.ndarray
    flow_ids: np.ndarray
    flow_groups: np.ndarray
    start_offsets: np.ndarray
    flow_start_timestamps: np.ndarray
    class_names: np.ndarray
    environment_names: np.ndarray
    group_names: np.ndarray
    fingerprint: str
    excluded_captures: tuple[dict[str, str], ...] = ()

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def subset(self, indices: np.ndarray) -> "SequenceDataset":
        return SequenceDataset(
            sizes=self.sizes[indices],
            iats=self.iats[indices],
            packet_counts=self.packet_counts[indices],
            labels=self.labels[indices],
            environment_ids=self.environment_ids[indices],
            group_ids=self.group_ids[indices],
            flow_ids=self.flow_ids[indices],
            flow_groups=self.flow_groups[indices],
            start_offsets=self.start_offsets[indices],
            flow_start_timestamps=self.flow_start_timestamps[indices],
            class_names=self.class_names,
            environment_names=self.environment_names,
            group_names=self.group_names,
            fingerprint=self.fingerprint,
            excluded_captures=self.excluded_captures,
        )

    def environment_id(self, name: str) -> int:
        matches = np.flatnonzero(self.environment_names == name)
        if matches.size == 0:
            raise ValueError(f"No extracted samples found for environment '{name}'")
        return int(matches[0])

    def validate(self) -> None:
        """Reject malformed datasets before their contents reach the study."""
        numeric_dtypes = {
            "sizes": np.dtype(np.int32),
            "iats": np.dtype(np.float32),
            "packet_counts": np.dtype(np.int32),
            "labels": np.dtype(np.int16),
            "environment_ids": np.dtype(np.int16),
            "group_ids": np.dtype(np.int32),
            "start_offsets": np.dtype(np.float64),
            "flow_start_timestamps": np.dtype(np.float64),
        }
        for name, expected_dtype in numeric_dtypes.items():
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.dtype != expected_dtype:
                raise ValueError(
                    f"Dataset field '{name}' must be a NumPy array with dtype {expected_dtype}"
                )

        string_fields = ("flow_ids", "flow_groups", "class_names", "environment_names", "group_names")
        for name in string_fields:
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.dtype.kind != "U":
                raise ValueError(f"Dataset field '{name}' must be a NumPy Unicode array")
            if value.ndim != 1:
                raise ValueError(f"Dataset field '{name}' must be one-dimensional")
            if np.any(value == ""):
                raise ValueError(f"Dataset field '{name}' contains an empty string")

        if self.sizes.ndim != 2 or self.iats.ndim != 2:
            raise ValueError("Dataset size and IAT matrices must be two-dimensional")
        if self.sizes.shape != self.iats.shape:
            raise ValueError("Dataset size and IAT matrices must have identical shapes")
        sequence_count, width = self.sizes.shape
        if sequence_count == 0 or width == 0:
            raise ValueError("The dataset must not be empty")

        row_fields = (
            "packet_counts",
            "labels",
            "environment_ids",
            "group_ids",
            "flow_ids",
            "flow_groups",
            "start_offsets",
            "flow_start_timestamps",
        )
        for name in row_fields:
            value = getattr(self, name)
            if value.ndim != 1 or value.shape[0] != sequence_count:
                raise ValueError(
                    f"Dataset field '{name}' must contain one value per sequence"
                )

        lookup_fields = ("class_names", "environment_names", "group_names")
        for name in lookup_fields:
            value = getattr(self, name)
            if value.size == 0:
                raise ValueError(f"Dataset lookup '{name}' must not be empty")
            if np.unique(value).size != value.size:
                raise ValueError(f"Dataset lookup '{name}' contains duplicate names")

        if not isinstance(self.fingerprint, str) or not self.fingerprint:
            raise ValueError(
                "The dataset fingerprint must be a non-empty string"
            )
        if np.any(self.packet_counts < 1) or np.any(self.packet_counts > width):
            raise ValueError("Dataset packet counts fall outside the sequence width")
        if np.any(self.labels < 0) or np.any(self.labels >= self.class_names.size):
            raise ValueError("Dataset labels reference an unknown class")
        if np.any(self.environment_ids < 0) or np.any(
            self.environment_ids >= self.environment_names.size
        ):
            raise ValueError("Dataset environment IDs reference an unknown environment")
        if np.any(self.group_ids < 0) or np.any(
            self.group_ids >= self.group_names.size
        ):
            raise ValueError("Dataset group IDs reference an unknown capture")
        if not np.all(np.isfinite(self.iats)) or np.any(self.iats < 0.0):
            raise ValueError("Dataset IAT values must be finite and non-negative")
        if not np.all(np.isfinite(self.start_offsets)) or np.any(self.start_offsets < 0.0):
            raise ValueError("Flow start offsets must be finite and non-negative")
        if not np.all(np.isfinite(self.flow_start_timestamps)) or np.any(self.flow_start_timestamps < 0.0):
            raise ValueError("Flow start timestamps must be finite and non-negative")
        if np.any(self.iats[:, 0] != 0.0):
            raise ValueError("Every sequence must begin with a zero IAT")

        padding = np.arange(width)[None, :] >= self.packet_counts[:, None]
        if np.any(self.sizes[padding] != 0) or np.any(self.iats[padding] != 0.0):
            raise ValueError("Dataset padding positions must contain zeros")
        if np.any(self.sizes[~padding] == 0):
            raise ValueError("Dataset packet positions must contain non-zero frame sizes")
        if np.any(self.sizes == _INT32_MIN):
            raise ValueError("Frame sizes cannot use the signed int32 minimum")


def from_records(
    records: list[SequenceRecord],
    class_names: list[str],
    environment_names: list[str],
    width: int,
    fingerprint: str,
    excluded_captures: tuple[dict[str, str], ...] = (),
) -> SequenceDataset:
    if not records:
        raise ValueError("No eligible packet sequences were extracted")
    size_matrix = np.zeros((len(records), width), dtype=np.int32)
    iat_matrix = np.zeros((len(records), width), dtype=np.float32)
    counts = np.empty(len(records), dtype=np.int32)
    for row, record in enumerate(records):
        count = min(record.sizes.size, width)
        size_matrix[row, :count] = record.sizes[:count]
        iat_matrix[row, :count] = record.iats[:count]
        counts[row] = count

    class_lookup = {name: index for index, name in enumerate(class_names)}
    environment_lookup = {name: index for index, name in enumerate(environment_names)}
    group_names, group_ids = np.unique(
        np.asarray([record.group for record in records], dtype=str),
        return_inverse=True,
    )
    dataset = SequenceDataset(
        sizes=size_matrix,
        iats=iat_matrix,
        packet_counts=counts,
        labels=np.asarray(
            [class_lookup[record.category] for record in records], dtype=np.int16
        ),
        environment_ids=np.asarray(
            [environment_lookup[record.environment] for record in records],
            dtype=np.int16,
        ),
        group_ids=group_ids.astype(np.int32, copy=False),
        flow_ids=np.asarray([record.flow_id for record in records], dtype=str),
        flow_groups=np.asarray([record.flow_group or record.flow_id for record in records], dtype=str),
        start_offsets=np.asarray([record.start_offset for record in records], dtype=np.float64),
        flow_start_timestamps=np.asarray([record.flow_start_timestamp for record in records], dtype=np.float64),
        class_names=np.asarray(class_names, dtype=str),
        environment_names=np.asarray(environment_names, dtype=str),
        group_names=group_names,
        fingerprint=fingerprint,
        excluded_captures=excluded_captures,
    )
    dataset.validate()
    return dataset
