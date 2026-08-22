from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


CACHE_SCHEMA_VERSION = 2
_INT32_MIN = np.iinfo(np.int32).min

_CACHE_ARRAY_FIELDS = (
    "sizes",
    "iats",
    "packet_counts",
    "labels",
    "environment_ids",
    "group_ids",
    "flow_ids",
    "class_names",
    "environment_names",
    "group_names",
)


@dataclass(slots=True)
class SequenceRecord:
    sizes: np.ndarray
    iats: np.ndarray
    environment: str
    category: str
    group: str
    flow_id: str


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
    class_names: np.ndarray
    environment_names: np.ndarray
    group_names: np.ndarray
    fingerprint: str

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
            class_names=self.class_names,
            environment_names=self.environment_names,
            group_names=self.group_names,
            fingerprint=self.fingerprint,
        )

    def environment_id(self, name: str) -> int:
        matches = np.flatnonzero(self.environment_names == name)
        if matches.size == 0:
            raise ValueError(f"No extracted samples found for environment '{name}'")
        return int(matches[0])

    def validate(self) -> None:
        """Reject malformed caches before their contents reach the experiment."""
        numeric_dtypes = {
            "sizes": np.dtype(np.int32),
            "iats": np.dtype(np.float32),
            "packet_counts": np.dtype(np.int32),
            "labels": np.dtype(np.int16),
            "environment_ids": np.dtype(np.int16),
            "group_ids": np.dtype(np.int32),
        }
        for name, expected_dtype in numeric_dtypes.items():
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.dtype != expected_dtype:
                raise ValueError(
                    f"Cache field '{name}' must be a NumPy array with dtype {expected_dtype}"
                )

        string_fields = ("flow_ids", "class_names", "environment_names", "group_names")
        for name in string_fields:
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.dtype.kind != "U":
                raise ValueError(f"Cache field '{name}' must be a NumPy Unicode array")
            if value.ndim != 1:
                raise ValueError(f"Cache field '{name}' must be one-dimensional")
            if np.any(value == ""):
                raise ValueError(f"Cache field '{name}' contains an empty string")

        if self.sizes.ndim != 2 or self.iats.ndim != 2:
            raise ValueError("Cache size and IAT matrices must be two-dimensional")
        if self.sizes.shape != self.iats.shape:
            raise ValueError("Cache size and IAT matrices must have identical shapes")
        sequence_count, width = self.sizes.shape
        if sequence_count == 0 or width == 0:
            raise ValueError("The sequence cache must not be empty")

        row_fields = (
            "packet_counts",
            "labels",
            "environment_ids",
            "group_ids",
            "flow_ids",
        )
        for name in row_fields:
            value = getattr(self, name)
            if value.ndim != 1 or value.shape[0] != sequence_count:
                raise ValueError(
                    f"Cache field '{name}' must contain one value per sequence"
                )

        lookup_fields = ("class_names", "environment_names", "group_names")
        for name in lookup_fields:
            value = getattr(self, name)
            if value.size == 0:
                raise ValueError(f"Cache lookup '{name}' must not be empty")
            if np.unique(value).size != value.size:
                raise ValueError(f"Cache lookup '{name}' contains duplicate names")

        if not isinstance(self.fingerprint, str) or not self.fingerprint:
            raise ValueError(
                "The sequence cache fingerprint must be a non-empty string"
            )
        if np.any(self.packet_counts < 1) or np.any(self.packet_counts > width):
            raise ValueError("Cache packet counts fall outside the sequence width")
        if np.any(self.labels < 0) or np.any(self.labels >= self.class_names.size):
            raise ValueError("Cache labels reference an unknown class")
        if np.any(self.environment_ids < 0) or np.any(
            self.environment_ids >= self.environment_names.size
        ):
            raise ValueError("Cache environment IDs reference an unknown environment")
        if np.any(self.group_ids < 0) or np.any(
            self.group_ids >= self.group_names.size
        ):
            raise ValueError("Cache group IDs reference an unknown capture")
        if not np.all(np.isfinite(self.iats)) or np.any(self.iats < 0.0):
            raise ValueError("Cache IAT values must be finite and non-negative")
        if np.any(self.iats[:, 0] != 0.0):
            raise ValueError("Every cached sequence must begin with a zero IAT")

        padding = np.arange(width)[None, :] >= self.packet_counts[:, None]
        if np.any(self.sizes[padding] != 0) or np.any(self.iats[padding] != 0.0):
            raise ValueError("Cache padding positions must contain zeros")
        if np.any(self.sizes[~padding] == 0):
            raise ValueError("Cache packet positions must contain non-zero frame sizes")
        if np.any(self.sizes == _INT32_MIN):
            raise ValueError("Cache frame sizes cannot use the signed int32 minimum")

    def save(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{target.stem}-",
            suffix=".npz",
            dir=target.parent,
            delete=False,
        )
        temporary = Path(handle.name)
        handle.close()
        try:
            np.savez_compressed(
                temporary,
                schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
                fingerprint=np.asarray(self.fingerprint),
                sizes=self.sizes,
                iats=self.iats,
                packet_counts=self.packet_counts,
                labels=self.labels,
                environment_ids=self.environment_ids,
                group_ids=self.group_ids,
                flow_ids=self.flow_ids,
                class_names=self.class_names,
                environment_names=self.environment_names,
                group_names=self.group_names,
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load_if_current(
        cls, path: str | Path, fingerprint: str
    ) -> "SequenceDataset | None":
        try:
            return cls.load(path, expected_fingerprint=fingerprint)
        except (FileNotFoundError, ValueError):
            return None

    @classmethod
    def load(
        cls, path: str | Path, expected_fingerprint: str | None = None
    ) -> "SequenceDataset":
        try:
            with np.load(path, allow_pickle=False) as data:
                schema_version = int(data["schema_version"].item())
                fingerprint = str(data["fingerprint"].item())
                arrays = {name: data[name] for name in _CACHE_ARRAY_FIELDS}
        except FileNotFoundError:
            raise
        except PermissionError:
            raise
        except (
            BadZipFile,
            EOFError,
            KeyError,
            OSError,
            OverflowError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(f"Invalid sequence cache at {path}; rebuild it") from error

        if schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported cache schema {schema_version}; "
                f"expected {CACHE_SCHEMA_VERSION}. Rebuild it."
            )
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise ValueError(
                "The sequence cache does not match the configuration or PCAP files; rebuild it"
            )
        dataset = cls(**arrays, fingerprint=fingerprint)
        dataset.validate()
        return dataset


def from_records(
    records: list[SequenceRecord],
    class_names: list[str],
    environment_names: list[str],
    width: int,
    fingerprint: str,
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
        class_names=np.asarray(class_names, dtype=str),
        environment_names=np.asarray(environment_names, dtype=str),
        group_names=group_names,
        fingerprint=fingerprint,
    )
    dataset.validate()
    return dataset
