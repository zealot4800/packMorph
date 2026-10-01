"""Causal, packet-level metadata traces used only by the comparison runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import SequenceDataset


@dataclass(slots=True)
class TrafficTrace:
    sizes: np.ndarray  # signed transmitted frame lengths
    times: np.ndarray  # seconds relative to this source sequence's first packet
    source_index: np.ndarray  # -1 for dummies, otherwise source packet index
    valid_bytes: np.ndarray  # source-byte capacity carried by each frame

    def __post_init__(self) -> None:
        n = len(self.sizes)
        if any(len(x) != n for x in (self.times, self.source_index, self.valid_bytes)):
            raise ValueError("Traffic trace arrays have different lengths")
        if n == 0 or np.any(self.sizes == 0) or not np.all(np.isfinite(self.times)):
            raise ValueError("Traffic trace has invalid packets or timestamps")
        if np.any(np.diff(self.times) < -1e-9) or self.times[0] < -1e-9:
            raise ValueError("Traffic trace times must be nonnegative and ordered")
        if np.any(self.valid_bytes < 0) or np.any(self.valid_bytes > np.abs(self.sizes)):
            raise ValueError("Invalid frame payload capacity")
        if np.any((self.source_index == -1) != (self.valid_bytes == 0)):
            raise ValueError("Dummy and real packet accounting disagree")

    @property
    def padding_bytes(self) -> int:
        return int(np.abs(self.sizes).sum() - self.valid_bytes.sum() - self.dummy_bytes)

    @property
    def dummy_bytes(self) -> int:
        return int(np.abs(self.sizes[self.source_index == -1]).sum())


def source_trace(dataset: SequenceDataset, row: int) -> TrafficTrace:
    count = int(dataset.packet_counts[row])
    sizes = dataset.sizes[row, :count].copy()
    times = np.cumsum(dataset.iats[row, :count].astype(np.float64))
    return TrafficTrace(sizes, times, np.arange(count, dtype=np.int32), np.abs(sizes))


def validate_conservation(source: TrafficTrace, defended: TrafficTrace, *, global_order: bool | None = True, require_complete: bool = True) -> None:
    """Verify causality, direction, per-packet byte capacity and ordered fragments."""
    if np.any(defended.source_index < -1) or np.any(defended.source_index >= len(source.sizes)):
        raise ValueError("Unknown source packet in defended trace")
    real = defended.source_index >= 0
    indices = defended.source_index[real]
    if global_order is True:
        if np.any(np.diff(indices) < 0):
            raise ValueError("Source packet order changed")
    elif global_order is False:
        for direction in (1, -1):
            same_direction = indices[np.sign(source.sizes[indices]) == direction]
            if np.any(np.diff(same_direction) < 0):
                raise ValueError("Source packet FIFO order changed")
    if np.any(defended.times[real] + 1e-8 < source.times[indices]):
        raise ValueError("A source byte was transmitted before arrival")
    if np.any(np.sign(defended.sizes[real]) != np.sign(source.sizes[indices])):
        raise ValueError("Source packet direction changed")
    capacity = np.bincount(indices, weights=defended.valid_bytes[real], minlength=len(source.sizes))
    expected = np.abs(source.sizes).astype(np.int64)
    if (not np.array_equal(capacity.astype(np.int64), expected) if require_complete else np.any(capacity > expected)):
        raise ValueError("Source packet byte capacity changed")
    if defended.padding_bytes < 0 or defended.dummy_bytes < 0:
        raise ValueError("Invalid padding or dummy accounting")


def make_trace(packets: list[tuple[float, int, int, int]], *, stable_timestamp: bool = False) -> TrafficTrace:
    """Order packets; FRONT requests a stable timestamp-only merge."""
    if stable_timestamp:
        packets.sort(key=lambda p: p[0])
    else:
        packets.sort(key=lambda p: (p[0], p[2] if p[2] >= 0 else 2**31))
    return TrafficTrace(
        np.asarray([p[1] for p in packets], dtype=np.int32),
        np.asarray([p[0] for p in packets], dtype=np.float64),
        np.asarray([p[2] for p in packets], dtype=np.int32),
        np.asarray([p[3] for p in packets], dtype=np.int32),
    )


def windows(source: SequenceDataset, traces: list[TrafficTrace]) -> tuple[SequenceDataset, np.ndarray]:
    """Rewindow every transmitted frame and retain a source-row mapping."""
    if len(traces) != len(source):
        raise ValueError("Need one complete trace per source sequence")
    width = source.sizes.shape[1]
    chunks: list[tuple[np.ndarray, np.ndarray, int, str, float]] = []
    for row, trace in enumerate(traces):
        for start in range(0, len(trace.sizes), width):
            stop = min(start + width, len(trace.sizes))
            times = trace.times[start:stop]
            iats = np.diff(times, prepend=times[0]).astype(np.float32)
            chunks.append((trace.sizes[start:stop], iats, row,
                           f"{source.flow_ids[row]}-c{start // width}", float(times[0])))
    mapping = np.asarray([chunk[2] for chunk in chunks], dtype=np.int64)
    sizes = np.zeros((len(chunks), width), dtype=np.int32)
    iats = np.zeros((len(chunks), width), dtype=np.float32)
    counts = np.empty(len(chunks), dtype=np.int32)
    for row, (chunk_sizes, chunk_iats, _, _, _) in enumerate(chunks):
        counts[row] = len(chunk_sizes)
        sizes[row, :counts[row]] = chunk_sizes
        iats[row, :counts[row]] = chunk_iats
    result = SequenceDataset(
        sizes=sizes, iats=iats, packet_counts=counts,
        labels=source.labels[mapping], environment_ids=source.environment_ids[mapping],
        group_ids=source.group_ids[mapping],
        flow_ids=np.asarray([chunk[3] for chunk in chunks], dtype=str),
        flow_groups=source.flow_groups[mapping],
        start_offsets=np.asarray([
            source.start_offsets[chunk[2]] + chunk[4] for chunk in chunks
        ], dtype=np.float64),
        flow_start_timestamps=source.flow_start_timestamps[mapping],
        class_names=source.class_names, environment_names=source.environment_names,
        group_names=source.group_names, fingerprint=source.fingerprint,
        excluded_captures=source.excluded_captures,
    )
    result.validate()
    return result, mapping


def costs(source: TrafficTrace, defended: TrafficTrace) -> dict[str, float | int | bool | None]:
    src_bytes = int(np.abs(source.sizes).sum())
    src_duration = float(source.times[-1] - source.times[0])
    out_duration = float(defended.times[-1] - defended.times[0])
    real = defended.source_index >= 0
    delivered = int(np.count_nonzero(real))
    delivered_sources = int(np.unique(defended.source_index[real]).size)
    capacity = np.bincount(defended.source_index[real], weights=defended.valid_bytes[real], minlength=len(source.sizes))
    completed = np.array_equal(capacity.astype(np.int64), np.abs(source.sizes).astype(np.int64))
    last_source_release = float(defended.times[real].max()) if completed else None
    original_completion = float(source.times[-1])
    return {
        "bandwidth_overhead": (int(np.abs(defended.sizes).sum()) - src_bytes) / src_bytes,
        "packet_count_overhead": (len(defended.sizes) - len(source.sizes)) / len(source.sizes),
        "source_packets": len(source.sizes),
        "transmitted_packets": len(defended.sizes),
        "sequence_duration_overhead": ((out_duration - src_duration) / src_duration if src_duration > 1e-9 else 0.0),
        "duration_valid": src_duration > 1e-9,
        "zero_duration_extended": src_duration <= 1e-9 and out_duration > 1e-9,
        "last_source_byte_delay_seconds": max(0.0, last_source_release - original_completion) if completed else None,
        "genuine_completion_overhead": ((last_source_release - original_completion) / src_duration if completed and src_duration > 1e-9 else None),
        "completed": completed,
        "delivered_fraction": delivered_sources / len(source.sizes),
        "delivered_source_packets": delivered_sources,
        "padding_bytes": defended.padding_bytes,
        "dummy_bytes": defended.dummy_bytes,
        "dummy_packets": int(np.count_nonzero(defended.source_index == -1)),
        "retained_real_packets": delivered,
        "outgoing_dummy_packets": int(np.count_nonzero((defended.source_index == -1) & (defended.sizes > 0))),
        "incoming_dummy_packets": int(np.count_nonzero((defended.source_index == -1) & (defended.sizes < 0))),
        "source_bytes": src_bytes,
        "transmitted_bytes": int(np.abs(defended.sizes).sum()),
    }
