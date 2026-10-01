"""Finite observation of a continuously scheduled ditto-style link pattern."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .dataset import SequenceDataset
from .traffic import TrafficTrace, make_trace, validate_conservation


@dataclass(slots=True)
class DittoOutcome:
    trace: TrafficTrace
    state_load: list[dict]
    queue_overflow_packets: int
    oversize_drops: int
    dropped_packets: int
    unfinished_packets: int
    outside_horizon_packets: int
    reordering_inversions: int
    mean_genuine_packet_delay_seconds: float | None


def fit_pattern(training: SequenceDataset, length: int, maximum_packet_size: int) -> list[int]:
    """Fit unlabeled packet-size percentiles from pooled training records."""
    if not 3 <= length <= 6 or maximum_packet_size <= 0:
        raise ValueError("Ditto candidate length must be 3–6 and maximum packet size positive")
    mask = np.arange(training.sizes.shape[1])[None, :] < training.packet_counts[:, None]
    lengths = np.abs(training.sizes[mask]).astype(np.int64)
    if not len(lengths):
        raise ValueError("Ditto pattern needs training packets")
    pattern = [int(np.percentile(lengths, 100 * (i + 1) / length, method="higher")) for i in range(length)]
    pattern[-1] = max(pattern[-1], maximum_packet_size)
    return pattern


def simulate(source: TrafficTrace, rng: np.random.Generator, parameters: dict) -> DittoOutcome:
    pattern = np.asarray(parameters["pattern"], dtype=np.int64)
    length = len(pattern)
    maximum_size = int(parameters["maximum_packet_size"])
    rate = float(parameters["rate_packets_per_second"])
    horizon = float(parameters["observation_horizon_seconds"])
    capacity = float(parameters["link_capacity_bps"])
    queue_limit = int(parameters["queue_limit_packets"])
    phase = int(parameters.get("phase_slots", 0))
    if (length < 1 or length > 16 or np.any(pattern <= 0) or np.any(np.diff(pattern) < 0)
            or pattern[-1] < maximum_size or np.any(pattern > np.iinfo(np.int32).max)
            or maximum_size <= 0 or queue_limit <= 0 or phase < 0 or phase >= length
            or not np.isfinite([rate, horizon, capacity]).all() or rate <= 0 or horizon <= 0
            or capacity <= 0 or rate * int(pattern[-1]) * 8 > capacity + 1e-6):
        raise ValueError("Invalid ditto pattern, clock, queue limit, or link capacity")
    # A full-sized packet must serialize before the next opportunity.
    slots = int(np.ceil(horizon * rate - 1e-12))
    packets: list[tuple[float, int, int, int]] = []
    state_load: list[dict] = []
    dropped = unfinished = outside = overflow = oversize = inversions = 0
    delays: list[float] = []
    for direction in (1, -1):
        indices = np.flatnonzero(np.sign(source.sizes) == direction)
        queues = [deque() for _ in range(length)]
        load = [{"direction": direction, "state": state, "size_bytes": int(pattern[state]),
                 "arrivals": 0, "accepted": 0, "delivered": 0, "overflow_drops": 0,
                 "unfinished": 0, "slots": 0, "max_queue_depth": 0} for state in range(length)]
        cursor = 0
        delivered_indices: list[int] = []

        def admit(until: float, *, boundary: bool = False) -> None:
            nonlocal cursor, dropped, overflow, oversize
            while cursor < len(indices) and (source.times[indices[cursor]] < until if boundary else source.times[indices[cursor]] <= until + 1e-12):
                index = int(indices[cursor])
                cursor += 1
                size = abs(int(source.sizes[index]))
                fits = np.flatnonzero(pattern >= size)
                if not len(fits):
                    dropped += 1
                    oversize += 1
                    continue
                smallest = pattern[fits[0]]
                tied = fits[pattern[fits] == smallest]
                state = int(rng.choice(tied)) if len(tied) > 1 else int(tied[0])
                item = load[state]
                item["arrivals"] += 1
                if len(queues[state]) >= queue_limit:
                    item["overflow_drops"] += 1
                    overflow += 1
                    dropped += 1
                else:
                    queues[state].append(index)
                    item["accepted"] += 1
                    item["max_queue_depth"] = max(item["max_queue_depth"], len(queues[state]))

        for slot in range(slots):
            when = slot / rate
            if when >= horizon:
                break
            admit(when)
            state = (slot + phase) % length
            item = load[state]
            item["slots"] += 1
            if queues[state]:
                index = queues[state].popleft()
                delivered_indices.append(index)
                original_size = abs(int(source.sizes[index]))
                packets.append((when, direction * int(pattern[state]), index, original_size))
                item["delivered"] += 1
                delays.append(when - float(source.times[index]))
            else:
                packets.append((when, direction * int(pattern[state]), -1, 0))
        # Packets arriving after the last opportunity but before the boundary
        # are queued or dropped, never silently removed from accounting.
        admit(horizon, boundary=True)
        outside += len(indices) - cursor
        for state, queue in enumerate(queues):
            load[state]["unfinished"] = len(queue)
            load[state]["offered_load_ratio"] = (load[state]["arrivals"] / load[state]["slots"] if load[state]["slots"] else None)
            unfinished += len(queue)
        # Count packets overtaken within a direction, including by other states.
        inversions += sum(earlier > later for pos, earlier in enumerate(delivered_indices)
                          for later in delivered_indices[pos + 1:])
        state_load.extend(load)

    result = make_trace(packets)
    validate_conservation(source, result, global_order=None, require_complete=False)
    real_indices = result.source_index[result.source_index >= 0]
    if len(real_indices) != len(np.unique(real_indices)):
        raise RuntimeError("Ditto delivered a real packet more than once")
    if len(real_indices) + dropped + unfinished + outside != len(source.sizes):
        raise RuntimeError("Ditto packet accounting disagrees")
    return DittoOutcome(result, state_load, overflow, oversize, dropped, unfinished,
                        outside, inversions, float(np.mean(delays)) if delays else None)
