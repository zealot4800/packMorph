"""Packet-metadata Securitas adaptation; no live packet construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .traffic import TrafficTrace, make_trace, validate_conservation


ACTION_SIZES = (0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88)
OPERATIONS = ("none", "fragment", "insert")


@dataclass(slots=True)
class SecuritasOutcome:
    trace: TrafficTrace
    action_counts: dict[str, int]
    fake_ttls: tuple[int, ...]
    icmp_reply_times: tuple[float, ...]


def validate_policy(policy: dict, strategy_length: int) -> None:
    if not isinstance(policy, dict) or set(policy) != {"action_sizes", "operations"}:
        raise ValueError("Securitas policy needs action_sizes and operations")
    sizes, operations = policy["action_sizes"], policy["operations"]
    if not isinstance(sizes, list) or not isinstance(operations, list) or len(sizes) != strategy_length or len(operations) != strategy_length:
        raise ValueError("Securitas policy length mismatch")
    if any(type(size) is not int or size not in ACTION_SIZES for size in sizes):
        raise ValueError("Securitas action size outside reference set")
    if any(op not in OPERATIONS or (size == 0) != (op == "none") for size, op in zip(sizes, operations)):
        raise ValueError("Securitas action operations disagree with sizes")


def freeze_policy(action_sizes: list[int], rng: np.random.Generator, fragment_preference: float) -> dict:
    if not 0 <= fragment_preference <= 1 or any(size not in ACTION_SIZES for size in action_sizes):
        raise ValueError("Invalid Securitas action sizes or fragment preference")
    operations = ["none" if size == 0 else "fragment" if rng.random() < fragment_preference else "insert"
                  for size in action_sizes]
    result = {"action_sizes": [int(size) for size in action_sizes], "operations": operations}
    validate_policy(result, len(action_sizes))
    return result


def simulate(source: TrafficTrace, policy: dict, rng: np.random.Generator, parameters: dict) -> SecuritasOutcome:
    strategy_length = int(parameters["strategy_length"])
    validate_policy(policy, strategy_length)
    real_header = int(parameters["modeled_real_header_bytes"])
    fake_header = int(parameters["modeled_fake_header_bytes"])
    icmp_size = int(parameters["icmp_observed_size_bytes"])
    reply_probability = float(parameters["icmp_reply_probability"])
    ttl_min = int(parameters["fake_ttl_min"])
    ttl_max = int(parameters["fake_ttl_max"])
    no_inbound_delay = float(parameters["no_inbound_reply_max_delay_seconds"])
    if (real_header < 0 or fake_header <= 0 or icmp_size <= 28 or ttl_min < 1 or ttl_max < ttl_min
            or not np.isfinite([reply_probability, no_inbound_delay]).all()
            or not 0 <= reply_probability <= 1 or no_inbound_delay < 0):
        raise ValueError("Invalid Securitas packet or network parameters")
    packets: list[tuple[float, int, int, int]] = []
    counts = {"zero_actions": 0, "inbound_skipped": 0, "beyond_strategy": 0,
              "fragmented_packets": 0, "inserted_fakes": 0,
              "fragment_fallback_insertions": 0, "icmp_replies": 0}
    fake_ttls: list[int] = []
    reply_times: list[float] = []
    inbound_indices = np.flatnonzero(source.sizes < 0)
    for original_index, (time, signed_size) in enumerate(zip(source.times, source.sizes)):
        time = float(time)
        size = abs(int(signed_size))
        if original_index >= strategy_length:
            counts["beyond_strategy"] += 1
            packets.append((time, int(signed_size), original_index, size))
            continue
        action_size = int(policy["action_sizes"][original_index])
        operation = policy["operations"][original_index]
        if signed_size < 0:
            counts["inbound_skipped"] += 1
            packets.append((time, int(signed_size), original_index, size))
            continue
        if action_size == 0:
            counts["zero_actions"] += 1
            packets.append((time, int(signed_size), original_index, size))
            continue
        header = min(real_header, size)
        payload = size - header
        if operation == "fragment" and 0 < action_size < payload:
            # The header is transmitted twice but counted only once as genuine.
            first_payload = action_size
            second_payload = payload - action_size
            packets.append((time, header + first_payload, original_index, header + first_payload))
            packets.append((time, header + second_payload, original_index, second_payload))
            counts["fragmented_packets"] += 1
            continue
        if operation == "fragment":
            counts["fragment_fallback_insertions"] += 1
        packets.append((time, int(signed_size), original_index, size))
        packets.append((time, fake_header + action_size, -1, 0))
        counts["inserted_fakes"] += 1
        fake_ttls.append(int(rng.integers(ttl_min, ttl_max + 1)))
        if rng.random() < reply_probability:
            later = inbound_indices[inbound_indices > original_index]
            upper = float(source.times[later[0]]) if len(later) else time + no_inbound_delay
            reply = float(rng.uniform(time, upper)) if upper > time else time
            packets.append((reply, -icmp_size, -1, 0))
            reply_times.append(reply)
            counts["icmp_replies"] += 1
    defended = make_trace(packets, stable_timestamp=True)
    validate_conservation(source, defended)
    return SecuritasOutcome(defended, counts, tuple(fake_ttls), tuple(reply_times))


def normalized_sequence_distance(source: TrafficTrace, defended: TrafficTrace) -> float:
    """Align valid signed log-size/log-IAT/direction vectors by emitted index."""
    def features(trace: TrafficTrace) -> np.ndarray:
        gaps = np.diff(trace.times, prepend=trace.times[0])
        return np.column_stack((np.log1p(np.abs(trace.sizes)),
                                np.log1p(1000 * gaps), np.sign(trace.sizes))).astype(np.float64)

    original = features(source)
    changed = features(defended)
    aligned = np.zeros((max(len(original), len(changed)), 3), dtype=np.float64)
    aligned[:len(changed)] += changed
    aligned[:len(original)] -= original
    return float(np.linalg.norm(aligned) / max(np.linalg.norm(original), 1e-9))
