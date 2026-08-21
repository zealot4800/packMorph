from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import SequenceDataset


@dataclass(slots=True)
class ProfileLibrary:
    training: SequenceDataset
    indices_by_class: dict[int, np.ndarray]
    alternatives_by_class: dict[int, np.ndarray]
    nearest_by_class: dict[int, int]

    @classmethod
    def fit(cls, training: SequenceDataset) -> "ProfileLibrary":
        width = training.sizes.shape[1]
        positions = np.arange(width)[None, :]
        class_ids = np.unique(training.labels).astype(int)
        indices: dict[int, np.ndarray] = {}
        centroids: dict[int, np.ndarray] = {}
        for label in class_ids:
            rows = np.flatnonzero(training.labels == label)
            indices[int(label)] = rows
            counts = training.packet_counts[rows]
            class_sizes = training.sizes[rows]
            class_iats = training.iats[rows]
            valid = positions < counts[:, None]
            absolute_sizes = np.abs(class_sizes)
            packet_sizes = absolute_sizes[valid]
            timed = valid & (positions > 0)
            packet_iats = class_iats[timed]
            byte_counts = (absolute_sizes * valid).sum(axis=1)
            centroids[int(label)] = np.asarray(
                [
                    np.median(packet_sizes),
                    (
                        np.median(np.log1p(packet_iats * 1000.0))
                        if packet_iats.size
                        else 0.0
                    ),
                    np.mean(counts),
                    np.mean(byte_counts),
                ],
                dtype=np.float64,
            )

        ordered_classes = sorted(indices)
        alternatives = {
            label: np.asarray(
                [candidate for candidate in ordered_classes if candidate != label],
                dtype=np.int16,
            )
            for label in ordered_classes
        }
        centroid_matrix = np.stack([centroids[label] for label in ordered_classes])
        scale = np.maximum(centroid_matrix.std(axis=0), 1e-9)
        nearest = {
            label: min(
                alternatives[label].tolist(),
                key=lambda candidate: float(
                    np.linalg.norm((centroids[candidate] - centroids[label]) / scale)
                ),
            )
            for label in ordered_classes
        }
        return cls(
            training=training,
            indices_by_class=indices,
            alternatives_by_class=alternatives,
            nearest_by_class=nearest,
        )

    def select_target(self, source: int, policy: str, rng: np.random.Generator) -> int:
        alternatives = self.alternatives_by_class[source]
        if alternatives.size == 0:
            raise ValueError("Morphing requires at least two service classes")
        if policy == "nearest":
            return self.nearest_by_class[source]
        return int(rng.choice(alternatives))

    def sample_template(
        self,
        target: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row = int(rng.choice(self.indices_by_class[target]))
        count = int(self.training.packet_counts[row])
        signed_sizes = self.training.sizes[row, :count]
        return (
            np.abs(signed_sizes),
            self.training.iats[row, :count],
            np.sign(signed_sizes).astype(np.int8),
        )


@dataclass(slots=True)
class MorphResult:
    dataset: SequenceDataset
    target_labels: np.ndarray
    source_indices: np.ndarray
    statistics: list[dict]


def _partition_capacity(
    required: int,
    budget: int,
    minimum_size: int,
    maximum_size: int,
    minimum_packets: int = 1,
) -> np.ndarray | None:
    """Find the smallest valid packet capacity covering required bytes within budget."""
    if required <= 0:
        return np.empty(0, dtype=np.int64)
    minimum_packets = max(
        minimum_packets,
        (required + maximum_size - 1) // maximum_size,
    )
    maximum_packets = budget // minimum_size
    for packet_count in range(minimum_packets, maximum_packets + 1):
        capacity = max(required, packet_count * minimum_size)
        if capacity > min(budget, packet_count * maximum_size):
            continue
        base, extra = divmod(capacity, packet_count)
        sizes = np.full(packet_count, base, dtype=np.int64)
        sizes[:extra] += 1
        return sizes
    return None


def _direction_capacity(
    sizes: np.ndarray,
    directions: np.ndarray,
    direction: int,
) -> int:
    return int(sizes[directions == direction].sum())


def _directional_fallback(
    forward_bytes: int,
    reverse_bytes: int,
    budget: int,
    minimum_size: int,
    maximum_size: int,
    minimum_slots: int,
    target_iats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a feasible two-direction plan when a target prefix cannot carry the source."""
    forward_minimum = 1 if forward_bytes else 0
    reverse_minimum = 1 if reverse_bytes else 0
    if forward_bytes and not reverse_bytes:
        forward_minimum = max(forward_minimum, minimum_slots)
    elif reverse_bytes and not forward_bytes:
        reverse_minimum = max(reverse_minimum, minimum_slots)

    forward = _partition_capacity(
        forward_bytes,
        budget,
        minimum_size,
        maximum_size,
        forward_minimum,
    )
    reverse = _partition_capacity(
        reverse_bytes,
        budget,
        minimum_size,
        maximum_size,
        reverse_minimum,
    )
    if forward is None or reverse is None:
        raise ValueError(
            "The packet-size bounds and padding ratio cannot represent the source directions"
        )
    capacity = int(forward.sum() + reverse.sum())
    if capacity > budget:
        raise ValueError(
            "The padding ratio is too small to represent both source directions"
        )

    packet_count = forward.size + reverse.size
    directions = np.full(packet_count, -1, dtype=np.int8)
    if forward.size:
        # Spread forward slots across the plan instead of creating one artificial burst.
        forward_positions = np.floor(
            (np.arange(forward.size, dtype=np.float64) + 0.5)
            * packet_count
            / forward.size
        ).astype(np.int64)
        directions[forward_positions] = 1
    sizes = np.empty(packet_count, dtype=np.int64)
    sizes[directions == 1] = forward
    sizes[directions == -1] = reverse

    positive_iats = target_iats[target_iats > 0]
    fallback_iat = float(np.median(positive_iats)) if positive_iats.size else 1e-6
    iats = np.resize(target_iats.astype(np.float64), packet_count)
    if packet_count > target_iats.size:
        repeated = np.arange(packet_count) >= target_iats.size
        iats[repeated] = np.maximum(iats[repeated], fallback_iat)
    return sizes, iats, directions


def _enforce_directional_capacity(
    sizes: np.ndarray,
    iats: np.ndarray,
    directions: np.ndarray,
    forward_bytes: int,
    reverse_bytes: int,
    budget: int,
    minimum_size: int,
    maximum_size: int,
    minimum_slots: int,
    target_iats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    requirements = ((1, forward_bytes), (-1, reverse_bytes))
    repaired_sizes = sizes
    repaired_iats = iats
    repaired_directions = directions
    positive_iats = target_iats[target_iats > 0]
    fallback_iat = float(np.median(positive_iats)) if positive_iats.size else 1e-6

    for direction, required in requirements:
        deficit = required - _direction_capacity(
            repaired_sizes, repaired_directions, direction
        )
        if deficit <= 0:
            continue
        remaining_budget = budget - int(repaired_sizes.sum())
        addition = _partition_capacity(
            deficit,
            remaining_budget,
            minimum_size,
            maximum_size,
        )
        if addition is None:
            return _directional_fallback(
                forward_bytes,
                reverse_bytes,
                budget,
                minimum_size,
                maximum_size,
                minimum_slots,
                target_iats,
            )
        repaired_sizes = np.concatenate((repaired_sizes, addition))
        repaired_iats = np.concatenate(
            (repaired_iats, np.full(addition.size, fallback_iat))
        )
        repaired_directions = np.concatenate(
            (
                repaired_directions,
                np.full(addition.size, direction, dtype=np.int8),
            )
        )

    if repaired_sizes.size < minimum_slots:
        return _directional_fallback(
            forward_bytes,
            reverse_bytes,
            budget,
            minimum_size,
            maximum_size,
            minimum_slots,
            target_iats,
        )
    return repaired_sizes, repaired_iats, repaired_directions


def _target_slots(
    source_bytes: int,
    forward_bytes: int,
    reverse_bytes: int,
    target_sizes: np.ndarray,
    target_iats: np.ndarray,
    target_directions: np.ndarray,
    config: dict,
    minimum_slots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum_size = config["minimum_packet_size"]
    maximum_size = config["maximum_packet_size"]
    budget = max(
        source_bytes, int(np.floor(source_bytes * config["maximum_padding_ratio"]))
    )
    template_sizes = np.clip(target_sizes.astype(np.int64), minimum_size, maximum_size)
    if template_sizes.size == 0:
        raise ValueError("Target profile contained no packets")

    template_capacity = int(template_sizes.sum())
    repeats = max(1, (source_bytes + template_capacity - 1) // template_capacity)
    sizes = np.tile(template_sizes, repeats)
    iats = np.tile(target_iats.astype(np.float64), repeats)
    directions = np.tile(target_directions.astype(np.int8), repeats)
    directions[directions == 0] = 1
    if template_capacity < source_bytes:
        end = int(np.searchsorted(np.cumsum(sizes), source_bytes, side="left")) + 1
        sizes, iats, directions = sizes[:end], iats[:end], directions[:end]

    positive_iats = target_iats[target_iats > 0]
    fallback_iat = float(np.median(positive_iats)) if positive_iats.size else 1e-6
    if repeats > 1:
        repeated = np.arange(iats.size) >= template_sizes.size
        iats[repeated] = np.maximum(iats[repeated], fallback_iat)

    cumulative = np.cumsum(sizes)
    keep = int(np.searchsorted(cumulative, budget, side="right"))
    sizes, iats, directions = sizes[:keep], iats[:keep], directions[:keep]
    capacity = int(sizes.sum())

    while capacity < source_bytes:
        addition = _partition_capacity(
            source_bytes - capacity,
            budget - capacity,
            minimum_size,
            maximum_size,
        )
        if addition is not None:
            start = sizes.size
            added_directions = np.resize(target_directions, start + addition.size)[
                start:
            ].astype(np.int8)
            added_directions[added_directions == 0] = 1
            sizes = np.concatenate((sizes, addition))
            iats = np.concatenate((iats, np.full(addition.size, fallback_iat)))
            directions = np.concatenate((directions, added_directions))
            capacity = int(sizes.sum())
            break
        if sizes.size == 0:
            raise ValueError(
                "The packet-size bounds and padding ratio cannot represent the source byte volume"
            )
        capacity -= int(sizes[-1])
        sizes, iats, directions = sizes[:-1], iats[:-1], directions[:-1]

    if capacity < source_bytes or capacity > budget:
        raise RuntimeError("Morphing capacity violated its byte-volume policy")
    sizes, iats, directions = _enforce_directional_capacity(
        sizes,
        iats,
        directions,
        forward_bytes,
        reverse_bytes,
        budget,
        minimum_size,
        maximum_size,
        minimum_slots,
        target_iats,
    )
    capacity = int(sizes.sum())
    if capacity < source_bytes or capacity > budget:
        raise RuntimeError("Directional repair violated the byte-volume policy")
    if _direction_capacity(sizes, directions, 1) < forward_bytes:
        raise RuntimeError("Morphing failed to provide enough forward capacity")
    if _direction_capacity(sizes, directions, -1) < reverse_bytes:
        raise RuntimeError("Morphing failed to provide enough reverse capacity")
    if np.any(sizes < minimum_size) or np.any(sizes > maximum_size):
        raise RuntimeError("Morphing generated an out-of-policy packet size")
    return sizes, iats, directions


def _morph_one(
    source_sizes: np.ndarray,
    source_iats: np.ndarray,
    target_sizes: np.ndarray,
    target_iats: np.ndarray,
    target_directions: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    source_bytes = int(np.abs(source_sizes).sum())
    forward_bytes = int(source_sizes[source_sizes > 0].sum())
    reverse_bytes = int(-source_sizes[source_sizes < 0].sum())
    source_packets = int(source_sizes.size)
    source_duration = float(source_iats.sum())
    packet_sizes, morphed_iats, directions = _target_slots(
        source_bytes,
        forward_bytes,
        reverse_bytes,
        target_sizes,
        target_iats,
        target_directions,
        config,
        minimum_slots=2 if source_duration > 0.0 else 1,
    )

    morphed_iats[0] = 0.0
    target_duration = float(morphed_iats.sum())
    if (
        source_duration > 0.0
        and target_duration < source_duration
        and morphed_iats.size > 1
    ):
        if target_duration <= 0.0:
            morphed_iats[1:] = source_duration / (morphed_iats.size - 1)
        else:
            morphed_iats[1:] *= source_duration / target_duration
        target_duration = float(morphed_iats.sum())
    maximum_duration = source_duration * config["maximum_fct_ratio"]
    if (
        source_duration > 0.0
        and target_duration > maximum_duration
        and morphed_iats.size > 1
    ):
        morphed_iats[1:] *= maximum_duration / target_duration
        target_duration = float(morphed_iats.sum())

    capacity = int(packet_sizes.sum())
    real_bytes = np.zeros_like(packet_sizes)
    accounted_by_direction: dict[int, int] = {}
    for direction, required in ((1, forward_bytes), (-1, reverse_bytes)):
        indices = np.flatnonzero(directions == direction)
        if indices.size == 0:
            accounted_by_direction[direction] = 0
            continue
        directional_sizes = packet_sizes[indices]
        starts = np.concatenate(
            (np.asarray([0], dtype=np.int64), np.cumsum(directional_sizes[:-1]))
        )
        allocated = np.clip(required - starts, 0, directional_sizes)
        real_bytes[indices] = allocated
        accounted_by_direction[direction] = int(allocated.sum())
    accounted_bytes = int(real_bytes.sum())
    forward_capacity = _direction_capacity(packet_sizes, directions, 1)
    reverse_capacity = _direction_capacity(packet_sizes, directions, -1)
    directional_accounting_ok = (
        accounted_by_direction[1] == forward_bytes
        and accounted_by_direction[-1] == reverse_bytes
    )
    dummy_packets = int(np.count_nonzero(real_bytes == 0))
    real_packets = packet_sizes.size - dummy_packets
    signed_sizes = packet_sizes.astype(np.int32) * directions.astype(np.int32)
    statistics = {
        "original_bytes": source_bytes,
        "morphed_bytes": capacity,
        "original_forward_bytes": forward_bytes,
        "original_reverse_bytes": reverse_bytes,
        "forward_capacity": forward_capacity,
        "reverse_capacity": reverse_capacity,
        "accounted_forward_bytes": accounted_by_direction[1],
        "accounted_reverse_bytes": accounted_by_direction[-1],
        "padding_and_dummy_bytes": capacity - source_bytes,
        "original_packets": source_packets,
        "morphed_packets": int(packet_sizes.size),
        "dummy_packets": dummy_packets,
        "fragmentation_delta": max(0, int(real_packets) - source_packets),
        "original_duration": source_duration,
        "morphed_duration": target_duration,
        "bandwidth_overhead": (capacity - source_bytes) / max(source_bytes, 1),
        "packet_overhead": (packet_sizes.size - source_packets)
        / max(source_packets, 1),
        "fct_overhead": (
            (target_duration - source_duration) / source_duration
            if source_duration > 1e-9
            else 0.0
        ),
        "fct_overhead_valid": source_duration > 1e-9,
        "directional_byte_accounting_ok": directional_accounting_ok,
        "byte_accounting_ok": (
            directional_accounting_ok and accounted_bytes == source_bytes
        ),
    }
    return signed_sizes, morphed_iats.astype(np.float32), statistics


def morph_dataset(
    source: SequenceDataset,
    library: ProfileLibrary,
    config: dict,
    seed: int | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> MorphResult:
    if rng is not None and seed is not None:
        raise ValueError("Pass either a morphing seed or an RNG, not both")
    if rng is None:
        if seed is None:
            raise ValueError("Morphing requires a seed or RNG")
        rng = np.random.default_rng(seed)
    width = source.sizes.shape[1]
    window_sizes: list[np.ndarray] = []
    window_iats: list[np.ndarray] = []
    source_indices: list[int] = []
    window_flow_ids: list[str] = []
    targets = np.empty(len(source), dtype=source.labels.dtype)
    statistics: list[dict] = []

    for row in range(len(source)):
        count = int(source.packet_counts[row])
        source_label = int(source.labels[row])
        target = library.select_target(source_label, config["target_policy"], rng)
        template_sizes, template_iats, template_directions = library.sample_template(
            target, rng
        )
        sizes, iats, stats = _morph_one(
            source.sizes[row, :count],
            source.iats[row, :count],
            template_sizes,
            template_iats,
            template_directions,
            config,
        )
        window_count = (sizes.size + width - 1) // width
        for window in range(window_count):
            start, stop = window * width, min((window + 1) * width, sizes.size)
            chunk_iats = iats[start:stop].copy()
            chunk_iats[0] = 0.0
            window_sizes.append(sizes[start:stop])
            window_iats.append(chunk_iats)
            source_indices.append(row)
            window_flow_ids.append(f"{source.flow_ids[row]}-m{window}")
        targets[row] = target
        stats.update(
            {
                "source_class": source_label,
                "source_class_name": str(source.class_names[source_label]),
                "target_class": target,
                "target_class_name": str(source.class_names[target]),
                "capture": source.group_for_row(row),
                "flow_id": str(source.flow_ids[row]),
                "morphed_windows": window_count,
            }
        )
        statistics.append(stats)

    mapping = np.asarray(source_indices, dtype=np.int64)
    size_matrix = np.zeros((mapping.size, width), dtype=np.int32)
    iat_matrix = np.zeros((mapping.size, width), dtype=np.float32)
    counts = np.empty(mapping.size, dtype=np.int32)
    for row, (sizes, iats) in enumerate(zip(window_sizes, window_iats)):
        counts[row] = sizes.size
        size_matrix[row, : sizes.size] = sizes
        iat_matrix[row, : iats.size] = iats

    dataset = SequenceDataset(
        sizes=size_matrix,
        iats=iat_matrix,
        packet_counts=counts,
        labels=source.labels[mapping],
        environment_ids=source.environment_ids[mapping],
        group_ids=source.group_ids[mapping],
        flow_ids=np.asarray(window_flow_ids, dtype=str),
        class_names=source.class_names,
        environment_names=source.environment_names,
        group_names=source.group_names,
        fingerprint=source.fingerprint,
    )
    return MorphResult(
        dataset=dataset,
        target_labels=targets,
        source_indices=mapping,
        statistics=statistics,
    )
