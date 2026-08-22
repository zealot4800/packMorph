from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import SequenceDataset


_MAX_GUIDED_ASSIGNMENT_ROWS = 512


class MorphInfeasibleError(ValueError):
    """Raised when a valid source/template pair cannot meet morphing constraints."""


def _minimum_cost_assignment(cost: np.ndarray) -> np.ndarray:
    """Return the exact row-to-column assignment for a finite square cost matrix."""
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Assignment cost matrix must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Assignment cost matrix must contain only finite values")
    size = matrix.shape[0]
    if size == 0:
        return np.empty(0, dtype=np.int64)

    row_potential = np.zeros(size + 1, dtype=np.float64)
    column_potential = np.zeros(size + 1, dtype=np.float64)
    matched_row = np.zeros(size + 1, dtype=np.int64)
    predecessor = np.zeros(size + 1, dtype=np.int64)

    for row in range(1, size + 1):
        matched_row[0] = row
        minimum = np.full(size + 1, np.inf, dtype=np.float64)
        used = np.zeros(size + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    matrix[active_row - 1, candidate - 1]
                    - row_potential[active_row]
                    - column_potential[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    predecessor[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            if not np.isfinite(delta):
                raise ValueError("Assignment problem has no perfect matching")
            for candidate in range(size + 1):
                if used[candidate]:
                    row_potential[matched_row[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break

        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    assignment = np.empty(size, dtype=np.int64)
    for column in range(1, size + 1):
        assignment[matched_row[column] - 1] = column - 1
    return assignment


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

    def plan_targets(
        self,
        source_labels: np.ndarray,
        policy: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Plan all decoy labels before morph batching changes sample order."""
        labels = np.asarray(source_labels)
        if labels.ndim != 1:
            raise ValueError("Source labels must be one-dimensional")
        targets = np.empty(labels.shape, dtype=labels.dtype)

        for source_value in np.unique(labels):
            source = int(source_value)
            if source not in self.alternatives_by_class:
                raise ValueError(f"No training profile exists for source class {source}")
            alternatives = self.alternatives_by_class[source]
            if alternatives.size == 0:
                raise ValueError("Morphing requires at least two service classes")
            rows = np.flatnonzero(labels == source_value)

            if policy == "nearest":
                targets[rows] = self.nearest_by_class[source]
                continue
            if policy == "random":
                targets[rows] = rng.choice(alternatives, size=rows.size)
                continue
            if policy == "balanced_guided":
                raise ValueError(
                    "balanced_guided targets require experiment-level candidate scoring"
                )
            if policy != "balanced":
                raise ValueError(f"Unknown target policy: {policy}")

            # Every legal target occurs either floor(N/(K-1)) or
            # ceil(N/(K-1)) times. Random permutations prevent the remainder
            # and row ordering from systematically favoring low class IDs.
            alternative_order = rng.permutation(alternatives)
            assignments = np.resize(alternative_order, rows.size)
            targets[rng.permutation(rows)] = assignments

        if np.any(targets == labels):
            raise RuntimeError("Target planning selected a source class as its own decoy")
        return targets

    def plan_guided_targets(
        self,
        source_labels: np.ndarray,
        utilities: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Maximize candidate utility subject to exact per-source target balance."""
        labels = np.asarray(source_labels)
        scores = np.asarray(utilities, dtype=np.float64)
        class_count = len(self.training.class_names)
        if labels.ndim != 1:
            raise ValueError("Source labels must be one-dimensional")
        if scores.shape != (labels.size, class_count):
            raise ValueError(
                "Guided utilities must have shape "
                f"{(labels.size, class_count)}, received {scores.shape}"
            )
        self.validate_guided_assignment_size(labels)
        targets = np.empty(labels.shape, dtype=labels.dtype)

        for source_value in np.unique(labels):
            source = int(source_value)
            if source not in self.alternatives_by_class:
                raise ValueError(f"No training profile exists for source class {source}")
            rows = np.flatnonzero(labels == source_value)
            alternatives = rng.permutation(self.alternatives_by_class[source])
            target_count = alternatives.size
            if target_count == 0:
                raise ValueError("Guided morphing requires at least two service classes")

            quotient, remainder = divmod(rows.size, target_count)
            repeats = quotient + int(remainder > 0)
            slots = np.repeat(alternatives, repeats)
            matrix_size = slots.size
            dummy_count = matrix_size - rows.size
            optional_columns = (
                np.arange(target_count, dtype=np.int64) * repeats + quotient
                if remainder
                else np.empty(0, dtype=np.int64)
            )
            real_utilities = scores[rows][:, slots]
            finite = np.isfinite(real_utilities)
            finite_values = real_utilities[finite]
            if finite_values.size == 0:
                raise ValueError(
                    f"No feasible guided candidates exist for source class {source}"
                )
            scale = max(float(np.max(np.abs(finite_values))), 1.0)
            scaled = real_utilities / scale
            maximum = float(np.max(scaled[finite]))
            invalid_cost = float(2 * matrix_size + 1)
            cost = np.full((matrix_size, matrix_size), invalid_cost, dtype=np.float64)
            real_cost = np.full(real_utilities.shape, invalid_cost, dtype=np.float64)
            real_cost[finite] = maximum - scaled[finite]
            cost[: rows.size] = real_cost
            if dummy_count:
                cost[rows.size :, optional_columns] = 0.0

            assignment = _minimum_cost_assignment(cost)
            selected_columns = assignment[: rows.size]
            selected_targets = slots[selected_columns]
            selected_finite = np.isfinite(
                scores[rows, selected_targets.astype(np.int64)]
            )
            dummy_valid = (
                not dummy_count
                or np.all(np.isin(assignment[rows.size :], optional_columns))
            )
            if not np.all(selected_finite) or not dummy_valid:
                available = {
                    int(target): int(np.count_nonzero(np.isfinite(scores[rows, target])))
                    for target in alternatives
                }
                raise ValueError(
                    "Balanced guided assignment is infeasible for source class "
                    f"{source}; rows={rows.size}, lower_quota={quotient}, "
                    f"upper_quota={quotient + (remainder > 0)}, "
                    f"available_candidates={available}"
                )
            targets[rows] = selected_targets

        if np.any(targets == labels):
            raise RuntimeError("Guided assignment selected a source class as its own decoy")
        return targets

    def validate_guided_assignment_size(self, source_labels: np.ndarray) -> None:
        """Fail before candidate scoring when an exact quota solve would be too large."""
        labels = np.asarray(source_labels)
        if labels.ndim != 1:
            raise ValueError("Source labels must be one-dimensional")
        for source_value in np.unique(labels):
            source = int(source_value)
            if source not in self.alternatives_by_class:
                raise ValueError(f"No training profile exists for source class {source}")
            target_count = self.alternatives_by_class[source].size
            if target_count == 0:
                raise ValueError("Guided morphing requires at least two service classes")
            source_count = int(np.count_nonzero(labels == source_value))
            quotient, remainder = divmod(source_count, target_count)
            matrix_size = (
                source_count if remainder == 0 else target_count * (quotient + 1)
            )
            if matrix_size > _MAX_GUIDED_ASSIGNMENT_ROWS:
                raise ValueError(
                    "balanced_guided exact assignment currently supports a maximum "
                    f"matrix dimension of {_MAX_GUIDED_ASSIGNMENT_ROWS}; source "
                    f"class {source} has {source_count} sequences and requires a "
                    f"{matrix_size}-row solve. "
                    "Use a bounded extraction or the non-guided balanced policy."
                )

    def sample_template_row(self, target: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.indices_by_class[target]))

    def template_for_row(
        self,
        row: int,
        expected_target: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if row < 0 or row >= len(self.training):
            raise ValueError(f"Template row {row} is outside the training profile library")
        actual_target = int(self.training.labels[row])
        if actual_target != expected_target:
            raise ValueError(
                f"Template row {row} belongs to class {actual_target}, "
                f"not requested target {expected_target}"
            )
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
    source_indices: np.ndarray
    bandwidth_overheads: np.ndarray
    fct_overheads: np.ndarray
    fct_overhead_valid: np.ndarray


def _append_target_padding(
    sizes: np.ndarray,
    iats: np.ndarray,
    directions: np.ndarray,
    template_sizes: np.ndarray,
    target_iats: np.ndarray,
    target_directions: np.ndarray,
    budget: int,
    minimum_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append the longest in-order target prefix that fits the remaining budget."""
    remaining = budget - int(sizes.sum())
    maximum_extra = remaining // minimum_size
    if maximum_extra <= 0:
        return sizes, iats, directions

    absolute = sizes.size + np.arange(maximum_extra, dtype=np.int64)
    indices = absolute % template_sizes.size
    candidates = template_sizes[indices]
    count = int(np.searchsorted(np.cumsum(candidates), remaining, side="right"))
    if count == 0:
        return sizes, iats, directions

    indices = indices[:count]
    extra_iats = target_iats[indices].astype(np.float64, copy=True)
    positive_iats = target_iats[target_iats > 0]
    fallback_iat = float(np.median(positive_iats)) if positive_iats.size else 1e-6
    repeated = absolute[:count] >= template_sizes.size
    extra_iats[repeated] = np.maximum(extra_iats[repeated], fallback_iat)
    extra_directions = target_directions[indices].astype(np.int8, copy=True)
    extra_directions[extra_directions == 0] = 1
    return (
        np.concatenate((sizes, candidates[:count])),
        np.concatenate((iats, extra_iats)),
        np.concatenate((directions, extra_directions)),
    )


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
        raise MorphInfeasibleError(
            "The packet-size bounds and padding ratio cannot represent the source directions"
        )
    capacity = int(forward.sum() + reverse.sum())
    if capacity > budget:
        raise MorphInfeasibleError(
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
    padding_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(padding_fraction) or not 0.0 <= padding_fraction <= 1.0:
        raise ValueError("Padding fraction must be finite and in [0, 1]")
    minimum_size = config["minimum_packet_size"]
    maximum_size = config["maximum_packet_size"]
    budget = max(
        source_bytes, int(np.floor(source_bytes * config["maximum_padding_ratio"]))
    )
    optional_budget = source_bytes + int(
        np.floor((budget - source_bytes) * padding_fraction)
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
    # Begin with the shortest target prefix that can carry the source. Resource
    # fractions may then add target-pattern traffic up to their effective cap.
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
            raise MorphInfeasibleError(
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
    if padding_fraction > 0.0:
        sizes, iats, directions = _append_target_padding(
            sizes,
            iats,
            directions,
            template_sizes,
            target_iats,
            target_directions,
            optional_budget,
            minimum_size,
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
    padding_fraction: float,
    fct_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
    if not np.isfinite(fct_fraction) or not 0.0 <= fct_fraction <= 1.0:
        raise ValueError("FCT fraction must be finite and in [0, 1]")
    source_bytes = int(np.abs(source_sizes).sum())
    forward_bytes = int(source_sizes[source_sizes > 0].sum())
    reverse_bytes = int(-source_sizes[source_sizes < 0].sum())
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
        padding_fraction=padding_fraction,
    )

    morphed_iats[0] = 0.0
    target_duration = float(morphed_iats.sum())
    if source_duration <= 0.0:
        morphed_iats.fill(0.0)
        target_duration = 0.0
    elif target_duration < source_duration and morphed_iats.size > 1:
        if target_duration <= 0.0:
            morphed_iats[1:] = source_duration / (morphed_iats.size - 1)
        else:
            morphed_iats[1:] *= source_duration / target_duration
        target_duration = source_duration
    global_maximum_duration = source_duration * config["maximum_fct_ratio"]
    maximum_duration = source_duration + (
        global_maximum_duration - source_duration
    ) * fct_fraction
    if (
        source_duration > 0.0
        and target_duration > maximum_duration
        and morphed_iats.size > 1
    ):
        morphed_iats[1:] *= maximum_duration / target_duration
        target_duration = maximum_duration

    capacity = int(packet_sizes.sum())
    signed_sizes = packet_sizes.astype(np.int32) * directions.astype(np.int32)
    fct_valid = source_duration > 1e-9
    return (
        signed_sizes,
        morphed_iats.astype(np.float32),
        (capacity - source_bytes) / max(source_bytes, 1),
        (
            (target_duration - source_duration) / source_duration
            if fct_valid
            else 0.0
        ),
        fct_valid,
    )


def morph_dataset(
    source: SequenceDataset,
    library: ProfileLibrary,
    config: dict,
    *,
    target_labels: np.ndarray,
    rng: np.random.Generator | None = None,
    template_rows: np.ndarray | None = None,
    padding_fraction: np.ndarray | None = None,
    fct_fraction: np.ndarray | None = None,
) -> MorphResult:
    """Morph planned targets with replayable per-source resource fractions.

    Fractions interpolate from the minimum feasible transformation toward the
    configured padding and FCT caps. Omitted arrays select zero for every row.
    """
    prescribed = np.asarray(target_labels)
    if prescribed.ndim != 1 or prescribed.shape != source.labels.shape:
        raise ValueError(
            "Prescribed target labels must have shape "
            f"{source.labels.shape}, received {prescribed.shape}"
        )
    if not np.issubdtype(prescribed.dtype, np.integer):
        raise ValueError("Prescribed target labels must be integers")
    targets = prescribed.astype(source.labels.dtype, copy=True)
    if np.any(targets.astype(np.int64) != prescribed.astype(np.int64)):
        raise ValueError("A prescribed target label exceeds the supported dtype")
    if np.any(targets == source.labels):
        raise ValueError("A source sequence cannot target its own service class")
    known_targets = np.asarray(sorted(library.indices_by_class), dtype=np.int64)
    if not np.all(np.isin(targets, known_targets)):
        raise ValueError("A prescribed target has no training profile")
    if template_rows is None:
        prescribed_templates = None
        if rng is None:
            raise ValueError("Automatic template sampling requires a seed or RNG")
    else:
        raw_templates = np.asarray(template_rows)
        if raw_templates.ndim != 1 or raw_templates.shape != source.labels.shape:
            raise ValueError(
                "Prescribed template rows must have shape "
                f"{source.labels.shape}, received {raw_templates.shape}"
            )
        if not np.issubdtype(raw_templates.dtype, np.integer):
            raise ValueError("Prescribed template rows must be integers")
        if np.issubdtype(raw_templates.dtype, np.unsignedinteger) and np.any(
            raw_templates > np.iinfo(np.int64).max
        ):
            raise ValueError("A prescribed template row exceeds the supported dtype")
        prescribed_templates = raw_templates.astype(np.int64, copy=True)
        if np.any(prescribed_templates < 0) or np.any(
            prescribed_templates >= len(library.training)
        ):
            raise ValueError("A prescribed template row is outside the profile library")
        if np.any(library.training.labels[prescribed_templates] != targets):
            raise ValueError("A prescribed template row does not match its target class")
    fractions: dict[str, np.ndarray] = {}
    for name, values in (
        ("padding_fraction", padding_fraction),
        ("fct_fraction", fct_fraction),
    ):
        if values is None:
            fractions[name] = np.zeros(len(source), dtype=np.float64)
            continue
        raw_values = np.asarray(values)
        if (
            raw_values.shape != source.labels.shape
            or not np.issubdtype(raw_values.dtype, np.floating)
            or not np.all(np.isfinite(raw_values))
            or np.any(raw_values < 0.0)
            or np.any(raw_values > 1.0)
        ):
            raise ValueError(
                f"{name} must be a finite floating-point array in [0, 1] "
                "with one value per source"
            )
        fractions[name] = raw_values.astype(np.float64, copy=True)
    width = source.sizes.shape[1]
    window_sizes: list[np.ndarray] = []
    window_iats: list[np.ndarray] = []
    source_indices: list[int] = []
    window_flow_ids: list[str] = []
    bandwidth_overheads = np.empty(len(source), dtype=np.float64)
    fct_overheads = np.empty(len(source), dtype=np.float64)
    fct_overhead_valid = np.empty(len(source), dtype=np.bool_)

    for row in range(len(source)):
        count = int(source.packet_counts[row])
        target = int(targets[row])
        template_row = (
            int(prescribed_templates[row])
            if prescribed_templates is not None
            else library.sample_template_row(target, rng)
        )
        template_sizes, template_iats, template_directions = library.template_for_row(
            template_row,
            target,
        )
        sizes, iats, bandwidth, fct, fct_valid = _morph_one(
            source.sizes[row, :count],
            source.iats[row, :count],
            template_sizes,
            template_iats,
            template_directions,
            config,
            float(fractions["padding_fraction"][row]),
            float(fractions["fct_fraction"][row]),
        )
        bandwidth_overheads[row] = bandwidth
        fct_overheads[row] = fct
        fct_overhead_valid[row] = fct_valid
        window_count = (sizes.size + width - 1) // width
        for window in range(window_count):
            start, stop = window * width, min((window + 1) * width, sizes.size)
            chunk_iats = iats[start:stop].copy()
            chunk_iats[0] = 0.0
            window_sizes.append(sizes[start:stop])
            window_iats.append(chunk_iats)
            source_indices.append(row)
            window_flow_ids.append(f"{source.flow_ids[row]}-m{window}")
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
        source_indices=mapping,
        bandwidth_overheads=bandwidth_overheads,
        fct_overheads=fct_overheads,
        fct_overhead_valid=fct_overhead_valid,
    )
