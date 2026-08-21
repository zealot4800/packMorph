from __future__ import annotations

import numpy as np

from .dataset import SequenceDataset


def _holdout_groups_by_sequence_count(
    groups: np.ndarray,
    sequence_counts: np.ndarray,
    holdout_fraction: float,
    minimum_retained_groups: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose capture groups whose combined sequence count best matches the target."""
    order = rng.permutation(groups.size)
    groups = groups[order]
    sequence_counts = sequence_counts[order]
    target = float(sequence_counts.sum()) * holdout_fraction
    selected: list[int] = []
    remaining = list(range(groups.size))
    selected_count = 0

    maximum_selected = groups.size - minimum_retained_groups
    while remaining and len(selected) < maximum_selected:
        candidate_totals = selected_count + sequence_counts[remaining]
        candidate = remaining[int(np.argmin(np.abs(candidate_totals - target)))]
        candidate_count = selected_count + int(sequence_counts[candidate])
        if selected and abs(candidate_count - target) >= abs(selected_count - target):
            break
        selected.append(candidate)
        remaining.remove(candidate)
        selected_count = candidate_count

    return groups[np.asarray(selected, dtype=np.int64)]


def stratified_group_split(
    dataset: SequenceDataset,
    holdout_fraction: float,
    seed: int,
    minimum_retained_groups: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratify by class while keeping every source capture in exactly one partition."""
    rng = np.random.default_rng(seed)
    retained: list[int] = []
    holdout: list[int] = []
    expected_labels = set(range(len(dataset.class_names)))
    present_labels = set(np.unique(dataset.labels).astype(int).tolist())
    if missing := expected_labels - present_labels:
        names = ", ".join(str(dataset.class_names[label]) for label in sorted(missing))
        raise ValueError(f"The dataset has no sequences for: {names}")

    for label in sorted(expected_labels):
        class_indices = np.flatnonzero(dataset.labels == label)
        groups, sequence_counts = np.unique(
            dataset.group_ids[class_indices], return_counts=True
        )
        required_groups = minimum_retained_groups + 1
        if groups.size < required_groups:
            name = str(dataset.class_names[label])
            raise ValueError(
                f"{name} needs at least {required_groups} eligible capture groups for a "
                "leakage-free split; "
                f"found {groups.size}"
            )
        holdout_groups = _holdout_groups_by_sequence_count(
            groups,
            sequence_counts,
            holdout_fraction,
            minimum_retained_groups,
            rng,
        )
        is_holdout = np.isin(dataset.group_ids[class_indices], holdout_groups)
        retained.extend(class_indices[~is_holdout].tolist())
        holdout.extend(class_indices[is_holdout].tolist())

    retained_indices = np.asarray(retained, dtype=np.int64)
    holdout_indices = np.asarray(holdout, dtype=np.int64)
    rng.shuffle(retained_indices)
    rng.shuffle(holdout_indices)
    return retained_indices, holdout_indices


def class_weights(labels: np.ndarray) -> dict[int, float]:
    classes, counts = np.unique(labels, return_counts=True)
    total = float(labels.size)
    return {
        int(label): total / (classes.size * float(count))
        for label, count in zip(classes, counts)
    }
