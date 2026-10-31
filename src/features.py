"""Balanced flow datasets loaded from the six fixed pickle splits."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

@dataclass(slots=True)
class FlowDataset:
    """Model inputs and the original flow identities used to prevent leakage."""

    features: np.ndarray
    labels: np.ndarray
    captures: np.ndarray
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    row_indices: np.ndarray
    record_ids: np.ndarray
    source_fingerprint: str

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, indices: np.ndarray) -> "FlowDataset":
        return FlowDataset(self.features[indices], self.labels[indices],
                           self.captures[indices], self.class_names, self.feature_names,
                           self.row_indices[indices], self.record_ids[indices],
                           self.source_fingerprint)


def load_pickle_splits(config: dict, environment: str) -> dict[str, FlowDataset]:
    """Load the fixed, balanced PKLs and reject inconsistent or leaking splits."""
    names = tuple(config['model']['features'][environment])
    classes = tuple(config['data']['categories'])
    splits = {}
    for split, filename in (('train', 'train'), ('validation', 'val'), ('test', 'test')):
        path = Path(config['data']['root']) / environment / f'{filename}.pkl'
        with path.open('rb') as handle:
            saved = pickle.load(handle)
        try:
            features = np.asarray(saved['X'])
            labels = np.asarray(saved['y'])
            captures = np.asarray(saved['captures'])
            rows = np.asarray(saved['row_indices'])
            records = np.asarray(saved['record_ids'])
            fingerprint = saved['source_fingerprint']
            valid = (saved['environment'] == environment and saved['split'] == filename
                     and tuple(saved['feature_names']) == names and tuple(saved['class_names']) == classes
                     and features.shape == (len(labels), 16) and len(labels) > 0
                     and features.dtype == np.float32 and np.isfinite(features).all()
                     and np.all(features >= 0) and np.issubdtype(labels.dtype, np.integer)
                     and labels.shape == captures.shape == rows.shape == records.shape == (len(labels),)
                     and np.all((labels >= 0) & (labels < len(classes)))
                     and np.issubdtype(rows.dtype, np.integer) and np.all(rows >= 0)
                     and captures.dtype.kind == 'U' and np.all(captures != '')
                     and records.dtype.kind in 'SU' and np.all(records != '')
                     and isinstance(fingerprint, str) and len(fingerprint) == 64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Invalid balanced pickle: {path}') from error
        if not valid or len(set(np.bincount(labels, minlength=len(classes)))) != 1:
            raise ValueError(f'{path}: expected balanced classes and the configured 16 raw features')
        unique, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
        if (any(not np.array_equal(a, a[first[inverse]]) for a in (features, labels, captures, records))
                or len(np.unique(records)) != len(unique)
                or (split != 'train' and len(unique) != len(rows))):
            raise ValueError(f'{path}: inconsistent copies or repeated holdout records')
        splits[split] = FlowDataset(features, labels, captures, classes, names, rows, records, fingerprint)
    if len({part.source_fingerprint for part in splits.values()}) != 1:
        raise ValueError('Pickle splits refer to different source datasets')
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        for field in ('captures', 'row_indices', 'record_ids'):
            if np.intersect1d(getattr(splits[left], field), getattr(splits[right], field)).size:
                raise ValueError(f'{environment}: {field} overlap between {left} and {right}')
    return splits
