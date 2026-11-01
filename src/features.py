"""Dataset-specific flow loaders with explicit provenance and split validation."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

@dataclass(slots=True)
class FlowDataset:
    """Model inputs and the original flow identities used to prevent leakage."""

    features: np.ndarray
    labels: np.ndarray
    captures: np.ndarray | None
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    row_indices: np.ndarray
    record_ids: np.ndarray
    source_fingerprint: str
    schema: str = 'cic-pkl-v1'
    capture_disjoint_verified: bool = True

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, indices: np.ndarray) -> "FlowDataset":
        return replace(self, features=self.features[indices], labels=self.labels[indices],
                       captures=None if self.captures is None else self.captures[indices],
                       row_indices=self.row_indices[indices], record_ids=self.record_ids[indices])


def load_pickle_splits(config: dict, environment: str) -> dict[str, FlowDataset]:
    """Load the fixed, balanced PKLs and reject inconsistent or leaking splits."""
    if config['data'].get('schema') == 'iscx-time-features-v1':
        return _load_time_splits(config, environment)
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
                     and features.shape == (len(labels), len(names)) and len(labels) > 0
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
            raise ValueError(f'{path}: expected balanced classes and the configured {len(names)} raw features')
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


def _load_time_splits(config: dict, environment: str) -> dict[str, FlowDataset]:
    """Validate natural-class flow holdouts without inventing capture identities."""
    names = tuple(config['model']['features'][environment])
    classes = tuple(config['data']['categories'])
    saved_parts = {}
    for split, filename in (('train', 'train'), ('validation', 'val'), ('test', 'test')):
        path = Path(config['data']['root']) / environment / f'{filename}.pkl'
        with path.open('rb') as handle:
            saved = pickle.load(handle)
        try:
            x, y = np.asarray(saved['X']), np.asarray(saved['y'])
            ids = np.asarray(saved['record_ids'])
            files, lines = np.asarray(saved['source_files']), np.asarray(saved['source_lines'])
            raw, mask = np.asarray(saved['X_raw']), np.asarray(saved['missing_mask'])
            raw_names = tuple(saved['raw_feature_names'])
            fingerprint = saved['source_fingerprint']
            valid = (
                saved['schema'] == 'iscx-time-features-v1'
                and saved['environment'] == environment and saved['split'] == filename
                and saved['split_unit'] == 'unique_feature_vector'
                and saved['capture_disjoint_verified'] is False
                and tuple(saved['feature_names']) == names and tuple(saved['class_names']) == classes
                and names == raw_names
                and y.ndim == 1 and len(y) > 0 and np.issubdtype(y.dtype, np.integer)
                and x.shape == (len(y), len(names)) and x.dtype == np.float32
                and np.isfinite(x).all() and (x >= 0).all()
                and set(y.tolist()) == set(range(len(classes)))
                and raw.shape == mask.shape == (len(y), len(raw_names))
                and raw.dtype == np.float32 and mask.dtype == bool
                and np.array_equal(mask, np.isnan(raw)) and not np.isinf(raw).any()
                and (raw[~mask] >= 0).all()
                and ids.shape == files.shape == lines.shape == y.shape
                and ids.dtype.kind in 'SU' and np.all(ids != '')
                and files.dtype.kind == 'U' and np.all(files != '')
                and np.issubdtype(lines.dtype, np.integer) and np.all(lines >= 1)
                and len(set(zip(files.tolist(), lines.tolist()))) == len(y)
                and len(np.unique(ids)) == len(y)
                and isinstance(fingerprint, str) and len(fingerprint) == 64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Invalid time-feature pickle: {path}') from error
        if not valid:
            raise ValueError(f'{path}: inconsistent time-feature data or provenance')
        saved_parts[split] = saved
    if len({s['source_fingerprint'] for s in saved_parts.values()}) != 1:
        raise ValueError('Pickle splits refer to different source datasets')
    # Validate saved imputation against training data, using raw X rather than X_scaled.
    train = saved_parts['train']['X_raw']
    medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0
                        for c in train.T], dtype=np.float32)
    raw_names = tuple(saved_parts['train']['raw_feature_names'])
    # Source line numbers remain real provenance, even when files share line numbers.
    provenance = sorted({(str(f), int(n)) for s in saved_parts.values()
                         for f, n in zip(s['source_files'], s['source_lines'])})
    row_map = {key: i for i, key in enumerate(provenance)}
    splits = {}
    identities, input_vectors, source_rows = set(), set(), set()
    for split, saved in saved_parts.items():
        if tuple(saved['raw_feature_names']) != raw_names:
            raise ValueError('Raw feature schemas differ between splits')
        raw = saved['X_raw']
        expected = np.where(np.isnan(raw), medians, raw)
        if not np.array_equal(expected, saved['X']):
            raise ValueError(f'{environment}/{split}: X differs from training-only imputation')
        ids = set(saved['record_ids'].tolist())
        vectors = {row.tobytes() for row in saved['X']}
        sources = set(zip(saved['source_files'].tolist(), saved['source_lines'].tolist()))
        if identities & ids or input_vectors & vectors or source_rows & sources:
            raise ValueError(f'{environment}: flow overlap between splits')
        identities.update(ids)
        input_vectors.update(vectors)
        source_rows.update(sources)
        rows = np.array([row_map[(str(f), int(n))] for f, n in
                         zip(saved['source_files'], saved['source_lines'])], dtype=np.int64)
        splits[split] = FlowDataset(saved['X'], saved['y'], None, classes, names, rows,
                                    saved['record_ids'], saved['source_fingerprint'],
                                    'iscx-time-features-v1', False)
    return splits


def require_packet_dataset(config: dict) -> None:
    """Reject unsupported aggregate inputs before policy fitting or cache writes."""
    if config['data'].get('schema') == 'iscx-time-features-v1':
        raise ValueError('CIC-ISCX contains aggregate time features without packet flow_context. '
                         'Use baseline classification; packet defenses require original PCAP data.')
