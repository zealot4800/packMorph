"""BLANKET-inspired learned flow controls, fitted against a frozen classifier.

Size moments and equal splitting of directional IATs have closed-form updates.
Packet-derived bulk coefficients describe the same offline insertion schedule.
Clock dilation preserves its burst boundaries; duration is a timestamp span.
This aggregate adaptation does not reproduce the paper's packet-position model.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .transform import (DIRECTIONS, FEATURE_NAMES, FlowPolicy, _context_rows,
                        transform_flow, validate_policy_parameters)


class Blanket(FlowPolicy):
    name = 'blanket'
    VERSION = 'blanket-flow-v3'
    required = {'iterations': 'positive_count', 'population': 'positive_count',
                'batch_size': 'positive_count', 'byte_penalty': 'nonnegative',
                'duration_penalty': 'nonnegative'}
    ASSUMPTIONS = {
        'sizes': 'exact uniform padding and grouped dummy moments',
        'timing': 'offline equal gap splitting, with zero, one or two dummies per gap',
        'bulk': 'packet-derived coefficients; clock dilation preserves burst boundaries',
        'duration': 'timestamp span only; link capacity and queueing are not simulated',
    }

    @classmethod
    def validate(cls, parameters: dict, source: str) -> dict:
        validate_policy_parameters(cls.name, parameters, cls.required, {}, source,
                                   byte_fraction_kind=None)
        if parameters['population'] < 4:
            raise ValueError('blanket needs population >= 4')
        return parameters

    def __init__(self, parameters, controls, columns, seed):
        super().__init__(parameters)
        self.parameters = parameters
        self.controls = np.asarray(controls, dtype=np.float64)
        if self.controls.shape != (14,) or not np.isfinite(self.controls).all():
            raise ValueError('Invalid learned Blanket controls')
        self.columns = tuple(columns)
        self.seed = seed
        self.reset()

    def reset(self):
        self._costs = np.zeros(7, dtype=np.float64)

    def _noise(self, row):
        key = f"{self.seed}:{row['source_capture']}:{row['flow_key']}"
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')
        return np.random.default_rng(seed).standard_normal(7)

    def _apply(self, row, noise):
        control = self.controls[:7] + np.exp(np.clip(self.controls[7:], -8, 2)) * noise
        sigmoid = 1 / (1 + np.exp(-np.clip(control, -40, 40)))
        maxima = np.asarray([row[f'{d}_pkt_len_max'] for d in DIRECTIONS])
        self._padding = np.floor(np.minimum(1500 * sigmoid[:2], np.maximum(0, 1500 - maxima)))
        gaps = np.minimum(2, np.floor(np.logaddexp(0, control[2:4]))).astype(np.int64)
        if not row['flow_duration']:
            gaps[:] = 0
        sizes = np.rint(64 + (1500 - 64) * sigmoid[4:6]).astype(np.int64)
        scale = 1 + np.logaddexp(0, control[6])
        table = np.asarray(row.get('blanket_bulk'))
        if table.shape != (3, 3, 2, 2, 6):
            raise ValueError('Blanket needs packet-derived blanket_bulk metadata in flow_context')
        pads = [int(self._padding[i] > 0 and row[f'tot_{d}_pkts'] > 0)
                for i, d in enumerate(DIRECTIONS)]
        payload, real, dummy, bulks, elapsed, maximum_gap = table[gaps[0], gaps[1], pads[0], pads[1]]
        if maximum_gap:
            scale = min(scale, max(1., (1 - 2e-9) / maximum_gap))
        self.extension = (scale - 1) * float(row['flow_duration'])
        for i, direction in enumerate(DIRECTIONS):
            count = float(row[f'tot_{direction}_pkts'])
            added = int(gaps[i]) * max(0, int(count) - 1)
            self.insertions[direction] = ((int(sizes[i]), added),)
        result = transform_flow(row, self, FEATURE_NAMES)
        if not self._padding.any() and not gaps.any() and scale == 1:
            return {name: float(row[name]) for name in FEATURE_NAMES}
        payload += real * self._padding[0] + dummy * (sizes[0] - row['dummy_header_bytes'])
        result['fwd_byts_b_avg'] = payload / bulks if bulks else 0.
        result['fwd_pkts_b_avg'] = (real + dummy) / bulks if bulks else 0.
        result['fwd_blk_rate_avg'] = payload / (elapsed * scale) if elapsed else 0.
        return result

    def edits(self, direction, count, mean):
        return self._padding[DIRECTIONS.index(direction)], self.insertions[direction]

    def __call__(self, row):
        result = self._apply(row, self._noise(row))
        before = sum(float(row[f'totlen_{d}_pkts']) for d in DIRECTIONS)
        after = sum(result[f'totlen_{d}_pkts'] for d in DIRECTIONS)
        old_time = float(row['flow_duration'])
        added_time = result['flow_duration'] - old_time
        self._costs[:4] += (1, before, after, added_time)
        if old_time > 0:
            self._costs[4] += added_time / old_time
            self._costs[5] += 1
        self._costs[6] = max(self._costs[6], added_time)
        return {name: result[name] for name in self.columns}

    @property
    def overhead(self):
        count, before, after, added, relative, timed, maximum = self._costs
        return dict(extra_byte_fraction=float(after / before - 1) if before else 0.,
                    mean_added_span_s=float(added / count) if count else 0.,
                    max_added_span_s=float(maximum),
                    mean_span_fraction=float(relative / timed) if timed else 0.,
                    zero_duration_flows=int(count - timed))

    @staticmethod
    def _weights(data):
        weights = np.empty(len(data), dtype=np.float64)
        for label in range(len(data.class_names)):
            members = data.labels == label
            _, inverse, counts = np.unique(data.captures[members], return_inverse=True,
                                            return_counts=True)
            if not len(counts):
                raise ValueError(f'Blanket has no examples for class {label}')
            weights[members] = 1 / (len(data.class_names) * len(counts) * counts[inverse])
        return weights

    @staticmethod
    def _prepared(data, context):
        rows = list(_context_rows(data, context))
        _, first = np.unique(data.row_indices, return_index=True)
        rows = [rows[i] for i in first]
        for row in rows:
            table = np.asarray(row.get('blanket_bulk'))
            header = row.get('dummy_header_bytes', 0)
            if (table.shape != (3, 3, 2, 2, 6) or not np.isfinite(table).all()
                    or np.any(table < 0) or not 28 <= header <= 60):
                raise ValueError('Blanket needs valid packet-derived bulk metadata and dummy headers')
        return data.subset(first), rows

    @classmethod
    def signature(cls, training, train_rows, validation, val_rows, frozen, parameters, seed):
        digest = hashlib.sha256(json.dumps([cls.VERSION, 'accuracy-first-v1', parameters, seed,
            training.feature_names, training.class_names, training.source_fingerprint],
            sort_keys=True).encode())
        arrays = list(frozen.model.embedding.get_weights())
        arrays += [frozen.model.prototypes, frozen.model.diagonal_variances,
                   frozen.scaler.mean, frozen.scaler.scale]
        for data, rows in ((training, train_rows), (validation, val_rows)):
            arrays += [data.features, data.labels, data.captures, data.row_indices, data.record_ids,
                       np.asarray([[r[n] for n in FEATURE_NAMES] for r in rows], dtype=np.float64),
                       np.asarray([r['blanket_bulk'] for r in rows]),
                       np.asarray([r['dummy_header_bytes'] for r in rows])]
        for array in arrays:
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    @classmethod
    def verify(cls, policy, signature):
        if not isinstance(policy, dict) or policy.get('version') != cls.VERSION or policy.get('signature') != signature:
            raise ValueError('Blanket policy is missing or stale; run packmorph-obfuscate blanket')
        controls = np.asarray(policy.get('controls'), dtype=np.float64)
        fingerprint = hashlib.sha256(signature.encode() + controls.tobytes()).hexdigest()
        if controls.shape != (14,) or not np.isfinite(controls).all() or policy.get('fingerprint') != fingerprint:
            raise ValueError('Learned Blanket policy failed integrity validation')

    @classmethod
    def fit(cls, training, train_context, validation, val_context, frozen, parameters,
            seed=42, cached=None):
        """Fit a stochastic universal controller using training-only CEM updates.

        Validation accuracy chooses the checkpoint, with cost breaking ties.
        Existing controls remain a candidate. Test rows are never used.
        Training copies are deduplicated and batches balance classes/captures.
        """
        cls.validate(parameters, 'Blanket fitting')
        if (training.feature_names != validation.feature_names
                or training.class_names != validation.class_names
                or training.source_fingerprint != validation.source_fingerprint):
            raise ValueError('Blanket training and validation schemas must match')
        for field in ('captures', 'row_indices', 'record_ids'):
            if np.intersect1d(getattr(training, field), getattr(validation, field)).size:
                raise ValueError(f'Blanket training and validation have overlapping {field}')
        training, train_rows = cls._prepared(training, train_context)
        validation, val_rows = cls._prepared(validation, val_context)
        signature = cls.signature(training, train_rows, validation, val_rows, frozen, parameters, seed)
        if isinstance(cached, dict) and cached.get('signature') == signature:
            cls.verify(cached, signature)
            return cached
        if parameters['batch_size'] < len(training.class_names):
            raise ValueError('Blanket batch_size must cover every service class')
        train_weights = cls._weights(training)
        val_weights = cls._weights(validation)
        rng = np.random.default_rng(seed)
        identity = np.r_[[-40., -40., -40., -40., -4., -4., -40.], np.full(7, -8.)]
        initial_center = np.r_[[-4., -4., -1., -1., -3., -3., -6.], np.full(7, -1.8)]
        center = initial_center.copy()
        spread = np.r_[np.full(7, 1.5), np.full(7, .4)]
        identity_policy = cls(parameters, identity, training.feature_names, seed)
        train_noise = np.asarray([identity_policy._noise(r) for r in train_rows])
        val_noise = np.asarray([identity_policy._noise(r) for r in val_rows])

        def assess(controls, data, rows, noise, weights):
            matrices, byte_costs, time_costs = [], [], []
            original_bytes = np.asarray([sum(float(r[f'totlen_{d}_pkts']) for d in DIRECTIONS) for r in rows])
            original_time = np.asarray([r['flow_duration'] for r in rows])
            if np.any(original_bytes <= 0):
                raise ValueError('Blanket fitting requires positive flow byte totals')
            for control in controls:
                policy = cls(parameters, control, data.feature_names, seed)
                result = [policy._apply(row, z) for row, z in zip(rows, noise)]
                matrices.append(np.asarray([[r[n] for n in data.feature_names] for r in result], dtype=np.float32))
                byte_costs.append(np.asarray([sum(r[f'totlen_{d}_pkts'] for d in DIRECTIONS) for r in result]) / original_bytes - 1)
                times = np.asarray([r['flow_duration'] for r in result])
                time_costs.append(np.divide(times - original_time, original_time,
                    out=np.zeros_like(times), where=original_time > 0))
            encoded = frozen.scaler.transform(np.concatenate(matrices))
            probabilities = frozen.model.predict_probabilities(encoded, 512).reshape(len(controls), len(data), -1)
            true_probability = probabilities[:, np.arange(len(data)), data.labels]
            rivals = probabilities.copy()
            rivals[:, np.arange(len(data)), data.labels] = -1
            margin = true_probability - rivals.max(axis=2)
            correct = probabilities.argmax(axis=2) == data.labels
            adversarial = correct + .1 * np.maximum(margin, -.05)
            byte_costs, time_costs = np.asarray(byte_costs), np.asarray(time_costs)
            objective = ((adversarial + parameters['byte_penalty'] * byte_costs
                          + parameters['duration_penalty'] * time_costs) * weights).sum(axis=1)
            accuracy = correct @ weights
            return objective, accuracy, byte_costs, time_costs

        initial = [identity]
        if isinstance(cached, dict) and cached.get('version') == cls.VERSION:
            cls.verify(cached, cached.get('signature'))
            center = np.asarray(cached['controls'], dtype=np.float64)
            initial.append(center)
        score, accuracy, bytes_, times = assess(initial, validation, val_rows, val_noise, val_weights)
        chosen = min(range(len(initial)), key=lambda i: (accuracy[i], score[i]))
        best = initial[chosen].copy()
        best_key = (float(accuracy[chosen]), float(score[chosen]))
        checkpoint = dict(iteration=0, objective=best_key[1], accuracy=best_key[0],
                          mean_extra_byte_fraction=float(bytes_[chosen].mean()),
                          mean_span_fraction=float(times[chosen].mean()))
        population = parameters['population']
        for iteration in range(parameters['iterations']):
            indices = []
            for label in range(len(training.class_names)):
                pool = np.flatnonzero(training.labels == label)
                count = parameters['batch_size'] // len(training.class_names) + (label < parameters['batch_size'] % len(training.class_names))
                p = train_weights[pool];p = p / p.sum()
                indices.extend(rng.choice(pool, count, replace=True, p=p))
            indices = np.asarray(indices)
            batch = training.subset(indices)
            candidates = rng.normal(center, spread, size=(population, 14))
            candidates[0] = center
            candidates[1] = best
            # Keep exploring beyond the incumbent's local neighborhood.
            candidates[2:2 + population // 4] = rng.normal(initial_center, 2.5,
                                                          size=(population // 4, 14))
            scores, _, _, _ = assess(candidates, batch, [train_rows[i] for i in indices],
                                    train_noise[indices], np.full(len(indices), 1 / len(indices)))
            elite = candidates[np.argsort(scores, kind='stable')[:max(2, population // 4)]]
            center = .6 * center + .4 * elite.mean(axis=0)
            spread = np.maximum(.05, .7 * spread + .3 * elite.std(axis=0))
            candidate = candidates[int(np.argmin(scores))]
            score, accuracy, bytes_, times = assess([candidate], validation, val_rows, val_noise, val_weights)
            key = (float(accuracy[0]), float(score[0]))
            if key < best_key:
                best, best_key = candidate.copy(), key
                checkpoint = dict(iteration=iteration + 1, objective=key[1], accuracy=key[0],
                                  mean_extra_byte_fraction=float(bytes_[0].mean()),
                                  mean_span_fraction=float(times[0].mean()))
            if (iteration + 1) % 10 == 0 or iteration + 1 == parameters['iterations']:
                print(f'  Blanket iteration {iteration + 1}/{parameters["iterations"]}: validation accuracy={checkpoint["accuracy"]:.4f}, mean extra bytes={checkpoint["mean_extra_byte_fraction"]:.2%}', flush=True)
        fingerprint = hashlib.sha256(signature.encode() + best.tobytes()).hexdigest()
        return dict(version=cls.VERSION, signature=signature, fingerprint=fingerprint,
                    controls=best.tolist(), validation=checkpoint,
                    unique_training_flows=len(training), assumptions=cls.ASSUMPTIONS)
