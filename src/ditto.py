"""Ditto-inspired isolated carrier windows, computed from aggregate queue envelopes.

Training packet-size percentiles define a repeating pattern (NDSS 2022, §V).
Both directions send at a fixed common packet rate, with chaff for empty queues.
Upper hulls of queue arrivals give exact drain times without replaying packets.
Carrier-window boundaries remain visible; this is not continuous WAN anonymity.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .transform import (DIRECTIONS, FEATURE_NAMES, SizeMoments, _context_rows,
                        flow_statistics, validate_policy_parameters)


class Ditto:
    name = 'ditto'
    VERSION = 'ditto-carrier-flow-v1'
    LENGTHS = (1, 3, 6)
    required = {'pattern_length': 'positive_count', 'target_utilization': 'fraction'}
    ASSUMPTIONS = {
        'scope': 'one isolated bidirectional carrier per source row; continuous shared-WAN anonymity is not modeled',
        'sizes': 'IP-equivalent observer frame lengths; Ethernet/MACsec/encapsulation bytes excluded',
        'pattern': 'training-only packet-weighted percentiles, preserving duplicate states; maximum state is 1500',
        'rate': 'estimated from the packet-weighted 95th percentile of training queue demand, not measured link capacity',
        'timing': 'fixed slots; backward phase is half a slot; unbounded FIFO queues per state; no TCP feedback or hardware jitter',
        'window': 'both directions send full cycles until every real packet has departed; start and stop remain visible',
        'bulk': 'all carrier frames have payload and directions alternate; forward bulk fields are zero',
        'fct': 'open-loop queue completion delay; real network FCT is not measured',
    }

    @classmethod
    def validate(cls, parameters, source):
        validate_policy_parameters(cls.name, parameters, cls.required, {}, source, byte_fraction_kind=None)
        if parameters['pattern_length'] not in cls.LENGTHS or parameters['target_utilization'] < .01:
            raise ValueError('Ditto needs pattern_length in (1, 3, 6) and target_utilization in [0.01, 1]')
        return parameters

    @classmethod
    def pattern_key(cls, pattern, seed):
        return hashlib.sha256(json.dumps([cls.VERSION, [int(s) for s in pattern], seed]).encode()).hexdigest()

    @staticmethod
    def _offset(key, flow_key, direction, state, copies):
        digest = hashlib.sha256(f'{key}:{flow_key}:{direction}:{state}'.encode()).digest()
        return int.from_bytes(digest[:8], 'big') % copies

    @classmethod
    def queue_counts(cls, histogram, pattern, key, flow_key):
        """Balance equal-sized states without counting oversampled training rows."""
        counts = np.zeros((2, len(pattern)), dtype=np.int64)
        mapped = np.searchsorted(pattern, histogram[:, 0])
        for state in np.unique(mapped):
            copies = int(np.count_nonzero(pattern == pattern[state]))
            for direction in (0, 1):
                total = int(histogram[mapped == state, direction + 1].sum())
                counts[direction, state:state + copies] = total // copies
                offset = cls._offset(key, flow_key, direction, int(state), copies)
                counts[direction, state + (np.arange(total % copies) + offset) % copies] += 1
        return counts

    @staticmethod
    def _histogram(row):
        table = np.asarray(row.get('ditto_histogram'))
        if (table.ndim != 2 or table.shape[1] != 3 or not len(table)
                or not np.issubdtype(table.dtype, np.integer)
                or np.any((table[:, 0] < 1) | (table[:, 0] > 1500))
                or np.any(np.diff(table[:, 0]) <= 0) or np.any(table[:, 1:] < 0)
                or np.any(table[:, 1:].sum(axis=1) == 0)):
            raise ValueError('Ditto needs a sparse packet-size histogram in flow_context')
        for direction, name in enumerate(DIRECTIONS):
            if (table[:, direction + 1].sum() != row[f'tot_{name}_pkts']
                    or table[:, 0] @ table[:, direction + 1] != row[f'totlen_{name}_pkts']):
                raise ValueError('Ditto histogram does not match original flow counts and byte totals')
        return table

    @classmethod
    def prepare_context(cls, times, directions, sizes, flow_key, patterns, seed=42):
        """Prepare rate-independent queue hulls once; timestamps are integer ns.

        For queue arrivals t[j], only vertices maximizing t[j] - j*cycle
        affect the last departure. Interior points can therefore be discarded.
        """
        t, d, s = (np.asarray(v) for v in (times, directions, sizes))
        if (not len(t) or any(v.ndim != 1 or v.shape != t.shape
                or not np.issubdtype(v.dtype, np.integer) for v in (t, d, s))
                or np.any(np.diff(t) < 0) or not np.isin(d, (0, 1)).all()
                or np.any((s < 1) | (s > 1500))):
            raise ValueError('Ditto preparation needs ordered integer packet times, directions and sizes')
        t = t - t[0]
        envelopes = {}
        for states in patterns:
            pattern = np.asarray(states, dtype=np.int64)
            if (len(pattern) not in cls.LENGTHS or np.any(np.diff(pattern) < 0)
                    or pattern[0] < 64 or pattern[-1] != 1500):
                raise ValueError('Invalid Ditto preparation pattern')
            key = cls.pattern_key(pattern.tolist(), seed)
            queues = []
            for direction in (0, 1):
                members = d == direction
                tt = t[members]
                assigned = np.searchsorted(pattern, s[members])
                for state in np.unique(assigned):
                    selected = np.flatnonzero(assigned == state)
                    copies = int(np.count_nonzero(pattern == pattern[state]))
                    offset = cls._offset(key, flow_key, direction, int(state), copies)
                    assigned[selected] = state + (np.arange(len(selected)) + offset) % copies
                for state in range(len(pattern)):
                    vertices = []
                    for j, timestamp in enumerate(tt[assigned == state]):
                        timestamp = int(timestamp)
                        while len(vertices) > 1:
                            a, b = vertices[-2:]
                            if (b[0] - a[0]) * (timestamp - b[1]) < (b[1] - a[1]) * (j - b[0]):
                                break
                            vertices.pop()
                        vertices.append((j, timestamp))
                    queues.append(np.asarray(vertices, dtype=np.int64).reshape(-1, 2))
            envelopes[key] = queues
        return dict(ditto_signature=f'{cls.VERSION}:{flow_key}', ditto_hulls=envelopes)

    @classmethod
    def _prepared(cls, data, context):
        rows = list(_context_rows(data, context))
        _, first = np.unique(data.row_indices, return_index=True)
        rows = [rows[i] for i in first]
        for row in rows:
            cls._histogram(row)
        return data.subset(first), rows

    @classmethod
    def signature(cls, data, rows, parameters, seed):
        digest = hashlib.sha256(json.dumps([cls.VERSION, 'packet-weighted-p95', parameters,
                                            seed, data.source_fingerprint], sort_keys=True).encode())
        for array in (data.row_indices, data.record_ids, np.asarray([r['flow_duration'] for r in rows])):
            digest.update(np.ascontiguousarray(array).tobytes())
        for row in rows:
            table = cls._histogram(row)
            digest.update(str(table.shape).encode())
            digest.update(table.astype('<i8', copy=False).tobytes())
        return digest.hexdigest()

    @staticmethod
    def _fingerprint(artifact):
        fields = [artifact[n] for n in ('signature', 'pattern', 'slot_ns', 'seed', 'parameters')]
        return hashlib.sha256(json.dumps(fields).encode()).hexdigest()

    @classmethod
    def verify(cls, artifact, signature):
        if (not isinstance(artifact, dict) or artifact.get('version') != cls.VERSION
                or not isinstance(signature, str) or len(signature) != 64 or artifact.get('signature') != signature):
            raise ValueError('Ditto policy is missing or stale; run packmorph-obfuscate ditto')
        cls.validate(artifact.get('parameters'), 'Ditto artifact')
        pattern = artifact.get('pattern')
        if (not isinstance(pattern, list) or len(pattern) not in cls.LENGTHS
                or any(type(s) is not int or not 64 <= s <= 1500 for s in pattern)
                or pattern != sorted(pattern) or pattern[-1] != 1500
                or type(artifact.get('slot_ns')) is not int or artifact['slot_ns'] < 2 or artifact['slot_ns'] % 2
                or type(artifact.get('seed')) is not int or artifact['seed'] < 0
                or artifact.get('fingerprint') != cls._fingerprint(artifact)):
            raise ValueError('Invalid learned Ditto pattern or slot period')

    @classmethod
    def fit(cls, training, context, parameters, seed=42, cached=None):
        """Learn the pattern and virtual carrier rate without labels or model queries."""
        cls.validate(parameters, 'Ditto fitting')
        data, rows = cls._prepared(training, context)
        signature = cls.signature(data, rows, parameters, seed)
        if isinstance(cached, dict) and cached.get('signature') == signature:
            cls.verify(cached, signature)
            return cached
        histogram = np.zeros(1501, dtype=np.int64)
        for row in rows:
            table = cls._histogram(row)
            histogram[table[:, 0]] += table[:, 1:].sum(axis=1)
        cumulative = np.cumsum(histogram)
        length = parameters['pattern_length']
        pattern = [max(64, int(np.searchsorted(cumulative, cumulative[-1] * (i + 1) / length)))
                   for i in range(length)]
        pattern[-1] = 1500
        states = np.asarray(pattern)
        key = cls.pattern_key(pattern, seed)
        demands, weights = [], []
        for row in rows:
            if row['flow_duration'] > 0:
                counts = cls.queue_counts(cls._histogram(row), states, key, row['flow_key'])
                demands.append(length * counts.max() / row['flow_duration'])
                weights.append(int(counts.sum()))
        if not demands:
            raise ValueError('Ditto cannot estimate a carrier rate from zero-duration training flows alone')
        order = np.argsort(demands)
        cumulative = np.cumsum(np.asarray(weights)[order])
        rate = float(np.asarray(demands)[order[np.searchsorted(cumulative, .95 * cumulative[-1])]])
        rate /= parameters['target_utilization']
        slot = max(2, int(round(1e9 / rate / 2)) * 2)
        artifact = dict(version=cls.VERSION, signature=signature, parameters=dict(parameters), pattern=pattern,
                        slot_ns=slot, seed=seed, assumptions=cls.ASSUMPTIONS)
        artifact['fingerprint'] = cls._fingerprint(artifact)
        cls.verify(artifact, signature)
        return artifact

    def __init__(self, parameters, artifact, columns=FEATURE_NAMES, seed=42):
        self.validate(parameters, 'Ditto')
        self.verify(artifact, artifact.get('signature') if isinstance(artifact, dict) else None)
        if (artifact['parameters'] != parameters or len(artifact['pattern']) != parameters['pattern_length']
                or artifact['seed'] != seed):
            raise ValueError('Ditto policy must match the configured pattern length and seed')
        self.parameters, self.artifact = parameters, artifact
        self.pattern = np.asarray(artifact['pattern'], dtype=np.int64)
        self.key = self.pattern_key(artifact['pattern'], seed)
        self.columns, self.seed = tuple(columns), seed
        self.reset()

    def reset(self):
        self._costs = np.zeros(10, dtype=np.float64)

    def __call__(self, row):
        table = self._histogram(row)
        envelopes = row.get('ditto_hulls')
        if (row.get('ditto_signature') != f"{self.VERSION}:{row['flow_key']}"
                or not isinstance(envelopes, dict) or self.key not in envelopes):
            raise ValueError('Ditto needs packet-derived queue envelopes for the training pattern in flow_context')
        queues = envelopes[self.key]
        length, slot = len(self.pattern), self.artifact['slot_ns']
        counts = self.queue_counts(table, self.pattern, self.key, row['flow_key'])
        if len(queues) != 2 * length:
            raise ValueError('Invalid Ditto queue envelope count')
        cycles, completion = 1, 0
        for index, raw in enumerate(queues):
            vertices = np.asarray(raw)
            count = int(counts.flat[index])
            if (vertices.ndim != 2 or vertices.shape[1] != 2 or not np.issubdtype(vertices.dtype, np.integer)
                    or np.any(vertices < 0) or (count and (not len(vertices) or vertices[0, 0] != 0
                        or vertices[-1, 0] != count - 1 or np.any(np.diff(vertices[:, 0]) <= 0)
                        or np.any(np.diff(vertices[:, 1]) < 0))) or (not count and len(vertices))):
                raise ValueError('Invalid Ditto queue arrival envelope')
            if not count:
                continue
            direction, state = divmod(index, length)
            phase = state * slot + direction * (slot // 2)
            maximum = max(int(t) - int(j) * length * slot for j, t in vertices)
            last = count - 1 + max(0, (maximum - phase + length * slot - 1) // (length * slot))
            cycles = max(cycles, last + 1)
            completion = max(completion, last * length * slot + phase)
        duration = ((cycles * length - 1) * slot + slot // 2) / 1e9
        summaries = [SizeMoments(cycles * length, cycles * int(self.pattern.sum()),
                                 float(self.pattern.var()), int(self.pattern[0]), 1500) for _ in DIRECTIONS]
        values = {n: float(row[n]) for n in FEATURE_NAMES}
        result = flow_statistics(values, summaries, duration, timing='uniform')
        for n in ('fwd_byts_b_avg', 'fwd_pkts_b_avg', 'fwd_blk_rate_avg'):
            result[n] = 0.
        before = sum(values[f'totlen_{d}_pkts'] for d in DIRECTIONS)
        after = sum(s.total for s in summaries)
        padding = int((counts * self.pattern).sum()) - before
        chaff = 2 * cycles * length - int(counts.sum())
        delay = max(0., completion / 1e9 - values['flow_duration'])
        timed = values['flow_duration'] > 0
        self._costs[:9] += (1, before, after, padding, chaff, duration - values['flow_duration'],
                           delay, delay / values['flow_duration'] if timed else 0., int(timed))
        self._costs[9] = max(self._costs[9], delay)
        return {n: result[n] for n in self.columns}

    @property
    def overhead(self):
        count, before, after, padding, chaff, span, delay, relative, timed, maximum = self._costs
        return dict(extra_byte_fraction=float((after - before) / before) if before else 0.,
                    padding_byte_fraction=float(padding / before) if before else 0.,
                    chaff_byte_fraction=float((after - before - padding) / before) if before else 0.,
                    mean_dummy_packets=float(chaff / count) if count else 0.,
                    mean_added_span_s=float(span / count) if count else 0.,
                    mean_completion_delay_s=float(delay / count) if count else 0.,
                    mean_completion_fraction=float(relative / timed) if timed else 0.,
                    max_completion_delay_s=float(maximum), zero_duration_flows=int(count - timed))
