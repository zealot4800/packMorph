"""Minos-inspired single-flow morphing, not encrypted-tunnel flow mixing.

Section 8's Bernoulli insertion is limited to a packet-count front window.
Optional bounded random padding reflects the separate IoT use case (Appendix A).
One-time packet-derived coefficients preserve sizes, timing and bulk membership;
routine transformations operate entirely on the stored flow context.
An optional backward dummy size extends the aggregate adaptation without
changing the seeded insertion positions or directions.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .transform import DIRECTIONS, FEATURE_NAMES, SizeMoments, flow_statistics, validate_policy_parameters


class Minos:
    name = 'minos'
    VERSION = 'minos-front-flow-v1'
    WINDOWS = (500, 1000)
    PROBABILITIES = (0., .3, .6)
    PADDING_MAXIMA = (0, 16, 32)
    required = {'front_window': 'positive_count', 'insertion_probability': 'fraction',
                'padding_max_bytes': 'size_zero', 'dummy_bytes': 'size'}
    optional = {'backward_dummy_bytes': 'size'}
    ASSUMPTIONS = {
        'scope': 'single-flow traffic-morphing ablation; concurrent tunnel mixing is unavailable',
        'proxy': 'header encryption and the proxy eight-byte header are not simulated',
        'dummy': 'seeded Bernoulli front insertion; random direction; timestamp of triggering packet',
        'dummy_size': 'fixed IP length per direction; backward length defaults to forward length',
        'padding': 'seeded uniform integer padding, clipped per packet to the 1500-byte IP limit',
        'duration': 'original timestamp span; link capacity, queues and real FCT are not simulated',
    }

    @classmethod
    def validate(cls, parameters, source):
        validate_policy_parameters(cls.name, parameters, cls.required, cls.optional, source, byte_fraction_kind=None)
        if (parameters['front_window'] not in cls.WINDOWS
                or parameters['insertion_probability'] not in cls.PROBABILITIES
                or parameters['padding_max_bytes'] not in cls.PADDING_MAXIMA
                or parameters['dummy_bytes'] < 64
                or parameters.get('backward_dummy_bytes', parameters['dummy_bytes']) < 64):
            raise ValueError('Minos requires a supported front/padding profile and dummy_bytes >= 64')
        return parameters

    @classmethod
    def _signature(cls, key, seed):
        return hashlib.sha256(f'{cls.VERSION}:{seed}:{key}'.encode()).hexdigest()

    @classmethod
    def prepare_context(cls, times, directions, sizes, payloads, flow_key, seed=42):
        """Measure reusable coefficients once; times are integer nanoseconds.

        No labels or classifier outputs are used. Dummy sizes remain configurable
        at runtime; only their locations, directions and bulk membership are fixed.
        """
        arrays = [np.asarray(v) for v in (times, directions, sizes, payloads)]
        if (not len(arrays[0]) or any(v.ndim != 1 or v.shape != arrays[0].shape for v in arrays)
                or any(not np.issubdtype(v.dtype, np.integer) for v in arrays)
                or np.any(np.diff(arrays[0]) < 0) or not np.isin(arrays[1], (0, 1)).all()
                or np.any((arrays[2] < 0) | (arrays[2] > 1500)) or np.any(arrays[3] < 0)):
            raise ValueError('Minos preparation needs ordered integer packet times, directions, sizes and payloads')
        times, directions, sizes, payloads = arrays
        times = times - times[0]
        signature = cls._signature(flow_key, seed)
        rng = np.random.default_rng(int(signature[:16], 16))
        random_padding = rng.random(len(times))
        front = min(max(cls.WINDOWS), len(times))
        triggers = rng.random(front)
        dummy_directions = rng.integers(0, 2, front)
        padding = np.zeros((3, 2, 5), dtype=np.float64)
        padded_payloads = []
        for pi, maximum in enumerate(cls.PADDING_MAXIMA):
            added = np.minimum(np.floor(random_padding * (maximum + 1)).astype(np.int64), 1500 - sizes)
            padded_payloads.append(payloads + added)
            for d in (0, 1):
                members = directions == d
                if not np.any(members):
                    continue
                s, a = sizes[members], added[members]
                padding[pi, d] = a.sum(), np.square(a).sum(), (s * a).sum(), (s + a).min(), (s + a).max()
        front_stats = np.zeros((2, 3, 2, 2), dtype=np.float64)
        bulk = np.zeros((2, 3, 3, 5), dtype=np.float64)
        for wi, window in enumerate(cls.WINDOWS):
            for qi, probability in enumerate(cls.PROBABILITIES):
                selected = np.flatnonzero(triggers[:min(window, front)] < probability)
                # Originals precede their injected dummy when timestamps are tied.
                ordering = np.r_[np.arange(len(times)) * 2, selected * 2 + 1]
                t = np.r_[times, times[selected]]
                d = np.r_[directions, dummy_directions[selected]]
                order = np.lexsort((ordering, t))
                t, d = t[order], d[order]
                original = np.r_[np.ones(len(times), dtype=bool), np.zeros(len(selected), dtype=bool)][order]
                for direction in (0, 1):
                    gaps = np.diff(t[d == direction]).astype(np.float64) / 1e9
                    front_stats[wi, qi, direction] = ((dummy_directions[selected] == direction).sum(),
                                                    gaps.std() if len(gaps) else 0.)
                for pi, payload in enumerate(padded_payloads):
                    p = np.r_[payload, np.zeros(len(selected), dtype=np.int64)][order]
                    keep = (p > 0) | ~original
                    bt, bd, bp, real = t[keep], d[keep], p[keep], original[keep]
                    if not len(bt):
                        continue
                    starts = np.r_[0, np.flatnonzero((np.diff(bd) != 0) | (np.diff(bt) > 1_000_000_000)) + 1]
                    ends = np.r_[starts[1:], len(bt)]
                    valid = ((ends - starts) >= 4) & (bd[starts] == 0)
                    if not np.any(valid):
                        continue
                    real_count = np.add.reduceat(real.astype(np.int64), starts)[valid].sum()
                    dummy_count = (ends - starts)[valid].sum() - real_count
                    bulk[wi, qi, pi] = (np.add.reduceat(bp, starts)[valid].sum(), real_count,
                        dummy_count, valid.sum(), (bt[ends[valid] - 1] - bt[starts[valid]]).sum() / 1e9)
        return dict(minos_signature=signature, minos_padding=padding, minos_front=front_stats, minos_bulk=bulk)

    def __init__(self, parameters, columns=FEATURE_NAMES, seed=42):
        self.validate(parameters, 'Minos')
        self.parameters = parameters
        self.columns, self.seed = tuple(columns), seed
        self.indices = (self.WINDOWS.index(parameters['front_window']),
                        self.PROBABILITIES.index(parameters['insertion_probability']),
                        self.PADDING_MAXIMA.index(parameters['padding_max_bytes']))
        self._totals = np.zeros(4, dtype=np.float64)

    def __call__(self, row):
        if row.get('minos_signature') != self._signature(row['flow_key'], self.seed):
            raise ValueError('Minos needs matching packet-derived front metadata in flow_context')
        metadata = [np.asarray(row.get(n)) for n in ('minos_padding', 'minos_front', 'minos_bulk')]
        for values, shape in zip(metadata, ((3, 2, 5), (2, 3, 2, 2), (2, 3, 3, 5))):
            if values.shape != shape or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Invalid packet-derived Minos front metadata')
        wi, qi, pi = self.indices
        values = {n: float(row[n]) for n in FEATURE_NAMES}
        summaries = [SizeMoments.from_values(values, d) for d in DIRECTIONS]
        sizes = (self.parameters['dummy_bytes'],
                 self.parameters.get('backward_dummy_bytes', self.parameters['dummy_bytes']))
        header = float(row.get('dummy_header_bytes', 0))
        if not 28 <= header <= 60:
            raise ValueError('Minos needs dummy_header_bytes from the packet extractor')
        for d, summary in enumerate(summaries):
            added, squared, product, minimum, maximum = metadata[0][pi, d]
            if summary.count:
                mean_pad = added / summary.count
                variance = (summary.variance + squared / summary.count - mean_pad ** 2
                            + 2 * (product / summary.count - summary.mean * mean_pad))
                summary.variance = max(0., variance)
                summary.total += added
                summary.minimum, summary.maximum = minimum, maximum
            summary.merge(SizeMoments.constant(metadata[1][wi, qi, d, 0], sizes[d]))
        result = flow_statistics(values, summaries, values['flow_duration'])
        for d, direction in enumerate(DIRECTIONS):
            result[f'{direction}_iat_std'] = metadata[1][wi, qi, d, 1]
        payload, real, dummy, groups, elapsed = metadata[2][wi, qi, pi]
        payload += dummy * (sizes[0] - header)
        result['fwd_byts_b_avg'] = payload / groups if groups else 0.
        result['fwd_pkts_b_avg'] = (real + dummy) / groups if groups else 0.
        result['fwd_blk_rate_avg'] = payload / elapsed if elapsed else 0.
        before = values['totlen_fwd_pkts'] + values['totlen_bwd_pkts']
        self._totals += (1, before, sum(s.total for s in summaries), sum(metadata[1][wi, qi, :, 0]))
        return {n: result[n] for n in self.columns}

    @property
    def report(self):
        count, before, after, dummy = self._totals
        return dict(generator=self.VERSION, assumptions=self.ASSUMPTIONS,
            overhead=dict(extra_byte_fraction=float(after / before - 1) if before else 0.,
                mean_dummy_packets=float(dummy / count) if count else 0.,
                mean_added_span_s=0., max_added_span_s=0.))
