"""MiniPatch-inspired, service-specific patches for aggregate flow classifiers.

Dual annealing minimizes training accuracy and margin; validation chooses a patch.
Locations use a proportional grid and align to the next same-direction burst
tail. Packet-derived summaries give exact aggregate updates for simultaneous
tail insertions. This is an offline flow adaptation, not a Tor WF reproduction.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .transform import (DIRECTIONS, FEATURE_NAMES, SizeMoments, _context_rows,
                        flow_statistics, forward_bulk_statistics, payload_segments,
                        validate_policy_parameters)


class Minipatch:
    name = 'minipatch'
    VERSION = 'minipatch-learned-flow-v1'
    GRID = 17
    required = {'max_patches': 'positive_count', 'max_patch_packets': 'positive_count',
                'iterations': 'positive_count', 'max_queries': 'positive_count',
                'fit_flows_per_service': 'positive_count', 'success_threshold': 'fraction',
                'dummy_bytes': 'size'}
    ASSUMPTIONS = {
        'scope': 'service-specific aggregate adaptation; sender must know its service',
        'learning': 'black-box training confidence; validation selection; no test fitting',
        'location': '17 proportional positions in the first 5000 packets, aligned forward to same-direction burst tails',
        'size': 'fixed IP dummy length; original packets and payload sizes are preserved',
        'timing': 'dummies share burst-tail timestamps; link serialization, queueing and real FCT are unavailable',
    }

    @classmethod
    def validate(cls, parameters, source):
        validate_policy_parameters(cls.name, parameters, cls.required, {}, source, byte_fraction_kind=None)
        if (parameters['max_patches'] > 8 or parameters['max_patch_packets'] > 64
                or parameters['dummy_bytes'] < 64 or not parameters['success_threshold']):
            raise ValueError('MiniPatch needs at most 8 patches of 64 packets, dummy_bytes >= 64 and a positive threshold')
        return parameters

    @classmethod
    def prepare_context(cls, times, directions, payloads, flow_key):
        """Compress packets into candidate tails and mergeable payload groups once.

        Segment rows: region, direction, count, payload bytes, first/last ns.
        Direction -1 summarizes untouched interior groups: region, -1, number
        of forward bulks, payload bytes, packet count, summed bulk duration ns.
        No packet vectors or precomputed classifier inputs are retained.
        """
        t, d, p = (np.asarray(v) for v in (times, directions, payloads))
        if (not len(t) or any(v.ndim != 1 or v.shape != t.shape
                or not np.issubdtype(v.dtype, np.integer) for v in (t, d, p))
                or np.any(np.diff(t) < 0) or not np.isin(d, (0, 1)).all() or np.any(p < 0)):
            raise ValueError('MiniPatch preparation needs ordered integer packet times, directions and payloads')
        t = t - t[0]
        ends = np.r_[np.flatnonzero(np.diff(d) != 0), len(t) - 1]
        probes = np.rint(np.linspace(0, min(len(t), 5000) - 1, cls.GRID)).astype(np.int64)
        resolved = np.full((2, cls.GRID), -1, dtype=np.int64)
        for direction in (0, 1):
            tails = ends[d[ends] == direction]
            matches = np.searchsorted(tails, probes)
            valid = matches < len(tails)
            resolved[direction, valid] = tails[matches[valid]]
        cuts = np.unique(resolved[resolved >= 0])
        tails = np.zeros((2, cls.GRID, 2), dtype=np.float64)
        tails[:, :, 0] = -1
        valid = resolved >= 0
        tails[valid, 0] = np.searchsorted(cuts, resolved[valid])
        tails[valid, 1] = t[resolved[valid]]
        segments = payload_segments(t, d, p, np.searchsorted(cuts, np.arange(len(t))))
        iat = np.zeros((2, 2), dtype=np.float64)
        for direction in (0, 1):
            gaps = np.diff(t[d == direction]).astype(np.float64) / 1e9
            iat[direction] = gaps.sum(), np.square(gaps).sum()
        return dict(minipatch_signature=f'{cls.VERSION}:{flow_key}', minipatch_tails=tails,
                    minipatch_segments=segments,
                    minipatch_iat=iat)

    @classmethod
    def _metadata(cls, row):
        if row.get('minipatch_signature') != f"{cls.VERSION}:{row['flow_key']}":
            raise ValueError('MiniPatch needs matching packet-derived burst metadata in flow_context')
        tails, segments, iat = [np.asarray(row.get(n)) for n in
                               ('minipatch_tails', 'minipatch_segments', 'minipatch_iat')]
        if (tails.shape != (2, cls.GRID, 2) or iat.shape != (2, 2)
                or segments.ndim != 2 or segments.shape[1] != 6
                or any(not np.issubdtype(v.dtype, np.number) for v in (tails, segments, iat))
                or any(not np.isfinite(v).all() for v in (tails, segments, iat))
                or np.any(tails[:, :, 0] < -1) or np.any(tails[:, :, 1] < 0)
                or np.any(tails[:, :, 0] != np.floor(tails[:, :, 0]))
                or np.any(iat < 0) or np.any(segments[:, 0] < 0) or np.any(segments[:, 2:] < 0)
                or not np.isin(segments[:, 1], (-1, 0, 1)).all()
                or not 28 <= row.get('dummy_header_bytes', 0) <= 60):
            raise ValueError('Invalid packet-derived MiniPatch metadata')
        return tails, segments, iat

    def __init__(self, parameters, artifact=None, columns=FEATURE_NAMES, seed=42):
        self.validate(parameters, 'MiniPatch')
        self.parameters, self.artifact = parameters, artifact
        self.columns, self.seed = tuple(columns), seed
        self.reset()

    def reset(self):
        self._totals = np.zeros(5, dtype=np.float64)

    def _apply(self, row, patches):
        tails, segments, iat = self._metadata(row)
        insertions = {}
        for position, length in patches:
            if not length:
                continue
            direction = int(length < 0)
            region, timestamp = tails[direction, position]
            if region >= 0 and region not in insertions:
                insertions[region] = (direction, abs(length), timestamp)
        if not insertions:
            return {n: float(row[n]) for n in FEATURE_NAMES}
        values = {n: float(row[n]) for n in FEATURE_NAMES}
        summaries = [SizeMoments.from_values(values, d) for d in DIRECTIONS]
        size = self.parameters['dummy_bytes']
        dummy = np.asarray([[region + .5, d, k, k * (size - row['dummy_header_bytes']), t, t]
                            for region, (d, k, t) in insertions.items()])
        for direction, summary in enumerate(summaries):
            count = sum(k for d, k, _ in insertions.values() if d == direction)
            summary.merge(SizeMoments.constant(count, size))
        result = flow_statistics(values, summaries, values['flow_duration'])
        for direction, name in enumerate(DIRECTIONS):
            intervals = summaries[direction].count - 1
            total, squared = iat[direction]
            variance = squared / intervals - (total / intervals) ** 2 if intervals > 0 else 0.
            result[f'{name}_iat_std'] = np.sqrt(max(0., variance))
        events = np.concatenate((segments, dummy))
        result.update(forward_bulk_statistics(events))
        return result

    def __call__(self, row, service):
        if self.artifact is None or service not in self.artifact['patches']:
            raise ValueError('MiniPatch needs a learned patch for the known source service')
        result = self._apply(row, self.artifact['patches'][service])
        before = sum(float(row[f'totlen_{d}_pkts']) for d in DIRECTIONS)
        after = sum(result[f'totlen_{d}_pkts'] for d in DIRECTIONS)
        added = sum(result[f'tot_{d}_pkts'] - float(row[f'tot_{d}_pkts']) for d in DIRECTIONS)
        packets = sum(float(row[f'tot_{d}_pkts']) for d in DIRECTIONS)
        self._totals += (1, before, after, added, packets)
        return {n: result[n] for n in self.columns}

    @property
    def overhead(self):
        count, before, after, dummy, packets = self._totals
        return dict(extra_byte_fraction=float(after / before - 1) if before else 0.,
                    extra_packet_fraction=float(dummy / packets) if packets else 0.,
                    mean_dummy_packets=float(dummy / count) if count else 0.,
                    mean_added_span_s=0., max_added_span_s=0.)

    @classmethod
    def _prepared(cls, data, context):
        rows = list(_context_rows(data, context))
        _, first = np.unique(data.row_indices, return_index=True)
        rows = [rows[i] for i in first]
        for row in rows:
            cls._metadata(row)
        return data.subset(first), rows

    @classmethod
    def signature(cls, training, train_rows, validation, val_rows, frozen, parameters, seed):
        digest = hashlib.sha256(json.dumps([cls.VERSION, 'accuracy-first-v1', parameters, seed,
            training.feature_names, training.class_names, training.source_fingerprint], sort_keys=True).encode())
        arrays = list(frozen.model.embedding.get_weights()) + [frozen.model.prototypes,
            frozen.model.diagonal_variances, frozen.scaler.mean, frozen.scaler.scale]
        for data, rows in ((training, train_rows), (validation, val_rows)):
            arrays += [data.features, data.labels, data.captures, data.row_indices, data.record_ids,
                       np.asarray([[r[n] for n in FEATURE_NAMES] for r in rows]),
                       np.asarray([r['dummy_header_bytes'] for r in rows])]
            for row in rows:
                arrays.extend(cls._metadata(row))
        for array in arrays:
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    @classmethod
    def verify(cls, artifact, signature, classes, parameters):
        if (not isinstance(artifact, dict) or artifact.get('version') != cls.VERSION
                or artifact.get('signature') != signature or set(artifact.get('patches', {})) != set(classes)):
            raise ValueError('MiniPatch policy is missing or stale; run packmorph-obfuscate minipatch')
        for patches in artifact['patches'].values():
            if (not isinstance(patches, list) or len(patches) > parameters['max_patches']
                    or any(not isinstance(p, list) or len(p) != 2 or any(type(v) is not int for v in p)
                           or not 0 <= p[0] < cls.GRID or abs(p[1]) > parameters['max_patch_packets'] for p in patches)):
                raise ValueError('Invalid learned MiniPatch vector')
        fingerprint = hashlib.sha256(signature.encode() + json.dumps(artifact['patches'], sort_keys=True).encode()).hexdigest()
        if artifact.get('fingerprint') != fingerprint:
            raise ValueError('Learned MiniPatch policy failed integrity validation')

    @classmethod
    def fit(cls, training, train_context, validation, val_context, frozen, parameters, seed=42, cached=None):
        """Fit once using black-box probabilities; never inspect test samples.

        Training copies are deduplicated. Fit subsets balance captures within
        each service. Successful training bounds are halved as in Algorithm 3;
        validation chooses accuracy, then protection per inserted packet.
        Identity and existing patches remain candidates. Several training search
        outcomes are retained so validation can select a generalizing patch.
        """
        from scipy.optimize import dual_annealing

        cls.validate(parameters, 'MiniPatch fitting')
        if (training.feature_names != validation.feature_names or training.class_names != validation.class_names
                or training.source_fingerprint != validation.source_fingerprint):
            raise ValueError('MiniPatch training and validation schemas must match')
        for field in ('captures', 'row_indices', 'record_ids'):
            if np.intersect1d(getattr(training, field), getattr(validation, field)).size:
                raise ValueError(f'MiniPatch training and validation have overlapping {field}')
        training, train_rows = cls._prepared(training, train_context)
        validation, val_rows = cls._prepared(validation, val_context)
        signature = cls.signature(training, train_rows, validation, val_rows, frozen, parameters, seed)
        if isinstance(cached, dict) and cached.get('signature') == signature:
            cls.verify(cached, signature, training.class_names, parameters)
            return cached
        if isinstance(cached, dict) and cached.get('version') == cls.VERSION:
            cls.verify(cached, cached.get('signature'), training.class_names, parameters)
        rng = np.random.default_rng(seed)
        policy = cls(parameters)

        def predict(matrix):
            return frozen.model.predict_probabilities(frozen.scaler.transform(matrix), 512)

        train_correct = predict(training.features).argmax(axis=1) == training.labels
        val_correct = predict(validation.features).argmax(axis=1) == validation.labels
        learned, diagnostics = {}, {}
        for label, service in enumerate(training.class_names):
            indices = np.flatnonzero((training.labels == label) & train_correct)
            if len(indices) > parameters['fit_flows_per_service']:
                _, inverse, counts = np.unique(training.captures[indices], return_inverse=True, return_counts=True)
                weights = 1 / counts[inverse]
                indices = rng.choice(indices, parameters['fit_flows_per_service'], replace=False, p=weights / weights.sum())
            fitting = [train_rows[i] for i in indices]
            holdout = [val_rows[i] for i in np.flatnonzero(validation.labels == label)]
            correct = val_correct[validation.labels == label]
            candidates = [[]]
            if isinstance(cached, dict) and cached.get('version') == cls.VERSION:
                candidates.append(cached['patches'][service])
            nodes, seen, queries = [(parameters['max_patches'], parameters['max_patch_packets'])], set(), 0

            def assess(patches, rows, weights=None):
                results = [policy._apply(r, patches) for r in rows]
                matrix = np.asarray([[r[n] for n in training.feature_names] for r in results], dtype=np.float32)
                probabilities = predict(matrix)
                success = probabilities.argmax(axis=1) != label
                rivals = probabilities.copy()
                rivals[:, label] = -1
                margin = probabilities[:, label] - rivals.max(axis=1)
                loss = float(np.average(~success, weights=weights)
                             + .1 * np.average(np.maximum(margin, -.05), weights=weights))
                added = np.asarray([sum(r[f'tot_{d}_pkts'] - float(old[f'tot_{d}_pkts']) for d in DIRECTIONS)
                                    for r, old in zip(results, rows)])
                return loss, success, float(added.mean())

            while nodes and fitting:
                patches, maximum = nodes.pop(0)
                if (patches, maximum) in seen:
                    continue
                seen.add((patches, maximum))
                cache = {}
                _, inverse, counts = np.unique(training.captures[indices], return_inverse=True, return_counts=True)
                weights = 1 / counts[inverse]

                def decode(vector):
                    a = np.asarray(vector).reshape(-1, 2)
                    return tuple((min(cls.GRID - 1, int(p)), int(np.clip(k, -maximum, maximum))) for p, k in a)

                def objective(vector):
                    key = decode(vector)
                    if key not in cache:
                        cache[key] = assess(key, fitting, weights)
                    return cache[key][0]

                def stop(vector, value, context):
                    return bool(cache[decode(vector)][1].mean() >= parameters['success_threshold'])

                result = dual_annealing(objective, [(0, cls.GRID - 1e-6), (-maximum - .999, maximum + .999)] * patches,
                    maxiter=parameters['iterations'], maxfun=parameters['max_queries'], accept=-1000,
                    rng=rng, callback=stop)
                best = decode(result.x)
                for vector in sorted(cache, key=lambda key: cache[key][0])[:3]:
                    candidate = [list(p) for p in vector if p[1]]
                    if candidate not in candidates:
                        candidates.append(candidate)
                rate = float(cache[best][1].mean())
                queries += len(cache)
                print(f'  MiniPatch {service}: bounds={patches}x{maximum}, train success={rate:.2%}, queries={len(cache)}', flush=True)
                if rate >= parameters['success_threshold']:
                    if patches > 1:
                        nodes.append((max(1, patches // 2), maximum))
                    if maximum > 1:
                        nodes.append((patches, max(1, maximum // 2)))
            best_key = None
            for patches in candidates:
                confidence, success, added = assess(patches, holdout)
                rate = float(success[correct].mean()) if correct.any() else 0.
                accuracy = float(1 - success.mean())
                key = (accuracy, -rate / max(added, 1), confidence, added)
                if best_key is None or key < best_key:
                    best_key, chosen = key, patches
                    stats = dict(accuracy=accuracy, protected_correct_fraction=rate,
                                 mean_dummy_packets=added, train_queries=queries)
            learned[service], diagnostics[service] = chosen, stats
            print(f'  MiniPatch {service}: selected {len(chosen)} patches; validation accuracy={stats["accuracy"]:.2%}', flush=True)
        artifact = dict(version=cls.VERSION, signature=signature, patches=learned,
                        validation=diagnostics, assumptions=cls.ASSUMPTIONS)
        artifact['fingerprint'] = hashlib.sha256(signature.encode() + json.dumps(learned, sort_keys=True).encode()).hexdigest()
        cls.verify(artifact, signature, training.class_names, parameters)
        return artifact
