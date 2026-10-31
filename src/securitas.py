"""Securitas-inspired flow policy, learned with an independent proxy classifier.

A compact categorical policy replaces the paper's packet-sequence LSTM: eight
blocks cover the first 64 original packet positions. Only forward packets are
edited. Packet-derived moments, payload groups and ICMP gap certificates make
routine feature updates independent of full packet traces.
The aggregate adaptation extends payload actions to 1440 bytes and optionally
repeats fakes; neither extension changes real-packet fragmentation eligibility.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .transform import (DIRECTIONS, FEATURE_NAMES, SizeMoments, _context_rows,
                        flow_statistics, forward_bulk_statistics, payload_segments,
                        validate_policy_parameters)


class Securitas:
    name = 'securitas'
    VERSION = 'securitas-proxy-flow-v1'
    WINDOW, BLOCKS = 64, 8
    # Extended aggregate adaptation; original 8..88-byte actions stay available.
    CHOICES = np.r_[np.arange(0, 89, 8), 128, 256, 512, 768, 1024, 1440].astype(np.int64)
    POINT = np.dtype([('position', 'i2'), ('marker', 'i2'), ('reply_marker', 'i2'),
        ('size', 'i2'), ('header', 'i2'), ('transport', 'i2'), ('payload', 'i2'),
        ('fragmentable', '?'), ('fragment_draw', 'f8'), ('reply_draw', 'f8'),
        ('time', 'i8'), ('reply_time', 'i8'), ('gap', 'i4'), ('left', 'i8'), ('right', 'i8')])
    required = {'iterations': 'positive_count', 'population': 'positive_count',
                'batch_size': 'positive_count', 'byte_penalty': 'nonnegative',
                'similarity_reward': 'nonnegative', 'fragment_probability': 'fraction',
                'icmp_probability': 'fraction'}
    optional = {'fake_copies': 'positive_count'}
    ASSUMPTIONS = {
        'scope': 'flow-window adaptation, not a reproduction of packet-sequence LSTM or live P4 execution',
        'learning': 'independent MLP proxy; categorical policy gradient; training updates and validation selection; no evaluation-classifier queries',
        'direction': 'source-flow forward is treated as user outbound; eight blocks of eight original packet positions',
        'fragment': 'IPv4 without options, DF or existing fragmentation; aligned first payload contains the complete transport header; otherwise insert a fake',
        'fake': 'extended 8..1440-byte payload action grid plus a measured header, at most 1500 bytes; TTL and invalid checksum are conceptual, not replayed',
        'fake_copies': 'optional 1..8 identical fakes per edit; grouped replies share the seeded response decision and timestamp',
        'icmp': '56-byte IPv4 Time Exceeded packets counted in the associated tunnel window, not the original five-tuple; probability is assumed, not measured',
        'timing': 'fragments/fakes share original timestamps; seeded ICMP time before the next original inbound packet; no reply invented without that bound',
        'fct': 'timestamp span unchanged; link serialization, routing, reassembly, TCP feedback and actual FCT are not measured',
    }

    @classmethod
    def validate(cls, parameters, source):
        validate_policy_parameters(cls.name, parameters, cls.required, cls.optional, source, byte_fraction_kind=None)
        if (parameters['population'] < 4 or parameters['batch_size'] < 5
                or parameters.get('fake_copies', 1) > 8):
            raise ValueError('Securitas needs population >= 4, batch_size >= 5 and at most 8 fake copies')
        return parameters

    @classmethod
    def _signature(cls, key, seed):
        return hashlib.sha256(f'{cls.VERSION}:{seed}:{key}'.encode()).hexdigest()

    @classmethod
    def prepare_context(cls, times, directions, sizes, headers, payloads, fragmentable, key, seed=42):
        """Prepare exact residual moments and edit neighborhoods once, in ns.

        ``fragmentable`` must come from the original IPv4 version/DF/MF/offset
        fields. Native packets outside edit neighborhoods become payload groups.
        """
        t, d, s, h, p = (np.asarray(a) for a in (times, directions, sizes, headers, payloads))
        eligible = np.asarray(fragmentable, dtype=bool)
        if (not len(t) or any(a.ndim != 1 or a.shape != t.shape
                or not np.issubdtype(a.dtype, np.integer) for a in (t, d, s, h, p))
                or eligible.shape != t.shape or np.any(np.diff(t) < 0)
                or not np.isin(d, (0, 1)).all() or np.any((s < h) | (s > 1500))
                or np.any(h < 20) or np.any((p < 0) | (p > s - h))):
            raise ValueError('Securitas preparation needs valid ordered packets and measured fragment eligibility')
        t = t - t[0]
        selected = np.flatnonzero((d == 0) & (np.arange(len(t)) < cls.WINDOW))
        points = np.zeros(len(selected), dtype=cls.POINT)
        points['reply_marker'] = points['gap'] = points['left'] = -1
        signature = cls._signature(key, seed)
        rng = np.random.default_rng(int(signature[:16], 16))
        random = rng.random((len(selected), 3))
        inbound = np.flatnonzero(d == 1)
        markers = []
        for j, i in enumerate(selected):
            point = points[j]
            point['position'], point['time'], point['size'] = i, t[i], s[i]
            point['header'], point['transport'], point['payload'] = h[i], s[i] - h[i] - p[i], p[i]
            point['fragmentable'] = eligible[i] and h[i] == 20 and point['transport'] >= 8
            point['fragment_draw'], point['reply_draw'] = random[j, :2]
            markers.append((int(t[i]), int(i * 4), j, 'marker'))
            gap = int(np.searchsorted(inbound, i, side='right'))
            if gap < len(inbound):
                right = int(t[inbound[gap]])
                reply = int(t[i]) + int(random[j, 2] * (right - int(t[i])))
                point['gap'], point['right'], point['reply_time'] = gap, right, reply
                if gap:
                    point['left'] = t[inbound[gap - 1]]
                markers.append((reply, int(i * 4 + 2), j, 'reply_marker'))
        markers.sort()
        for rank, (_, _, j, field) in enumerate(markers):
            points[j][field] = 2 * rank + 1
        untouched = np.ones(len(t), dtype=bool)
        untouched[selected] = False
        residual = np.zeros((2, 5), dtype=np.float64)
        iat = np.zeros((2, 2), dtype=np.float64)
        for direction in (0, 1):
            sizes_left = s[untouched & (d == direction)]
            if len(sizes_left):
                residual[direction] = len(sizes_left), sizes_left.sum(), sizes_left.var(), sizes_left.min(), sizes_left.max()
            gaps = np.diff(t[d == direction]).astype(np.float64) / 1e9
            iat[direction] = gaps.sum(), np.square(gaps).sum()
        dtype = np.dtype([('time', 'i8'), ('order', 'i8')])
        boundaries = np.asarray([(a, b) for a, b, _, _ in markers], dtype=dtype)
        native = np.empty(int(untouched.sum()), dtype=dtype)
        native['time'], native['order'] = t[untouched], np.flatnonzero(untouched) * 4
        regions = 2 * np.searchsorted(boundaries, native)
        segments = payload_segments(t[untouched], d[untouched], p[untouched], regions)
        return dict(securitas_signature=signature, securitas_points=points,
                    securitas_residual=residual, securitas_iat=iat, securitas_segments=segments)

    @classmethod
    def _metadata(cls, row, seed):
        if row.get('securitas_signature') != cls._signature(row['flow_key'], seed):
            raise ValueError('Securitas needs matching packet-derived edit metadata in flow_context')
        points, residual, iat, segments = [np.asarray(row.get(n)) for n in
            ('securitas_points', 'securitas_residual', 'securitas_iat', 'securitas_segments')]
        if (points.ndim != 1 or points.dtype != cls.POINT or len(points) > cls.WINDOW
                or residual.shape != (2, 5) or iat.shape != (2, 2)
                or segments.ndim != 2 or segments.shape[1] != 6
                or any(not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all() for a in (residual, iat, segments))
                or np.any(residual < 0) or np.any(iat < 0) or np.any(segments[:, 2:] < 0)
                or not np.isin(segments[:, 1], (-1, 0, 1)).all()
                or np.any((points['position'] < 0) | (points['position'] >= cls.WINDOW))
                or np.any(np.diff(points['position']) <= 0)
                or np.any((points['fragment_draw'] < 0) | (points['fragment_draw'] >= 1))
                or np.any((points['reply_draw'] < 0) | (points['reply_draw'] >= 1))
                or not 28 <= row.get('dummy_header_bytes', 0) <= 60):
            raise ValueError('Invalid packet-derived Securitas edit metadata')
        if (residual[0, 0] + len(points) != row['tot_fwd_pkts'] or residual[1, 0] != row['tot_bwd_pkts']
                or residual[0, 1] + points['size'].sum() != row['totlen_fwd_pkts']
                or residual[1, 1] != row['totlen_bwd_pkts']):
            raise ValueError('Securitas residual moments do not match original flow counts and bytes')
        return points, residual, iat, segments

    def __init__(self, parameters, artifact=None, columns=FEATURE_NAMES, seed=42):
        self.validate(parameters, 'Securitas')
        if artifact is not None:
            self.verify(artifact, artifact.get('signature'))
        self.parameters, self.artifact = parameters, artifact
        self.columns, self.seed = tuple(columns), seed
        self.reset()

    def reset(self):
        self._totals = np.zeros(7, dtype=np.float64)

    def _apply(self, row, controls):
        points, residual, iat, segments = self._metadata(row, self.seed)
        if not np.any(controls) or not len(points):
            return {n: float(row[n]) for n in FEATURE_NAMES}, np.zeros(4)
        summaries = [SizeMoments(int(a[0]), *map(float, a[1:])) for a in residual]
        additions, reply_gaps = [], {}
        counts = np.zeros(4, dtype=np.float64)  # fragments, fakes, ICMP, fallbacks
        copies = self.parameters.get('fake_copies', 1)
        for point in points:
            amount = int(controls[int(point['position']) // (self.WINDOW // self.BLOCKS)])
            size, header, transport, payload, timestamp = [int(point[n]) for n in ('size', 'header', 'transport', 'payload', 'time')]
            fragment = (amount and point['fragment_draw'] < self.parameters['fragment_probability'])
            valid = (point['fragmentable'] and transport <= amount < size - header)
            if fragment and valid:
                parts = [(header + amount, amount - transport), (size - amount, size - header - amount)]
                counts[0] += 1
            else:
                parts = [(size, payload)]
                if amount:
                    parts.extend([(int(row['dummy_header_bytes']) + amount, amount)] * copies)
                    counts[1] += copies
                    counts[3] += int(bool(fragment))
                    if point['reply_marker'] >= 0 and point['reply_draw'] < self.parameters['icmp_probability']:
                        summaries[1].merge(SizeMoments.constant(copies, 56))
                        reply = int(point['reply_time'])
                        additions.append([point['reply_marker'], 1, copies, 28 * copies, reply, reply])
                        gap = reply_gaps.setdefault(int(point['gap']), [int(point['left']), int(point['right']), []])
                        gap[2].extend([reply] * copies)
                        counts[2] += copies
            for part_size, part_payload in parts:
                summaries[0].merge(SizeMoments.constant(1, part_size))
                if part_payload:
                    additions.append([point['marker'], 0, 1, part_payload, timestamp, timestamp])
        result = flow_statistics({n: float(row[n]) for n in FEATURE_NAMES}, summaries, float(row['flow_duration']))
        sums = iat.copy()
        for left, right, replies in reply_gaps.values():
            endpoints = sorted(replies) + [right]
            if left >= 0:
                endpoints.insert(0, left)
                sums[1, 1] -= ((right - left) / 1e9) ** 2
            else:
                sums[1, 0] += (right - endpoints[0]) / 1e9
            gaps = np.diff(np.asarray(endpoints, dtype=np.int64)).astype(np.float64) / 1e9
            sums[1, 1] += np.square(gaps).sum()
        for direction, name in enumerate(DIRECTIONS):
            intervals = summaries[direction].count - 1
            total, squared = sums[direction]
            variance = squared / intervals - (total / intervals) ** 2 if intervals > 0 else 0.
            result[f'{name}_iat_std'] = float(np.sqrt(max(0., variance)))
        events = np.concatenate((segments, np.asarray(additions, dtype=np.float64).reshape(-1, 6)))
        result.update(forward_bulk_statistics(events))
        return result, counts

    def __call__(self, row):
        if self.artifact is None:
            raise ValueError('Securitas needs an independent-proxy fitted policy')
        result, counts = self._apply(row, self.artifact['controls'])
        before = sum(float(row[f'totlen_{d}_pkts']) for d in DIRECTIONS)
        after = sum(result[f'totlen_{d}_pkts'] for d in DIRECTIONS)
        self._totals += (1, before, after, *counts)
        return {n: result[n] for n in self.columns}

    @property
    def overhead(self):
        n, before, after, fragments, fakes, replies, fallback = self._totals
        return dict(extra_byte_fraction=float(after / before - 1) if before else 0.,
                    mean_fragmented_packets=float(fragments / n) if n else 0.,
                    mean_fake_packets=float(fakes / n) if n else 0.,
                    mean_icmp_replies=float(replies / n) if n else 0.,
                    mean_fragment_fallbacks=float(fallback / n) if n else 0., mean_added_span_s=0.)

    @classmethod
    def _prepared(cls, data, context, seed=42):
        rows = list(_context_rows(data, context))
        _, first = np.unique(data.row_indices, return_index=True)
        rows = [rows[i] for i in first]
        for row in rows:
            cls._metadata(row, seed)
        return data.subset(first), rows

    @classmethod
    def signature(cls, training, train_rows, validation, val_rows, parameters, seed):
        digest = hashlib.sha256(json.dumps([cls.VERSION, 'wide-actions-v2', parameters, seed, training.feature_names,
                                            training.class_names, training.source_fingerprint], sort_keys=True).encode())
        for data, rows in ((training, train_rows), (validation, val_rows)):
            arrays = [data.features, data.labels, data.captures, data.row_indices, data.record_ids,
                      np.asarray([[r[n] for n in FEATURE_NAMES] for r in rows]),
                      np.asarray([r['dummy_header_bytes'] for r in rows])]
            for row in rows:
                arrays.extend(cls._metadata(row, seed))
            for array in arrays:
                digest.update(str((array.shape, array.dtype)).encode())
                digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    @classmethod
    def _fingerprint(cls, signature, controls):
        # Retain integrity compatibility with existing small-action artifacts.
        encoded = (bytes(controls) if max(controls, default=0) <= 255 else
                   b'wide-actions:' + np.asarray(controls, dtype='<i2').tobytes())
        return hashlib.sha256(signature.encode() + encoded).hexdigest()

    @classmethod
    def verify(cls, artifact, signature):
        if (not isinstance(artifact, dict) or artifact.get('version') != cls.VERSION
                or not isinstance(signature, str) or len(signature) != 64 or artifact.get('signature') != signature):
            raise ValueError('Securitas policy is missing or stale; run packmorph-obfuscate securitas')
        controls = artifact.get('controls')
        if (not isinstance(controls, list) or len(controls) != cls.BLOCKS
                or any(type(a) is not int or a not in cls.CHOICES for a in controls)
                or artifact.get('fingerprint') != cls._fingerprint(signature, controls)):
            raise ValueError('Learned Securitas policy failed integrity validation')

    @staticmethod
    def _weights(data):
        weights = np.zeros(len(data), dtype=np.float64)
        for label in range(len(data.class_names)):
            members = data.labels == label
            _, inverse, counts = np.unique(data.captures[members], return_inverse=True, return_counts=True)
            if not len(counts):
                raise ValueError('Securitas proxy needs every service class')
            weights[members] = 1 / (len(data.class_names) * len(counts) * counts[inverse])
        return weights

    @classmethod
    def fit(cls, training, train_context, validation, val_context, parameters, seed=42, cached=None, device='cpu'):
        """Fit an independent proxy and a universal categorical policy gradient.

        The evaluation classifier is deliberately absent from this interface.
        The flow adaptation uses eight categorical blocks instead of the paper's
        recurrent packet controller. Validation proxy accuracy selects the policy;
        byte cost and dissimilarity break ties. Existing controls remain a candidate.
        """
        cls.validate(parameters, 'Securitas fitting')
        if (training.feature_names != validation.feature_names or training.class_names != validation.class_names
                or training.source_fingerprint != validation.source_fingerprint):
            raise ValueError('Securitas training and validation schemas must match')
        for field in ('captures', 'row_indices', 'record_ids'):
            if np.intersect1d(getattr(training, field), getattr(validation, field)).size:
                raise ValueError(f'Securitas training and validation have overlapping {field}')
        training, train_rows = cls._prepared(training, train_context, seed)
        validation, val_rows = cls._prepared(validation, val_context, seed)
        signature = cls.signature(training, train_rows, validation, val_rows, parameters, seed)
        if isinstance(cached, dict) and cached.get('signature') == signature:
            cls.verify(cached, signature)
            return cached
        from .baseline import BehaviorScaler
        from .runtime import _tensorflow, prepare_runtime

        prepare_runtime(seed + 37, device)
        tf = _tensorflow()
        rng = np.random.default_rng(seed)
        scaler = BehaviorScaler.fit(training.features)
        proxy = tf.keras.Sequential([tf.keras.Input(shape=(16,)), tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(len(training.class_names), activation='softmax')])
        optimizer = tf.keras.optimizers.Adam(.001)
        train_weights, val_weights = cls._weights(training), cls._weights(validation)
        encoded, heldout = scaler.transform(training.features), scaler.transform(validation.features)

        @tf.function(reduce_retracing=True)
        def proxy_step(x, y):
            with tf.GradientTape() as tape:
                probability = proxy(x, training=True)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, probability))
            optimizer.apply_gradients(zip(tape.gradient(loss, proxy.trainable_variables), proxy.trainable_variables))

        best_proxy = None
        best_proxy_key = None
        for step in range(400):
            indices = rng.choice(len(training), parameters['batch_size'], replace=True, p=train_weights)
            proxy_step(encoded[indices], training.labels[indices])
            if (step + 1) % 50 == 0:
                probabilities = np.asarray(proxy(heldout, training=False))
                accuracy = float(np.dot(probabilities.argmax(axis=1) == validation.labels, val_weights))
                nll = float(np.dot(-np.log(np.maximum(probabilities[np.arange(len(validation)), validation.labels], 1e-12)), val_weights))
                key = (-accuracy, nll)
                if best_proxy_key is None or key < best_proxy_key:
                    best_proxy_key, best_proxy = key, proxy.get_weights()
        proxy.set_weights(best_proxy)
        print(f'  Securitas independent proxy: validation capture accuracy={-best_proxy_key[0]:.2%}', flush=True)
        policy = cls(parameters, seed=seed)

        def assess(controls, data, rows, weights):
            matrices, byte_costs = [], []
            original_bytes = np.asarray([sum(float(r[f'totlen_{d}_pkts']) for d in DIRECTIONS) for r in rows])
            for control in controls:
                results = [policy._apply(row, control)[0] for row in rows]
                matrices.append(np.asarray([[r[n] for n in data.feature_names] for r in results], dtype=np.float32))
                byte_costs.append(np.asarray([sum(r[f'totlen_{d}_pkts'] for d in DIRECTIONS) for r in results]) / original_bytes - 1)
            matrix = np.concatenate(matrices)
            probability = np.asarray(proxy(scaler.transform(matrix), training=False)).reshape(len(controls), len(data), -1)
            confidence = probability[:, np.arange(len(data)), data.labels]
            rivals = probability.copy()
            rivals[:, np.arange(len(data)), data.labels] = -1
            margin = confidence - rivals.max(axis=2)
            correct = probability.argmax(axis=2) == data.labels
            adversarial = correct + .1 * np.maximum(margin, -.05)
            logged = np.log1p(np.asarray(matrices))
            original = np.log1p(data.features)
            distance = np.linalg.norm(logged - original, axis=2) / np.maximum(1e-12, np.linalg.norm(original, axis=1))
            distance = distance / (1 + distance)
            costs = np.asarray(byte_costs)
            loss = (.6 * adversarial + parameters['byte_penalty'] * costs - parameters['similarity_reward'] * distance) @ weights
            accuracy = correct @ weights
            return loss, accuracy, costs @ weights, distance @ weights

        initial = [np.zeros(cls.BLOCKS, dtype=np.int64)]
        if isinstance(cached, dict) and cached.get('version') == cls.VERSION:
            cls.verify(cached, cached.get('signature'))
            initial.append(np.asarray(cached['controls'], dtype=np.int64))
        initial.extend(np.full(cls.BLOCKS, size, dtype=np.int64) for size in cls.CHOICES[1:])
        score, accuracy, cost, distance = assess(initial, validation, val_rows, val_weights)
        chosen = min(range(len(initial)), key=lambda i: (accuracy[i], score[i]))
        best = initial[chosen].copy()
        best_key = (float(accuracy[chosen]), float(score[chosen]))
        checkpoint = dict(objective=best_key[1], proxy_accuracy=best_key[0], mean_extra_byte_fraction=float(cost[chosen]), iteration=0)
        logits = np.zeros((cls.BLOCKS, len(cls.CHOICES)))
        logits[np.arange(cls.BLOCKS), np.searchsorted(cls.CHOICES, best)] = 1.5
        for iteration in range(parameters['iterations']):
            indices = []
            for label in range(len(training.class_names)):
                pool = np.flatnonzero(training.labels == label)
                count = parameters['batch_size'] // len(training.class_names) + (label < parameters['batch_size'] % len(training.class_names))
                weights = train_weights[pool];weights = weights / weights.sum()
                indices.extend(rng.choice(pool, count, replace=True, p=weights))
            indices = np.asarray(indices)
            batch = training.subset(indices)
            probability = np.exp(logits - logits.max(axis=1, keepdims=True));probability /= probability.sum(axis=1, keepdims=True)
            draws = rng.random((parameters['population'], cls.BLOCKS, 1))
            codes = np.minimum(len(cls.CHOICES) - 1, (draws > probability.cumsum(axis=1)).sum(axis=2))
            controls = cls.CHOICES[codes]
            losses, _, _, _ = assess(controls, batch, [train_rows[i] for i in indices], np.full(len(indices), 1 / len(indices)))
            advantage = (losses - losses.mean()) / max(float(losses.std()), 1e-6)
            one_hot = np.eye(len(cls.CHOICES))[codes]
            gradient = (advantage[:, None, None] * (one_hot - probability)).mean(axis=0)
            logits = np.clip(logits - .2 * gradient, -8, 8)
            candidates = np.stack((controls[int(np.argmin(losses))], cls.CHOICES[logits.argmax(axis=1)]))
            scores, accuracies, costs, distances = assess(candidates, validation, val_rows, val_weights)
            for i, candidate in enumerate(candidates):
                key = (float(accuracies[i]), float(scores[i]))
                if key < best_key:
                    best, best_key = candidate.copy(), key
                    checkpoint = dict(objective=key[1], proxy_accuracy=key[0], mean_extra_byte_fraction=float(costs[i]),
                                      dissimilarity=float(distances[i]), iteration=iteration + 1)
            if (iteration + 1) % 10 == 0 or iteration + 1 == parameters['iterations']:
                print(f'  Securitas {iteration + 1}/{parameters["iterations"]}: validation proxy accuracy={checkpoint["proxy_accuracy"]:.2%}, mean extra bytes={checkpoint["mean_extra_byte_fraction"]:.2%}', flush=True)
        controls = best.tolist()
        artifact = dict(version=cls.VERSION, signature=signature, controls=controls,
            fingerprint=cls._fingerprint(signature, controls), assumptions=cls.ASSUMPTIONS,
            independent_proxy_accuracy=-best_proxy_key[0], validation=checkpoint)
        cls.verify(artifact, signature)
        return artifact
