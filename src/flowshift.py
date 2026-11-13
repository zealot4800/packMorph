"""Fit and apply FlowShift policies to prepared packet-derived flow features."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

import numpy as np

from .baseline import BehaviorScaler
from .minos import Minos
from .securitas import Securitas
from .transform import (DIRECTIONS, FEATURE_NAMES, SizeMoments, flow_statistics,
                        _context_rows, model_columns, validate_policy_parameters)


CONTEXT_FIELDS = ('blanket_bulk', 'dummy_header_bytes', 'minipatch_signature',
                  'minipatch_tails', 'minipatch_segments', 'minipatch_iat',
                  'securitas_signature', 'securitas_points', 'securitas_residual',
                  'securitas_iat', 'securitas_segments', 'minos_signature',
                  'minos_padding', 'minos_front', 'minos_bulk')
COST_PENALTIES = (0., .1, .3, 1., 3., 5., 6., 8., 10., 15., 20.)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def build_profiles(training, context):
    """Deduplicate physical flows; give captures equal weight within a service."""
    _, first = np.unique(training.row_indices, return_index=True)
    retained = set(first)
    rows = {i: row for i, row in enumerate(_context_rows(training, context)) if i in retained}
    data = training.subset(first)
    values = np.asarray([[rows[i][n] for n in FEATURE_NAMES] for i in first], dtype=np.float64)
    scaler = BehaviorScaler.fit(data.features)
    encoded = scaler.transform(data.features)
    profiles, centers = [], []
    for label in range(len(data.class_names)):
        members = data.labels == label
        if not members.any():
            raise ValueError('FlowShift profiles need every training service')
        _, inverse, counts = np.unique(data.captures[members], return_inverse=True, return_counts=True)
        weights = 1 / (len(counts) * counts[inverse])
        profiles.append(np.average(values[members], axis=0, weights=weights).tolist())
        centers.append(np.average(encoded[members], axis=0, weights=weights).tolist())
    # Bind every input used to estimate profiles, including original identities.
    digest = hashlib.sha256()
    for array in (data.features, values, data.labels, data.captures, data.row_indices, data.record_ids):
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    for i in first:
        for name in CONTEXT_FIELDS:
            array = np.asarray(rows[i].get(name))
            digest.update(name.encode())
            digest.update(str((array.shape, array.dtype)).encode())
            # Missing optional metadata must have a stable representation.
            digest.update(b'missing' if array.dtype.kind == 'O' else np.ascontiguousarray(array).tobytes())
    return dict(columns=list(data.feature_names), classes=list(data.class_names),
                source=data.source_fingerprint, training_digest=digest.hexdigest(),
                mean=scaler.mean.tolist(), scale=scaler.scale.tolist(),
                profiles=profiles, centers=centers)


def pareto_indices(objectives):
    """Indices of non-dominated rows; every objective is minimized."""
    values = np.asarray(objectives, dtype=np.float64)
    return [i for i, row in enumerate(values)
            if not np.any(np.all(values <= row, axis=1) & np.any(values < row, axis=1))]


def teacher_fingerprint(frozen):
    if frozen is None:
        return None
    digest = hashlib.sha256()
    arrays = list(frozen.model.embedding.get_weights()) + [frozen.model.prototypes,
        frozen.model.diagonal_variances, frozen.scaler.mean, frozen.scaler.scale]
    for array in arrays:
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def guide_probabilities(encoded, weights):
    """NumPy-only inference; the evaluation classifier is absent at runtime."""
    from scipy.special import softmax
    values = np.clip(encoded, -12, 12)
    for i in range(0, len(weights), 2):
        values = values @ weights[i] + weights[i + 1]
        if i < len(weights) - 2:
            values = np.maximum(values, 0)
    return softmax(values, axis=1)


def guided_choice(probabilities, costs, parameters, key, byte_scale=1.):
    """Choose a cost-aware non-dominated point, with identity at index zero."""
    source = probabilities[0].argmax()
    rivals = probabilities.copy()
    rivals[:, source] = -1
    risk = np.maximum(probabilities[:, source] - rivals.max(axis=1), -.1)
    scores = -risk - costs @ np.array([parameters['byte_penalty'] * byte_scale, parameters['duration_penalty']])
    ties = np.flatnonzero(scores == scores.max())
    # Among equal scalar objectives, prefer fewer bytes and then less span.
    # Nonnegative cost weights make this choice Pareto non-dominated.
    cheapest = min(ties, key=lambda i: tuple(costs[i]))
    ties = [i for i in ties if np.array_equal(costs[i], costs[cheapest])]
    return int(ties[int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % len(ties)])


def context_signature(data, context):
    """Bind checkpoint selection to the exact validation inputs and identities."""
    digest = hashlib.sha256()
    for array in (data.features, data.labels, data.captures, data.row_indices, data.record_ids):
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    for row in _context_rows(data, context):
        for name in FEATURE_NAMES + CONTEXT_FIELDS:
            array = np.asarray(row.get(name))
            digest.update(name.encode())
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(b'missing' if array.dtype.kind == 'O' else np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def policy_inputs(policy, data, context):
    """Build a reusable feasible bank; labels never participate in generation."""
    matrices, costs, offsets, keys, before, durations, packets = [], [], [0], [], [], [], []
    for row in _context_rows(data, context):
        candidates = policy.candidates(row)
        matrices.extend([[row[n] for n in policy.columns]] +
                        [list(policy.project(row, c[0]).values()) for c in candidates])
        costs.extend([(0., 0.)] + [(c[2], c[3]) for c in candidates])
        count = sum(row[f'tot_{d}_pkts'] for d in DIRECTIONS)
        packets.extend([0.] + [sum(c[0][f'tot_{d}_pkts'] for d in DIRECTIONS) - count for c in candidates])
        offsets.append(len(matrices))
        keys.append(f"{policy.seed}:{row['source_capture']}:{row['flow_key']}")
        before.append(sum(row[f'totlen_{d}_pkts'] for d in DIRECTIONS))
        durations.append(row['flow_duration'])
    return dict(features=np.asarray(matrices, dtype=np.float32), costs=np.asarray(costs),
                offsets=np.asarray(offsets), keys=keys, before=np.asarray(before),
                durations=np.asarray(durations), added_packets=np.asarray(packets), byte_reference=policy.byte_reference)


def policy_choices(probabilities, bank, parameters):
    scales = bank['before'] / bank['byte_reference'] if parameters.get('cost_basis', 'relative') == 'aggregate' else np.ones(len(bank['keys']))
    choices = []
    for start, end, key, scale in zip(bank['offsets'][:-1], bank['offsets'][1:], bank['keys'], scales):
        chosen = start + guided_choice(probabilities[start:end], bank['costs'][start:end], parameters, key, scale)
        choices.append(chosen)
    return np.asarray(choices)


def policy_overhead(bank, chosen):
    costs = bank['costs'][chosen]
    spans = bank['durations'] * costs[:, 1]
    return dict(extra_byte_fraction=max(0., float(np.dot(bank['before'], costs[:, 0]) / bank['before'].sum())),
                mean_added_packets=float(bank['added_packets'][chosen].mean()),
                mean_added_span_s=float(spans.mean()), max_added_span_s=float(spans.max()),
                max_extra_byte_fraction=float(costs[:, 0].max()), max_span_fraction=float(costs[:, 1].max()))


def policy_capability(bank, predictions, data):
    """Label-aware diagnostic only; never use oracle choices to transform flows.

    Count flows for which every candidate remains correctly classified.
    Shared byte credit and causal selection are ignored, so the resulting
    accuracy is an optimistic bound for this bank on these records.
    """
    blocked = np.asarray([np.all(predictions[start:end] == label)
                          for start, end, label in zip(bank['offsets'][:-1], bank['offsets'][1:], data.labels)])
    return dict(flows=len(data), candidate_evaluations=len(predictions),
        optimistic_accuracy_floor=float(blocked.mean()), unavoidable_correct_flows=int(blocked.sum()),
        services={name: dict(flows=int(np.sum(data.labels == label)),
                            unavoidable_correct_flows=int(np.sum(blocked & (data.labels == label))))
                  for label, name in enumerate(data.class_names)},
        scope='label-aware oracle over this finite bank on these flows; shared byte cap ignored; '
              'not runtime selection, measured defense accuracy, or a bound for other packet operations')


def fit_guide(policy, training, context, validation, validation_context, frozen, seed, steps):
    """Fit on training only; restore the lowest-cost/effective validation checkpoint."""
    from .metrics import capture_macro_accuracy
    from .runtime import _tensorflow, _runtime_device, prepare_runtime

    if (training.feature_names != validation.feature_names or training.class_names != validation.class_names
            or training.source_fingerprint != validation.source_fingerprint):
        raise ValueError('FlowShift training/validation schemas must match')
    for field in ('captures', 'row_indices', 'record_ids'):
        if np.intersect1d(getattr(training, field), getattr(validation, field)).size:
            raise ValueError(f'FlowShift training/validation have overlapping {field}')
    settings = policy.parameters
    _, first = np.unique(training.row_indices, return_index=True)
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(len(training.class_names)):
        pool = first[training.labels[first] == label]
        _, inverse, counts = np.unique(training.captures[pool], return_inverse=True, return_counts=True)
        weights = 1 / counts[inverse]
        selected.extend(rng.choice(pool, min(settings.get('fit_flows_per_service', 256), len(pool)),
                                   replace=False, p=weights / weights.sum()))
    fitting = training.subset(np.asarray(selected))
    retained = set(selected)
    rows = {i: row for i, row in enumerate(_context_rows(training, context)) if i in retained}

    print(f'  FlowShift: building complete feasible training pools for {len(selected)} unique flows', flush=True)
    train_bank = policy_inputs(policy, fitting, [rows[i] for i in selected])
    val_bank = policy_inputs(policy, validation, validation_context)
    encoded = np.clip(policy.scaler.transform(train_bank['features']), -12, 12)
    targets = frozen.model.predict_probabilities(frozen.scaler.transform(train_bank['features']), 512)
    heldout = np.clip(policy.scaler.transform(val_bank['features']), -12, 12)
    outcomes = frozen.model.predict_probabilities(frozen.scaler.transform(val_bank['features']), 512)
    offsets = train_bank['offsets']
    # Equal class/capture mass, equal original-flow mass inside a capture, then
    # equal candidate mass inside that flow. Large banks cannot dominate fitting.
    flow_weights = np.zeros(len(fitting))
    for label in range(len(fitting.class_names)):
        members = fitting.labels == label
        _, inverse, counts = np.unique(fitting.captures[members], return_inverse=True, return_counts=True)
        flow_weights[members] = 1 / (len(fitting.class_names) * len(counts) * counts[inverse])
    sizes = np.diff(offsets)
    sampling = np.repeat(flow_weights / sizes, sizes)
    sampling /= sampling.sum()
    sources = np.repeat(targets[offsets[:-1]].argmax(axis=1), sizes)
    teacher_rivals = targets.copy()
    teacher_rivals[np.arange(len(targets)), sources] = -1
    margins = targets[np.arange(len(targets)), sources] - teacher_rivals.max(axis=1)
    actual = outcomes.argmax(axis=1)
    capability = policy_capability(val_bank, actual, validation)
    print(f'  FlowShift: optimistic validation bank floor={capability["optimistic_accuracy_floor"]:.2%} '
          '(perfect label-aware choice; shared byte cap ignored)', flush=True)
    prepare_runtime(seed + 311, _runtime_device or 'cpu')
    tf = _tensorflow()
    model = tf.keras.Sequential([tf.keras.Input(shape=(encoded.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(.1),
        tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(.1),
        tf.keras.layers.Dense(targets.shape[1])])
    optimizer = tf.keras.optimizers.Adam(.001, weight_decay=.0001, global_clipnorm=1.)

    @tf.function(reduce_retracing=True)
    def step(x, y, source, target_margin):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            probability = tf.nn.softmax(logits)
            mask = tf.one_hot(source, targets.shape[1])
            predicted_margin = tf.reduce_sum(probability * mask, axis=1) - tf.reduce_max(
                tf.where(mask > 0, -1., probability), axis=1)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
                                  + .5 * tf.square(predicted_margin - target_margin))
        optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

    def assess(weights):
        probability = guide_probabilities(heldout, weights)
        points = []
        pairs = {(p, p) for p in COST_PENALTIES} | {(settings['byte_penalty'], settings['duration_penalty'])}
        for byte_penalty, duration_penalty in sorted(pairs):
            costs_setting = settings | dict(byte_penalty=byte_penalty, duration_penalty=duration_penalty)
            chosen = policy_choices(probability, val_bank, costs_setting)
            score = capture_macro_accuracy(validation.labels, actual[chosen], validation.captures, len(validation.class_names))
            overhead = policy_overhead(val_bank, chosen)
            byte, span = overhead['extra_byte_fraction'], overhead['mean_added_span_s']
            limit = settings.get('fit_byte_budget', .0025)
            key = (byte > limit + 1e-12, max(0., byte - limit), score, byte, span)
            points.append((key, dict(accuracy=float(np.mean(actual[chosen] == validation.labels)),
                capture_macro_accuracy=score, extra_byte_fraction=byte, mean_added_span_s=span,
                byte_penalty=byte_penalty, duration_penalty=duration_penalty)))
        return min(points, key=lambda point: point[0])

    history, best, best_key, stale, accumulated = [], None, None, 0, 0.
    interval = settings.get('guide_validation_interval', 250)
    for iteration in range(1, steps + 1):
        indices = rng.choice(len(encoded), 256, replace=True, p=sampling)
        accumulated += float(step(encoded[indices], targets[indices], sources[indices], margins[indices]))
        if iteration % interval and iteration != steps:
            continue
        weights = model.get_weights()
        key, metrics = assess(weights)
        history.append(dict(step=iteration, loss=accumulated / (iteration - (history[-1]['step'] if history else 0)),
                            learning_rate=float(optimizer.learning_rate.numpy()), **metrics))
        accumulated = 0.
        if best_key is None or key < best_key:
            best, best_key, best_step, stale = weights, key, iteration, 0
        else:
            stale += 1
            if stale % 2 == 0:
                optimizer.learning_rate.assign(max(1e-5, float(optimizer.learning_rate.numpy()) * .5))
        print(f'  FlowShift {iteration}/{steps}: validation accuracy={metrics["accuracy"]:.2%}, '
              f'bytes={metrics["extra_byte_fraction"]:.3%}; best step={best_step}', flush=True)
        if stale >= settings.get('guide_patience', 4):
            break
    agreement = float(np.mean(guide_probabilities(encoded, best).argmax(axis=1) == targets.argmax(axis=1)))
    capture_counts = {name: int(len(np.unique(training.captures[first][training.labels[first] == label])))
                      for label, name in enumerate(training.class_names)}
    return dict(weights=[w.tolist() for w in best], teacher_queries=len(encoded), validation_queries=len(heldout),
                training_agreement=agreement, width=128, selected_step=best_step, steps_run=iteration,
                stop_reason='early_stopping' if iteration < steps else 'step_limit', history=history,
                checkpoint_penalties=list(COST_PENALTIES),
                validation_capability=capability,
                training_capture_counts=capture_counts, validation_metrics=history[[r['step'] for r in history].index(best_step)])

class FlowShift:
    name = 'flowshift'
    VERSION = 'flowshift-policy-flow-v4'
    required = {'max_extra_byte_fraction': 'nonnegative', 'max_span_fraction': 'nonnegative',
                'max_added_span_s': 'nonnegative', 'min_displacement': 'nonnegative',
                'byte_penalty': 'nonnegative', 'duration_penalty': 'nonnegative'}
    ASSUMPTIONS = {
        'scope': 'offline whole-flow adaptation; no epochs, queues or VM execution',
        'profiles': 'unique training flows and capture-balanced class means; no test labels',
        'source': 'centroid association in profile mode; original guide prediction in distilled mode',
        'size': 'uniform directional or packet-derived bounded padding; output IP lengths at most 1500 bytes',
        'volume': 'zero, one or two dummies at equal subdivisions of each original directional gap',
        'sparse': 'small dummy groups at packet-derived burst tails when tail metadata is available',
        'fragment': 'reuse exact packet edit kernel; valid IPv4 splits or fake fallback; no simulated ICMP',
        'timing': 'equal-gap splitting and bounded clock dilation, preserving measured bulk boundaries',
        'selection': 'validation-selected source-margin model; non-dominated cost-aware policy choice',
        'training': 'capture-balanced unique training pools; validation policy accuracy/cost select checkpoints',
        'penalty': 'aggregate mode scales added bytes by a fixed training-derived byte reference',
        'teacher': 'guided mode queries frozen classifier only on training originals/edits during fitting',
        'cost': 'IP bytes and timestamp span only; excludes encapsulation and application FCT',
    }

    @classmethod
    def validate(cls, parameters, source):
        numeric = {k: v for k, v in parameters.items() if k not in ('selection', 'fragmentation', 'cost_basis')} if isinstance(parameters, dict) else parameters
        validate_policy_parameters(cls.name, numeric, cls.required,
            {'guide_steps': 'positive_count', 'guide_seed': 'positive_count',
             'guide_validation_interval': 'positive_count', 'guide_patience': 'positive_count',
             'fit_flows_per_service': 'positive_count', 'fit_byte_budget': 'nonnegative'}, source, byte_fraction_kind=None)
        if parameters.get('selection', 'profile') not in ('profile', 'distilled'):
            raise ValueError('FlowShift selection must be profile or distilled')
        if type(parameters.get('fragmentation', False)) is not bool:
            raise ValueError('FlowShift fragmentation must be boolean')
        if parameters.get('cost_basis', 'relative') not in ('relative', 'aggregate'):
            raise ValueError('FlowShift cost_basis must be relative or aggregate')
        return parameters

    @classmethod
    def signature(cls, training, context, parameters, seed=42, frozen=None,
                  validation=None, validation_context=None):
        if parameters.get('selection') == 'distilled' and frozen is None:
            raise ValueError('Distilled FlowShift requires its frozen training classifier')
        if parameters.get('selection') == 'distilled' and validation is None:
            raise ValueError('FlowShift checkpoint selection requires validation data')
        return fingerprint([cls.VERSION, build_profiles(training, context), parameters, seed,
            teacher_fingerprint(frozen) if parameters.get('selection') == 'distilled' else None,
            context_signature(validation, validation_context) if parameters.get('selection') == 'distilled' else None])

    @classmethod
    def fit(cls, training, context, parameters, seed=42, cached=None, frozen=None,
            validation=None, validation_context=None):
        cls.validate(parameters, 'FlowShift fitting')
        profiles = build_profiles(training, context)
        guided = parameters.get('selection') == 'distilled'
        if guided and frozen is None:
            raise ValueError('Distilled FlowShift fitting requires a frozen training classifier')
        teacher = teacher_fingerprint(frozen) if guided else None
        if guided and validation is None:
            raise ValueError('FlowShift checkpoint selection requires validation data')
        heldout_signature = context_signature(validation, validation_context) if guided else None
        signature = fingerprint([cls.VERSION, profiles, parameters, seed, teacher, heldout_signature])
        if isinstance(cached, dict) and cached.get('signature') == signature:
            cls.verify(cached, signature)
            return cached
        artifact = dict(version=cls.VERSION, signature=signature, profiles=profiles,
                        assumptions=cls.ASSUMPTIONS, teacher=teacher, guide=None,
                        parameters=dict(parameters), seed=seed, validation_signature=heldout_signature)
        artifact['fingerprint'] = fingerprint(artifact)
        if guided:
            policy = cls(parameters, artifact, training.feature_names, seed)
            artifact['guide'] = fit_guide(policy, training, context, validation, validation_context, frozen,
                parameters.get('guide_seed', seed), parameters.get('guide_steps', 6000))
            artifact['fingerprint'] = fingerprint({k: v for k, v in artifact.items() if k != 'fingerprint'})
        return artifact

    @classmethod
    def verify(cls, artifact, signature):
        if (not isinstance(artifact, dict) or artifact.get('version') != cls.VERSION
                or artifact.get('signature') != signature
                or artifact.get('fingerprint') != fingerprint(
                    {k: v for k, v in artifact.items() if k != 'fingerprint'})):
            raise ValueError('FlowShift profiles are missing or stale; run packmorph-obfuscate flowshift')

    def __init__(self, parameters, artifact, columns=FEATURE_NAMES, seed=42):
        self.validate(parameters, 'FlowShift')
        self.verify(artifact, artifact.get('signature'))
        profiles = artifact['profiles']
        if (tuple(profiles['columns']) != tuple(columns)
                or artifact['signature'] != fingerprint([self.VERSION, profiles, parameters, seed, artifact.get('teacher'), artifact.get('validation_signature')])):
            raise ValueError('FlowShift profiles do not match columns, parameters or seed')
        self.parameters, self.columns, self.seed = dict(parameters), tuple(columns), seed
        self.scaler = BehaviorScaler(np.asarray(profiles['mean']), np.asarray(profiles['scale']))
        self.centers = np.asarray(profiles['centers'])
        self.profiles = [dict(zip(FEATURE_NAMES, values)) for values in profiles['profiles']]
        self.byte_reference = float(np.mean([p['totlen_fwd_pkts'] + p['totlen_bwd_pkts'] for p in self.profiles]))
        if not np.isfinite(self.byte_reference) or self.byte_reference <= 0:
            raise ValueError('FlowShift training profiles require positive reference bytes')
        guide = artifact.get('guide')
        self.guide = None if guide is None else [np.asarray(w, dtype=np.float32) for w in guide['weights']]
        if self.guide is not None:
            width = self.guide[0].shape[1] if self.guide and self.guide[0].ndim == 2 else 0
            shapes = ((len(columns), width), (width,), (width, width), (width,), (width, len(self.centers)), (len(self.centers),))
            if len(self.guide) != len(shapes) or any(w.shape != shape or not np.isfinite(w).all()
                                                    for w, shape in zip(self.guide, shapes)):
                raise ValueError('Invalid FlowShift guide weights')
        # Only the packet-statistics kernel is reused, not Securitas learning/policies.
        self.fragmenter = (Securitas(dict(iterations=1, population=4, batch_size=5, byte_penalty=0.,
            similarity_reward=0., fragment_probability=1., icmp_probability=0.), seed=seed)
            if parameters.get('fragmentation', False) else None)
        self.reset()

    def reset(self):
        self.diagnostics = Counter()
        self._totals = np.zeros(7, dtype=np.float64)

    def _apply(self, row, padding, gaps, sizes, scale, metadata):
        table, header = metadata
        values = {n: float(row[n]) for n in FEATURE_NAMES}
        summaries = [SizeMoments.from_values(values, d) for d in DIRECTIONS]
        for i, summary in enumerate(summaries):
            if summary.count:
                summary.pad(padding[i])
                summary.merge(SizeMoments.constant(gaps[i] * max(0, summary.count - 1), sizes[i]))
        result = flow_statistics(values, summaries, values['flow_duration'] * scale)
        pf, pb = map(int, padding > 0)
        payload, real, dummy, groups, elapsed, _ = table[gaps[0], gaps[1], pf, pb]
        payload += real * padding[0] + dummy * (sizes[0] - header)
        result.update(fwd_byts_b_avg=payload / groups if groups else 0.,
                      fwd_pkts_b_avg=(real + dummy) / groups if groups else 0.,
                      fwd_blk_rate_avg=payload / (elapsed * scale) if elapsed else 0.)
        return result

    def project(self, row, result):
        return model_columns(row, result, self.columns)

    def candidates(self, row):
        table = np.asarray(row.get('blanket_bulk'))
        header = row.get('dummy_header_bytes', 0)
        if (table.shape != (3, 3, 2, 2, 6) or not np.isfinite(table).all()
                or np.any(table < 0) or not 28 <= header <= 60):
            raise ValueError('FlowShift needs packet-derived blanket_bulk metadata and dummy headers')
        counts = np.asarray([row[f'tot_{d}_pkts'] for d in DIRECTIONS])
        totals = np.asarray([row[f'totlen_{d}_pkts'] for d in DIRECTIONS])
        if (not np.isfinite(counts).all() or np.any(counts != counts.astype(np.int64))
                or np.any(counts < 0) or counts.sum() <= 0 or totals.sum() <= 0):
            raise ValueError('FlowShift needs integer packet counts and positive original bytes')
        maxima = np.asarray([row[f'{d}_pkt_len_max'] for d in DIRECTIONS])
        if np.any(maxima > 1500):
            raise ValueError('FlowShift source packets exceed the 1500-byte IP limit')
        counts = counts.astype(np.int64)
        original = self.scaler.transform(np.asarray([[row[n] for n in self.columns]]))[0]
        source = int(np.argmin(np.linalg.norm(self.centers - original, axis=1)))
        settings = self.parameters
        before, duration = float(totals.sum()), float(row['flow_duration'])
        cap = math.floor(before * settings['max_extra_byte_fraction'])
        candidates = []
        for target, profile in enumerate(self.profiles):
            if target == source:
                continue
            seen = set()
            for strength in (.25, .5, 1.):
                for mode in ('size', 'volume', 'time', 'balanced'):
                    padding = np.zeros(2, dtype=np.int64)
                    if mode in ('size', 'balanced'):
                        desired = [max(0., profile[f'{d}_pkt_len_mean'] - row[f'{d}_pkt_len_mean']) for d in DIRECTIONS]
                        padding = np.floor(np.minimum(np.maximum(0., 1500 - maxima), strength * np.asarray(desired))).astype(np.int64)
                        padding[counts == 0] = 0
                        spent = int(padding @ counts)
                        if spent > cap:
                            padding = np.floor(padding * (cap / spent)).astype(np.int64)
                    remaining = cap - int(padding @ counts)
                    gaps = np.zeros(2, dtype=np.int64)
                    sizes = np.clip(np.rint([profile[f'{d}_pkt_len_mean'] for d in DIRECTIONS]), 64, 1500).astype(np.int64)
                    if mode in ('volume', 'balanced') and duration:
                        for i, d in enumerate(DIRECTIONS):
                            if counts[i] > 1:
                                ratio = max(0., profile[f'tot_{d}_pkts'] / counts[i] - 1)
                                gaps[i] = min(2, math.ceil(strength * ratio))
                        slots = np.maximum(0, counts - 1)
                        while int(gaps @ slots) * 64 > remaining:
                            i = max((i for i in (0, 1) if gaps[i]), key=lambda i: (slots[i], i))
                            gaps[i] -= 1
                        injected = gaps * slots
                        minimum, desired = int(injected.sum()) * 64, int(injected @ sizes)
                        if desired > remaining:
                            sizes = 64 + np.floor((sizes - 64) * ((remaining - minimum) / (desired - minimum))).astype(np.int64)
                    scale = 1.
                    if mode in ('time', 'balanced') and duration:
                        ratio = max(0., profile['flow_iat_mean'] / max(row['flow_iat_mean'], 1e-12) - 1)
                        scale += min(strength * ratio, settings['max_span_fraction'], settings['max_added_span_s'] / duration)
                    pf, pb = map(int, padding > 0)
                    maximum_gap = table[gaps[0], gaps[1], pf, pb, 5]
                    if maximum_gap:
                        scale = min(scale, max(1., (1 - 2e-9) / maximum_gap))
                    plan = (tuple(padding), tuple(gaps), tuple(sizes), scale)
                    if plan in seen or (not padding.any() and not gaps.any() and scale == 1):
                        continue
                    seen.add(plan)
                    result = self._apply(row, padding, gaps, sizes, scale, (table, header))
                    candidates.append(self.record(row, result, mode, original, target))
        candidates.extend(self.envelope_candidates(row, (table, header), counts, maxima, cap, original))
        if 'minipatch_signature' in row and duration:
            from .minipatch import apply_burst_insertions
            seen_sparse = set()
            for direction in (0, 1):
                if not counts[direction]:
                    continue
                for position in (0, 8, 16):
                    for amount in (1, 4, 16, 64):
                        for size in (64, 512, 1500):
                            if amount * size > cap:
                                continue
                            result = apply_burst_insertions(row, [(position, amount * (1 if direction == 0 else -1))], size)
                            columns = list(self.project(row, result).values())
                            key = tuple(columns)
                            if key in seen_sparse:
                                continue
                            seen_sparse.add(key)
                            candidates.append(self.record(row, result, 'sparse', original))
        if 'minos_signature' in row and duration:
            for window, probability in ((500, .3), (500, .6), (1000, .3), (1000, .6), (500, 0.)):
                for padding in (0, 16, 32):
                    for size in ((64, 512, 1500) if probability else (64,)):
                        if not probability and not padding:
                            continue
                        policy = Minos(dict(front_window=window, insertion_probability=probability,
                            padding_max_bytes=padding, dummy_bytes=size), FEATURE_NAMES, self.seed)
                        result = policy(row)
                        byte = sum(result[f'totlen_{d}_pkts'] for d in DIRECTIONS) / before - 1
                        if byte < 0 or byte > settings['max_extra_byte_fraction'] + 1e-10:
                            continue
                        candidates.append(self.record(row, result, 'front', original))
        if settings.get('fragmentation', False) and 'securitas_signature' in row and duration:
            seen_fragments = set()
            for amount in (8, 32, 64, 128, 256, 512, 1024, 1440):
                for block in (-1, 0, 4, 7):
                    controls = np.zeros(8, dtype=np.int64)
                    controls[:] = amount if block == -1 else 0
                    if block >= 0:
                        controls[block] = amount
                    result, _ = self.fragmenter._apply(row, controls)
                    byte = sum(result[f'totlen_{d}_pkts'] for d in DIRECTIONS) / before - 1
                    if byte < 0 or byte > settings['max_extra_byte_fraction'] + 1e-10:
                        continue
                    columns = tuple(self.project(row, result).values())
                    if columns in seen_fragments:
                        continue
                    seen_fragments.add(columns)
                    candidates.append(self.record(row, result, 'fragment', original))
        return [candidate for candidate in candidates if candidate is not None]

    def record(self, row, result, mode, original, target=None):
        """One accounting/filtering path for every physical edit kernel."""
        settings = self.parameters
        before = sum(float(row[f'totlen_{d}_pkts']) for d in DIRECTIONS)
        byte = sum(result[f'totlen_{d}_pkts'] for d in DIRECTIONS) / before - 1
        duration = float(row['flow_duration'])
        added = result['flow_duration'] - duration
        span = added / duration if duration else 0.
        if (byte < -1e-10 or byte > settings['max_extra_byte_fraction'] + 1e-10
                or span < -1e-10 or span > settings['max_span_fraction'] + 1e-10
                or added > settings['max_added_span_s'] + 1e-9):
            return None
        encoded = self.scaler.transform(np.asarray([list(self.project(row, result).values())]))[0]
        displacement = float(np.linalg.norm(encoded - original) / math.sqrt(len(self.columns)))
        if displacement < settings['min_displacement']:
            return None
        score = 0.
        if settings.get('selection', 'profile') == 'profile':
            source = int(np.argmin(np.linalg.norm(self.centers - original, axis=1)))
            targets = [self.centers[target]] if target is not None else [c for i, c in enumerate(self.centers) if i != source]
            progress = max((np.linalg.norm(original - c) - np.linalg.norm(encoded - c))
                           / math.sqrt(len(self.columns)) for c in targets)
            byte_cost = byte * before / self.byte_reference if settings.get('cost_basis', 'relative') == 'aggregate' else byte
            score = float(progress - settings['byte_penalty'] * byte_cost - settings['duration_penalty'] * span)
        return result, displacement, byte, span, score, mode

    def envelope_candidates(self, row, metadata, counts, maxima, cap, original):
        """Explore feasible edits even when every target mean/count is smaller."""
        table, header = metadata
        settings = self.parameters
        duration = float(row['flow_duration'])
        candidates, seen = [], set()
        for fraction in (.0625, .25, .5, 1.):
            for direction in (0, 1, 2):
                members = np.array([direction in (0, 2), direction in (1, 2)]) & (counts > 0)
                padding = np.zeros(2, dtype=np.int64)
                if counts[members].sum():
                    padding[members] = np.floor(np.minimum(1500 - maxima[members],
                        cap * fraction / counts[members].sum())).astype(np.int64)
                for g in (0, 1, 2):
                    gaps = np.where(members & (counts > 1) & (duration > 0), g, 0)
                    injected = gaps * np.maximum(0, counts - 1)
                    remaining = cap - int(padding @ counts)
                    if int(injected.sum()) * 64 > remaining:
                        continue
                    for requested in ((64, 256, 1500) if injected.any() else (64,)):
                        size = min(requested, remaining // int(injected.sum())) if injected.any() else 64
                        sizes = np.full(2, size, dtype=np.int64)
                        for timed in (False, True):
                            scale = 1. + (min(settings['max_span_fraction'], settings['max_added_span_s'] / duration)
                                          * fraction if timed and duration else 0.)
                            pf, pb = map(int, padding > 0)
                            maximum_gap = table[gaps[0], gaps[1], pf, pb, 5]
                            if maximum_gap:
                                scale = min(scale, max(1., (1 - 2e-9) / maximum_gap))
                            plan = (tuple(padding), tuple(gaps), size, scale)
                            if plan in seen or (not padding.any() and not gaps.any() and scale == 1):
                                continue
                            seen.add(plan)
                            result = self._apply(row, padding, gaps, sizes, scale, (table, header))
                            candidates.append(self.record(row, result, 'envelope', original))
        return candidates

    def __call__(self, row):
        candidates = self.candidates(row)
        settings = self.parameters
        before = sum(float(row[f'totlen_{d}_pkts']) for d in DIRECTIONS)
        duration = float(row['flow_duration'])
        count = sum(float(row[f'tot_{d}_pkts']) for d in DIRECTIONS)
        self.diagnostics['flows'] += 1
        self.diagnostics['eligible_candidates'] += len(candidates)
        result = {n: float(row[n]) for n in FEATURE_NAMES}
        applied = False
        if candidates:
            key = f"{self.seed}:{row['source_capture']}:{row['flow_key']}"
            chosen = None
            if settings.get('selection') == 'distilled':
                if self.guide is None:
                    raise ValueError('FlowShift is missing its fitted distilled guide')
                matrix = np.asarray([[row[n] for n in self.columns]] +
                    [list(self.project(row, c[0]).values()) for c in candidates], dtype=np.float32)
                probabilities = guide_probabilities(self.scaler.transform(matrix), self.guide)
                costs = np.asarray([(0., 0.)] + [(c[2], c[3]) for c in candidates])
                scale = before / self.byte_reference if settings.get('cost_basis', 'relative') == 'aggregate' else 1.
                index = guided_choice(probabilities, costs, settings, key, scale)
                chosen = index - 1 if index else None
            else:
                front = pareto_indices([(-c[1], c[2], c[3]) for c in candidates])
                self.diagnostics['pareto_candidates'] += len(front)
                scores = np.asarray([c[4] for c in candidates])
                best = max(scores[i] for i in front)
                if best > 0:
                    ties = [i for i in front if scores[i] == best]
                    chosen = ties[int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % len(ties)]
            if chosen is not None:
                result, _, _, _, _, mode = candidates[chosen]
                applied = True
                self.diagnostics['applied_flows'] += 1
                self.diagnostics['selected_' + mode] += 1
        if not applied:
            self.diagnostics['fallback_flows'] += 1
        after = sum(result[f'totlen_{d}_pkts'] for d in DIRECTIONS)
        added = result['flow_duration'] - duration
        self._totals[:4] += (before, after, added, sum(result[f'tot_{d}_pkts'] for d in DIRECTIONS) - count)
        self._totals[4:] = np.maximum(self._totals[4:], (after / before - 1, added / duration if duration else 0., added))
        return model_columns(row, result, self.columns)

    @property
    def overhead(self):
        before, after, added, dummy, max_byte, max_span, max_added = self._totals
        n = self.diagnostics['flows']
        return dict(extra_byte_fraction=float(after / before - 1) if before else 0.,
                    mean_added_packets=float(dummy / n) if n else 0.,
                    mean_added_span_s=float(added / n) if n else 0., max_added_span_s=float(max_added),
                    max_extra_byte_fraction=float(max_byte), max_span_fraction=float(max_span))
