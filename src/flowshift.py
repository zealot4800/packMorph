"""Budgeted, training-profile-driven aggregate FlowShift adaptation.

Original packets stay ordered and intact. Uniform directional padding, equal-gap
insertion and bounded clock dilation use packet-derived size/timing/bulk summaries.
This offline single-policy adaptation does not export VM frame schedules or model
link serialization, online epoch rotation, or application completion time.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from .policy import Candidate, FlowShiftPolicy, MorphingSpec
from .transform import CAPTURE_COLUMN, DIRECTIONS, FEATURE_NAMES, transform_flow


@dataclass(frozen=True, slots=True)
class EditPlan:
    padding: tuple[int, int]
    gaps: tuple[int, int]
    dummy_sizes: tuple[int, int]
    scale: float
    extension: float

    def byte_budget(self, counts, totals):
        # A finite implementation allowance; the plan's actual cap was enforced
        # before this shared moment updater is called.
        return sum(totals) + 4500 * sum(counts)

    def edits(self, direction, count, mean):
        i = DIRECTIONS.index(direction)
        return self.padding[i], ((self.dummy_sizes[i], self.gaps[i] * max(0, int(count) - 1)),)


def _metadata(row):
    table = np.asarray(row.get('blanket_bulk'))
    header = row.get('dummy_header_bytes', 0)
    if (table.shape != (3, 3, 2, 2, 6) or not np.isfinite(table).all()
            or np.any(table < 0) or not 28 <= header <= 60):
        raise ValueError('FlowShift needs packet-derived equal-gap bulk metadata and dummy headers')
    return table, header


def _plan(row, spec: MorphingSpec, settings, holding_s, metadata=None) -> EditPlan:
    table, _ = _metadata(row) if metadata is None else metadata
    counts = np.asarray([row[f'tot_{d}_pkts'] for d in DIRECTIONS], dtype=np.int64)
    before = sum(row[f'totlen_{d}_pkts'] for d in DIRECTIONS)
    budget = math.floor(before * settings['max_extra_byte_fraction'])
    pads = np.zeros(2, dtype=np.int64)
    if spec.mode in ('size', 'balanced'):
        headroom = np.asarray([max(0., 1500 - row[f'{d}_pkt_len_max']) for d in DIRECTIONS])
        pads = np.floor(np.minimum(headroom, spec.strength * np.asarray(spec.padding))).astype(np.int64)
        pads[counts == 0] = 0
        spent = int(pads @ counts)
        if spent > budget:
            pads = np.floor(pads * (budget / spent)).astype(np.int64)
        budget -= int(pads @ counts)
    gaps = np.zeros(2, dtype=np.int64)
    sizes = np.asarray(spec.dummy_sizes, dtype=np.int64)
    if spec.mode.startswith('volume') or spec.mode == 'balanced':
        allowed = ((0,) if spec.mode == 'volume-fwd' else (1,) if spec.mode == 'volume-bwd' else (0, 1))
        for d in allowed:
            if counts[d] > 1:
                gaps[d] = min(2, max(1, math.ceil(spec.strength * max(1., spec.volume_ratios[d] - 1))))
        # Keep only complete equal-gap insertion profiles covered by the metadata.
        slots = np.maximum(0, counts - 1)
        while int(gaps @ slots) * 64 > budget:
            d = max((i for i in (0, 1) if gaps[i]), key=lambda i: (slots[i], i))
            gaps[d] -= 1
        injected = gaps * slots
        minimum, desired = int(injected.sum()) * 64, int(injected @ sizes)
        if desired > budget:
            fraction = (budget - minimum) / (desired - minimum)
            sizes = 64 + np.floor(fraction * (sizes - 64)).astype(np.int64)
    if not row['flow_duration']:
        gaps[:] = 0
    sizes[gaps == 0] = 64
    scale = 1.
    if spec.mode in ('timing', 'balanced') and row['flow_duration'] > 0:
        relative = min(settings['max_fct_fraction'], max(0., spec.clock_ratio - 1)) * spec.strength
        scale += min(relative, holding_s / row['flow_duration'])
    pf, pb = [int(pads[i] > 0 and counts[i] > 0) for i in (0, 1)]
    maximum_gap = table[gaps[0], gaps[1], pf, pb, 5]
    if maximum_gap:
        scale = min(scale, max(1., (1 - 2e-9) / maximum_gap))
    return EditPlan(tuple(map(int, pads)), tuple(map(int, gaps)), tuple(map(int, sizes)),
                    float(scale), (scale - 1) * row['flow_duration'])


def _estimate(row, plan: EditPlan, metadata=None):
    """Exact size moments and certified equal-gap bulk/timing updates."""
    table, header = _metadata(row) if metadata is None else metadata
    if not any(plan.padding) and not any(plan.gaps) and plan.scale == 1:
        return {n: float(row[n]) for n in FEATURE_NAMES}
    # transform_flow uses the same source-mean and interval accounting as other
    # aggregate adapters; bulk fields below replace its generic approximation.
    result = transform_flow(row, plan, FEATURE_NAMES)
    pf, pb = [int(plan.padding[i] > 0 and row[f'tot_{d}_pkts'] > 0)
              for i, d in enumerate(DIRECTIONS)]
    payload, real, dummy, groups, elapsed, _ = table[plan.gaps[0], plan.gaps[1], pf, pb]
    payload += real * plan.padding[0] + dummy * (plan.dummy_sizes[0] - header)
    result['fwd_byts_b_avg'] = payload / groups if groups else 0.
    result['fwd_pkts_b_avg'] = (real + dummy) / groups if groups else 0.
    result['fwd_blk_rate_avg'] = payload / (elapsed * plan.scale) if elapsed else 0.
    return result


class FlowShift(FlowShiftPolicy):
    VERSION = 'flowshift-budgeted-flow-v2'
    ASSUMPTIONS = {
        'scope': 'offline single-policy aggregate adaptation; no VM schedule export or online epoch rotation',
        'size': 'uniform directional padding clipped to the 1500-byte IP limit; original packets are preserved',
        'volume': 'up to two dummies per directional gap; exact packet-derived bulk coefficients',
        'timing': 'equal directional gap splitting and bounded clock dilation; original burst boundaries preserved',
        'holding': 'HMAC-sampled holding allowance caps added timestamp span, not a VM epoch schedule',
        'cost': 'per-flow byte and modeled-span limits; no serialization, queueing, encapsulation or application-FCT model',
        'selection': 'known source service; training profiles and scaled feature displacement, no classifier queries',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostics = Counter()
        self._totals = np.zeros(11, dtype=np.float64)

    @classmethod
    def verify(cls, cached, fingerprint, seed):
        if (not isinstance(cached, dict) or cached.get('generator') != cls.VERSION
                or cached.get('profile_fingerprint') != fingerprint or cached.get('seed') != seed
                or not isinstance(cached.get('overhead'), dict)
                or not isinstance(cached.get('diagnostics'), dict)):
            raise ValueError('FlowShift features are stale; run packmorph-obfuscate flowshift')
        diagnostic = cached['diagnostics']
        if (diagnostic.get('flows', 0) <= 0 or diagnostic['flows'] !=
                diagnostic.get('applied_flows', 0) + diagnostic.get('fallback_flows', 0)):
            raise ValueError('Invalid FlowShift coverage diagnostics; regenerate features')

    def select(self, row: Mapping[str, object]) -> list[Candidate]:
        values = {name: float(row[name]) for name in FEATURE_NAMES}
        if any(not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError('FlowShift requires finite nonnegative behavior features')
        counts = [values[f'tot_{d}_pkts'] for d in DIRECTIONS]
        if any(n != int(n) for n in counts) or sum(counts) <= 0:
            raise ValueError('FlowShift requires positive total integer packet counts')
        if any(not n and values[f'totlen_{d}_pkts'] > 0 for d, n in zip(DIRECTIONS, counts)):
            raise ValueError('FlowShift bytes have no source packets in one direction')
        if any(n and (values[f'{d}_pkt_len_min'] <= 0 or values[f'{d}_pkt_len_max'] > 1500)
               for d, n in zip(DIRECTIONS, counts)):
            raise ValueError('FlowShift requires source packet sizes in 1..1500 bytes')
        before = sum(values[f'totlen_{d}_pkts'] for d in DIRECTIONS)
        if before <= 0:
            raise ValueError('FlowShift requires positive source bytes')
        metadata = _metadata(row)
        holding = self.holding_limit_s(str(row.get('flow_key', self.flows)))
        unique, feasible = set(), []
        for spec in self.bank:
            self.diagnostics['proposed_candidates'] += 1
            plan = _plan(row, spec, self.settings, holding, metadata)
            if plan in unique:
                self.diagnostics['duplicate_plans'] += 1
                continue
            unique.add(plan)
            if not any(plan.padding) and not any(plan.gaps) and plan.scale == 1:
                self.diagnostics['no_effect_candidates'] += 1
                continue
            result = _estimate(row, plan, metadata)
            byte = (sum(result[f'totlen_{d}_pkts'] for d in DIRECTIONS) - before) / before
            duration = values['flow_duration']
            span = result['flow_duration'] / duration - 1 if duration else 0.
            if (byte < -1e-12 or byte > self.settings['max_extra_byte_fraction'] + 1e-12
                    or span > self.settings['max_fct_fraction'] + 1e-12
                    or result['flow_duration'] - duration > holding + 1e-9):
                raise RuntimeError('FlowShift plan violated its byte, span or holding limit')
            feasible.append((spec, result, byte, span))
        if not feasible:
            self.diagnostics['fallback_no_effect'] += 1
            return []
        columns = self.profiles.columns
        encoded = self.profiles.scaler.transform(np.asarray(
            [[values[n] for n in columns]] + [[r[n] for n in columns] for _, r, _, _ in feasible]))
        candidates = []
        for (spec, result, byte, span), point in zip(feasible, encoded[1:]):
            displacement = float(np.linalg.norm(point - encoded[0]) / math.sqrt(len(columns)))
            if displacement >= self.settings['min_displacement']:
                candidates.append(Candidate(spec, result, displacement, byte, span))
            else:
                self.diagnostics['below_displacement'] += 1
        self.diagnostics['eligible_candidates'] += len(candidates)
        if not candidates:
            self.diagnostics['fallback_below_displacement'] += 1
        return candidates

    def __call__(self, row):
        self.capture = row.get(CAPTURE_COLUMN, self.capture)
        key = str(row.get('flow_key', self.flows))
        candidates = self.select(row)
        chosen = self.choose(candidates, key)
        self.flows += 1
        self.diagnostics['flows'] += 1
        if chosen is None:
            self.diagnostics['fallback_flows'] += 1
            result = {n: float(row[n]) for n in FEATURE_NAMES}
        else:
            self.diagnostics['applied_flows'] += 1
            self.diagnostics['selected_' + chosen.policy.mode] += 1
            result = chosen.row
        before = sum(float(row[f'totlen_{d}_pkts']) for d in DIRECTIONS)
        after = sum(result[f'totlen_{d}_pkts'] for d in DIRECTIONS)
        packets = sum(float(row[f'tot_{d}_pkts']) for d in DIRECTIONS)
        extra_packets = sum(result[f'tot_{d}_pkts'] for d in DIRECTIONS) - packets
        old_time = float(row['flow_duration']);added = result['flow_duration'] - old_time
        self._totals[:6] += (before, after, packets, extra_packets, added,
                            added / old_time if old_time else 0.)
        self._totals[6] = max(self._totals[6], added)
        self._totals[7] += int(old_time > 0)
        self._totals[8:] = np.maximum(self._totals[8:], (
            (after - before) / before, added / old_time if old_time else 0., extra_packets / packets))
        return {n: result[n] for n in self.profiles.columns}

    @staticmethod
    def summarize(transforms):
        transforms = tuple(transforms)
        diagnostics = Counter()
        totals = np.zeros(11, dtype=np.float64)
        fingerprints = set()
        for transform in transforms:
            diagnostics.update(transform.diagnostics);totals += transform._totals
            fingerprints.add(transform.fingerprint)
        # Maxima do not add across source-class adapters.
        totals[6] = max((t._totals[6] for t in transforms), default=0.)
        for i in (8, 9, 10):
            totals[i] = max((t._totals[i] for t in transforms), default=0.)
        if len(fingerprints) != 1:
            raise ValueError('FlowShift adapters must share profiles, settings and seed')
        before, after, packets, extra_packets, added, relative, maximum, timed, max_byte, max_span, max_packet = totals
        flows = diagnostics['flows']
        return dict(generator=FlowShift.VERSION, profile_fingerprint=next(iter(fingerprints)),
            diagnostics=dict(diagnostics), assumptions=FlowShift.ASSUMPTIONS,
            overhead=dict(extra_byte_fraction=float(after / before - 1),
                extra_packet_fraction=float(extra_packets / packets),
                mean_added_span_s=float(added / flows), max_added_span_s=float(maximum),
                mean_span_fraction=float(relative / timed) if timed else 0.,
                max_extra_byte_fraction=float(max_byte), max_span_fraction=float(max_span),
                max_extra_packet_fraction=float(max_packet),
                zero_duration_flows=int(flows - timed)))
