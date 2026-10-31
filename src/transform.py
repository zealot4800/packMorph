"""Shared flow validation, bounded aggregate edits and method dispatch.

Times are seconds and size variances are population variances. Model outputs
stay N-by-16 and require supporting flow_context. Timing and bulk membership
remain aggregate modeling assumptions.
"""
from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module

import numpy as np

from .features import FlowDataset


MAX_SIZE_BYTES = 1500
CAPTURE_COLUMN = "source_capture"
DIRECTIONS = ("fwd", "bwd")
FEATURE_NAMES = (
    "pkt_len_mean", "pkt_len_std", "pkt_len_min",
    "fwd_pkt_len_mean", "fwd_pkt_len_std", "fwd_pkt_len_min", "fwd_pkt_len_max",
    "bwd_pkt_len_mean", "bwd_pkt_len_std", "bwd_pkt_len_min", "bwd_pkt_len_max",
    "flow_duration", "flow_iat_mean", "fwd_iat_std", "bwd_iat_std",
    "fwd_byts_b_avg", "fwd_pkts_b_avg", "fwd_blk_rate_avg",
    "tot_fwd_pkts", "tot_bwd_pkts", "totlen_fwd_pkts", "totlen_bwd_pkts",
    "flow_pkts_s", "bwd_pkts_s", "down_up_ratio",
)


def _number(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _size(value: object, minimum: int = 1) -> bool:
    return type(value) is int and minimum <= value <= MAX_SIZE_BYTES


def validate_policy_parameters(strategy: str, parameters: dict, required: dict,
                               optional: dict, source: str, *, byte_fraction_kind='byte_budget') -> dict:
    """Validate a method's schema using shared numeric and shape rules."""
    if not isinstance(parameters, dict):
        raise ValueError(f'{source}: {strategy} parameters must be a mapping')
    rules = dict(required)
    if byte_fraction_kind is not None and (byte_fraction_kind != 'byte_budget' or 'max_extra_byte_fraction' in parameters):
        rules = {'max_extra_byte_fraction': byte_fraction_kind, **rules}
    rules.update({name: kind for name, kind in optional.items() if name in parameters})
    if set(parameters) != set(rules):
        raise ValueError(f"{source}: {strategy} requires exactly {', '.join(rules)}")
    for name, kind in rules.items():
        value = parameters[name]
        if kind == "fraction":
            valid = _number(value) and 0 <= value <= 1
        elif kind == "byte_budget":
            valid = value is None or (_number(value) and value >= 0)
        elif kind == "nonnegative":
            valid = _number(value) and value >= 0
        elif kind == 'positive_count':
            valid = type(value) is int and value >= 1
        elif kind in {"size", "size_zero"}:
            valid = _size(value, 0 if kind == "size_zero" else 1)
        elif kind == 'strengths':
            valid = (isinstance(value, list) and bool(value)
                     and all(_number(s) and 0 < s <= 1 for s in value)
                     and len(set(value)) == len(value))
        elif kind == 'bounds':
            valid = (isinstance(value, list) and len(value) == 2
                     and all(type(v) is int and v >= 1 for v in value)
                     and value[0] <= value[1])
        else:
            raise ValueError(f'Unknown policy validation rule: {kind}')
        if not valid:
            raise ValueError(f"Invalid {strategy}.{name} in {source}")
    return parameters


def payload_segments(times, directions, payloads, regions):
    """Compress untouched payload groups between possible edit markers.

    Rows are region, direction, count, payload bytes, first/last ns. Direction
    -1 carries fixed interior bulk totals: groups, bytes, packets, elapsed ns.
    Only the boundary groups can merge with edits in neighboring regions.
    """
    keep = np.flatnonzero(payloads > 0)
    if not len(keep):
        return np.empty((0, 6), dtype=np.float64)
    t, d, p, r = (a[keep] for a in (times, directions, payloads, regions))
    first = np.r_[0, np.flatnonzero((np.diff(r) != 0) | (np.diff(d) != 0)
                                    | (np.diff(t) > 1_000_000_000)) + 1]
    last = np.r_[first[1:], len(t)]
    groups = np.column_stack((r[first], d[first], last - first,
                             np.add.reduceat(p, first), t[first], t[last - 1]))
    boundaries = np.r_[0, np.flatnonzero(np.diff(groups[:, 0]) != 0) + 1, len(groups)]
    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        block = groups[start:end]
        segments.append(block[0])
        if len(block) > 2:
            middle = block[1:-1]
            bulk = middle[(middle[:, 1] == 0) & (middle[:, 2] >= 4)]
            segments.append([block[0, 0], -1, len(bulk), bulk[:, 3].sum(),
                             bulk[:, 2].sum(), (bulk[:, 5] - bulk[:, 4]).sum()])
        if len(block) > 1:
            segments.append(block[-1])
    return np.asarray(segments, dtype=np.float64).reshape(-1, 6)


def forward_bulk_statistics(events):
    """Merge edit groups with compressed untouched payload groups."""
    events = events[np.argsort(events[:, 0], kind='stable')]
    totals = np.zeros(4, dtype=np.float64)
    current = None

    def flush():
        if current is not None and current[0] == 0 and current[1] >= 4:
            totals[:] += (1, current[2], current[1], current[4] - current[3])

    for _, direction, count, payload, first, last in events:
        if direction == -1:
            flush()
            totals += (count, payload, first, last)
            current = None
        elif current is not None and direction == current[0] and first - current[4] <= 1e9:
            current[1] += count
            current[2] += payload
            current[4] = last
        else:
            flush()
            current = [direction, count, payload, first, last]
    flush()
    groups, payload, count, elapsed = totals
    return dict(fwd_byts_b_avg=payload / groups if groups else 0.,
                fwd_pkts_b_avg=count / groups if groups else 0.,
                fwd_blk_rate_avg=payload / (elapsed / 1e9) if elapsed else 0.)


class FlowPolicy:
    """Common settings and hooks for bounded directional packet edits."""

    name: str
    required = {}
    optional = {}
    padding = extension = 0

    @classmethod
    def validate(cls, parameters: dict, source: str) -> dict:
        return validate_policy_parameters(cls.name, parameters, cls.required, cls.optional, source)

    def __init__(self, parameters: dict):
        self.validate(parameters, 'flow policy')
        self.byte_fraction = parameters.get('max_extra_byte_fraction')
        self.insertions = {direction: () for direction in DIRECTIONS}

    def byte_budget(self, counts, totals):
        """Return a byte cap, or enough bytes for every configured edit."""
        if self.byte_fraction is not None:
            return math.floor(sum(totals) * self.byte_fraction)
        dummies = sum(size * count for groups in self.insertions.values()
                      for size, count in groups)
        # Cover headers, padding after a split, and every requested dummy.
        return 2 * (sum(counts) * (2 * MAX_SIZE_BYTES + 60) + dummies)

    def edits(self, direction, count, mean):
        if not count:
            return 0.0, ()
        return self.padding, self.insertions[direction]


@dataclass(slots=True)
class SizeMoments:
    """Sufficient statistics for size edits; no individual packets are stored."""

    count: int
    total: float
    variance: float
    minimum: float
    maximum: float

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @classmethod
    def from_values(cls, values: dict[str, float], direction: str) -> "SizeMoments":
        count = values[f"tot_{direction}_pkts"]
        total = values[f"totlen_{direction}_pkts"]
        if count != int(count) or count < 0 or total < 0:
            raise ValueError("Packet counts must be nonnegative integers and bytes nonnegative")
        if not count:
            if total:
                raise ValueError("A direction with no packets cannot contain bytes")
            return cls.constant(0, 0)
        # Counts and byte totals determine the mean. Preserve measured variance.
        mean = total / count
        prefix = f"{direction}_pkt_len_"
        return cls(int(count), total, values[prefix + "std"] ** 2,
                   min(values[prefix + "min"], mean), max(values[prefix + "max"], mean))

    @classmethod
    def constant(cls, count: int, size: float) -> "SizeMoments":
        if count != int(count) or count < 0 or not math.isfinite(size) or size < 0:
            raise ValueError("Invalid constant-size packet group")
        return cls(int(count), count * size, 0.0, size if count else 0.0, size if count else 0.0)

    def pad(self, size: float) -> None:
        """Adding the same size to every packet preserves its variance."""
        if not math.isfinite(size) or size < 0:
            raise ValueError("Padding must be finite and nonnegative")
        if self.count:
            self.total += self.count * size
            self.minimum += size
            self.maximum += size

    def merge(self, other: "SizeMoments") -> None:
        """Pool populations using centered moments to avoid cancellation."""
        if not other.count:
            return
        count = self.count + other.count
        delta = other.mean - self.mean
        m2 = (self.count * self.variance + other.count * other.variance
              + delta * delta * self.count * other.count / count)
        self.minimum = min(self.minimum, other.minimum) if self.count else other.minimum
        self.maximum = max(self.maximum, other.maximum) if self.count else other.maximum
        self.total += other.total
        self.count = count
        self.variance = max(0.0, m2 / count)

def flow_statistics(values: dict[str, float], directions: list[SizeMoments],
                    duration: float, *, timing: str = "dilated",
                    fwd_padding: float = 0) -> dict[str, float]:
    """Derive dependent CSV fields once for every policy adapter.

    ``dilated`` scales old spans and compresses intervals after insertion.
    ``uniform`` assumes equally spaced arrivals over the modeled duration.
    Neither model recovers packet order or bulk membership.
    """
    if len(directions) != 2 or timing not in ("dilated", "uniform"):
        raise ValueError("Expected two directions and a supported timing model")
    combined = SizeMoments.constant(0, 0)
    for summary in directions:
        combined.merge(summary)
    if not combined.count or not math.isfinite(duration) or duration < 0:
        raise ValueError("A flow needs packets and a finite nonnegative duration")
    if combined.count == 1:
        duration = 0.0  # A constant offset of one arrival is not a measured span.
    old_duration = values["flow_duration"]
    time_scale = duration / old_duration if old_duration else 1.0
    result = values.copy()

    def sizes(prefix: str, summary: SizeMoments) -> None:
        result.update({prefix + "mean": summary.mean, prefix + "std": math.sqrt(summary.variance),
                       prefix + "min": summary.minimum, prefix + "max": summary.maximum})

    for direction, summary in zip(DIRECTIONS, directions):
        result[f"tot_{direction}_pkts"] = summary.count
        result[f"totlen_{direction}_pkts"] = summary.total
        sizes(f"{direction}_pkt_len_", summary)
    # Retain bulk membership and model padding by its directional mean.
    # Adapters with packet-derived metadata replace these bulk estimates.
    bulk_bytes = values["fwd_byts_b_avg"]
    bulk_scale = 1 + fwd_padding * values["fwd_pkts_b_avg"] / bulk_bytes if bulk_bytes else 1.0
    result.update(pkt_len_mean=combined.mean, pkt_len_std=math.sqrt(combined.variance),
                  pkt_len_min=combined.minimum, flow_duration=duration,
                  flow_pkts_s=combined.count / duration if duration else 0.0,
                  flow_iat_mean=duration / (combined.count - 1) if combined.count > 1 else 0.0,
                  bwd_pkts_s=directions[1].count / duration if duration else 0.0,
                  fwd_byts_b_avg=bulk_bytes * bulk_scale,
                  fwd_blk_rate_avg=values["fwd_blk_rate_avg"] * bulk_scale / time_scale if time_scale else 0.0,
                  fwd_pkts_b_avg=values["fwd_pkts_b_avg"],
                  down_up_ratio=directions[1].count / directions[0].count if directions[0].count else 0.0)
    old_counts = [values[f"tot_{d}_pkts"] for d in DIRECTIONS]
    for direction, old_count, summary in zip(DIRECTIONS, old_counts, directions):
        factor = time_scale * max(old_count - 1, 0) / max(summary.count - 1, 1)
        result[f"{direction}_iat_std"] = (0.0 if timing == "uniform" or old_duration == 0 else
                                           values[f"{direction}_iat_std"] * factor)
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError("Aggregate edits produced invalid flow statistics")
    return result


def transform_flow(row: Mapping[str, object], policy: FlowPolicy,
                   feature_names: tuple[str, ...]) -> dict[str, float]:
    """Compute consistent edits and return only the configured model inputs."""
    missing = sorted(set(FEATURE_NAMES) - set(row))
    if missing:
        raise ValueError('Missing flow context: ' + ', '.join(missing))
    values = {name: float(row[name]) for name in FEATURE_NAMES}
    values = {name: max(0.0, value) if math.isfinite(value) and "_iat_" in name else value
              for name, value in values.items()}
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Flow behavior features must be finite and nonnegative")
    directions = [SizeMoments.from_values(values, d) for d in DIRECTIONS]
    counts = [s.count for s in directions]
    if not sum(counts):
        raise ValueError("Flow packet counts must have a positive total")
    totals = [s.total for s in directions]
    budget = policy.byte_budget(counts, totals)
    fwd_padding = 0
    for index, (direction, summary) in enumerate(zip(DIRECTIONS, directions)):
        if not summary.count:
            continue
        pad, requested = policy.edits(direction, counts[index], summary.mean)
        # The surrogate applies whole-byte padding uniformly to existing packets.
        pad = math.floor(min(pad, max(0.0, MAX_SIZE_BYTES - summary.maximum),
                             budget / summary.count))
        budget -= pad * summary.count
        if index == 0:
            fwd_padding += pad
        if pad:
            summary.pad(pad)
        for dummy_size, wanted in requested:
            dummy_size = max(1, round(dummy_size))
            added = min(wanted, budget // dummy_size)
            if added:
                summary.merge(SizeMoments.constant(added, dummy_size))
                budget -= added * dummy_size
    duration = values["flow_duration"] + policy.extension
    derived = flow_statistics(values, directions, duration, fwd_padding=fwd_padding)
    return {name: derived[name] for name in feature_names}


def _context_rows(data: FlowDataset, context: Mapping | Sequence | None) -> Iterator[dict]:
    """Combine model inputs with supplied context, preserving original precision."""
    required = set(FEATURE_NAMES)
    model_fields = set(data.feature_names)
    if (data.features.shape != (len(data), 16) or len(data.feature_names) != 16
            or len(model_fields) != 16 or not model_fields <= required):
        raise ValueError('Transformations require 16 supported model features in order')
    if isinstance(context, Mapping):
        if any(len(column) != len(data) for column in context.values()):
            raise ValueError('Flow context columns must have one value per PKL row')
        extra = ({name: column[i] for name, column in context.items()} for i in range(len(data)))
    elif context is None:
        extra = ({} for _ in range(len(data)))
    elif len(context) != len(data):
        raise ValueError('Flow context must have one mapping per PKL row')
    else:
        extra = context
    for i, auxiliary in enumerate(extra):
        if not isinstance(auxiliary, Mapping):
            raise ValueError(f'Flow context at PKL row {i} must be a mapping')
        row = dict(zip(data.feature_names, data.features[i]))
        row.update(auxiliary)
        missing = required.difference(row)
        if missing:
            raise ValueError('Fresh transformations need flow_context with missing statistics: '
                             + ', '.join(sorted(missing)))
        values = {name: float(row[name]) for name in FEATURE_NAMES}
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError(f'Flow context must be finite and nonnegative at PKL row {i}')
        if not np.array_equal(np.asarray([values[n] for n in data.feature_names], dtype=np.float32),
                              data.features[i]):
            raise ValueError(f'Flow context does not match model inputs at PKL row {i}')
        row.update(values)
        row[CAPTURE_COLUMN] = str(data.captures[i])
        record = data.record_ids[i]
        row.setdefault('flow_key', record.decode() if isinstance(record, bytes) else str(record))
        yield row


_POLICIES = {'flowshift': 'FlowShift', 'blanket': 'Blanket', 'ditto': 'Ditto',
             'minipatch': 'Minipatch', 'minos': 'Minos', 'securitas': 'Securitas'}
METHODS = tuple(_POLICIES)


def _policy_class(strategy: str):
    """Load methods on demand so they can import the shared engine safely."""
    if strategy not in _POLICIES:
        raise ValueError(f'Unknown strategy: {strategy}')
    return getattr(import_module(f'.{strategy}', __package__), _POLICIES[strategy])


def validate_parameters(strategy: str, parameters: dict, source: str) -> dict:
    """Validate parameters using the owning method's schema and constraints."""
    return _policy_class(strategy).validate(parameters, source)


def transform_dataset(data: FlowDataset, method: str, parameters: dict, *,
                      context: Mapping | Sequence | None = None,
                      training: FlowDataset | None = None,
                      training_context: Mapping | Sequence | None = None,
                      seed: int = 42, fitted_policy=None, summary: dict | None = None) -> np.ndarray:
    """Generate fresh features without classifier queries during transformation.

    ``context`` is a column-to-array mapping or a sequence of row dictionaries.
    Counts/duration cannot be recovered reliably from the 16 model features.
    FlowShift additionally needs train.pkl and its context to fit target profiles.
    Blanket, MiniPatch, Ditto and Securitas require fitted policies from the caller.
    Minos requires packet-derived front-window coefficients in flow_context.
    Original row order, labels and flow identities are retained by the caller.
    """
    policy_class = _policy_class(method)
    rows = _context_rows(data, context)
    if method in ('blanket', 'minipatch', 'ditto', 'securitas'):
        if (not isinstance(fitted_policy, policy_class)
                or fitted_policy.parameters != parameters
                or fitted_policy.columns != data.feature_names or fitted_policy.seed != seed):
            raise ValueError(f'{method} needs a matching fitted policy; use packmorph-obfuscate {method}')
        fitted_policy.reset()
    elif method == 'minos':
        policy = policy_class(parameters, data.feature_names, seed)
    elif method == 'flowshift':
        policy_class.validate(parameters, 'transformation')
        if training is None:
            raise ValueError('FlowShift needs train.pkl and its flow context')
        if (training.feature_names != data.feature_names or training.class_names != data.class_names
                or training.source_fingerprint != data.source_fingerprint):
            raise ValueError('FlowShift training data must match the input environment and features')
        from .policy import Profiles

        try:
            profiles = Profiles.fit(training, _context_rows(training, training_context))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f'FlowShift train.pkl flow_context: {error}') from error
        transforms = {}
    def values():
        for i, row in enumerate(rows):
            if method == 'flowshift':
                label = int(data.labels[i])
                if label not in transforms:
                    transforms[label] = policy_class(profiles, data.class_names[label], parameters, str(seed))
                output = transforms[label](row)
            elif method in ('blanket', 'ditto', 'securitas'):
                output = fitted_policy(row)
            elif method == 'minipatch':
                output = fitted_policy(row, data.class_names[int(data.labels[i])])
            elif method == 'minos':
                output = policy(row)
            # Both adapters return the configured model columns in order.
            yield from output.values()

    result = np.fromiter(values(), dtype=np.float32, count=len(data) * 16).reshape(len(data), 16)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError('Transformation produced invalid model features')
    if method == 'minos' and summary is not None:
        summary.update(policy.report)
    elif method == 'flowshift' and summary is not None:
        summary.update(policy_class.summarize(transforms.values()))
    return result
