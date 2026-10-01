"""Simulation adapters. Published method names are qualified in the report."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dataset import SequenceDataset
from .ditto import simulate as simulate_ditto
from .flowshift_schedule import FlowShiftSchedule
from .qos import isolated_completion
from .securitas import simulate as simulate_securitas
from .traffic import TrafficTrace, costs, make_trace, source_trace, validate_conservation


SUPPORTED = ("original", "front_adapted", "regulator_adapted", "flowshift_manuscript", "ditto_adapted", "securitas_adapted")


@dataclass(slots=True)
class TransformResult:
    trace: TrafficTrace
    target: int | None = None
    status: str = "applied"
    reason: str | None = None
    candidate_policies: int | None = None
    feasible_policies: int | None = None
    selected_policies: tuple[str, ...] = ()
    scheduled_policies: tuple[str, ...] = ()
    schedule_seed: str | None = None
    deadline_bypass_packets: int = 0
    policy_switches: int = 0
    upload_delay_violations: int = 0
    startup_download_dummy_packets: int = 0
    startup_upload_dummy_packets: int = 0
    main_download_dummy_packets: int = 0
    sampled_upload_dummy_packets: int = 0
    state_load: tuple[dict, ...] = ()
    queue_overflow_packets: int = 0
    oversize_drops: int = 0
    dropped_packets: int = 0
    unfinished_packets: int = 0
    outside_horizon_packets: int = 0
    reordering_inversions: int = 0
    mean_genuine_packet_delay_seconds: float | None = None
    action_counts: dict[str, int] = field(default_factory=dict)
    fake_ttls: tuple[int, ...] = ()
    icmp_reply_times: tuple[float, ...] = ()


def _dummy_packets(times: np.ndarray, direction: int, size: int) -> list[tuple[float, int, int, int]]:
    return [(float(t), direction * size, -1, 0) for t in times]


def front(source: TrafficTrace, rng: np.random.Generator, parameters: dict) -> TransformResult:
    """Apply FRONT once to the available trace, using inclusive dummy counts.

    Outgoing time zero is the first packet; incoming time zero is the first
    incoming real packet. A missing incoming packet suppresses that schedule.
    """
    minimum = int(parameters.get("minimum_dummy_packets", 1))
    client_maximum = int(parameters.get("client_dummy_packets", 1700))
    server_maximum = int(parameters.get("server_dummy_packets", 1700))
    window_min = float(parameters.get("minimum_window_seconds", 1.0))
    window_max = float(parameters.get("maximum_window_seconds", 14.0))
    size = int(parameters.get("dummy_size_bytes", 514))
    if minimum < 1 or client_maximum < minimum or server_maximum < minimum or not np.isfinite([window_min, window_max]).all() or window_min <= 0 or window_max < window_min or size <= 0 or size > np.iinfo(np.int32).max:
        raise ValueError("Invalid FRONT parameters")
    packets = [(float(t), int(s), int(i), int(abs(s))) for t, s, i in zip(source.times, source.sizes, source.source_index)]
    end = float(source.times[-1])
    arrivals = source.times[source.sizes < 0]
    for direction, maximum, origin in ((1, client_maximum, 0.0), (-1, server_maximum, float(arrivals[0]) if len(arrivals) else None)):
        if origin is None:
            continue
        count = int(rng.integers(minimum, maximum + 1))
        window = float(rng.uniform(window_min, window_max))
        # Rayleigh draws are absolute offsets, never inter-arrival times.
        dummy_times = origin + rng.rayleigh(window, count)
        dummy_times = dummy_times[dummy_times <= end]
        packets.extend(_dummy_packets(dummy_times, direction, size))
    result = make_trace(packets, stable_timestamp=True)
    validate_conservation(source, result)
    return TransformResult(result)


def regulator(source: TrafficTrace, rng: np.random.Generator, parameters: dict) -> TransformResult:
    """Schedule variable-size packets in independent download/upload FIFO queues."""
    rate = float(parameters.get("orig_rate", 277.0))
    decay = float(parameters.get("dep_rate", 0.94))
    budget = int(parameters.get("budget", 3550))
    threshold = float(parameters.get("threshold", 3.55))
    upload_ratio = float(parameters.get("upload_ratio", 3.95))
    delay_cap = float(parameters.get("delay_cap", 1.77))
    dummy_size = int(parameters.get("dummy_size_bytes", 514))
    if (not np.isfinite([rate, decay, threshold, upload_ratio, delay_cap]).all()
            or rate <= 0 or not 0 < decay <= 1 or budget < 0 or threshold <= 0
            or upload_ratio <= 0 or delay_cap < 0 or not 0 < dummy_size <= np.iinfo(np.int32).max):
        raise ValueError("Invalid RegulaTor parameters")
    down = np.flatnonzero(source.sizes < 0)
    up = np.flatnonzero(source.sizes > 0)
    # The paper says ten packets; the script's target_trace[10] is the 11th.
    # Use the 10th incoming packet, or the last one in a short trace.
    anchor_time = float(source.times[down[min(9, len(down) - 1)]]) if len(down) else 0.0
    dummy_budget = int(rng.integers(0, budget)) if budget else 0
    main_dummies = 0
    startup_down_dummies = 0
    startup_up_dummies = 0
    sampled_up_dummies = 0
    packets: list[tuple[float, int, int, int]] = []
    download_slots: list[float] = []
    next_down = 0

    def emit_real(index: int, when: float) -> None:
        size = int(source.sizes[index])
        packets.append((when, size, index, abs(size)))

    def emit_download_slot(when: float, *, startup: bool) -> None:
        nonlocal next_down, main_dummies, startup_down_dummies
        if next_down < len(down) and source.times[down[next_down]] <= when + 1e-12:
            emit_real(int(down[next_down]), when)
            next_down += 1
        elif startup:
            packets.append((when, -dummy_size, -1, 0))
            startup_down_dummies += 1
        elif main_dummies < dummy_budget:
            packets.append((when, -dummy_size, -1, 0))
            main_dummies += 1
        else:
            return
        download_slots.append(when)

    if len(down):
        # Startup padding is outside the sampled main allowance.
        for slot in range(int(np.ceil(anchor_time * 10 - 1e-12))):
            emit_download_slot(slot * 0.1, startup=True)
        emit_download_slot(anchor_time, startup=True)
        current = anchor_time
        surge_start = anchor_time
        while next_down < len(down):
            target_rate = max(1.0, rate * decay ** max(current - surge_start, 0.0))
            waiting = int(np.searchsorted(source.times[down], current, side="right") - next_down)
            # The current gap uses the pre-reset rate, as in the reference.
            gap = 1.0 / target_rate
            if waiting > threshold * target_rate:
                surge_start = current
            following = current + gap
            if following <= current:
                raise RuntimeError("RegulaTor clock did not advance")
            current = following
            emit_download_slot(current, startup=False)

    sampled_count = int(len(download_slots) / upload_ratio)
    sampled = rng.choice(download_slots, size=sampled_count, replace=True).tolist() if sampled_count else []
    # The reference also starts upload slots at 0.2-second intervals.
    upload_slots = [(float(t), False) for t in sampled]
    upload_slots.extend((slot * 0.2, True) for slot in range(int(np.ceil(anchor_time * 5 - 1e-12))))
    upload_slots.sort(key=lambda item: item[0])
    next_up = 0

    def emit_upload_deadlines(until: float, *, inclusive: bool) -> None:
        nonlocal next_up
        while next_up < len(up):
            index = int(up[next_up])
            deadline = float(source.times[index]) + delay_cap
            if deadline > until + 1e-12 or (not inclusive and deadline >= until - 1e-12):
                break
            emit_real(index, deadline)
            next_up += 1

    for slot, startup in upload_slots:
        emit_upload_deadlines(slot, inclusive=False)
        if next_up < len(up) and source.times[up[next_up]] <= slot + 1e-12:
            emit_real(int(up[next_up]), slot)
            next_up += 1
        else:
            packets.append((slot, dummy_size, -1, 0))
            if startup:
                startup_up_dummies += 1
            else:
                sampled_up_dummies += 1
        emit_upload_deadlines(slot, inclusive=True)
    emit_upload_deadlines(float("inf"), inclusive=True)

    result = make_trace(packets)
    validate_conservation(source, result, global_order=False)
    real = result.source_index >= 0
    if len(result.source_index[real]) != len(source.sizes) or len(np.unique(result.source_index[real])) != len(source.sizes):
        raise RuntimeError("RegulaTor lost or duplicated a real packet")
    upload_real = real & (result.sizes > 0)
    violations = int(np.count_nonzero(result.times[upload_real] - source.times[result.source_index[upload_real]] > delay_cap + 1e-8))
    if violations:
        raise RuntimeError("RegulaTor exceeded an upload deadline")
    return TransformResult(result, upload_delay_violations=violations,
                           startup_download_dummy_packets=startup_down_dummies,
                           startup_upload_dummy_packets=startup_up_dummies,
                           main_download_dummy_packets=main_dummies,
                           sampled_upload_dummy_packets=sampled_up_dummies)


@dataclass(slots=True)
class FlowShiftProfile:
    size: int
    iat: float
    count: float


@dataclass(frozen=True, slots=True)
class FlowShiftVariant:
    name: str
    size: float
    timing: float
    volume: float


def parse_flowshift_variants(value: object) -> tuple[FlowShiftVariant, ...]:
    """Validate the configured offline size, timing, and volume variants."""
    if not isinstance(value, list) or not value:
        raise ValueError("FlowShift policy_variants must be a nonempty list")
    variants = []
    names = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"name", "size", "timing", "volume"}:
            raise ValueError(f"FlowShift policy_variants[{index}] needs name, size, timing, and volume")
        name = item["name"]
        if (not isinstance(name, str) or not name or not all(character.isalnum() or character == "_" for character in name)
                or name.casefold() in names):
            raise ValueError(f"FlowShift policy_variants[{index}] has an invalid or duplicate name")
        names.add(name.casefold())
        strengths = []
        for key in ("size", "timing", "volume"):
            strength = item[key]
            if type(strength) not in (int, float) or not np.isfinite(strength) or not 0 <= strength <= 1:
                raise ValueError(f"FlowShift policy_variants[{index}].{key} must be in [0, 1]")
            strengths.append(float(strength))
        if not any(strengths):
            raise ValueError(f"FlowShift policy_variants[{index}] cannot be all zero")
        variants.append(FlowShiftVariant(name, *strengths))
    return tuple(variants)


@dataclass(frozen=True, slots=True)
class FlowShiftPolicy:
    source: int
    target: int
    variant: FlowShiftVariant
    size: int
    iat: float
    extra_ratio: float

    @property
    def name(self) -> str:
        return f"{self.source}->{self.target}:{self.variant.name}"


@dataclass(slots=True)
class FlowShiftLibrary:
    profiles: dict[int, FlowShiftProfile]
    scale: np.ndarray
    variants: tuple[FlowShiftVariant, ...]
    policy_bank: dict[int, tuple[FlowShiftPolicy, ...]]
    selection_cache: dict[tuple, tuple[tuple[FlowShiftPolicy, ...], int]] = field(default_factory=dict)

    def candidates(self, source: int) -> tuple[FlowShiftPolicy, ...]:
        """Return policies prepared from training data before evaluating flows."""
        return self.policy_bank[source]

    def reference_trace(self, source: int, first_packet_size: int | None = None) -> TrafficTrace:
        """Training profile conditioned on the first packet available to the sender."""
        profile = self.profiles[source]
        count = max(2, round(profile.count))
        directions = np.where(np.arange(count) % 2 == 0, 1, -1)
        sizes = (directions * profile.size).astype(np.int32)
        if first_packet_size is not None:
            if first_packet_size == 0 or first_packet_size == np.iinfo(np.int32).min:
                raise ValueError("First packet must have a nonzero signed frame size")
            sizes[0] = first_packet_size
        times = np.arange(count, dtype=np.float64) * profile.iat
        return TrafficTrace(sizes, times, np.arange(count, dtype=np.int32),
                            np.abs(sizes))

    def select(self, source: int, minimum_displacement: float,
               maximum_bandwidth: float, maximum_duration: float,
               minimum_size: int, maximum_size: int,
               first_packet_size: int | None = None,
               deadline_seconds: float | None = None,
               baseline_seconds: float | None = None,
               link_bps: float | None = None) -> tuple[tuple[FlowShiftPolicy, ...], int]:
        """Choose a Pareto set from offline profiles and the first observed packet."""
        # Cache by a bounded frame-size bucket. An exact-size cache would repeat
        # the full bank evaluation for nearly every frame length in a capture.
        if first_packet_size is not None:
            if first_packet_size == 0 or first_packet_size == np.iinfo(np.int32).min:
                raise ValueError("First packet must have a nonzero signed frame size")
            magnitude = min(2048, ((abs(first_packet_size) + 127) // 128) * 128)
            first_packet_size = int(np.copysign(magnitude, first_packet_size))
        key = (source, minimum_displacement, maximum_bandwidth, maximum_duration,
               minimum_size, maximum_size, first_packet_size,
               deadline_seconds, baseline_seconds, link_bps)
        if key not in self.selection_cache:
            reference = self.reference_trace(source, first_packet_size)
            baseline = _descriptor(reference)
            reference_completion = (isolated_completion(reference, link_bps)
                                    if link_bps is not None else None)
            feasible = []
            for policy in self.candidates(source):
                defended = _flowshift_simulate(reference, [policy] * len(reference.sizes),
                                               minimum_size, maximum_size)
                validate_conservation(reference, defended)
                cost = costs(reference, defended)
                displacement = float(np.linalg.norm((_descriptor(defended) - baseline) / self.scale))
                bandwidth = float(cost["bandwidth_overhead"])
                duration = float(cost["sequence_duration_overhead"])
                predicted_completion = (baseline_seconds * isolated_completion(defended, link_bps)
                                        / reference_completion
                                        if deadline_seconds is not None else None)
                if (displacement >= minimum_displacement
                        and (deadline_seconds is None or predicted_completion <= deadline_seconds + 1e-10)
                        and bandwidth <= maximum_bandwidth + 1e-10
                        and duration <= maximum_duration + 1e-10
                        and not cost["zero_duration_extended"]):
                    feasible.append((policy, (displacement, bandwidth, duration)))
            selected = tuple(policy for policy, metrics in feasible
                             if not any(_dominates(other, metrics) for _, other in feasible))
            self.selection_cache[key] = (selected, len(feasible))
        return self.selection_cache[key]

    @classmethod
    def fit(cls, training: SequenceDataset, policy_variants: list[dict]) -> "FlowShiftLibrary":
        variants = parse_flowshift_variants(policy_variants)
        profiles: dict[int, FlowShiftProfile] = {}
        descriptors = []
        for row in range(len(training)):
            trace = source_trace(training, row)
            descriptors.append(_descriptor(trace))
        scale = np.maximum(np.std(np.asarray(descriptors), axis=0), 1e-6)
        for label in np.unique(training.labels):
            rows = np.flatnonzero(training.labels == label)
            lengths = np.concatenate([np.abs(training.sizes[row, :training.packet_counts[row]]) for row in rows])
            iats = np.concatenate([training.iats[row, 1:training.packet_counts[row]] for row in rows])
            profiles[int(label)] = FlowShiftProfile(
                size=int(np.median(lengths)),
                iat=max(float(np.median(iats[iats > 0])) if np.any(iats > 0) else 0.001, 1e-6),
                count=float(np.mean(training.packet_counts[rows])),
            )
        policy_bank: dict[int, tuple[FlowShiftPolicy, ...]] = {}
        for source, source_profile in profiles.items():
            bank = []
            for target, target_profile in profiles.items():
                if target == source:
                    continue
                for variant in variants:
                    bank.append(FlowShiftPolicy(
                        source=source, target=target, variant=variant,
                        size=max(1, round(source_profile.size + variant.size * (target_profile.size - source_profile.size))),
                        iat=max(1e-6, source_profile.iat + variant.timing * (target_profile.iat - source_profile.iat)),
                        extra_ratio=max(0.0, (target_profile.count / source_profile.count - 1.0) * variant.volume),
                    ))
            policy_bank[source] = tuple(bank)
        return cls(profiles, scale, variants, policy_bank)


def _descriptor(trace: TrafficTrace) -> np.ndarray:
    gaps = np.diff(trace.times)
    return np.asarray([
        np.log1p(np.median(np.abs(trace.sizes))),
        np.log1p(1000.0 * np.median(gaps)) if len(gaps) else 0.0,
        np.log1p(len(trace.sizes)),
        np.log1p(np.abs(trace.sizes).sum()),
    ], dtype=np.float64)


def _flowshift_simulate(source: TrafficTrace, assigned: list[FlowShiftPolicy | None], minimum_size: int, maximum_size: int) -> TrafficTrace:
    packets: list[tuple[float, int, int, int]] = []
    previous = 0.0
    for index, (size, arrival) in enumerate(zip(source.sizes, source.times)):
        policy = assigned[index]
        if policy is None:
            previous = max(previous, float(arrival))
            packets.append((previous, int(size), index, abs(int(size))))
            continue
        frame_size = int(np.clip(policy.size, minimum_size, maximum_size))
        remaining = abs(int(size))
        direction = 1 if size > 0 else -1
        while remaining:
            valid = min(remaining, frame_size)
            release = max(float(arrival), previous + policy.iat)
            packets.append((release, direction * frame_size, index, valid))
            previous = release
            remaining -= valid
        # V requirement is the target class's mean packet-count ratio. Filling
        # is deliberately limited to candidate strength and explicit dummies.
        extras = policy.extra_ratio
        count = int(extras) + int((index + 1) * (extras % 1.0)) - int(index * (extras % 1.0))
        for _ in range(count):
            previous = max(previous, float(arrival)) + policy.iat
            packets.append((previous, direction * frame_size, -1, 0))
    return make_trace(packets)


def _dominates(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    return a[0] >= b[0] and a[1] <= b[1] and a[2] <= b[2] and a != b


def flowshift(source: TrafficTrace, label: int, library: FlowShiftLibrary, rng: np.random.Generator, parameters: dict) -> TransformResult:
    max_bw = float(parameters.get("maximum_bandwidth_overhead", 0.2))
    max_duration = float(parameters.get("maximum_sequence_duration_overhead", 0.2))
    minimum_displacement = float(parameters.get("minimum_displacement", 0.0))
    minimum_hold = float(parameters.get("minimum_hold_seconds", 0.1))
    maximum_hold = float(parameters.get("maximum_hold_seconds", 1.0))
    min_size = int(parameters.get("minimum_packet_size", 40))
    max_size = int(parameters.get("maximum_packet_size", 1514))
    if max_bw < 0 or max_duration < 0 or minimum_displacement < 0 or minimum_hold <= 0 or maximum_hold < minimum_hold or min_size <= 0 or max_size < min_size:
        raise ValueError("Invalid FlowShift parameters")
    deadline = parameters.get("_qos_deadline_seconds")
    baseline = parameters.get("_qos_baseline_seconds")
    link = parameters.get("_qos_link_capacity_bps")
    expected_bytes = parameters.get("_qos_expected_trace_bytes")
    if deadline is not None and (baseline is None or link is None or expected_bytes is None):
        raise ValueError("QoS deadline needs a baseline, link rate, and expected bytes")
    bank = library.candidates(label)
    policies, feasible_count = library.select(label, minimum_displacement, max_bw,
                                               max_duration, min_size, max_size,
                                               int(source.sizes[0]), deadline, baseline, link)
    if not policies:
        return TransformResult(source, status="fallback", reason="no estimated feasible Pareto policy",
                               candidate_policies=len(bank), feasible_policies=0)
    schedule_seed = rng.bytes(16).hex()
    scheduler = FlowShiftSchedule(schedule_seed, len(policies), minimum_hold, maximum_hold)
    assigned: list[FlowShiftPolicy | None] = []
    schedule = []
    observed_bytes = 0
    bypassed = 0
    boundary = 0.0
    current = policies[0]
    for arrival, packet_size in zip(source.times, source.sizes):
        while arrival >= boundary:
            policy_index, holding_time = scheduler.plan(len(schedule))
            current = policies[policy_index]
            schedule.append(current.name)
            boundary += holding_time
            if len(schedule) > 100_000:
                raise ValueError("FlowShift holding-time schedule exceeded 100000 intervals")
        choice: FlowShiftPolicy | None = current
        if deadline is not None:
            remaining = max(float(expected_bytes - observed_bytes), float(abs(int(packet_size))))
            slack = float(deadline - arrival)
            if slack <= 0:
                choice = None
            else:
                required_rate = remaining / slack
                def rate(policy: FlowShiftPolicy) -> float:
                    frame = min(max(policy.size, min_size), max_size)
                    return frame / (max(policy.iat, 8 * frame / link) * (1 + policy.extra_ratio))
                if rate(current) < required_rate:
                    fastest = max(policies, key=rate)
                    choice = fastest if rate(fastest) >= required_rate else None
        if choice is None:
            bypassed += 1
        assigned.append(choice)
        observed_bytes += abs(int(packet_size))
    defended = _flowshift_simulate(source, assigned, min_size, max_size)
    validate_conservation(source, defended)
    cost = costs(source, defended)
    names = tuple(policy.name for policy in policies)
    over_budget = (float(cost["bandwidth_overhead"]) > max_bw + 1e-10
                   or float(cost["sequence_duration_overhead"]) > max_duration + 1e-10
                   or cost["zero_duration_extended"])
    if over_budget and deadline is None:
        return TransformResult(source, status="fallback", reason="combined policy schedule exceeded overhead limit",
                               candidate_policies=len(bank), feasible_policies=feasible_count,
                               selected_policies=names, scheduled_policies=tuple(schedule),
                               schedule_seed=schedule_seed)
    switches = sum(a != b for a, b in zip(assigned, assigned[1:]))
    used = [policy for policy in assigned if policy is not None]
    chosen_target = (max(set(policy.target for policy in used),
                         key=lambda t: sum(x.target == t for x in used)) if used else None)
    status = "over_budget" if over_budget else "partial_fallback" if bypassed else "applied"
    return TransformResult(defended, target=chosen_target, status=status,
                           reason="realized overhead exceeded QoS limit" if over_budget else None,
                           candidate_policies=len(bank), feasible_policies=feasible_count,
                           selected_policies=names, scheduled_policies=tuple(schedule),
                           schedule_seed=schedule_seed, deadline_bypass_packets=bypassed,
                           policy_switches=switches)


def transform(method: str, source: TrafficTrace, label: int, rng: np.random.Generator, parameters: dict, library: FlowShiftLibrary | None = None) -> TransformResult:
    if method == "original":
        return TransformResult(source, status="original")
    if method == "front_adapted":
        return front(source, rng, parameters)
    if method == "regulator_adapted":
        return regulator(source, rng, parameters)
    if method == "ditto_adapted":
        outcome = simulate_ditto(source, rng, parameters)
        return TransformResult(outcome.trace, state_load=tuple(outcome.state_load),
                               queue_overflow_packets=outcome.queue_overflow_packets,
                               oversize_drops=outcome.oversize_drops,
                               dropped_packets=outcome.dropped_packets,
                               unfinished_packets=outcome.unfinished_packets,
                               outside_horizon_packets=outcome.outside_horizon_packets,
                               reordering_inversions=outcome.reordering_inversions,
                               mean_genuine_packet_delay_seconds=outcome.mean_genuine_packet_delay_seconds)
    if method == "securitas_adapted":
        outcome = simulate_securitas(source, parameters["policy"], rng, parameters)
        return TransformResult(outcome.trace, action_counts=outcome.action_counts,
                               fake_ttls=outcome.fake_ttls,
                               icmp_reply_times=outcome.icmp_reply_times)
    if method == "flowshift_manuscript":
        if library is None:
            raise ValueError("FlowShift requires training-only profiles")
        return flowshift(source, label, library, rng, parameters)
    raise ValueError(f"Unknown comparison method: {method}")
