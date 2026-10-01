"""Optional per-flow QoS contracts and packet-trace delivery accounting."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np

from .dataset import SequenceDataset
from .traffic import TrafficTrace, source_trace


CONTRACT_FIELDS = {
    "deadline_seconds", "expected_trace_bytes",
    "maximum_bandwidth_overhead", "maximum_duration_overhead",
}
CONTRACT_CSV_COLUMNS = (
    "environment", "capture", "flow_group", "deadline_seconds",
    "expected_trace_bytes", "maximum_bandwidth_overhead", "maximum_duration_overhead",
)
CSV_FIELDS = set(CONTRACT_CSV_COLUMNS)


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


@dataclass(frozen=True, slots=True)
class FlowContract:
    deadline_seconds: float | None = None
    expected_trace_bytes: int | None = None
    maximum_bandwidth_overhead: float | None = None
    maximum_duration_overhead: float | None = None

    @classmethod
    def parse(cls, values: dict) -> "FlowContract":
        if not isinstance(values, dict) or set(values) - CONTRACT_FIELDS:
            raise ValueError("Unknown QoS contract field")
        parsed = {}
        for name, value in values.items():
            if value is None or value == "":
                continue
            if name == "expected_trace_bytes":
                number = _number(value, name, positive=True)
                if not number.is_integer():
                    raise ValueError("expected_trace_bytes must be an integer")
                parsed[name] = int(number)
            else:
                parsed[name] = _number(value, name, positive=name == "deadline_seconds")
        return cls(**parsed)

    def merged(self, override: "FlowContract") -> "FlowContract":
        return FlowContract(**{
            name: getattr(override, name) if getattr(override, name) is not None else getattr(self, name)
            for name in CONTRACT_FIELDS
        })

    def active(self) -> bool:
        return any(getattr(self, name) is not None for name in CONTRACT_FIELDS)


def validate_qos_config(value: object, classes: list[str]) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "link_capacity_bps", "class_defaults", "flow_contracts_csv"
    }:
        raise ValueError("QoS requires link_capacity_bps, class_defaults, and flow_contracts_csv")
    link = _number(value["link_capacity_bps"], "link_capacity_bps", positive=True)
    defaults = value["class_defaults"]
    if not isinstance(defaults, dict) or set(defaults) - set(classes):
        raise ValueError("QoS class_defaults contains an unknown class")
    for contract in defaults.values():
        FlowContract.parse(contract)
    path = value["flow_contracts_csv"]
    if path is not None and (not isinstance(path, str) or not path):
        raise ValueError("flow_contracts_csv must be a path or null")
    return {"link_capacity_bps": link, "class_defaults": defaults,
            "flow_contracts_csv": path}


def load_contracts(path: str | None) -> dict[tuple[str, str, str], FlowContract]:
    if path is None:
        return {}
    contracts = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("QoS CSV columns must be: " + ", ".join(sorted(CSV_FIELDS)))
        for row in reader:
            key = tuple(row[name].strip() for name in ("environment", "capture", "flow_group"))
            if not all(key) or key in contracts:
                raise ValueError("QoS CSV has an empty or duplicate flow key")
            contract = FlowContract.parse({
                name: float(row[name]) if row[name].strip() else None
                for name in CONTRACT_FIELDS
            })
            if contract.active():
                contracts[key] = contract
    return contracts


def export_contract_template(data: SequenceDataset, path: str | Path) -> Path:
    """Create an edit-ready CSV of extracted flow keys without overwriting one."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with destination.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTRACT_CSV_COLUMNS)
        writer.writeheader()
        for row in range(len(data)):
            key = (str(data.environment_names[data.environment_ids[row]]),
                   str(data.group_names[data.group_ids[row]]),
                   str(data.flow_groups[row]))
            if key in seen:
                continue
            seen.add(key)
            writer.writerow(dict(zip(CONTRACT_CSV_COLUMNS[:3], key)))
    return destination


def isolated_completion(trace: TrafficTrace, link_bps: float) -> float:
    """Last genuine byte's delivery time on an otherwise idle serial link."""
    finish = last_real = 0.0
    for release, size, source_index in zip(trace.times, trace.sizes, trace.source_index):
        finish = max(finish, float(release)) + 8 * abs(int(size)) / link_bps
        if source_index >= 0:
            last_real = finish
    return last_real


def _flow_rows(data: SequenceDataset) -> dict[tuple[int, str], list[int]]:
    groups = {}
    for row in range(len(data)):
        groups.setdefault((int(data.group_ids[row]), str(data.flow_groups[row])), []).append(row)
    return groups


@dataclass(slots=True)
class QoSContext:
    settings: dict
    overrides: dict[tuple[str, str, str], FlowContract]
    baseline_seconds: dict[int, float]
    expected_bytes: dict[int, int]
    baseline_cache: dict[int, dict] = field(default_factory=dict)

    @classmethod
    def fit(cls, training: SequenceDataset, settings: dict,
            overrides: dict[tuple[str, str, str], FlowContract]) -> "QoSContext":
        durations: dict[int, list[float]] = {}
        byte_counts: dict[int, list[int]] = {}
        for rows in _flow_rows(training).values():
            label = int(training.labels[rows[0]])
            duration = max(float(training.start_offsets[row] + source_trace(training, row).times[-1])
                           for row in rows)
            byte_count = sum(int(np.abs(training.sizes[row, :training.packet_counts[row]]).sum())
                             for row in rows)
            durations.setdefault(label, []).append(duration)
            byte_counts.setdefault(label, []).append(byte_count)
        return cls(settings, overrides,
                   {label: max(float(np.percentile(values, 75)), 1e-6)
                    for label, values in durations.items()},
                   {label: max(round(float(np.percentile(values, 75))), 1)
                    for label, values in byte_counts.items()})

    def contract(self, data: SequenceDataset, row: int) -> FlowContract:
        label = str(data.class_names[data.labels[row]])
        default = FlowContract.parse(self.settings["class_defaults"].get(label, {}))
        key = (str(data.environment_names[data.environment_ids[row]]),
               str(data.group_names[data.group_ids[row]]), str(data.flow_groups[row]))
        return default.merged(self.overrides.get(key, FlowContract()))

    @property
    def link_bps(self) -> float:
        return float(self.settings["link_capacity_bps"])


def _deliver(data: SequenceDataset, traces: list[TrafficTrace], link_bps: float) -> dict:
    """FCFS single-link queue, independent for each capture."""
    events: dict[int, list[tuple[float, int, int, int, int]]] = {}
    for row, trace in enumerate(traces):
        capture = int(data.group_ids[row])
        origin = float(data.flow_start_timestamps[row] + data.start_offsets[row])
        for packet, (release, size, source_index) in enumerate(
            zip(trace.times, trace.sizes, trace.source_index)
        ):
            events.setdefault(capture, []).append(
                (origin + float(release), row, packet, abs(int(size)), int(source_index))
            )
    completion = {}
    for capture, packets in events.items():
        packets.sort()
        finish = 0.0
        for release, row, _, size, source_index in packets:
            finish = max(finish, release) + 8 * size / link_bps
            if source_index >= 0:
                key = (capture, str(data.flow_groups[row]))
                completion[key] = finish
    return completion


def evaluate_qos(data: SequenceDataset, traces: list[TrafficTrace],
                 context: QoSContext, results: list | None = None) -> tuple[dict, list[dict]]:
    """Compare original and defended completion on the same capture-local link."""
    key_for_cache = id(data)
    if key_for_cache not in context.baseline_cache:
        context.baseline_cache[key_for_cache] = _deliver(
            data, [source_trace(data, row) for row in range(len(data))], context.link_bps)
    baseline = context.baseline_cache[key_for_cache]
    defended = _deliver(data, traces, context.link_bps)
    for key, rows in _flow_rows(data).items():
        for row in rows:
            real = traces[row].source_index >= 0
            capacity = np.bincount(traces[row].source_index[real],
                                   weights=traces[row].valid_bytes[real],
                                   minlength=int(data.packet_counts[row]))
            if not np.array_equal(capacity.astype(np.int64),
                                  np.abs(data.sizes[row, :data.packet_counts[row]]).astype(np.int64)):
                defended.pop(key, None)
                break
    records = []
    for key, rows in _flow_rows(data).items():
        row = rows[0]
        contract = context.contract(data, row)
        if not contract.active():
            continue
        start = float(data.flow_start_timestamps[row])
        original = baseline.get(key)
        actual = defended.get(key)
        original = None if original is None else max(0.0, original - start)
        actual = None if actual is None else max(0.0, actual - start)
        deadline = contract.deadline_seconds
        source_bytes = sum(int(np.abs(data.sizes[item, :data.packet_counts[item]]).sum())
                           for item in rows)
        sent_bytes = sum(int(np.abs(traces[item].sizes).sum()) for item in rows)
        met = actual is not None and (deadline is None or actual <= deadline + 1e-9)
        baseline_met = original is not None and (deadline is None or original <= deadline + 1e-9)
        records.append({
            "environment": str(data.environment_names[data.environment_ids[row]]),
            "capture": str(data.group_names[key[0]]), "flow_group": key[1],
            "class": str(data.class_names[data.labels[row]]),
            "transform_status": results[row].status if results is not None else None,
            "policy_switches": results[row].policy_switches if results is not None else None,
            "deadline_bypass_packets": results[row].deadline_bypass_packets
                if results is not None else None,
            "deadline_seconds": deadline,
            "baseline_completion_seconds": original,
            "completion_seconds": actual,
            "deadline_met": met if deadline is not None else None,
            "baseline_deadline_met": baseline_met if deadline is not None else None,
            "lateness_seconds": max(0.0, actual - deadline) if actual is not None and deadline is not None else None,
            "trace_goodput_bytes_per_second": source_bytes / actual if actual and actual > 0 else None,
            "bandwidth_overhead": (sent_bytes - source_bytes) / source_bytes,
            "duration_overhead": (actual - original) / original
                if actual is not None and original and original > 0 else None,
            "duration_budget_met": (actual - original) / original <= contract.maximum_duration_overhead + 1e-9
                if actual is not None and original and original > 0
                and contract.maximum_duration_overhead is not None else None,
            "bandwidth_budget_met": (sent_bytes - source_bytes) / source_bytes <= contract.maximum_bandwidth_overhead + 1e-9
                if contract.maximum_bandwidth_overhead is not None else None,
        })
    deadlines = [record for record in records if record["deadline_seconds"] is not None]
    summary = {
        "link_capacity_bps": context.link_bps,
        "class_defaults": context.settings["class_defaults"],
        "flow_contracts_csv": context.settings["flow_contracts_csv"],
        "training_baseline_p75_seconds": {
            str(data.class_names[label]): value for label, value in context.baseline_seconds.items()
        },
        "contracted_flows": len(records),
        "deadline_flows": len(deadlines),
        "deadline_met_flows": sum(record["deadline_met"] for record in deadlines),
        "deadline_missed_flows": sum(not record["deadline_met"] for record in deadlines),
        "baseline_deadline_missed_flows": sum(not record["baseline_deadline_met"] for record in deadlines),
        "defense_induced_deadline_misses": sum(
            record["baseline_deadline_met"] and not record["deadline_met"] for record in deadlines
        ),
        "maximum_lateness_seconds": max(
            (record["lateness_seconds"] for record in deadlines if record["lateness_seconds"] is not None),
            default=None,
        ),
    }
    return summary, records
