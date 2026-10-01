"""Capture-disjoint comparison of five packet-trace defenses with one CNN."""

from __future__ import annotations

import logging
import hashlib
from collections import Counter
from dataclasses import replace
from typing import Iterator

import numpy as np

from .dataset import SequenceDataset
from .defenses import SUPPORTED, FlowShiftLibrary, TransformResult, parse_flowshift_variants, transform
from .ditto import fit_pattern
from .extract import extract_dataset
from .features import FeatureScaler
from .model import build_sequence_model, predict_probabilities, prepare_runtime, train_model
from .qos import QoSContext, evaluate_qos, load_contracts, validate_qos_config
from .split import class_weights, stratified_group_split
from .traffic import TrafficTrace, costs, source_trace, validate_conservation, windows


logger = logging.getLogger(__name__)
_runtime_device: str | None = None


def validate_comparison_config(comparison: dict, base: dict) -> dict:
    required = {"seeds", "budgets", "methods"}
    if not isinstance(comparison, dict) or not required <= set(comparison) or set(comparison) - required - {"qos"}:
        raise ValueError(f"Comparison config requires {', '.join(sorted(required))}")
    seeds = comparison["seeds"]
    if (not isinstance(seeds, list) or len(seeds) < 2
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Comparison needs at least two distinct nonnegative seeds")
    budgets = comparison["budgets"]
    if (not isinstance(budgets, dict)
            or set(budgets) != {"bandwidth_overhead", "sequence_duration_overhead"}
            or any(type(value) not in (int, float) or not np.isfinite(value) or value < 0
                   for value in budgets.values())):
        raise ValueError("Comparison budgets must be finite nonnegative ratios")
    methods = comparison["methods"]
    if not isinstance(methods, dict) or set(methods) != set(SUPPORTED) - {"original"}:
        raise ValueError("Comparison must configure all five defenses")
    for name, choices in methods.items():
        point = "learned" if name == "securitas_adapted" else "example" if name == "ditto_adapted" else "published"
        expected = {point, "candidates"}
        if name == "flowshift_manuscript":
            expected.add("policy_variants")
        if (not isinstance(choices, dict) or set(choices) != expected
                or not isinstance(choices[point], dict) or not isinstance(choices["candidates"], list)
                or not choices["candidates"] or any(not isinstance(item, dict) for item in choices["candidates"])):
            raise ValueError(f"Invalid {name} operating points")
        if name == "flowshift_manuscript":
            parse_flowshift_variants(choices["policy_variants"])
        if name == "front_adapted" and set(choices["published"]) != {"FT-1", "FT-2"}:
            raise ValueError("FRONT needs FT-1 and FT-2 published presets")
        if name == "regulator_adapted" and set(choices["published"]) != {"Light", "Heavy"}:
            raise ValueError("RegulaTor needs Light and Heavy published presets")
        if (name == "ditto_adapted"
                and {int(item.get("pattern_length", 0)) for item in choices["candidates"]} != {3, 4, 5, 6}):
            raise ValueError("ditto candidates must cover pattern lengths 3–6")
    comparison["qos"] = validate_qos_config(comparison.get("qos", {
        "link_capacity_bps": 250_000_000, "class_defaults": {}, "flow_contracts_csv": None
    }), base["data"]["categories"])
    return {"comparison": comparison, "base": base}


def _split(data: SequenceDataset, seed: int, base: dict) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    development_rows, test_rows = stratified_group_split(data, base["split"]["test_fraction"], seed,
                                                         minimum_retained_groups=2)
    development = data.subset(development_rows)
    train_rows, validation_rows = stratified_group_split(development,
                                                         base["model"]["validation_fraction"], seed + 1)
    parts = (development.subset(train_rows), development.subset(validation_rows), data.subset(test_rows))
    groups = [set(part.group_ids.tolist()) for part in parts]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Capture groups overlap across partitions")
    return parts


def _draw(method: str, data: SequenceDataset, library: FlowShiftLibrary, seed: int,
          role: int, parameters: dict,
          qos: QoSContext | None = None) -> tuple[list[TrafficTrace], list[TransformResult], list[dict]]:
    traces, results, overheads = [], [], []
    grouped_results = _draw_flowshift_flows(data, library, seed, role, parameters, qos) if method == "flowshift_manuscript" else None
    for row in range(len(data)):
        source = source_trace(data, row)
        if grouped_results is None:
            rng = np.random.default_rng(np.random.SeedSequence([seed, role, row]))
            result = transform(method, source, int(data.labels[row]), rng, parameters, library)
        else:
            result = grouped_results[row]
        validate_conservation(source, result.trace,
                              global_order=None if method == "ditto_adapted" else method != "regulator_adapted",
                              require_complete=method != "ditto_adapted")
        if method == "front_adapted":
            real = result.trace.source_index >= 0
            if not (np.array_equal(result.trace.source_index[real], source.source_index)
                    and np.array_equal(result.trace.sizes[real], source.sizes)
                    and np.array_equal(result.trace.times[real], source.times)):
                raise RuntimeError("FRONT modified an original packet")
        cost = costs(source, result.trace)
        cost.update(upload_delay_violations=result.upload_delay_violations,
                    queue_overflow_packets=result.queue_overflow_packets,
                    oversize_drops=result.oversize_drops, dropped_packets=result.dropped_packets,
                    unfinished_packets=result.unfinished_packets,
                    outside_horizon_packets=result.outside_horizon_packets,
                    reordering_inversions=result.reordering_inversions,
                    mean_genuine_packet_delay_seconds=result.mean_genuine_packet_delay_seconds,
                    startup_download_dummy_packets=result.startup_download_dummy_packets,
                    startup_upload_dummy_packets=result.startup_upload_dummy_packets,
                    main_download_dummy_packets=result.main_download_dummy_packets,
                    sampled_upload_dummy_packets=result.sampled_upload_dummy_packets)
        if method == "regulator_adapted":
            size = int(parameters["dummy_size_bytes"])
            for phase in ("startup_download", "main_download", "startup_upload", "sampled_upload"):
                cost[f"{phase}_dummy_bytes"] = cost[f"{phase}_dummy_packets"] * size
            if sum(cost[f"{phase}_dummy_bytes"] for phase in
                   ("startup_download", "main_download", "startup_upload", "sampled_upload")) != cost["dummy_bytes"]:
                raise RuntimeError("RegulaTor dummy byte accounting disagrees")
        traces.append(result.trace)
        results.append(result)
        overheads.append(cost)
    return traces, results, overheads


def _flow_chunk_number(flow_id: str) -> int:
    suffix = flow_id.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isdecimal() else 0


def _draw_flowshift_flows(data: SequenceDataset, library: FlowShiftLibrary,
                          seed: int, role: int, parameters: dict,
                          qos: QoSContext | None = None) -> list[TransformResult]:
    """Select and schedule once per original flow, then return its CNN windows."""
    groups: dict[tuple[int, str], list[int]] = {}
    for row in range(len(data)):
        groups.setdefault((int(data.group_ids[row]), str(data.flow_groups[row])), []).append(row)
    prepared: list[TransformResult | None] = [None] * len(data)
    for (capture, flow_group), rows in groups.items():
        rows.sort(key=lambda row: (float(data.start_offsets[row]),
                                   _flow_chunk_number(str(data.flow_ids[row]))))
        labels = {int(data.labels[row]) for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Flow {flow_group} has conflicting class labels")
        origins = np.asarray([data.start_offsets[row] for row in rows], dtype=np.float64)
        origin = float(origins[0])
        parts = [source_trace(data, row) for row in rows]
        sizes = np.concatenate([part.sizes for part in parts])
        times = np.maximum.accumulate(np.concatenate([
            part.times + float(offset - origin)
            for part, offset in zip(parts, origins)
        ]))
        full_source = TrafficTrace(
            sizes, times, np.arange(len(sizes), dtype=np.int32), np.abs(sizes)
        )
        capture_name = str(data.group_names[capture])
        digest = hashlib.sha256(f"{seed}:{role}:{capture_name}:{flow_group}".encode()).digest()
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, role, int.from_bytes(digest[:4], "big"), int.from_bytes(digest[4:8], "big")]
        ))
        label = labels.pop()
        settings = dict(parameters)
        if qos is not None:
            contract = qos.contract(data, rows[0])
            if contract.maximum_bandwidth_overhead is not None:
                settings["maximum_bandwidth_overhead"] = min(
                    float(settings["maximum_bandwidth_overhead"]), contract.maximum_bandwidth_overhead)
            if contract.maximum_duration_overhead is not None:
                settings["maximum_sequence_duration_overhead"] = min(
                    float(settings["maximum_sequence_duration_overhead"]), contract.maximum_duration_overhead)
            if contract.deadline_seconds is not None:
                settings.update(_qos_deadline_seconds=contract.deadline_seconds - origin,
                                _qos_baseline_seconds=qos.baseline_seconds[label],
                                _qos_expected_trace_bytes=contract.expected_trace_bytes
                                or qos.expected_bytes[label],
                                _qos_link_capacity_bps=qos.link_bps)
        full_result = transform("flowshift_manuscript", full_source, label,
                                rng, settings, library)
        validate_conservation(full_source, full_result.trace)
        if full_result.status == "fallback":
            for row in rows:
                prepared[row] = replace(full_result, trace=source_trace(data, row))
            continue
        owner_by_source = np.repeat(np.arange(len(rows)), [len(part.sizes) for part in parts])
        packet_owners = np.empty(len(full_result.trace.sizes), dtype=np.int32)
        previous_owner = 0
        for packet, source_index in enumerate(full_result.trace.source_index):
            if source_index >= 0:
                previous_owner = int(owner_by_source[source_index])
            packet_owners[packet] = previous_owner
        starts = np.cumsum([0] + [len(part.sizes) for part in parts[:-1]])
        for position, row in enumerate(rows):
            chosen = packet_owners == position
            global_indices = full_result.trace.source_index[chosen]
            local_indices = np.where(global_indices >= 0, global_indices - starts[position], -1)
            local_trace = TrafficTrace(
                full_result.trace.sizes[chosen].copy(),
                np.maximum(0.0, full_result.trace.times[chosen] - (origins[position] - origin)),
                local_indices.astype(np.int32),
                full_result.trace.valid_bytes[chosen].copy(),
            )
            prepared[row] = replace(full_result, trace=local_trace)
    if any(result is None for result in prepared):
        raise RuntimeError("FlowShift failed to cover every source sequence")
    return [result for result in prepared if result is not None]


def _predict(model, scaler: FeatureScaler, data: SequenceDataset,
             traces: list[TrafficTrace], batch_size: int) -> np.ndarray:
    expanded, mapping = windows(data, traces)
    probabilities = predict_probabilities(model, scaler.transform(expanded), batch_size)
    totals = np.zeros((len(data), len(data.class_names)), dtype=np.float64)
    np.add.at(totals, mapping, probabilities)
    counts = np.bincount(mapping, minlength=len(data))
    if np.any(counts == 0):
        raise RuntimeError("A transformed source lost all CNN windows")
    return (totals / counts[:, None]).astype(np.float32)


def _train(training: SequenceDataset, validation: SequenceDataset, seed: int,
           base: dict) -> tuple[object, FeatureScaler]:
    global _runtime_device
    device = base["model"]["device"]
    if _runtime_device is None:
        prepare_runtime(seed, device)
        _runtime_device = device
    else:
        if _runtime_device != device:
            raise RuntimeError("Changing TensorFlow device during a comparison is unsupported")
        import tensorflow as tf
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
    scaler = FeatureScaler.fit(training)
    model = build_sequence_model(training.sizes.shape[1], len(training.class_names),
                                 base["model"], class_names=training.class_names)
    train_model(model, scaler.transform(training), training.labels,
                scaler.transform(validation), validation.labels,
                class_weights(training.labels), base["model"], seed)
    return model, scaler


def _restore(weights: list[np.ndarray], data: SequenceDataset, base: dict):
    model = build_sequence_model(data.sizes.shape[1], len(data.class_names),
                                 base["model"], class_names=data.class_names)
    model.set_weights(weights)
    return model


def _distribution(values: list[float]) -> dict:
    data = np.asarray(values, dtype=float)
    return {"mean": float(data.mean()) if len(data) else None,
            "median": float(np.median(data)) if len(data) else None,
            "p90": float(np.percentile(data, 90)) if len(data) else None,
            "maximum": float(data.max()) if len(data) else None}


def _metrics(data: SequenceDataset, probabilities: np.ndarray,
             costs_by_source: list[dict], results: list[TransformResult], budgets: dict,
             qos_summary: dict | None = None) -> dict:
    labels = data.labels.astype(int)
    predicted = probabilities.argmax(axis=1).astype(int)
    confusion = np.zeros((len(data.class_names), len(data.class_names)), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    f1 = []
    for index in range(len(data.class_names)):
        true_positive = int(confusion[index, index])
        precision = true_positive / max(int(confusion[:, index].sum()), 1)
        recall = true_positive / max(int(confusion[index].sum()), 1)
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    source_bytes = sum(item["source_bytes"] for item in costs_by_source)
    emitted_bytes = sum(item["transmitted_bytes"] for item in costs_by_source)
    source_packets = sum(item["source_packets"] for item in costs_by_source)
    emitted_packets = sum(item["transmitted_packets"] for item in costs_by_source)
    targets = np.asarray([result.target if result.target is not None else -1 for result in results])
    targeted = targets >= 0
    report = {
        "samples": len(data), "accuracy": float(np.mean(labels == predicted)),
        "macro_f1": float(np.mean(f1)), "class_order": data.class_names.tolist(),
        "confusion_matrix": confusion.tolist(),
        "original_bytes": source_bytes, "emitted_bytes": emitted_bytes,
        "added_bytes_over_original_bytes": (emitted_bytes - source_bytes) / source_bytes,
        "original_packets": source_packets, "emitted_packets": emitted_packets,
        "packet_count_overhead": (emitted_packets - source_packets) / source_packets,
        "bandwidth_budget_exceeded_records": sum(
            item["bandwidth_overhead"] > budgets["bandwidth_overhead"] for item in costs_by_source),
        "duration_budget_exceeded_records": sum(
            item["duration_valid"] and item["sequence_duration_overhead"] > budgets["sequence_duration_overhead"]
            for item in costs_by_source),
        "bandwidth_overhead_per_record": _distribution([item["bandwidth_overhead"] for item in costs_by_source]),
        "sequence_duration_overhead": _distribution([
            item["sequence_duration_overhead"] for item in costs_by_source if item["duration_valid"]]),
        "last_real_packet_delay_seconds": _distribution([
            item["last_source_byte_delay_seconds"] for item in costs_by_source
            if item["last_source_byte_delay_seconds"] is not None]),
        "dummy_bytes": sum(item["dummy_bytes"] for item in costs_by_source),
        "padding_bytes": sum(item["padding_bytes"] for item in costs_by_source),
        "dummy_packets": sum(item["dummy_packets"] for item in costs_by_source),
        "outgoing_dummy_packets": sum(item["outgoing_dummy_packets"] for item in costs_by_source),
        "incoming_dummy_packets": sum(item["incoming_dummy_packets"] for item in costs_by_source),
        "retained_source_packets": sum(item["delivered_source_packets"] for item in costs_by_source),
        "emitted_real_packets": sum(item["retained_real_packets"] for item in costs_by_source),
        "retained_source_fraction": sum(
            item["delivered_fraction"] * int(data.packet_counts[i])
            for i, item in enumerate(costs_by_source)) / source_packets,
        "incomplete_records": sum(not item["completed"] for item in costs_by_source),
        "upload_delay_violations": sum(item["upload_delay_violations"] for item in costs_by_source),
        "queue_overflow_packets": sum(item["queue_overflow_packets"] for item in costs_by_source),
        "oversize_drops": sum(item["oversize_drops"] for item in costs_by_source),
        "dropped_packets": sum(item["dropped_packets"] for item in costs_by_source),
        "unfinished_packets": sum(item["unfinished_packets"] for item in costs_by_source),
        "outside_horizon_packets": sum(item["outside_horizon_packets"] for item in costs_by_source),
        "reordering_inversions": sum(item["reordering_inversions"] for item in costs_by_source),
        "genuine_packet_delay_seconds": _distribution([
            item["mean_genuine_packet_delay_seconds"] for item in costs_by_source
            if item["mean_genuine_packet_delay_seconds"] is not None]),
        "startup_download_dummy_packets": sum(item["startup_download_dummy_packets"] for item in costs_by_source),
        "main_download_dummy_packets": sum(item["main_download_dummy_packets"] for item in costs_by_source),
        "startup_upload_dummy_packets": sum(item["startup_upload_dummy_packets"] for item in costs_by_source),
        "sampled_upload_dummy_packets": sum(item["sampled_upload_dummy_packets"] for item in costs_by_source),
        "startup_download_dummy_bytes": sum(item.get("startup_download_dummy_bytes", 0) for item in costs_by_source),
        "main_download_dummy_bytes": sum(item.get("main_download_dummy_bytes", 0) for item in costs_by_source),
        "startup_upload_dummy_bytes": sum(item.get("startup_upload_dummy_bytes", 0) for item in costs_by_source),
        "sampled_upload_dummy_bytes": sum(item.get("sampled_upload_dummy_bytes", 0) for item in costs_by_source),
        "action_counts": {name: sum(result.action_counts.get(name, 0) for result in results)
                          for name in sorted(set().union(*(result.action_counts for result in results)))},
        "applied_records": sum(result.status == "applied" for result in results),
        "fallback_records": sum(result.status == "fallback" for result in results),
        "assigned_target_success": (
            float(np.mean(predicted[targeted] == targets[targeted])) if np.any(targeted) else None),
    }
    policy_results = []
    seen_flows = set()
    for row, result in enumerate(results):
        key = (int(data.group_ids[row]), str(data.flow_groups[row]))
        if result.feasible_policies is not None and key not in seen_flows:
            policy_results.append(result)
            seen_flows.add(key)
    if policy_results:
        report["flowshift_flows"] = len(policy_results)
        report["flowshift_policy_counts"] = {
            "bank": _distribution([result.candidate_policies for result in policy_results]),
            "feasible": _distribution([result.feasible_policies for result in policy_results]),
            "selected": _distribution([len(result.selected_policies) for result in policy_results]),
            "scheduled_intervals": _distribution([len(result.scheduled_policies) for result in policy_results]),
        }
        frequency = Counter(name for result in policy_results for name in result.selected_policies)
        report["flowshift_selected_policy_frequency"] = dict(sorted(frequency.items()))
        report["flowshift_policy_switches"] = _distribution([result.policy_switches for result in policy_results])
    if qos_summary is not None:
        report["qos"] = qos_summary
        report["flowshift_deadline_bypass_packets"] = sum(
            result.deadline_bypass_packets for result in policy_results)
    return report


def _choose(method: str, choices: dict, validation: SequenceDataset, library: FlowShiftLibrary,
            model, scaler: FeatureScaler, seed: int, budgets: dict, batch_size: int,
            qos: QoSContext | None = None) -> tuple[dict, list[dict]]:
    trials = []
    for index, parameters in enumerate(choices["candidates"]):
        traces, results, overheads = _draw(method, validation, library, seed, 200 + index, parameters, qos)
        prediction = _predict(model, scaler, validation, traces, batch_size).argmax(axis=1)
        bandwidth = float(np.mean([item["bandwidth_overhead"] for item in overheads]))
        durations = [item["sequence_duration_overhead"] for item in overheads if item["duration_valid"]]
        duration = float(np.mean(durations)) if durations else 0.0
        incomplete = sum(not item["completed"] for item in overheads)
        qos_summary = evaluate_qos(validation, traces, qos)[0] if qos is not None else None
        feasible = (bandwidth <= budgets["bandwidth_overhead"]
                    and duration <= budgets["sequence_duration_overhead"]
                    and (method != "ditto_adapted" or incomplete == 0)
                    and (qos_summary is None or qos_summary["defense_induced_deadline_misses"] == 0))
        trials.append({"index": index, "parameters": parameters,
                       "accuracy": float(np.mean(prediction == validation.labels)),
                       "bandwidth_overhead": bandwidth, "sequence_duration_overhead": duration,
                       "incomplete_records": incomplete, "budget_feasible": feasible,
                       "qos": qos_summary})
    feasible = [trial for trial in trials if trial["budget_feasible"]]
    selected = min(feasible, key=lambda trial: (
        trial["accuracy"], trial["bandwidth_overhead"], trial["index"])) if feasible else min(
        trials, key=lambda trial: (
                                  trial["qos"]["defense_induced_deadline_misses"] if trial["qos"] else 0,
                                  trial["incomplete_records"] if method == "ditto_adapted" else 0,
                                  max(0, trial["bandwidth_overhead"] - budgets["bandwidth_overhead"])
                                  + max(0, trial["sequence_duration_overhead"] - budgets["sequence_duration_overhead"]),
                                  trial["index"]))
    return selected["parameters"], trials


def _fit_ditto_choices(choices: dict, training: SequenceDataset) -> dict:
    resolved = {"example": dict(choices["example"]), "candidates": []}
    for candidate in choices["candidates"]:
        settings = dict(candidate)
        settings["pattern"] = fit_pattern(training, int(settings["pattern_length"]),
                                           int(settings["maximum_packet_size"]))
        resolved["candidates"].append(settings)
    return resolved


def _points(method: str, choices: dict, selected: dict) -> dict[str, dict]:
    if method in {"front_adapted", "regulator_adapted"}:
        return choices["published"] | {"validation_selected": selected}
    if method == "ditto_adapted":
        return {"repository_example": choices["example"], "validation_selected": selected}
    if method == "flowshift_manuscript":
        variants = choices["policy_variants"]
        return {"published": choices["published"] | {"policy_variants": variants},
                "validation_selected": selected | {"policy_variants": variants}}
    return {"published": choices["published"], "validation_selected": selected}


def run_comparison(config: dict) -> Iterator[dict]:
    """Yield complete result rows; CSV writing lives outside this package."""
    comparison, base = config["comparison"], config["base"]
    data = extract_dataset(base)
    qos_settings = comparison["qos"]
    qos_overrides = load_contracts(qos_settings["flow_contracts_csv"])
    available_flows = {
        (str(data.environment_names[data.environment_ids[row]]),
         str(data.group_names[data.group_ids[row]]), str(data.flow_groups[row]))
        for row in range(len(data))
    }
    unknown_contracts = set(qos_overrides) - available_flows
    if unknown_contracts:
        raise ValueError(f"QoS CSV refers to an unknown extracted flow: {sorted(unknown_contracts)[0]}")
    for environment in base["data"]["environments"]:
        environment_data = data.subset(np.flatnonzero(data.environment_ids == data.environment_id(environment)))
        for seed in comparison["seeds"]:
            logger.debug("%s, seed %s", environment, seed)
            training, validation, testing = _split(environment_data, seed, base)
            split = {name: {"records": len(part),
                            "captures": [str(part.group_names[index])
                                         for index in sorted(set(part.group_ids.tolist()))]}
                     for name, part in (("training", training), ("validation", validation), ("test", testing))}
            environment_overrides = {
                key: contract for key, contract in qos_overrides.items() if key[0] == environment
            }
            qos = (QoSContext.fit(training, qos_settings, environment_overrides)
                   if qos_settings["class_defaults"] or environment_overrides else None)
            library = FlowShiftLibrary.fit(
                training, comparison["methods"]["flowshift_manuscript"]["policy_variants"]
            )
            original_model, original_scaler = _train(training, validation, seed, base)
            original_weights = [np.array(value, copy=True) for value in original_model.get_weights()]
            original_traces, original_results, original_costs = _draw("original", testing, library, seed, 300, {}, qos)
            original_qos, original_qos_flows = (evaluate_qos(testing, original_traces, qos, original_results)
                                                if qos is not None else (None, []))
            original_report = _metrics(testing, _predict(original_model, original_scaler, testing,
                                                         original_traces, base["model"]["batch_size"]),
                                       original_costs, original_results, comparison["budgets"], original_qos)
            common = {"environment": environment, "seed": seed, "capture_fingerprint": data.fingerprint,
                      "split": split,
                      "model_settings": base["model"], "data_settings": base["data"],
                      "excluded_captures": data.excluded_captures}
            yield common | {"method": "original", "point": "baseline", "adversary": "original_trained",
                            "parameters": {}, "validation_selection": None, "metrics": original_report,
                            "qos_flows": original_qos_flows}
            for method, configured in comparison["methods"].items():
                choices = _fit_ditto_choices(configured, training) if method == "ditto_adapted" else configured
                if method == "securitas_adapted":
                    from .securitas_policy import train_policy
                    surrogate, surrogate_scaler = _train(training, validation, seed + 100_000, base)

                    def surrogate_predict(part, traces):
                        return _predict(surrogate, surrogate_scaler, part, traces, base["model"]["batch_size"])

                    trials = []
                    for index, override in enumerate(choices["candidates"]):
                        settings = choices["learned"] | override
                        policy, history = train_policy(training, validation, settings,
                                                       seed + 200_000, surrogate_predict)
                        trials.append({"index": index, "settings": settings, "policy": policy,
                                       "validation_objective": history["best_validation_objective"]})
                    best = min(trials, key=lambda trial: (trial["validation_objective"], trial["index"]))
                    points = {"learned": best["settings"] | {"policy": best["policy"]}}
                    selection = {"trials": trials, "selected_index": best["index"]}
                else:
                    original_model = _restore(original_weights, training, base)
                    selected, trials = _choose(method, choices, validation, library, original_model,
                                               original_scaler, seed, comparison["budgets"],
                                               base["model"]["batch_size"], qos)
                    points = _points(method, choices, selected)
                    selection = {"trials": trials, "selected_parameters": selected}
                for point, parameters in points.items():
                    original_model = _restore(original_weights, training, base)
                    traces, results, overheads = _draw(method, testing, library, seed, 400, parameters, qos)
                    qos_summary, qos_flows = (evaluate_qos(testing, traces, qos, results)
                                              if qos is not None else (None, []))
                    original_report = _metrics(testing, _predict(original_model, original_scaler, testing,
                                                                 traces, base["model"]["batch_size"]),
                                               overheads, results, comparison["budgets"], qos_summary)
                    report_base = common | {"method": method, "point": point,
                                            "parameters": parameters, "validation_selection": selection,
                                            "qos_flows": qos_flows}
                    yield report_base | {"adversary": "original_trained", "metrics": original_report}
                    defended_training, _, _ = _draw(method, training, library, seed, 100, parameters, qos)
                    defended_validation, _, _ = _draw(method, validation, library, seed, 200, parameters, qos)
                    training_windows, _ = windows(training, defended_training)
                    validation_windows, _ = windows(validation, defended_validation)
                    aware_model, aware_scaler = _train(training_windows, validation_windows, seed, base)
                    aware_report = _metrics(testing, _predict(aware_model, aware_scaler, testing, traces,
                                                              base["model"]["batch_size"]), overheads, results,
                                           comparison["budgets"], qos_summary)
                    yield report_base | {"adversary": "defense_aware", "metrics": aware_report}
