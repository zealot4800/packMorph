from __future__ import annotations

import json
import multiprocessing
import traceback
from pathlib import Path

import numpy as np

from .config import dump_config
from .dataset import SequenceDataset
from .extract import extraction_fingerprint
from .features import FeatureScaler
from .metrics import (
    classification_metrics,
    plot_confusion,
    plot_training_history,
    save_confusion_csv,
)
from .model import (
    build_sequence_model,
    predict_probabilities,
    prepare_runtime,
    train_model,
)
from .morph import ProfileLibrary, morph_dataset
from .split import class_weights, stratified_group_split


def _mean_statistics(statistics: list[dict]) -> dict:
    numeric_keys = (
        "bandwidth_overhead",
        "packet_overhead",
        "padding_and_dummy_bytes",
        "dummy_packets",
        "fragmentation_delta",
        "morphed_windows",
    )
    valid_fct = [row["fct_overhead"] for row in statistics if row["fct_overhead_valid"]]
    summary = {
        f"mean_{key}": float(np.mean([row[key] for row in statistics]))
        for key in numeric_keys
    }
    summary.update(
        {
            f"median_{key}": float(np.median([row[key] for row in statistics]))
            for key in ("bandwidth_overhead", "packet_overhead")
        }
    )
    summary["mean_fct_overhead"] = float(np.mean(valid_fct)) if valid_fct else 0.0
    summary["median_fct_overhead"] = float(np.median(valid_fct)) if valid_fct else 0.0
    summary["fct_valid_sequences"] = len(valid_fct)
    directional_accuracy = float(
        np.mean([row["directional_byte_accounting_ok"] for row in statistics])
    )
    summary["directional_byte_accounting_accuracy"] = directional_accuracy
    # Retain the existing key for consumers of earlier result files.
    summary["byte_accounting_accuracy"] = directional_accuracy
    return summary


def _validate_environment(dataset: SequenceDataset, environment: str) -> None:
    present = set(np.unique(dataset.labels).astype(int).tolist())
    expected = set(range(len(dataset.class_names)))
    if missing := expected - present:
        names = ", ".join(str(dataset.class_names[label]) for label in sorted(missing))
        raise ValueError(f"{environment} is missing extracted sequences for: {names}")


def _aggregate_probabilities(
    window_probabilities: np.ndarray,
    source_indices: np.ndarray,
    source_count: int,
) -> np.ndarray:
    totals = np.zeros((source_count, window_probabilities.shape[1]), dtype=np.float64)
    np.add.at(totals, source_indices, window_probabilities)
    counts = np.bincount(source_indices, minlength=source_count)
    if np.any(counts == 0):
        raise RuntimeError("At least one transformed sequence has no classifier window")
    return (totals / counts[:, None]).astype(np.float32)


def _morph_source_batch_size(testing: SequenceDataset, config: dict) -> int:
    """Bound each morph batch by a conservative expanded-window estimate."""
    byte_totals = np.abs(testing.sizes).sum(axis=1, dtype=np.int64)
    maximum_source_bytes = int(byte_totals.max())
    maximum_budget = max(
        maximum_source_bytes,
        int(np.floor(maximum_source_bytes * config["morph"]["maximum_padding_ratio"])),
    )
    maximum_slots = max(
        1,
        maximum_budget // config["morph"]["minimum_packet_size"],
    )
    width = testing.sizes.shape[1]
    maximum_windows = max(1, (maximum_slots + width - 1) // width)
    prediction_batch_size = config["model"]["batch_size"]
    target_windows = prediction_batch_size * 4
    return max(
        1,
        min(prediction_batch_size, target_windows // maximum_windows),
    )


def _predict_morphed_batches(
    model,
    testing: SequenceDataset,
    library: ProfileLibrary,
    scaler: FeatureScaler,
    config: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict], int]:
    source_count = len(testing)
    class_count = len(testing.class_names)
    morphed_probabilities = np.empty((source_count, class_count), dtype=np.float32)
    target_labels = np.empty(source_count, dtype=testing.labels.dtype)
    statistics: list[dict] = []
    morphed_window_count = 0
    rng = np.random.default_rng(seed)
    source_batch_size = _morph_source_batch_size(testing, config)
    prediction_batch_size = config["model"]["batch_size"]

    for start in range(0, source_count, source_batch_size):
        stop = min(start + source_batch_size, source_count)
        batch_indices = np.arange(start, stop, dtype=np.int64)
        source_batch = testing.subset(batch_indices)
        result = morph_dataset(
            source_batch,
            library,
            config["morph"],
            rng=rng,
        )
        features = scaler.transform(result.dataset)
        window_probabilities = predict_probabilities(
            model,
            features,
            prediction_batch_size,
        )
        morphed_probabilities[start:stop] = _aggregate_probabilities(
            window_probabilities,
            result.source_indices,
            len(source_batch),
        )
        target_labels[start:stop] = result.target_labels
        statistics.extend(result.statistics)
        morphed_window_count += len(result.dataset)
        del features, window_probabilities, result, source_batch

    return morphed_probabilities, target_labels, statistics, morphed_window_count


def run_environment(dataset: SequenceDataset, environment: str, config: dict) -> dict:
    environment_id = dataset.environment_id(environment)
    environment_indices = np.flatnonzero(dataset.environment_ids == environment_id)
    environment_data = dataset.subset(environment_indices)
    _validate_environment(environment_data, environment)

    seed = config["seed"]
    development_indices, test_indices = stratified_group_split(
        environment_data,
        config["split"]["test_fraction"],
        seed,
        minimum_retained_groups=2,
    )
    development = environment_data.subset(development_indices)
    testing = environment_data.subset(test_indices)
    del environment_data
    training_indices, validation_indices = stratified_group_split(
        development,
        config["model"]["validation_fraction"],
        seed + 1,
    )
    training = development.subset(training_indices)
    validation = development.subset(validation_indices)
    del development

    class_names = dataset.class_names.tolist()
    output_name = environment.lower().replace(" ", "_").replace("-", "_")
    output_dir = Path(config["output"]["directory"]) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\n[{environment}] train={len(training)}, validation={len(validation)}, "
        f"test={len(testing)}"
    )

    scaler = FeatureScaler.fit(training)
    training_features = scaler.transform(training)
    validation_features = scaler.transform(validation)
    test_features = scaler.transform(testing)

    runtime_device = prepare_runtime(seed, config["model"]["device"])
    model = build_sequence_model(
        sequence_length=training.sizes.shape[1],
        class_count=len(class_names),
        config=config["model"],
    )
    history = train_model(
        model,
        training_features,
        training.labels,
        validation_features,
        validation.labels,
        class_weights(training.labels),
        config["model"],
        output_dir,
    )
    del training_features, validation_features

    baseline_probabilities = predict_probabilities(
        model,
        test_features,
        config["model"]["batch_size"],
    )
    baseline_prediction = np.argmax(baseline_probabilities, axis=1).astype(np.int16)
    baseline = classification_metrics(testing.labels, baseline_prediction, class_names)
    del test_features

    library = ProfileLibrary.fit(training)
    (
        morphed_probabilities,
        target_labels,
        morph_statistics,
        morphed_window_count,
    ) = _predict_morphed_batches(
        model,
        testing,
        library,
        scaler,
        config,
        seed + 10_000,
    )
    morphed_prediction = np.argmax(morphed_probabilities, axis=1).astype(np.int16)
    morphed = classification_metrics(testing.labels, morphed_prediction, class_names)
    overhead = _mean_statistics(morph_statistics)

    baseline_correct = baseline_prediction == testing.labels
    baseline_correct_count = int(np.count_nonzero(baseline_correct))
    conditional_evasion = (
        float(
            np.mean(
                morphed_prediction[baseline_correct] != testing.labels[baseline_correct]
            )
        )
        if baseline_correct_count
        else None
    )
    target_hit_rate = float(np.mean(morphed_prediction == target_labels))
    prediction_change_rate = float(np.mean(morphed_prediction != baseline_prediction))

    baseline_matrix = np.asarray(baseline["confusion_matrix"], dtype=np.int64)
    morphed_matrix = np.asarray(morphed["confusion_matrix"], dtype=np.int64)
    save_confusion_csv(
        baseline_matrix, class_names, output_dir / "baseline_confusion.csv"
    )
    save_confusion_csv(
        morphed_matrix, class_names, output_dir / "morphed_confusion.csv"
    )
    plot_confusion(
        baseline_matrix,
        class_names,
        f"{environment}: original traffic (CNN-BiGRU)",
        output_dir / "baseline_confusion.png",
    )
    plot_confusion(
        morphed_matrix,
        class_names,
        f"{environment}: PackMorph simulation (CNN-BiGRU)",
        output_dir / "morphed_confusion.png",
    )
    plot_training_history(history.history, output_dir / "training_history.png")

    relative_reduction = (
        float((baseline["accuracy"] - morphed["accuracy"]) / baseline["accuracy"])
        if baseline["accuracy"] > 0.0
        else None
    )
    report = {
        "environment": environment,
        "train_sequences": len(training),
        "validation_sequences": len(validation),
        "test_sequences": len(testing),
        "morphed_classifier_windows": morphed_window_count,
        "train_captures": int(np.unique(training.group_ids).size),
        "validation_captures": int(np.unique(validation.group_ids).size),
        "test_captures": int(np.unique(testing.group_ids).size),
        "classifier": "padding-aware 1D CNN + bidirectional GRU",
        "runtime_device": runtime_device,
        "baseline": baseline,
        "morphed": morphed,
        "accuracy_reduction": float(baseline["accuracy"] - morphed["accuracy"]),
        "relative_accuracy_reduction": relative_reduction,
        "morphed_error_rate": float(1.0 - morphed["accuracy"]),
        "conditional_evasion_success": conditional_evasion,
        "conditional_evasion_denominator": baseline_correct_count,
        "prediction_change_rate": prediction_change_rate,
        "target_hit_rate": target_hit_rate,
        "overhead": overhead,
        "feature_scaler": scaler.as_dict(),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "morph_statistics.jsonl").open("w", encoding="utf-8") as handle:
        for row in morph_statistics:
            handle.write(json.dumps(row) + "\n")
    np.savez_compressed(
        output_dir / "predictions.npz",
        truth=testing.labels,
        target=target_labels,
        baseline_prediction=baseline_prediction,
        baseline_probabilities=baseline_probabilities,
        morphed_prediction=morphed_prediction,
        morphed_probabilities=morphed_probabilities,
        test_group_ids=testing.group_ids,
        group_names=testing.group_names,
    )
    print(
        f"[{environment}] baseline accuracy={baseline['accuracy']:.3f}, "
        f"morphed accuracy={morphed['accuracy']:.3f}, "
        f"conditional evasion={conditional_evasion if conditional_evasion is not None else 'n/a'}, "
        f"target-hit={target_hit_rate:.3f}"
    )
    print(
        f"[{environment}] mean bandwidth={overhead['mean_bandwidth_overhead']:.2%}, "
        f"FCT={overhead['mean_fct_overhead']:.2%}, "
        "directional byte accounting="
        f"{overhead['directional_byte_accounting_accuracy']:.2%}"
    )
    return report


def _environment_worker(config: dict, environment: str, sender) -> None:
    """Run one TensorFlow environment in a disposable native process."""
    try:
        expected_fingerprint = extraction_fingerprint(config["data"])
        dataset = SequenceDataset.load(
            Path(config["data"]["cache"]),
            expected_fingerprint=expected_fingerprint,
        )
        sender.send(("ok", run_environment(dataset, environment, config)))
    except Exception:
        sender.send(("error", traceback.format_exc()))
    finally:
        sender.close()


def _run_environment_isolated(
    config: dict,
    environment: str,
    maximum_attempts: int = 2,
) -> dict:
    context = multiprocessing.get_context("spawn")
    for attempt in range(1, maximum_attempts + 1):
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_environment_worker,
            args=(config, environment, sender),
            name=f"packmorph-{environment.lower().replace(' ', '-')}",
        )
        process.start()
        sender.close()
        try:
            process.join()
        except BaseException:
            if process.is_alive():
                process.terminate()
            process.join()
            receiver.close()
            raise

        message = None
        if receiver.poll():
            try:
                message = receiver.recv()
            except EOFError:
                pass
        receiver.close()
        if process.exitcode == 0 and message is not None:
            status, payload = message
            if status == "ok":
                return payload
            raise RuntimeError(
                f"{environment} experiment failed in its worker process:\n{payload}"
            )

        if attempt < maximum_attempts:
            print(
                f"[{environment}] TensorFlow worker exited with code "
                f"{process.exitcode}; retrying in a fresh process",
                flush=True,
            )
            continue
        raise RuntimeError(
            f"{environment} TensorFlow worker exited with code {process.exitcode} "
            f"after {maximum_attempts} attempts"
        )


def run_experiment(config: dict) -> dict:
    expected_fingerprint = extraction_fingerprint(config["data"])
    cache_path = Path(config["data"]["cache"])
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Extracted sequence cache not found: {cache_path}. Run 'extract' or 'all' first."
        )
    # Validate once in the parent for a fast, clear cache error. Each worker then
    # reloads independently so no TensorFlow-native state survives environments.
    SequenceDataset.load(cache_path, expected_fingerprint=expected_fingerprint)

    output_root = Path(config["output"]["directory"])
    output_root.mkdir(parents=True, exist_ok=True)
    dump_config(config, output_root / "effective_config.yaml")
    reports = {
        environment: _run_environment_isolated(config, environment)
        for environment in config["data"]["environments"]
    }
    summary = {
        "paper_scope": "dataset-driven simulation only",
        "model_replacement": "Random Forest -> padding-aware 1D CNN + bidirectional GRU",
        "environments": reports,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
