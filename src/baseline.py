"""Frozen prototypical classifier trained from dataset-specific pickle splits."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FlowDataset
from .metrics import evaluate
from .model import CicClassifier, build_embedding, fit_classifier, train_embedding
from .runtime import prepare_runtime


def _log_features(features: np.ndarray) -> np.ndarray:
    """Preprocess in one owned buffer so the caller's features stay unchanged."""
    logged = np.array(features, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(logged)):
        raise ValueError("CICFlowMeter features contain nonfinite values")
    np.maximum(logged, 0.0, out=logged)
    np.log1p(logged, out=logged)
    return logged


@dataclass(slots=True)
class BehaviorScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "BehaviorScaler":
        logged = _log_features(features)
        return cls(logged.mean(axis=0), np.maximum(logged.std(axis=0), 1e-6))

    def transform(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != len(self.mean):
            raise ValueError("CICFlowMeter feature count differs from saved schema")
        logged = _log_features(features)
        logged -= self.mean
        logged /= self.scale
        return logged.astype(np.float32)


@dataclass(slots=True)
class FrozenCic:
    model: CicClassifier
    scaler: BehaviorScaler


def artifact_path(output_dir: str | Path, environment: str, feature_count: int = 16) -> Path:
    slug = environment.casefold().replace(" ", "-")
    return (Path(output_dir) / "results" / "frozen_baseline"
            / f"cic-pkl-v2-config{feature_count}" / slug)


def _fingerprint(flows: FlowDataset) -> str:
    digest = hashlib.sha256()
    for value in (flows.features, flows.labels, flows.captures):
        if value is None:
            digest.update(b'no_capture_metadata')
        else:
            digest.update(memoryview(np.ascontiguousarray(value)))
    digest.update(json.dumps(flows.feature_names).encode())
    return digest.hexdigest()


def _weights_checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _manifest(config: dict, environment: str, splits: dict[str, FlowDataset]) -> dict:
    flows = splits['train']
    names = config['model']['features'][environment]
    if tuple(names) != flows.feature_names or flows.features.shape[1] != len(names):
        raise ValueError('Dataset feature order must match model.features in the configuration')
    capture_aware = flows.captures is not None
    manifest = {"version": 3, "environment": environment,
            "training_version": "regularized_validation_early_stop_v2",
            "dataset_schema": flows.schema,
            "capture_disjoint_verified": flows.capture_disjoint_verified,
            "split_fingerprints": {name: _fingerprint(part) for name, part in splits.items()},
            "split_identity_fingerprints": {name: hashlib.sha256(
                part.row_indices.tobytes() + part.record_ids.tobytes()).hexdigest()
                for name, part in splits.items()},
            "source_fingerprint": flows.source_fingerprint,
            "model_config": {**config["model"], "features": names},
            "feature_names": names, "class_names": list(flows.class_names),
            "preprocessing": "log1p_nonnegative_unique_training_standardize_v2",
            "sampling": ("equal_capture_probability_unique_training_flows_per_class_v1"
                         if capture_aware else "uniform_unique_training_flows_per_class_v1"),
            "split_strategy": ("fixed_balanced_pickles_v1" if capture_aware
                               else "stratified_unique_feature_vectors_v1"),
            "checkpoint_policy": ("validation_class_capture_macro_v1" if capture_aware
                                  else "validation_class_macro_v1")}
    return manifest


def prepare_baseline(config: dict, environment: str,
                     splits: dict[str, FlowDataset]) -> Path:
    settings = config["model"]
    destination = artifact_path(config["output_dir"], environment, len(config["model"]["features"][environment]))
    if destination.exists():
        raise FileExistsError(f"Frozen CICFlowMeter baseline already exists: {destination}")
    manifest = _manifest(config, environment, splits)
    balanced = splits['train']
    # Copies retain their original identity: never put the same physical flow
    # in both support and query, or count copies as independent prototypes.
    _, first = np.unique(balanced.row_indices, return_index=True)
    training = balanced.subset(np.sort(first))
    validation = splits['validation']
    print(f"{environment}: train.pkl={len(balanced)} rows ({len(training)} unique flows), "
          f"val.pkl={len(validation)}, test.pkl={len(splits['test'])}; {len(balanced.feature_names)} features", flush=True)
    scaler = BehaviorScaler.fit(training.features)
    training_features = scaler.transform(training.features)
    embedding, checkpoint = train_embedding(
        training_features, training.labels, len(training.class_names), settings,
        training.captures,
        (scaler.transform(validation.features), validation.labels, validation.captures))
    model = fit_classifier(embedding, training_features, training.labels,
                           len(training.class_names), settings, training.captures)
    history = checkpoint.pop('history')
    manifest["selected_checkpoint"] = checkpoint
    manifest['training_history_file'] = 'training_history.json'
    manifest['unique_training_metrics'] = evaluate(
        model, training_features, training.labels, training.captures,
        training.class_names, settings['batch_size'])
    for name, part in splits.items():
        manifest[name + '_metrics'] = evaluate(
            model, scaler.transform(part.features), part.labels, part.captures,
            part.class_names, settings['batch_size'])
    manifest['split_rows'] = {name: len(part) for name, part in splits.items()}
    manifest['unique_training_flows'] = len(training)
    weights = embedding.get_weights()
    manifest["weight_count"] = len(weights)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cic-baseline-", dir=destination.parent) as directory:
        staging = Path(directory)
        arrays = {f"weight_{index}": weight
                  for index, weight in enumerate(weights)}
        for name in ("prototypes", "diagonal_variances"):
            arrays[name] = getattr(model, name)
        arrays["preprocessing_mean"] = scaler.mean
        arrays["preprocessing_scale"] = scaler.scale
        weights_file = staging / "weights.npz"
        np.savez_compressed(weights_file, **arrays)
        manifest["weights_sha256"] = _weights_checksum(weights_file)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8")
        (staging / 'training_history.json').write_text(
            json.dumps(history, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(staging, destination)
    return destination


def _verified_artifact(config: dict, environment: str,
                       splits: dict[str, FlowDataset]) -> tuple[dict, Path]:
    location = artifact_path(config["output_dir"], environment, len(config["model"]["features"][environment]))
    if not location.is_dir():
        raise FileNotFoundError(f"Frozen CICFlowMeter baseline is missing: {location}")
    manifest = json.loads((location / "manifest.json").read_text(encoding="utf-8"))
    for key, value in _manifest(config, environment, splits).items():
        if manifest.get(key) != value:
            raise ValueError(f"Frozen CICFlowMeter baseline {location} does not match current {key}")
    selected = manifest.get("selected_checkpoint", {})
    metric_key = ('validation_macro_accuracy' if splits['train'].captures is None
                  else 'validation_capture_macro_accuracy')
    if (not isinstance(selected, dict)
            or type(selected.get("episode")) is not int
            or not 1 <= selected["episode"] <= config["model"]["episodes"]
            or type(selected.get(metric_key)) not in (int, float)
            or not 0 <= selected[metric_key] <= 1):
        raise ValueError(f"Frozen CICFlowMeter baseline {location} lacks a valid checkpoint")
    weights_file = location / "weights.npz"
    if _weights_checksum(weights_file) != manifest["weights_sha256"]:
        raise ValueError("Frozen CICFlowMeter weights failed checksum validation")
    return manifest, weights_file


def load_baseline(config: dict, environment: str,
                  splits: dict[str, FlowDataset]) -> FrozenCic:
    """Load model weights when predictions are actually needed."""
    manifest, weights_file = _verified_artifact(config, environment, splits)
    settings = manifest["model_config"]
    prepare_runtime(settings["split_seed"], settings["device"])
    embedding = build_embedding(settings, len(manifest["feature_names"]))
    with np.load(weights_file, allow_pickle=False) as arrays:
        weights = [arrays[f"weight_{index}"]
                   for index in range(manifest["weight_count"])]
        if (len(weights) != len(embedding.weights)
                or any(saved.shape != current.shape for saved, current in
                       zip(weights, embedding.weights))):
            raise ValueError("Frozen CICFlowMeter embedding dimensions are invalid")
        embedding.set_weights(weights)
        model = CicClassifier(embedding, arrays["prototypes"], arrays["diagonal_variances"])
        scaler = BehaviorScaler(arrays["preprocessing_mean"], arrays["preprocessing_scale"])
        if (scaler.mean.shape != (len(manifest["feature_names"]),)
                or scaler.scale.shape != (len(manifest["feature_names"]),)
                or not np.all(np.isfinite(scaler.mean))
                or not np.all(np.isfinite(scaler.scale))
                or np.any(scaler.scale <= 0)):
            raise ValueError("Frozen CICFlowMeter preprocessing artifact is invalid")
    return FrozenCic(model, scaler)


def ensure_baseline(config: dict, environment: str,
                    splits: dict[str, FlowDataset]) -> Path:
    destination = artifact_path(config["output_dir"], environment, len(config["model"]["features"][environment]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / f".{destination.name}.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if destination.exists():
            _verified_artifact(config, environment, splits)
            return destination
        return prepare_baseline(config, environment, splits)
