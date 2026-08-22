from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


_CHECKPOINT_FORMAT_VERSION = 2
_ARCHITECTURE_VERSION = 1


def _checkpoint_metadata(
    sequence_length: int,
    class_count: int,
    class_names: list[str] | np.ndarray,
    config: dict,
) -> tuple[str, str]:
    names = [str(value) for value in class_names]
    if len(names) != class_count:
        raise ValueError("Model class names must match the configured class count")
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Model class names must be non-empty and unique")
    metadata = {
        "architecture": {
            "name": "padding-aware-dilated-temporal-cnn",
            "version": _ARCHITECTURE_VERSION,
            "input_feature_count": 4,
            "conv_filters": [int(value) for value in config["conv_filters"]],
            "kernel_sizes": [int(value) for value in config["kernel_sizes"]],
            "context_filters": int(config["context_filters"]),
            "dilations": [int(value) for value in config["dilations"]],
            "dense_units": int(config["dense_units"]),
            "dropout": float(config["dropout"]),
        },
        "sequence_length": int(sequence_length),
        "class_names": names,
    }
    encoded = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return encoded, fingerprint


def _checkpoint_text(value: np.ndarray, name: str) -> str:
    scalar = np.asarray(value)
    if scalar.shape != () or scalar.dtype.kind not in {"U", "S"}:
        raise ValueError(f"Model weight checkpoint field {name} must be text")
    item = scalar.item()
    if isinstance(item, bytes):
        try:
            return item.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Model weight checkpoint field {name} is not valid UTF-8"
            ) from error
    return str(item)


def _tensorflow():
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # Conservative thread pools avoid native instability on hybrid-core CPUs.
    # The limits remain user-overridable and are also safe for GPU runs.
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # Keras may import Matplotlib opportunistically when it is installed even
    # though PackMorph does not depend on it.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-packmorph")
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    return tf


def prepare_runtime(seed: int, device: str) -> None:
    # Device visibility must be chosen before TensorFlow is imported. Each
    # experiment environment runs in a fresh spawned process, so this setting
    # cannot leak from one environment into the next.
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    tf = _tensorflow()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(
            int(os.environ["TF_NUM_INTRAOP_THREADS"])
        )
        tf.config.threading.set_inter_op_parallelism_threads(
            int(os.environ["TF_NUM_INTEROP_THREADS"])
        )
    except RuntimeError as error:
        raise RuntimeError(
            "TensorFlow initialized before PackMorph could configure its thread pools"
        ) from error
    physical_gpus = tf.config.list_physical_devices("GPU")

    if device == "gpu" and not physical_gpus:
        raise RuntimeError(
            "model.device is 'gpu', but TensorFlow found no usable GPU. "
            "Install the TensorFlow CUDA dependencies and verify that the GPU "
            "is visible, or set model.device to 'auto' or 'cpu'."
        )

    use_gpu = device != "cpu" and bool(physical_gpus)
    if use_gpu:
        try:
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            raise RuntimeError(
                "TensorFlow initialized the GPU before PackMorph could enable "
                "dynamic memory growth"
            ) from error
    else:
        tf.config.set_visible_devices([], "GPU")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def load_sequence_model(
    path: Path,
    sequence_length: int,
    class_count: int,
    config: dict,
    *,
    class_names: list[str] | np.ndarray,
):
    expected_metadata, expected_fingerprint = _checkpoint_metadata(
        sequence_length,
        class_count,
        class_names,
        config,
    )
    with np.load(path, allow_pickle=False) as archive:
        metadata_fields = {
            "format_version",
            "metadata_json",
            "metadata_sha256",
        }
        if not metadata_fields.issubset(archive.files):
            raise ValueError("Model weight checkpoint has an invalid field set")
        version = np.asarray(archive["format_version"])
        if version.shape != () or int(version) != _CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Model weight checkpoint has an unsupported format")
        metadata = _checkpoint_text(archive["metadata_json"], "metadata_json")
        stored_fingerprint = _checkpoint_text(
            archive["metadata_sha256"], "metadata_sha256"
        )
        actual_fingerprint = hashlib.sha256(metadata.encode("utf-8")).hexdigest()
        if stored_fingerprint != actual_fingerprint:
            raise ValueError("Model weight checkpoint metadata fingerprint is invalid")
        if (
            metadata != expected_metadata
            or stored_fingerprint != expected_fingerprint
        ):
            raise ValueError(
                "Model weight checkpoint does not match the configured architecture, "
                "class order, or sequence length"
            )
        model = build_sequence_model(
            sequence_length,
            class_count,
            config,
            class_names=class_names,
        )
        expected_weights = model.get_weights()
        expected_fields = metadata_fields | {
            f"weight_{index:03d}" for index in range(len(expected_weights))
        }
        if set(archive.files) != expected_fields:
            raise ValueError("Model weight checkpoint has an invalid field set")
        loaded_weights = [
            np.asarray(archive[f"weight_{index:03d}"])
            for index in range(len(expected_weights))
        ]
    for index, (loaded, expected) in enumerate(zip(loaded_weights, expected_weights)):
        if loaded.shape != expected.shape or not np.all(np.isfinite(loaded)):
            raise ValueError(f"Model weight {index} has an invalid shape or value")
    model.set_weights(loaded_weights)
    return model


def build_sequence_model(
    sequence_length: int,
    class_count: int,
    config: dict,
    *,
    class_names: list[str] | np.ndarray,
):
    """Build a padding-aware dilated temporal CNN over packet sequences."""
    tf = _tensorflow()
    inputs = tf.keras.Input(shape=(sequence_length, 4), name="packet_sequence")
    masking = tf.keras.layers.Masking(mask_value=0.0, name="padding_mask")
    sequence_mask = masking.compute_mask(inputs)
    validity = inputs[:, :, 3:4]
    x = inputs
    for index, (filter_count, kernel_size) in enumerate(
        zip(config["conv_filters"], config["kernel_sizes"]),
        start=1,
    ):
        x = tf.keras.layers.Conv1D(
            filter_count,
            kernel_size,
            padding="same",
            activation="relu",
            name=f"temporal_conv_{index}",
        )(x)
        x = tf.keras.layers.LayerNormalization(name=f"layer_norm_{index}")(x)
        x = tf.keras.layers.Multiply(name=f"validity_gate_{index}")([x, validity])

    for index, dilation in enumerate(config["dilations"], start=1):
        residual = x
        x = tf.keras.layers.Conv1D(
            config["context_filters"],
            3,
            padding="same",
            dilation_rate=dilation,
            use_bias=False,
            activation="relu",
            name=f"dilated_context_{index}",
        )(x)
        x = tf.keras.layers.LayerNormalization(
            name=f"context_norm_{index}"
        )(x)
        x = tf.keras.layers.Multiply(name=f"context_gate_{index}")([x, validity])
        if int(residual.shape[-1]) != config["context_filters"]:
            residual = tf.keras.layers.Conv1D(
                config["context_filters"],
                1,
                padding="same",
                use_bias=False,
                name=f"residual_projection_{index}",
            )(residual)
            residual = tf.keras.layers.Multiply(
                name=f"residual_gate_{index}"
            )([residual, validity])
        x = tf.keras.layers.Add(name=f"context_residual_{index}")([x, residual])
    x = tf.keras.layers.GlobalAveragePooling1D(name="masked_temporal_pool")(
        x,
        mask=sequence_mask,
    )
    x = tf.keras.layers.Dense(config["dense_units"], activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(config["dropout"], name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        class_count, activation="softmax", name="service_class"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="packmorph_temporal_cnn")
    metadata, fingerprint = _checkpoint_metadata(
        sequence_length,
        class_count,
        class_names,
        config,
    )
    object.__setattr__(model, "_packmorph_checkpoint_metadata", metadata)
    object.__setattr__(model, "_packmorph_checkpoint_fingerprint", fingerprint)
    return model


def train_model(
    model,
    training_features: np.ndarray,
    training_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    weights: dict[int, float],
    config: dict,
    checkpoint_path: Path,
    seed: int,
):
    tf = _tensorflow()
    output_dir = checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_prefix = f".packmorph-{checkpoint_path.stem}-"
    for stale_directory in output_dir.glob(f"{temporary_prefix}*"):
        if stale_directory.is_dir():
            shutil.rmtree(stale_directory, ignore_errors=True)
    with tempfile.TemporaryDirectory(
        prefix=temporary_prefix, dir=output_dir
    ) as temporary_directory:
        temporary_checkpoint = Path(temporary_directory) / checkpoint_path.name
        batch_size = config["batch_size"]
        optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
        training_count = training_labels.size
        validation_count = validation_labels.size
        if training_count == 0 or validation_count == 0:
            raise ValueError("Training and validation partitions must be non-empty")

        sample_weights = np.asarray(
            [weights[int(label)] for label in training_labels],
            dtype=np.float32,
        )
        rng = np.random.default_rng(seed)
        best_validation_loss = np.inf
        best_weights: list[np.ndarray] | None = None
        epochs_without_improvement = 0

        # Bounded batch calls avoid Keras' implicit tf.data worker pools, which
        # can crash natively on some CPU/TensorFlow combinations. Direct eager
        # gradient steps also avoid Keras' compiled train-function adapter. This
        # path is backend-neutral and used for both CPU and GPU execution.
        for _ in range(config["epochs"]):
            order = rng.permutation(training_count)
            for start in range(0, training_count, batch_size):
                rows = order[start : start + batch_size]
                batch_features = tf.convert_to_tensor(training_features[rows])
                batch_labels = tf.convert_to_tensor(training_labels[rows])
                batch_weights = tf.convert_to_tensor(sample_weights[rows])
                with tf.GradientTape() as tape:
                    probabilities = model(batch_features, training=True)
                    losses = tf.keras.losses.sparse_categorical_crossentropy(
                        batch_labels,
                        probabilities,
                    )
                    weighted_losses = losses * batch_weights
                    batch_loss = tf.reduce_mean(weighted_losses)
                gradients = tape.gradient(batch_loss, model.trainable_variables)
                if not bool(tf.math.is_finite(batch_loss).numpy()) or any(
                    gradient is not None
                    and not bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())
                    for gradient in gradients
                ):
                    raise RuntimeError(
                        "Training produced a non-finite loss or gradient; the "
                        "previous checkpoint was preserved"
                    )
                optimizer.apply_gradients(
                    (gradient, variable)
                    for gradient, variable in zip(gradients, model.trainable_variables)
                    if gradient is not None
                )
            validation_loss_total = 0.0
            for start in range(0, validation_count, batch_size):
                stop = min(start + batch_size, validation_count)
                batch_labels = tf.convert_to_tensor(validation_labels[start:stop])
                probabilities = model(
                    validation_features[start:stop],
                    training=False,
                )
                losses = tf.keras.losses.sparse_categorical_crossentropy(
                    batch_labels,
                    probabilities,
                )
                validation_loss_total += float(tf.reduce_sum(losses).numpy())
            validation_loss = validation_loss_total / validation_count
            if not np.isfinite(validation_loss):
                raise RuntimeError(
                    "Training produced a non-finite metric; the previous "
                    "checkpoint was preserved"
                )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_weights = [np.array(value, copy=True) for value in model.get_weights()]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config["patience"]:
                    break

        if best_weights is None:
            raise RuntimeError(
                "Training did not produce finite best weights; the previous "
                "checkpoint was preserved"
            )
        model.set_weights(best_weights)
        checkpoint_arrays = {
            "format_version": np.asarray(
                _CHECKPOINT_FORMAT_VERSION, dtype=np.int16
            ),
            "metadata_json": np.asarray(model._packmorph_checkpoint_metadata),
            "metadata_sha256": np.asarray(
                model._packmorph_checkpoint_fingerprint
            ),
            **{
                f"weight_{index:03d}": value
                for index, value in enumerate(best_weights)
            },
        }
        with temporary_checkpoint.open("wb") as handle:
            np.savez_compressed(handle, **checkpoint_arrays)
        if (
            not temporary_checkpoint.is_file()
            or temporary_checkpoint.stat().st_size == 0
        ):
            raise RuntimeError(
                "Training did not produce a checkpoint; the previous checkpoint "
                "was preserved"
            )
        os.replace(temporary_checkpoint, checkpoint_path)


def predict_probabilities(model, features: np.ndarray, batch_size: int) -> np.ndarray:
    if features.shape[0] == 0:
        return np.empty((0, int(model.output_shape[-1])), dtype=np.float32)
    tf = _tensorflow()
    feature_shape = tuple(int(value) for value in features.shape[1:])
    cache_key = (batch_size, feature_shape)
    predictors = getattr(model, "_packmorph_predictors", None)
    if predictors is None:
        predictors = {}
        object.__setattr__(model, "_packmorph_predictors", predictors)
    predictor = predictors.get(cache_key)
    if predictor is None:
        predictor = tf.function(
            lambda batch: model(batch, training=False),
            input_signature=[
                tf.TensorSpec(
                    shape=(batch_size, *feature_shape),
                    dtype=tf.float32,
                )
            ],
            autograph=False,
            jit_compile=False,
        )
        predictors[cache_key] = predictor

    batches = []
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        valid_count = stop - start
        batch = features[start:stop]
        if valid_count < batch_size:
            fixed_batch = np.repeat(batch[:1], batch_size, axis=0)
            fixed_batch[:valid_count] = batch
        else:
            fixed_batch = batch
        probabilities = predictor(fixed_batch)
        batches.append(np.asarray(probabilities[:valid_count], dtype=np.float32))
    return np.concatenate(batches, axis=0)
