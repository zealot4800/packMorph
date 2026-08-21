from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


def _tensorflow():
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-packmorph")
    import tensorflow as tf

    return tf


def prepare_runtime(seed: int, device: str) -> str:
    # Device visibility must be chosen before TensorFlow is imported. Each
    # experiment environment runs in a fresh spawned process, so this setting
    # cannot leak from one environment into the next.
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    tf = _tensorflow()
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
        runtime_device = f"GPU ({len(physical_gpus)} visible)"
    else:
        tf.config.set_visible_devices([], "GPU")
        runtime_device = "CPU"

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    print(f"TensorFlow runtime device: {runtime_device}", flush=True)
    return runtime_device


def build_sequence_model(sequence_length: int, class_count: int, config: dict):
    """Build a padding-aware CNN-BiGRU adversary over packet sequences."""
    tf = _tensorflow()
    inputs = tf.keras.Input(shape=(sequence_length, 4), name="packet_sequence")
    masking = tf.keras.layers.Masking(mask_value=0.0, name="padding_mask")
    sequence_mask = masking.compute_mask(inputs)
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
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(config["gru_units"]),
        name="bidirectional_gru",
    )(x, mask=sequence_mask)
    x = tf.keras.layers.Dense(config["dense_units"], activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(config["dropout"], name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        class_count, activation="softmax", name="service_class"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="packmorph_cnn_bigru")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model,
    training_features: np.ndarray,
    training_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    weights: dict[int, float],
    config: dict,
    output_dir: Path,
):
    tf = _tensorflow()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_directory in output_dir.glob(".packmorph-checkpoint-*"):
        if stale_directory.is_dir():
            shutil.rmtree(stale_directory, ignore_errors=True)
    final_checkpoint = output_dir / "best_model.keras"
    with tempfile.TemporaryDirectory(
        prefix=".packmorph-checkpoint-", dir=output_dir
    ) as temporary_directory:
        temporary_checkpoint = Path(temporary_directory) / "best_model.keras"
        callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config["patience"],
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(temporary_checkpoint),
                monitor="val_loss",
                save_best_only=True,
            ),
        ]
        history = model.fit(
            training_features,
            training_labels,
            validation_data=(validation_features, validation_labels),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            class_weight=weights,
            callbacks=callbacks,
            verbose=2,
            shuffle=True,
        )
        for metric in ("loss", "val_loss"):
            values = np.asarray(history.history.get(metric, ()), dtype=np.float64)
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"Training did not produce a finite {metric} history; "
                    "the previous checkpoint was preserved"
                )
        if (
            not temporary_checkpoint.is_file()
            or temporary_checkpoint.stat().st_size == 0
        ):
            raise RuntimeError(
                "Training did not produce a checkpoint; the previous checkpoint "
                "was preserved"
            )
        os.replace(temporary_checkpoint, final_checkpoint)
    return history


def predict_probabilities(model, features: np.ndarray, batch_size: int) -> np.ndarray:
    return model.predict(features, batch_size=batch_size, verbose=0)
