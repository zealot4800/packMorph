"""The shared padding-aware temporal CNN and its unchanged training loop."""

from __future__ import annotations

import os

import numpy as np


def _tensorflow():
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    return tf


def prepare_runtime(seed: int, device: str) -> None:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf = _tensorflow()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["TF_NUM_INTRAOP_THREADS"]))
        tf.config.threading.set_inter_op_parallelism_threads(int(os.environ["TF_NUM_INTEROP_THREADS"]))
    except RuntimeError as error:
        raise RuntimeError("TensorFlow initialized before thread configuration") from error
    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not gpus:
        raise RuntimeError("model.device is gpu, but TensorFlow found no usable GPU")
    if device != "cpu" and gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        tf.config.set_visible_devices([], "GPU")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def build_sequence_model(sequence_length: int, class_count: int, config: dict,
                         *, class_names: list[str] | np.ndarray):
    """Build the original four-feature padding-aware dilated temporal CNN."""
    if len(class_names) != class_count:
        raise ValueError("Class names do not match the CNN output count")
    tf = _tensorflow()
    inputs = tf.keras.Input(shape=(sequence_length, 4), name="packet_sequence")
    masking = tf.keras.layers.Masking(mask_value=0.0, name="padding_mask")
    sequence_mask = masking.compute_mask(inputs)
    validity = inputs[:, :, 3:4]
    x = inputs
    for index, (filter_count, kernel_size) in enumerate(zip(config["conv_filters"], config["kernel_sizes"]), start=1):
        x = tf.keras.layers.Conv1D(filter_count, kernel_size, padding="same",
                                   activation="relu", name=f"temporal_conv_{index}")(x)
        x = tf.keras.layers.LayerNormalization(name=f"layer_norm_{index}")(x)
        x = tf.keras.layers.Multiply(name=f"validity_gate_{index}")([x, validity])
    for index, dilation in enumerate(config["dilations"], start=1):
        residual = x
        x = tf.keras.layers.Conv1D(config["context_filters"], 3, padding="same",
                                   dilation_rate=dilation, use_bias=False, activation="relu",
                                   name=f"dilated_context_{index}")(x)
        x = tf.keras.layers.LayerNormalization(name=f"context_norm_{index}")(x)
        x = tf.keras.layers.Multiply(name=f"context_gate_{index}")([x, validity])
        if int(residual.shape[-1]) != config["context_filters"]:
            residual = tf.keras.layers.Conv1D(config["context_filters"], 1, padding="same",
                                              use_bias=False, name=f"residual_projection_{index}")(residual)
            residual = tf.keras.layers.Multiply(name=f"residual_gate_{index}")([residual, validity])
        x = tf.keras.layers.Add(name=f"context_residual_{index}")([x, residual])
    x = tf.keras.layers.GlobalAveragePooling1D(name="masked_temporal_pool")(x, mask=sequence_mask)
    x = tf.keras.layers.Dense(config["dense_units"], activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(config["dropout"], name="dropout")(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="service_class")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="packmorph_temporal_cnn")


def train_model(model, training_features: np.ndarray, training_labels: np.ndarray,
                validation_features: np.ndarray, validation_labels: np.ndarray,
                weights: dict[int, float], config: dict, seed: int):
    """Train with the original weighted batches and validation early stopping."""
    tf = _tensorflow()
    batch_size = config["batch_size"]
    optimizer = tf.keras.optimizers.Adam(config["learning_rate"])
    training_count = training_labels.size
    validation_count = validation_labels.size
    if training_count == 0 or validation_count == 0:
        raise ValueError("Training and validation partitions must be nonempty")
    sample_weights = np.asarray([weights[int(label)] for label in training_labels], dtype=np.float32)
    rng = np.random.default_rng(seed)
    best_validation_loss = np.inf
    best_weights: list[np.ndarray] | None = None
    epochs_without_improvement = 0
    for _ in range(config["epochs"]):
        order = rng.permutation(training_count)
        for start in range(0, training_count, batch_size):
            rows = order[start:start + batch_size]
            batch_features = tf.convert_to_tensor(training_features[rows])
            batch_labels = tf.convert_to_tensor(training_labels[rows])
            batch_weights = tf.convert_to_tensor(sample_weights[rows])
            with tf.GradientTape() as tape:
                probabilities = model(batch_features, training=True)
                losses = tf.keras.losses.sparse_categorical_crossentropy(batch_labels, probabilities)
                batch_loss = tf.reduce_mean(losses * batch_weights)
            gradients = tape.gradient(batch_loss, model.trainable_variables)
            if not bool(tf.math.is_finite(batch_loss).numpy()) or any(
                gradient is not None and not bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())
                for gradient in gradients
            ):
                raise RuntimeError("Training produced a nonfinite loss or gradient")
            optimizer.apply_gradients((gradient, variable)
                                      for gradient, variable in zip(gradients, model.trainable_variables)
                                      if gradient is not None)
        validation_total = 0.0
        for start in range(0, validation_count, batch_size):
            stop = min(start + batch_size, validation_count)
            batch_labels = tf.convert_to_tensor(validation_labels[start:stop])
            probabilities = model(validation_features[start:stop], training=False)
            losses = tf.keras.losses.sparse_categorical_crossentropy(batch_labels, probabilities)
            validation_total += float(tf.reduce_sum(losses).numpy())
        validation_loss = validation_total / validation_count
        if not np.isfinite(validation_loss):
            raise RuntimeError("Training produced a nonfinite validation loss")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_weights = [np.array(value, copy=True) for value in model.get_weights()]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                break
    if best_weights is None:
        raise RuntimeError("Training did not produce finite best weights")
    model.set_weights(best_weights)
    return model


def predict_probabilities(model, features: np.ndarray, batch_size: int) -> np.ndarray:
    if features.shape[0] == 0:
        return np.empty((0, int(model.output_shape[-1])), dtype=np.float32)
    tf = _tensorflow()
    shape = tuple(int(value) for value in features.shape[1:])
    cache_key = (batch_size, shape)
    predictors = getattr(model, "_packmorph_predictors", None)
    if predictors is None:
        predictors = {}
        object.__setattr__(model, "_packmorph_predictors", predictors)
    predictor = predictors.get(cache_key)
    if predictor is None:
        predictor = tf.function(lambda batch: model(batch, training=False),
                                input_signature=[tf.TensorSpec(shape=(batch_size, *shape), dtype=tf.float32)],
                                autograph=False, jit_compile=False)
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
