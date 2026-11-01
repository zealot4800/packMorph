"""Four-layer prototypical neural classifier for CICFlowMeter flow features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import capture_macro_accuracy
from .runtime import _tensorflow, prepare_runtime


@dataclass
class CheckpointSelection:
    """Track actual best weights separately from meaningful stopping progress."""

    patience: int
    min_delta: float
    best_score: float = -np.inf
    best_episode: int = 0
    progress_score: float = -np.inf
    stale_checks: int = 0
    lr_stale_checks: int = 0

    def observe(self, episode: int, score: float) -> bool:
        if not np.isfinite(score):
            raise RuntimeError('Validation produced a nonfinite selection score')
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_episode = score, episode
        if score > self.progress_score + self.min_delta:
            self.progress_score = score
            self.stale_checks = self.lr_stale_checks = 0
        else:
            self.stale_checks += 1
            self.lr_stale_checks += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.stale_checks >= self.patience


def build_embedding(config: dict, feature_count: int):
    """Four 64-unit ReLU layers; configured dropout after layers two and three."""
    tf = _tensorflow()
    inputs = tf.keras.Input(shape=(feature_count,), name="cic_flow_features")
    x = inputs
    for index in range(4):
        x = tf.keras.layers.Dense(64, activation="relu", name=f"embedding_{index + 1}")(x)
        if index in (1, 2):
            x = tf.keras.layers.Dropout(config["dropout"], name=f"dropout_{index + 1}")(x)
    return tf.keras.Model(inputs, x, name="cic_prototypical_embedding")


def _class_pools(labels: np.ndarray, class_count: int) -> list[np.ndarray]:
    """Sample flows uniformly within each class."""
    pools = []
    for label in range(class_count):
        rows = np.flatnonzero(labels == label)
        if not len(rows):
            raise ValueError(f"CICFlowMeter training has no examples of class {label}")
        pools.append(rows)
    return pools


def _capture_weights(captures: np.ndarray,
                     pool: np.ndarray) -> np.ndarray:
    """Give each source CSV equal total probability within its class."""
    if captures is None:
        return np.full(len(pool), 1.0 / len(pool))
    _, inverse, counts = np.unique(captures[pool], return_inverse=True,
                                    return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.sum()


def _support_rows(labels: np.ndarray, class_count: int, config: dict,
                  captures: np.ndarray) -> np.ndarray:
    """Use the same frozen support rows for checkpoint scoring and final prototypes."""
    rng = np.random.default_rng(config["split_seed"] + 2)
    size = config["calibration_support_per_class"]
    return np.concatenate([
        rng.choice(pool, size=min(size, len(pool)), replace=False,
                   p=_capture_weights(captures, pool))
        for pool in _class_pools(labels, class_count)
    ])


def _checkpoint_scores(model, training: np.ndarray, training_labels: np.ndarray,
                      support_rows: np.ndarray, validation: np.ndarray,
                      validation_labels: np.ndarray, validation_captures: np.ndarray,
                      class_count: int, batch_size: int,
                      training_captures: np.ndarray | None) -> dict:
    """Evaluate one frozen calibration classifier; only validation selects weights."""
    classifier = _fit_support(model, training, training_labels, support_rows,
                              class_count, batch_size)
    predicted = classifier.predict_probabilities(validation, batch_size).argmax(axis=1)
    train_predicted = classifier.predict_probabilities(training, batch_size).argmax(axis=1)
    return dict(validation_score=capture_macro_accuracy(
        validation_labels, predicted, validation_captures, class_count),
        train_score=capture_macro_accuracy(
        training_labels, train_predicted, training_captures, class_count))


def train_embedding(features: np.ndarray, labels: np.ndarray,
                    class_count: int, config: dict,
                    captures: np.ndarray,
                    validation: tuple[np.ndarray, np.ndarray, np.ndarray]):
    """Train episodically, balancing queries across the configured service classes."""
    seed = config["split_seed"]
    prepare_runtime(seed, config["device"])
    tf = _tensorflow()
    model = build_embedding(config, features.shape[1])
    rng = np.random.default_rng(seed)
    support_size = config["support_per_class"]
    query_size = config["queries_per_episode"]
    pools = _class_pools(labels, class_count)
    pool_weights = [_capture_weights(captures, pool) for pool in pools]
    if any(len(pool) < support_size + 1 for pool in pools):
        raise ValueError(f"CICFlowMeter training needs more than {support_size} examples per class")
    optimizer = tf.keras.optimizers.Adam(
        config["learning_rate"], weight_decay=config['weight_decay'],
        global_clipnorm=config['gradient_clipnorm'])
    checkpoint_support_rows = _support_rows(labels, class_count, config, captures)
    selection = CheckpointSelection(config['early_stopping_patience'], config['min_delta'])
    best_weights = None
    checkpoint_interval = config['validation_interval']
    history = []
    interval_loss = 0.0
    interval_steps = 0

    @tf.function(reduce_retracing=True)
    def step(support, query, query_labels):
        with tf.GradientTape() as tape:
            encoded = model(tf.concat((support, query), axis=0), training=True)
            prototypes = tf.reduce_mean(
                tf.reshape(encoded[:class_count * support_size],
                           (class_count, support_size, 64)), axis=1)
            distances = tf.reduce_sum(tf.square(
                encoded[class_count * support_size:, None, :] - prototypes[None, :, :]),
                axis=2)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                query_labels, -distances, from_logits=True))
        gradients = tape.gradient(loss, model.trainable_variables)
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, 'Training gradient is nonfinite')
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for episode in range(config["episodes"]):
        support_by_class = [rng.choice(pool, size=support_size, replace=False,
                                       p=weights)
                            for pool, weights in zip(pools, pool_weights)]
        support_rows = np.concatenate(support_by_class)
        # Each class contributes to the query loss. Support examples are excluded.
        query_by_class = []
        for label, pool in enumerate(pools):
            count = query_size // class_count + (label < query_size % class_count)
            if count == 0:
                continue
            available = ~np.isin(pool, support_by_class[label])
            candidates = pool[available]
            weights = pool_weights[label][available]
            weights = weights / weights.sum()
            query_by_class.append(rng.choice(
                candidates, size=count, replace=len(candidates) < count,
                p=weights))
        query_rows = np.concatenate(query_by_class)
        rng.shuffle(query_rows)
        loss = step(features[support_rows], features[query_rows], labels[query_rows])
        if not bool(tf.math.is_finite(loss)):
            raise RuntimeError("CICFlowMeter episodic training produced a nonfinite loss")
        interval_loss += float(loss)
        interval_steps += 1
        if ((episode + 1) % checkpoint_interval == 0
                or episode + 1 == config["episodes"]):
            scores = _checkpoint_scores(
                model, features, labels, checkpoint_support_rows, *validation,
                class_count, config["batch_size"], captures)
            score = scores['validation_score']
            improved = selection.observe(episode + 1, score)
            current_lr = float(optimizer.learning_rate.numpy())
            history.append(dict(episode=episode + 1, train_loss=interval_loss / interval_steps,
                                learning_rate=current_lr, **scores))
            interval_loss = 0.0
            interval_steps = 0
            metric_name = 'class macro accuracy' if captures is None else 'capture macro accuracy'
            print(f"  episode {episode + 1}/{config['episodes']}: "
                  f"loss={history[-1]['train_loss']:.4f}; train {metric_name}={scores['train_score']:.4f}; "
                  f"validation={score:.4f}; lr={current_lr:.3g}; "
                  f"patience={selection.stale_checks}/{selection.patience}", flush=True)
            if improved:
                best_weights = model.get_weights()
            if selection.should_stop:
                print(f'  Early stopping; restoring episode {selection.best_episode}.', flush=True)
                break
            if selection.lr_stale_checks >= config['lr_patience']:
                reduced_lr = max(config['min_learning_rate'], current_lr * config['lr_factor'])
                optimizer.learning_rate.assign(min(current_lr, reduced_lr))
                selection.lr_stale_checks = 0
    if best_weights is None:
        raise RuntimeError('Training did not produce a valid validation checkpoint')
    model.set_weights(best_weights)
    metric_key = ('validation_macro_accuracy' if captures is None
                  else 'validation_capture_macro_accuracy')
    return model, {"episode": selection.best_episode,
                   metric_key: selection.best_score,
                   "checkpoint_interval": checkpoint_interval,
                   "episodes_run": episode + 1,
                   "stopped_early": episode + 1 < config['episodes'],
                   "stop_reason": 'validation_plateau' if selection.should_stop else 'episode_limit',
                   "history": history}


@dataclass(slots=True)
class CicClassifier:
    embedding: object
    prototypes: np.ndarray
    diagonal_variances: np.ndarray

    def predict_probabilities(self, features: np.ndarray, batch_size: int) -> np.ndarray:
        probabilities = np.empty((len(features), len(self.prototypes)), dtype=np.float32)
        for start in range(0, len(features), batch_size):
            embedded = np.asarray(self.embedding(features[start:start + batch_size], training=False),
                                  dtype=np.float64)
            distances = np.sum(np.square(embedded[:, None, :] - self.prototypes[None, :, :])
                               / self.diagonal_variances[None, :, :], axis=2)
            logits = -distances
            weights = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities[start:start + batch_size] = weights / weights.sum(axis=1, keepdims=True)
        return probabilities


def fit_classifier(embedding, training: np.ndarray, training_labels: np.ndarray,
                   class_count: int, config: dict,
                   captures: np.ndarray) -> CicClassifier:
    """Freeze class prototypes and their diagonal variances."""
    support_rows = _support_rows(training_labels, class_count, config, captures)
    return _fit_support(embedding, training, training_labels, support_rows,
                        class_count, config["batch_size"])


def _fit_support(embedding, training: np.ndarray, training_labels: np.ndarray,
                 support_rows: np.ndarray, class_count: int, batch_size: int) -> CicClassifier:
    """Build the same classifier for validation and the frozen artifact."""
    support_labels = training_labels[support_rows]
    support = np.concatenate([
        np.asarray(embedding(training[support_rows[start:start + batch_size]], training=False),
                   dtype=np.float64)
        for start in range(0, len(support_rows), batch_size)
    ])
    prototypes = np.stack([support[support_labels == label].mean(axis=0)
                           for label in range(class_count)])
    diagonal_variances = np.stack([
        np.maximum(support[support_labels == label].var(axis=0), 1e-5)
        for label in range(class_count)
    ])
    return CicClassifier(embedding, prototypes, diagonal_variances)
