"""Training-only categorical LSTM policy for the Securitas simulation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .securitas import ACTION_SIZES, freeze_policy, normalized_sequence_distance, simulate
from .traffic import source_trace


class CategoricalPolicy(tf.keras.Model):
    """Autoregressive action distribution, conditioned on preceding actions."""

    def __init__(self, hidden_units: int = 32):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(len(ACTION_SIZES) + 1, hidden_units)
        self.cell = tf.keras.layers.LSTMCell(hidden_units)
        self.head = tf.keras.layers.Dense(len(ACTION_SIZES))
        self.hidden_units = hidden_units

    def rollout(self, length: int, generator: tf.random.Generator | None = None):
        hidden = tf.zeros((1, self.hidden_units), tf.float32)
        cell = tf.zeros_like(hidden)
        previous = tf.constant([len(ACTION_SIZES)], tf.int32)
        actions = []
        log_probabilities = []
        for _ in range(length):
            output, [hidden, cell] = self.cell(self.embedding(previous), states=[hidden, cell])
            logits = self.head(output)
            if generator is None:
                choice = tf.argmax(logits, axis=-1, output_type=tf.int32)
            else:
                uniform = generator.uniform(tf.shape(logits), minval=1e-7, maxval=1.0 - 1e-7)
                gumbel = -tf.math.log(-tf.math.log(uniform))
                choice = tf.argmax(logits + gumbel, axis=-1, output_type=tf.int32)
            actions.append(int(choice.numpy()[0]))
            log_probabilities.append(-tf.nn.sparse_softmax_cross_entropy_with_logits(labels=choice, logits=logits)[0])
            previous = choice
        return actions, tf.add_n(log_probabilities)


def _score(data, row: int, policy: dict, settings: dict, network_rng: np.random.Generator,
           surrogate_predict) -> tuple[float, dict]:
    source = source_trace(data, row)
    defended = simulate(source, policy, network_rng, settings).trace
    probability = float(surrogate_predict(data.subset(np.asarray([row])), [defended])[0, int(data.labels[row])])
    distance = normalized_sequence_distance(source, defended)
    ratio = float(np.abs(defended.sizes).sum() / np.abs(source.sizes).sum())
    weights = settings["objective_weights"]
    value = (float(weights["confidence"]) * probability
             - float(weights["distance"]) * distance
             + float(weights["bandwidth"]) * ratio)
    return value, {"true_class_probability": probability, "normalized_sequence_distance": distance,
                   "defended_to_original_byte_ratio": ratio, "objective": value}


def train_policy(training, validation, settings: dict, seed: int, surrogate_predict) -> tuple[dict, dict]:
    """Use training rewards and validation-only policy selection.

    The categorical sample is discrete. REINFORCE multiplies its summed log
    probability by reward minus a moving baseline; no test label or reporting
    adversary prediction enters this function.
    """
    length = int(settings["strategy_length"])
    steps = int(settings["policy_steps"])
    candidates = int(settings["validation_examples"])
    preference = float(settings["fragmentation_preference"])
    weights = settings["objective_weights"]
    if (length <= 0 or steps <= 0 or candidates <= 0 or len(validation) == 0
            or not 0 <= preference <= 1 or set(weights) != {"confidence", "distance", "bandwidth"}
            or any(not np.isfinite(float(x)) or float(x) < 0 for x in weights.values())
            or not np.isfinite(float(settings["policy_learning_rate"]))
            or float(settings["policy_learning_rate"]) <= 0):
        raise ValueError("Invalid Securitas policy training settings")
    tf.keras.utils.set_random_seed(seed)
    action_rng = tf.random.Generator.from_seed(seed + 11)
    choice_rng = np.random.default_rng(np.random.SeedSequence([seed, 12]))
    network_rng = np.random.default_rng(np.random.SeedSequence([seed, 13]))
    policy = CategoricalPolicy(int(settings.get("policy_hidden_units", 32)))
    policy.rollout(1)
    optimizer = tf.keras.optimizers.Adam(float(settings["policy_learning_rate"]))
    baseline = None
    best = float("inf")
    best_policy = None
    history = []
    validation_rows = np.random.default_rng(np.random.SeedSequence([seed, 17])).permutation(len(validation))[:min(candidates, len(validation))]
    validation_every = max(1, int(settings.get("validation_every_steps", 5)))
    for step in range(steps):
        row = int(choice_rng.integers(len(training)))
        with tf.GradientTape() as tape:
            indices, log_probability = policy.rollout(length, action_rng)
            sizes = [ACTION_SIZES[index] for index in indices]
            trial = freeze_policy(sizes, choice_rng, preference)
            reward, components = _score(training, row, trial, settings, network_rng, surrogate_predict)
            advantage = reward - (baseline if baseline is not None else reward)
            loss = tf.cast(advantage, tf.float32) * log_probability
        gradients = tape.gradient(loss, policy.trainable_variables)
        optimizer.apply_gradients([(gradient, variable) for gradient, variable in zip(gradients, policy.trainable_variables) if gradient is not None])
        baseline = reward if baseline is None else 0.9 * baseline + 0.1 * reward
        record = {"step": step + 1, "training": components}
        if (step + 1) % validation_every == 0 or step + 1 == steps:
            greedy, _ = policy.rollout(length)
            fixed = freeze_policy([ACTION_SIZES[index] for index in greedy],
                                  np.random.default_rng(np.random.SeedSequence([seed, 14, step])), preference)
            scores = [_score(validation, int(row), fixed, settings,
                             np.random.default_rng(np.random.SeedSequence([seed, 15, step, int(row)])),
                             surrogate_predict)[0] for row in validation_rows]
            value = float(np.mean(scores))
            record["validation_objective"] = value
            if value < best:
                best = value
                best_policy = fixed
        history.append(record)
    if best_policy is None:
        raise RuntimeError("Securitas policy produced no validation candidate")
    return best_policy, {"seed": seed, "best_validation_objective": best,
                    "objective_convention": "minimize confidence - distance + byte ratio",
                    "history": history}
