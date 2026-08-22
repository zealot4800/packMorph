from __future__ import annotations

import copy
import json
import logging
import multiprocessing
import os
import shutil
import tempfile
from dataclasses import dataclass
from multiprocessing.connection import wait as wait_for_connection
from pathlib import Path

import numpy as np

from .config import environment_slug
from .dataset import SequenceDataset
from .extract import extraction_fingerprint
from .features import FeatureScaler
from .model import (
    build_sequence_model,
    load_sequence_model,
    predict_probabilities,
    prepare_runtime,
    train_model,
)
from .morph import MorphInfeasibleError, ProfileLibrary, morph_dataset
from .split import class_weights, stratified_group_split


logger = logging.getLogger(__name__)


def _configure_worker_logging() -> None:
    level_name = os.environ.get("PACKMORPH_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(message)s", force=True)


def _worker_error(error: Exception) -> str:
    return f"{type(error).__name__}: {error or 'no details available'}"


@dataclass(slots=True)
class GuidedSelection:
    target_labels: np.ndarray
    template_rows: np.ndarray
    padding_fractions: np.ndarray
    fct_fractions: np.ndarray
    bandwidth_overheads: np.ndarray
    fct_overheads: np.ndarray


def _collect_worker_message(process, receiver):
    """Drain a worker pipe before join so a large message cannot deadlock send()."""
    message = None
    try:
        ready = wait_for_connection((receiver, process.sentinel))
        if receiver in ready:
            try:
                message = receiver.recv()
            except EOFError:
                pass
        process.join()
        if message is None and receiver.poll():
            try:
                message = receiver.recv()
            except EOFError:
                pass
        return message
    except BaseException:
        if process.is_alive():
            process.terminate()
        process.join()
        raise
    finally:
        receiver.close()


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
    # Morph several inference batches at once. Prediction still runs in fixed
    # model-sized chunks, while this larger bound avoids thousands of tiny
    # selector calls during the resource-grid search.
    target_windows = prediction_batch_size * 32
    return max(
        1,
        min(prediction_batch_size, target_windows // maximum_windows),
    )


def _candidate_template_rows(
    profile_rows: np.ndarray,
    source_count: int,
    candidate_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = np.empty((source_count, candidate_count), dtype=np.int64)
    replace = profile_rows.size < candidate_count
    for row in range(source_count):
        candidates[row] = rng.choice(
            profile_rows,
            size=candidate_count,
            replace=replace,
        )
    return candidates


def _normalized_overhead(values: np.ndarray, maximum_ratio: float) -> np.ndarray:
    allowance = maximum_ratio - 1.0
    non_negative = np.maximum(values.astype(np.float64), 0.0)
    if allowance <= 1e-12:
        return np.where(non_negative <= 1e-12, 0.0, np.inf)
    return non_negative / allowance


def _evaluate_guided_candidate_batch(
    selector_models: tuple,
    source_batch: SequenceDataset,
    library: ProfileLibrary,
    scaler: FeatureScaler,
    morph_config: dict,
    targets: np.ndarray,
    templates: np.ndarray,
    padding_fraction: float,
    fct_fraction: float,
    prediction_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = morph_dataset(
        source_batch,
        library,
        morph_config,
        target_labels=targets,
        template_rows=templates,
        padding_fraction=np.full(
            len(source_batch), padding_fraction, dtype=np.float64
        ),
        fct_fraction=np.full(len(source_batch), fct_fraction, dtype=np.float64),
    )
    features = scaler.transform(result.dataset)
    probabilities = []
    for selector_model in selector_models:
        window_probabilities = predict_probabilities(
            selector_model,
            features,
            prediction_batch_size,
        )
        probabilities.append(
            _aggregate_probabilities(
                window_probabilities,
                result.source_indices,
                len(source_batch),
            )
        )
    return (
        np.stack(probabilities),
        result.bandwidth_overheads,
        result.fct_overheads,
    )


def _plan_guided_selection(
    selector_models: tuple,
    testing: SequenceDataset,
    library: ProfileLibrary,
    scaler: FeatureScaler,
    config: dict,
    seed: int,
) -> GuidedSelection:
    expected_selectors = config["model"]["selector_count"]
    if len(selector_models) != expected_selectors:
        raise ValueError(
            "Guided selection requires exactly "
            f"{expected_selectors} selector model(s)"
        )
    source_count = len(testing)
    class_count = len(testing.class_names)
    morph_config = config["morph"]
    candidate_count = morph_config["candidates_per_target"]
    padding_fractions = np.asarray(
        morph_config["padding_fractions"], dtype=np.float64
    )
    fct_fractions = np.asarray(morph_config["fct_fractions"], dtype=np.float64)
    attack_tolerance = float(morph_config["attack_utility_tolerance"])
    for name, values in (
        ("padding_fractions", padding_fractions),
        ("fct_fractions", fct_fractions),
    ):
        if (
            values.ndim != 1
            or values.size == 0
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError(f"morph.{name} must contain finite fractions in [0, 1]")
    if not np.isfinite(attack_tolerance) or attack_tolerance < 0.0:
        raise ValueError("morph.attack_utility_tolerance must be non-negative")
    prediction_batch_size = config["model"]["batch_size"]
    source_batch_size = _morph_source_batch_size(testing, config)
    library.validate_guided_assignment_size(testing.labels)

    best_scores = np.full((source_count, class_count), -np.inf, dtype=np.float64)
    best_templates = np.full((source_count, class_count), -1, dtype=np.int64)
    best_padding_fractions = np.zeros(
        (source_count, class_count), dtype=np.float64
    )
    best_fct_fractions = np.ones((source_count, class_count), dtype=np.float64)
    best_bandwidth = np.full((source_count, class_count), np.inf, dtype=np.float64)
    best_fct = np.full((source_count, class_count), np.inf, dtype=np.float64)
    epsilon = 1e-7
    log_probability_floor = float(np.log(epsilon))
    bandwidth_weight = float(morph_config["bandwidth_cost_weight"])
    fct_weight = float(morph_config["fct_cost_weight"])
    maximum_cost_weight = max(bandwidth_weight, fct_weight)
    variant_count = (
        candidate_count * padding_fractions.size * fct_fractions.size
    )
    if variant_count * len(selector_models) > 256:
        raise ValueError(
            "Guided candidate/resource variants multiplied by selector models "
            "cannot exceed 256 evaluations per source-target pair"
        )

    for target in range(class_count):
        eligible = np.flatnonzero(testing.labels != target)
        target_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 2, target])
        )
        candidate_rows = _candidate_template_rows(
            library.indices_by_class[target],
            eligible.size,
            candidate_count,
            target_rng,
        )
        attacks = np.full((variant_count, eligible.size), -np.inf, dtype=np.float64)
        target_log_probabilities = np.full_like(attacks, -np.inf)
        overhead_costs = np.full_like(attacks, np.inf)
        candidate_bandwidth = np.full_like(attacks, np.inf)
        candidate_fct = np.full_like(attacks, np.inf)
        candidate_templates = np.full(
            (variant_count, eligible.size), -1, dtype=np.int64
        )
        candidate_padding_fractions = np.empty(variant_count, dtype=np.float64)
        candidate_fct_fractions = np.empty(variant_count, dtype=np.float64)
        variant = 0
        for candidate in range(candidate_count):
            for padding_fraction in padding_fractions:
                for fct_fraction in fct_fractions:
                    candidate_padding_fractions[variant] = padding_fraction
                    candidate_fct_fractions[variant] = fct_fraction
                    for start in range(0, eligible.size, source_batch_size):
                        stop = min(start + source_batch_size, eligible.size)
                        positions = np.arange(start, stop, dtype=np.int64)
                        global_rows = eligible[start:stop]
                        templates = candidate_rows[start:stop, candidate]
                        pending = [(positions, global_rows, templates)]
                        while pending:
                            batch_positions, batch_rows, batch_templates = pending.pop()
                            source_batch = testing.subset(batch_rows)
                            targets = np.full(
                                batch_rows.size,
                                target,
                                dtype=testing.labels.dtype,
                            )
                            try:
                                probabilities, bandwidth, fct = (
                                    _evaluate_guided_candidate_batch(
                                        selector_models,
                                        source_batch,
                                        library,
                                        scaler,
                                        morph_config,
                                        targets,
                                        batch_templates,
                                        float(padding_fraction),
                                        float(fct_fraction),
                                        prediction_batch_size,
                                    )
                                )
                            except MorphInfeasibleError:
                                if batch_rows.size > 1:
                                    midpoint = batch_rows.size // 2
                                    pending.append(
                                        (
                                            batch_positions[midpoint:],
                                            batch_rows[midpoint:],
                                            batch_templates[midpoint:],
                                        )
                                    )
                                    pending.append(
                                        (
                                            batch_positions[:midpoint],
                                            batch_rows[:midpoint],
                                            batch_templates[:midpoint],
                                        )
                                    )
                                    continue
                                continue

                            source_classes = testing.labels[batch_rows].astype(
                                np.int64
                            )
                            row_indices = np.arange(batch_rows.size)
                            source_probabilities = probabilities[
                                :, row_indices, source_classes
                            ].astype(np.float64)
                            target_probabilities = probabilities[:, :, target].astype(
                                np.float64
                            )
                            selector_target_log_probability = np.log(
                                np.clip(target_probabilities, epsilon, 1.0)
                            )
                            pairwise_target_confidence = (
                                (target_probabilities + epsilon)
                                / (
                                    target_probabilities
                                    + source_probabilities
                                    + 2.0 * epsilon
                                )
                            )
                            # Reward both absolute target resemblance and the
                            # target-versus-source odds. The absolute term avoids
                            # treating a tiny target probability as successful
                            # merely because the source probability is even lower.
                            selector_attacks = selector_target_log_probability + np.log(
                                np.clip(pairwise_target_confidence, epsilon, 1.0)
                            )
                            # A candidate is only as strong as its weakest
                            # independently seeded selector. This reduces the
                            # single-model winner's curse before evaluator replay.
                            attack = np.min(selector_attacks, axis=0)
                            target_log_probability = np.min(
                                selector_target_log_probability,
                                axis=0,
                            )
                            normalized_bandwidth = _normalized_overhead(
                                bandwidth,
                                morph_config["maximum_padding_ratio"],
                            )
                            normalized_fct = _normalized_overhead(
                                fct,
                                morph_config["maximum_fct_ratio"],
                            )
                            overhead_cost = np.maximum(
                                bandwidth_weight * normalized_bandwidth,
                                fct_weight * normalized_fct,
                            )
                            attacks[variant, batch_positions] = attack
                            target_log_probabilities[variant, batch_positions] = (
                                target_log_probability
                            )
                            overhead_costs[variant, batch_positions] = overhead_cost
                            candidate_bandwidth[variant, batch_positions] = bandwidth
                            candidate_fct[variant, batch_positions] = fct
                            candidate_templates[variant, batch_positions] = (
                                batch_templates
                            )
                    variant += 1

        if variant != variant_count:
            raise RuntimeError("Guided search evaluated an invalid variant count")
        finite = (
            np.isfinite(attacks)
            & np.isfinite(target_log_probabilities)
            & np.isfinite(overhead_costs)
            & (candidate_templates >= 0)
        )
        best_attack = np.max(np.where(finite, attacks, -np.inf), axis=0)
        # Selector probabilities are float32, so include a small comparison
        # tolerance at the configured epsilon boundary.
        near_best = finite & (
            attacks >= best_attack[None, :] - attack_tolerance - 1e-7
        )
        for local_row, global_row in enumerate(eligible):
            choices = np.flatnonzero(near_best[:, local_row])
            if choices.size == 0:
                continue
            normalized_choice_bandwidth = _normalized_overhead(
                candidate_bandwidth[choices, local_row],
                morph_config["maximum_padding_ratio"],
            )
            normalized_choice_fct = _normalized_overhead(
                candidate_fct[choices, local_row],
                morph_config["maximum_fct_ratio"],
            )
            weighted_resource_sum = (
                bandwidth_weight * normalized_choice_bandwidth
                + fct_weight * normalized_choice_fct
            )
            # After the primary evasion margin is constrained to its epsilon-near-
            # best set, minimize weighted Chebyshev overhead. The resource sum
            # breaks Chebyshev plateaus so a resource-dominated choice cannot win
            # merely through target confidence.
            order = np.lexsort(
                (
                    choices,
                    -attacks[choices, local_row],
                    -target_log_probabilities[choices, local_row],
                    weighted_resource_sum,
                    overhead_costs[choices, local_row],
                )
            )
            selected = int(choices[order[0]])
            target_confidence = float(
                np.clip(
                    (
                        target_log_probabilities[selected, local_row]
                        - log_probability_floor
                    )
                    / -log_probability_floor,
                    0.0,
                    1.0,
                )
            )
            selected_cost = float(overhead_costs[selected, local_row])
            normalized_selected_cost = (
                float(np.clip(selected_cost / maximum_cost_weight, 0.0, 1.0))
                if maximum_cost_weight > 0.0
                else 0.0
            )
            best_scores[global_row, target] = (
                attacks[selected, local_row]
                - attack_tolerance * normalized_selected_cost
                + 1e-9 * target_confidence
            )
            best_templates[global_row, target] = candidate_templates[
                selected, local_row
            ]
            best_padding_fractions[global_row, target] = (
                candidate_padding_fractions[selected]
            )
            best_fct_fractions[global_row, target] = candidate_fct_fractions[
                selected
            ]
            best_bandwidth[global_row, target] = candidate_bandwidth[
                selected, local_row
            ]
            best_fct[global_row, target] = candidate_fct[selected, local_row]

    assignment_rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    target_labels = library.plan_guided_targets(
        testing.labels,
        best_scores,
        assignment_rng,
    )
    source_rows = np.arange(source_count, dtype=np.int64)
    selected_templates = best_templates[source_rows, target_labels]
    selected_padding_fractions = best_padding_fractions[source_rows, target_labels]
    selected_fct_fractions = best_fct_fractions[source_rows, target_labels]
    selected_scores = best_scores[source_rows, target_labels]
    selected_bandwidth = best_bandwidth[source_rows, target_labels]
    selected_fct = best_fct[source_rows, target_labels]
    if np.any(selected_templates < 0) or not np.all(np.isfinite(selected_scores)):
        raise RuntimeError("Guided assignment selected an unavailable candidate")

    return GuidedSelection(
        target_labels=target_labels,
        template_rows=selected_templates,
        padding_fractions=selected_padding_fractions,
        fct_fractions=selected_fct_fractions,
        bandwidth_overheads=selected_bandwidth,
        fct_overheads=selected_fct,
    )


def _save_guided_selection(selection: GuidedSelection, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}-",
        suffix=".npz",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(
            temporary,
            target_labels=selection.target_labels,
            template_rows=selection.template_rows,
            padding_fractions=selection.padding_fractions,
            fct_fractions=selection.fct_fractions,
            bandwidth_overheads=selection.bandwidth_overheads,
            fct_overheads=selection.fct_overheads,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_guided_selection(
    path: Path,
    testing: SequenceDataset,
    training: SequenceDataset,
    config: dict,
) -> GuidedSelection:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    required = {
        "target_labels",
        "template_rows",
        "padding_fractions",
        "fct_fractions",
        "bandwidth_overheads",
        "fct_overheads",
    }
    if set(arrays) != required:
        raise ValueError(
            f"Guided selection artifact has invalid fields: {sorted(arrays)}"
        )
    source_count = arrays["target_labels"].size
    if any(value.ndim != 1 or value.size != source_count for value in arrays.values()):
        raise ValueError("Guided selection arrays must contain one value per source")
    if source_count != len(testing):
        raise ValueError("Guided selection artifact does not match the test partition")
    for name in ("target_labels", "template_rows"):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"Guided selection field {name} must contain integers")
    for name in ("padding_fractions", "fct_fractions"):
        values = arrays[name]
        if (
            not np.issubdtype(values.dtype, np.floating)
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError(
                f"Guided selection field {name} must contain fractions in [0, 1]"
            )
        configured = np.asarray(config["morph"][name], dtype=np.float64)
        if not np.all(np.any(values[:, None] == configured[None, :], axis=1)):
            raise ValueError(
                f"Guided selection field {name} contains an unconfigured mode"
            )
    class_count = len(testing.class_names)
    targets = arrays["target_labels"].astype(np.int64)
    templates = arrays["template_rows"].astype(np.int64)
    if np.any(targets < 0) or np.any(targets >= class_count):
        raise ValueError("Guided selection contains an invalid target class")
    if np.any(targets == testing.labels):
        raise ValueError("Guided selection contains a self-target assignment")
    for source_value in np.unique(testing.labels):
        source = int(source_value)
        source_rows = testing.labels == source_value
        selected_counts = np.bincount(
            targets[source_rows], minlength=class_count
        )
        legal_counts = np.delete(selected_counts, source)
        if selected_counts[source] != 0 or (
            legal_counts.size and int(legal_counts.max() - legal_counts.min()) > 1
        ):
            raise ValueError(
                "Guided selection violates the exact per-source target quotas "
                f"for source class {source}"
            )
    if np.any(templates < 0) or np.any(templates >= len(training)):
        raise ValueError("Guided selection contains an invalid training-template row")
    if np.any(training.labels[templates] != targets):
        raise ValueError("Guided selection template classes do not match their targets")
    overhead_limits = {
        "bandwidth_overheads": config["morph"]["maximum_padding_ratio"] - 1.0,
        "fct_overheads": config["morph"]["maximum_fct_ratio"] - 1.0,
    }
    for name, limit in overhead_limits.items():
        values = arrays[name]
        if not np.issubdtype(values.dtype, np.floating) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError(f"Guided selection field {name} must be finite floats")
        if np.any(values < -1e-12) or np.any(values > limit + 1e-9):
            raise ValueError(
                f"Guided selection field {name} violates its configured hard cap"
            )
    return GuidedSelection(**arrays)


def _training_worker(
    training: SequenceDataset,
    validation: SequenceDataset,
    scaler: FeatureScaler,
    config: dict,
    output_dir: Path,
    seed: int,
    role: str,
    selector_index: int,
    sender,
) -> None:
    _configure_worker_logging()
    try:
        is_selector = role == "selector"
        if not is_selector and role != "evaluator":
            raise ValueError(f"Unknown training role: {role}")
        if is_selector:
            selector_count = config["model"]["selector_count"]
            if selector_index < 0 or selector_index >= selector_count:
                raise ValueError("Selector index is outside the configured ensemble")
            training_seed = seed + config["model"]["surrogate_seed_offset"] * (
                selector_index + 1
            )
            checkpoint_name = f"selector_{selector_index}_weights.npz"
        else:
            if selector_index != -1:
                raise ValueError("Evaluator training cannot use a selector index")
            training_seed = seed
            checkpoint_name = "evaluator_weights.npz"
        prepare_runtime(training_seed, config["model"]["device"])
        training_features = scaler.transform(training)
        validation_features = scaler.transform(validation)
        model = build_sequence_model(
            sequence_length=training.sizes.shape[1],
            class_count=len(training.class_names),
            config=config["model"],
            class_names=training.class_names,
        )
        train_model(
            model,
            training_features,
            training.labels,
            validation_features,
            validation.labels,
            class_weights(training.labels),
            config["model"],
            output_dir / checkpoint_name,
            training_seed,
        )
        sender.send(("ok", None))
    except Exception as error:
        sender.send(("error", _worker_error(error)))
    finally:
        sender.close()


def _selector_planning_worker(
    training: SequenceDataset,
    testing: SequenceDataset,
    scaler: FeatureScaler,
    config: dict,
    output_dir: Path,
    seed: int,
    sender,
) -> None:
    _configure_worker_logging()
    try:
        selector_seed = seed + config["model"]["surrogate_seed_offset"]
        prepare_runtime(selector_seed, config["model"]["device"])
        selector_models = tuple(
            load_sequence_model(
                output_dir / f"selector_{index}_weights.npz",
                training.sizes.shape[1],
                len(training.class_names),
                config["model"],
                class_names=training.class_names,
            )
            for index in range(config["model"]["selector_count"])
        )
        selection = _plan_guided_selection(
            selector_models,
            testing,
            ProfileLibrary.fit(training),
            scaler,
            config,
            seed + 10_000,
        )
        _save_guided_selection(selection, output_dir / "guided_selection.npz")
        sender.send(("ok", None))
    except Exception as error:
        sender.send(("error", _worker_error(error)))
    finally:
        sender.close()


def _run_isolated_worker(
    target,
    arguments: tuple,
    process_name: str,
    environment: str,
    phase_name: str,
    maximum_attempts: int,
    retry_cleanup=None,
):
    context = multiprocessing.get_context("spawn")
    for attempt in range(1, maximum_attempts + 1):
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=target,
            args=(*arguments, sender),
            name=process_name,
        )
        try:
            process.start()
        except BaseException:
            receiver.close()
            sender.close()
            raise
        sender.close()
        message = _collect_worker_message(process, receiver)
        if process.exitcode == 0 and message is not None:
            status, payload = message
            if status == "ok":
                return payload
            raise RuntimeError(
                f"{environment} {phase_name} failed in its worker process:\n"
                f"{payload}"
            )

        if attempt < maximum_attempts:
            logger.warning(
                f"[{environment}] {phase_name} worker exited with code "
                f"{process.exitcode}; retrying in a fresh process"
            )
            if retry_cleanup is not None:
                retry_cleanup()
            continue
        raise RuntimeError(
            f"{environment} {phase_name} worker exited with code {process.exitcode} "
            f"after {maximum_attempts} attempts"
        )


def _run_selector_isolated(
    training: SequenceDataset,
    validation: SequenceDataset,
    testing: SequenceDataset,
    scaler: FeatureScaler,
    config: dict,
    output_dir: Path,
    seed: int,
    environment: str,
    maximum_attempts: int = 5,
) -> GuidedSelection:
    slug = environment.lower().replace(" ", "-")
    selector_count = config["model"]["selector_count"]
    for selector_index in range(selector_count):
        _run_isolated_worker(
            _training_worker,
            (
                training,
                validation,
                scaler,
                config,
                output_dir,
                seed,
                "selector",
                selector_index,
            ),
            f"packmorph-selector-{selector_index + 1}-training-{slug}",
            environment,
            f"selector {selector_index + 1}/{selector_count} training",
            maximum_attempts,
        )
    _run_isolated_worker(
        _selector_planning_worker,
        (training, testing, scaler, config, output_dir, seed),
        f"packmorph-selector-planning-{slug}",
        environment,
        "selector planning",
        maximum_attempts,
    )
    selection = _load_guided_selection(
        output_dir / "guided_selection.npz",
        testing,
        training,
        config,
    )
    for selector_index in range(selector_count):
        (output_dir / f"selector_{selector_index}_weights.npz").unlink(
            missing_ok=True
        )
    (output_dir / "guided_selection.npz").unlink(missing_ok=True)
    return selection


def _run_evaluator_training_isolated(
    training: SequenceDataset,
    validation: SequenceDataset,
    scaler: FeatureScaler,
    config: dict,
    output_dir: Path,
    seed: int,
    environment: str,
    maximum_attempts: int = 5,
) -> None:
    slug = environment.lower().replace(" ", "-")
    _run_isolated_worker(
        _training_worker,
        (training, validation, scaler, config, output_dir, seed, "evaluator", -1),
        f"packmorph-evaluator-training-{slug}",
        environment,
        "evaluator training",
        maximum_attempts,
    )


def _predict_morphed_batches(
    model,
    testing: SequenceDataset,
    library: ProfileLibrary,
    scaler: FeatureScaler,
    config: dict,
    seed: int,
    *,
    planned_targets: np.ndarray | None = None,
    template_rows: np.ndarray | None = None,
    padding_fraction: np.ndarray | None = None,
    fct_fraction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_count = len(testing)
    class_count = len(testing.class_names)
    morphed_probabilities = np.empty((source_count, class_count), dtype=np.float32)
    bandwidth_overheads = np.empty(source_count, dtype=np.float64)
    fct_overheads = np.empty(source_count, dtype=np.float64)
    fct_overhead_valid = np.empty(source_count, dtype=np.bool_)
    if planned_targets is None:
        target_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
        planned_targets = library.plan_targets(
            testing.labels,
            config["morph"]["target_policy"],
            target_rng,
        )
    elif planned_targets.shape != testing.labels.shape:
        raise ValueError("Planned targets must contain one label per test sequence")
    template_rng = (
        np.random.default_rng(np.random.SeedSequence([seed, 1]))
        if template_rows is None
        else None
    )
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
            rng=template_rng,
            target_labels=planned_targets[start:stop],
            template_rows=(
                None if template_rows is None else template_rows[start:stop]
            ),
            padding_fraction=(
                None if padding_fraction is None else padding_fraction[start:stop]
            ),
            fct_fraction=(
                None if fct_fraction is None else fct_fraction[start:stop]
            ),
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
        bandwidth_overheads[start:stop] = result.bandwidth_overheads
        fct_overheads[start:stop] = result.fct_overheads
        fct_overhead_valid[start:stop] = result.fct_overhead_valid
        del features, window_probabilities, result, source_batch

    return (
        morphed_probabilities,
        bandwidth_overheads,
        fct_overheads,
        fct_overhead_valid,
    )


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

    output_name = environment_slug(environment)
    output_dir = Path(config["output"]["directory"]) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler = FeatureScaler.fit(training)
    guided_selection: GuidedSelection | None = None
    if config["morph"]["target_policy"] == "balanced_guided":
        guided_selection = _run_selector_isolated(
            training,
            validation,
            testing,
            scaler,
            config,
            output_dir,
            seed,
            environment,
        )

    evaluator_checkpoint = output_dir / "evaluator_weights.npz"
    _run_evaluator_training_isolated(
        training,
        validation,
        scaler,
        config,
        output_dir,
        seed,
        environment,
    )
    prepare_runtime(seed, config["model"]["device"])
    model = load_sequence_model(
        evaluator_checkpoint,
        training.sizes.shape[1],
        len(training.class_names),
        config["model"],
        class_names=training.class_names,
    )
    evaluator_checkpoint.unlink(missing_ok=True)
    test_features = scaler.transform(testing)
    baseline_probabilities = predict_probabilities(
        model,
        test_features,
        config["model"]["batch_size"],
    )
    baseline_prediction = np.argmax(baseline_probabilities, axis=1).astype(np.int16)
    baseline_accuracy = float(np.mean(testing.labels == baseline_prediction))
    del baseline_probabilities, test_features

    library = ProfileLibrary.fit(training)

    (
        morphed_probabilities,
        bandwidth_overheads,
        fct_overheads,
        fct_overhead_valid,
    ) = _predict_morphed_batches(
        model,
        testing,
        library,
        scaler,
        config,
        seed + 10_000,
        planned_targets=(
            None if guided_selection is None else guided_selection.target_labels
        ),
        template_rows=(
            None if guided_selection is None else guided_selection.template_rows
        ),
        padding_fraction=(
            None
            if guided_selection is None
            else guided_selection.padding_fractions
        ),
        fct_fraction=(
            None if guided_selection is None else guided_selection.fct_fractions
        ),
    )
    if guided_selection is not None:
        if not np.allclose(
            bandwidth_overheads,
            guided_selection.bandwidth_overheads,
            rtol=0.0,
            atol=1e-12,
        ) or not np.allclose(
            fct_overheads,
            guided_selection.fct_overheads,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("Guided replay did not reproduce selected overheads")
    morphed_prediction = np.argmax(morphed_probabilities, axis=1).astype(np.int16)
    morphed_accuracy = float(np.mean(testing.labels == morphed_prediction))
    valid_fct = fct_overheads[fct_overhead_valid]
    report = {
        "baseline_accuracy": baseline_accuracy,
        "morphed_accuracy": morphed_accuracy,
        "bandwidth_overhead": float(np.mean(bandwidth_overheads)),
        "fct_overhead": float(np.mean(valid_fct)) if valid_fct.size else 0.0,
    }
    shutil.rmtree(output_dir)
    logger.info(
        "[%s] baseline=%.2f%% morphed=%.2f%% bandwidth=%.2f%% FCT=%.2f%%",
        environment,
        report["baseline_accuracy"] * 100.0,
        report["morphed_accuracy"] * 100.0,
        report["bandwidth_overhead"] * 100.0,
        report["fct_overhead"] * 100.0,
    )
    return report


def _environment_worker(config: dict, environment: str, sender) -> None:
    """Run one TensorFlow environment in a disposable native process."""
    _configure_worker_logging()
    try:
        expected_fingerprint = extraction_fingerprint(config["data"])
        dataset = SequenceDataset.load(
            Path(config["data"]["cache"]),
            expected_fingerprint=expected_fingerprint,
        )
        sender.send(("ok", run_environment(dataset, environment, config)))
    except Exception as error:
        sender.send(("error", _worker_error(error)))
    finally:
        sender.close()


def _run_environment_isolated(
    config: dict,
    environment: str,
    maximum_attempts: int = 3,
) -> dict:
    output_name = environment_slug(environment)
    return _run_isolated_worker(
        _environment_worker,
        (config, environment),
        f"packmorph-{environment.lower().replace(' ', '-')}",
        environment,
        "experiment",
        maximum_attempts,
        retry_cleanup=lambda: shutil.rmtree(
            Path(config["output"]["directory"]) / output_name,
            ignore_errors=True,
        ),
    )


def _publish_output_directory(staging: Path, destination: Path) -> None:
    """Replace a completed output tree while preserving the prior tree on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup_holder = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-previous-", dir=destination.parent)
    )
    previous = backup_holder / "output"
    had_previous = destination.exists()
    try:
        if had_previous:
            os.replace(destination, previous)
        try:
            os.replace(staging, destination)
        except BaseException:
            if had_previous and previous.exists() and not destination.exists():
                os.replace(previous, destination)
            raise
    finally:
        shutil.rmtree(backup_holder, ignore_errors=True)


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
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}-run-", dir=output_root.parent)
    )
    try:
        staged_config = copy.deepcopy(config)
        staged_config["output"]["directory"] = str(staging)
        reports = {
            environment: _run_environment_isolated(staged_config, environment)
            for environment in config["data"]["environments"]
        }
        result_fields = {
            "baseline_accuracy",
            "morphed_accuracy",
            "bandwidth_overhead",
            "fct_overhead",
        }
        if any(set(report) != result_fields for report in reports.values()):
            raise RuntimeError("Experiment produced an invalid result field set")
        if any(staging.iterdir()):
            raise RuntimeError("Experiment left unexpected intermediate output files")
        with (staging / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(reports, handle, indent=2, allow_nan=False)
            handle.write("\n")
        if {path.name for path in staging.iterdir()} != {"summary.json"}:
            raise RuntimeError("Experiment output must contain only summary.json")
        _publish_output_directory(staging, output_root)
        return reports
    finally:
        shutil.rmtree(staging, ignore_errors=True)
