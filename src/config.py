from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


_INT32_MAX = 2_147_483_647

_SCHEMA = {
    "seed": None,
    "data": {
        "root",
        "cache",
        "environments",
        "categories",
        "max_sequence_length",
        "min_packets",
        "flow_idle_timeout_seconds",
        "max_windows_per_capture",
        "extraction_workers",
        "tshark_binary",
    },
    "split": {"test_fraction"},
    "model": {
        "device",
        "conv_filters",
        "kernel_sizes",
        "context_filters",
        "dilations",
        "dense_units",
        "dropout",
        "learning_rate",
        "batch_size",
        "epochs",
        "patience",
        "validation_fraction",
        "selector_count",
        "surrogate_seed_offset",
    },
    "morph": {
        "target_policy",
        "candidates_per_target",
        "padding_fractions",
        "fct_fractions",
        "attack_utility_tolerance",
        "bandwidth_cost_weight",
        "fct_cost_weight",
        "maximum_padding_ratio",
        "maximum_fct_ratio",
        "minimum_packet_size",
        "maximum_packet_size",
    },
    "output": {"directory"},
}


def _positive_integer(value: Any, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _fraction_levels(values: Any, name: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    fractions = [
        _finite_number(value, f"{name}[{index}]")
        for index, value in enumerate(values)
    ]
    if any(value < 0.0 or value > 1.0 for value in fractions):
        raise ValueError(f"{name} values must be in [0, 1]")
    if any(left >= right for left, right in zip(fractions, fractions[1:])):
        raise ValueError(f"{name} must be sorted with unique values")
    return fractions


def _unique_strings(values: Any, name: str) -> list[str]:
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(item, str) and item for item in values)
    ):
        raise ValueError(f"{name} must be a non-empty list of strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _path_components(values: Any, name: str) -> list[str]:
    names = _unique_strings(values, name)
    if any(
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).name != value
        for value in names
    ):
        raise ValueError(f"{name} values must be single directory names")
    if len({value.casefold() for value in names}) != len(names):
        raise ValueError(f"{name} must not contain case-insensitive duplicates")
    return names


def environment_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _validate_keys(config: dict[str, Any]) -> None:
    expected_top = set(_SCHEMA)
    actual_top = set(config)
    if actual_top != expected_top:
        missing = sorted(expected_top - actual_top)
        unknown = sorted(actual_top - expected_top)
        raise ValueError(
            f"Invalid top-level configuration keys; missing={missing}, unknown={unknown}"
        )
    for section, expected in _SCHEMA.items():
        if expected is None:
            continue
        value = config[section]
        if not isinstance(value, dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping")
        actual = set(value)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise ValueError(
                f"Invalid '{section}' keys; missing={missing}, unknown={unknown}"
            )


def validate_config(config: dict[str, Any]) -> None:
    _validate_keys(config)
    _positive_integer(config["seed"], "seed", allow_zero=True)

    data = config["data"]
    environments = _path_components(data["environments"], "data.environments")
    categories = _path_components(data["categories"], "data.categories")
    environment_slugs = [environment_slug(name) for name in environments]
    if any(not slug.strip("_") for slug in environment_slugs) or len(
        set(environment_slugs)
    ) != len(environment_slugs):
        raise ValueError("data.environments produce invalid or duplicate output names")
    if len(categories) < 2:
        raise ValueError("data.categories must contain at least two service classes")
    if len(categories) > 32_767 or len(environments) > 32_767:
        raise ValueError(
            "data category/environment counts exceed the supported int16 identifiers"
        )
    for key in ("root", "cache"):
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError(f"data.{key} must be a non-empty path string")
    width = _positive_integer(data["max_sequence_length"], "data.max_sequence_length")
    minimum_packets = _positive_integer(data["min_packets"], "data.min_packets")
    if width < minimum_packets:
        raise ValueError("data.max_sequence_length must be >= data.min_packets")
    if width > _INT32_MAX:
        raise ValueError(
            "data.max_sequence_length exceeds the supported int32 packet count"
        )
    timeout = _finite_number(
        data["flow_idle_timeout_seconds"], "data.flow_idle_timeout_seconds"
    )
    if timeout <= 0:
        raise ValueError("data.flow_idle_timeout_seconds must be positive")
    window_limit = data["max_windows_per_capture"]
    if window_limit is not None:
        _positive_integer(window_limit, "data.max_windows_per_capture")
    _positive_integer(data["extraction_workers"], "data.extraction_workers")
    if not isinstance(data["tshark_binary"], str) or not data["tshark_binary"]:
        raise ValueError("data.tshark_binary must be a non-empty string")
    if (
        not isinstance(config["output"]["directory"], str)
        or not config["output"]["directory"]
    ):
        raise ValueError("output.directory must be a non-empty path string")

    test_fraction = _finite_number(
        config["split"]["test_fraction"], "split.test_fraction"
    )
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("split.test_fraction must be between zero and one")

    model = config["model"]
    if model["device"] not in {"auto", "cpu", "gpu"}:
        raise ValueError("model.device must be auto, cpu, or gpu")
    filters = model["conv_filters"]
    kernels = model["kernel_sizes"]
    if not isinstance(filters, list) or not filters:
        raise ValueError("model.conv_filters must be a non-empty list")
    if not isinstance(kernels, list) or len(filters) != len(kernels):
        raise ValueError("model.kernel_sizes must match model.conv_filters")
    for index, value in enumerate(filters):
        _positive_integer(value, f"model.conv_filters[{index}]")
    for index, value in enumerate(kernels):
        kernel = _positive_integer(value, f"model.kernel_sizes[{index}]")
        if kernel > width:
            raise ValueError(
                "model kernel sizes cannot exceed data.max_sequence_length"
            )
    _positive_integer(model["context_filters"], "model.context_filters")
    dilations = model["dilations"]
    if not isinstance(dilations, list) or not dilations:
        raise ValueError("model.dilations must be a non-empty list")
    for index, value in enumerate(dilations):
        dilation = _positive_integer(value, f"model.dilations[{index}]")
        if dilation > width:
            raise ValueError(
                "model dilations cannot exceed data.max_sequence_length"
            )
    if len(dilations) != len(set(dilations)):
        raise ValueError("model.dilations must not contain duplicates")
    _positive_integer(model["dense_units"], "model.dense_units")
    _positive_integer(model["batch_size"], "model.batch_size")
    _positive_integer(model["epochs"], "model.epochs")
    _positive_integer(model["patience"], "model.patience", allow_zero=True)
    selector_count = _positive_integer(
        model["selector_count"], "model.selector_count"
    )
    if selector_count > 3:
        raise ValueError("model.selector_count cannot exceed 3")
    _positive_integer(model["surrogate_seed_offset"], "model.surrogate_seed_offset")
    dropout = _finite_number(model["dropout"], "model.dropout")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    learning_rate = _finite_number(model["learning_rate"], "model.learning_rate")
    if learning_rate <= 0:
        raise ValueError("model.learning_rate must be positive")
    validation_fraction = _finite_number(
        model["validation_fraction"], "model.validation_fraction"
    )
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("model.validation_fraction must be between zero and one")

    morph = config["morph"]
    if morph["target_policy"] not in {
        "balanced_guided",
        "balanced",
        "random",
        "nearest",
    }:
        raise ValueError(
            "morph.target_policy must be balanced_guided, balanced, random, or nearest"
        )
    candidate_count = _positive_integer(
        morph["candidates_per_target"], "morph.candidates_per_target"
    )
    if candidate_count > 32:
        raise ValueError("morph.candidates_per_target cannot exceed 32")
    padding_fractions = _fraction_levels(
        morph["padding_fractions"], "morph.padding_fractions"
    )
    fct_fractions = _fraction_levels(
        morph["fct_fractions"], "morph.fct_fractions"
    )
    variant_count = candidate_count * len(padding_fractions) * len(fct_fractions)
    if variant_count * selector_count > 256:
        raise ValueError(
            "morph candidate/resource variants multiplied by model.selector_count "
            "cannot exceed 256 evaluations per source-target pair"
        )
    tolerance = _finite_number(
        morph["attack_utility_tolerance"], "morph.attack_utility_tolerance"
    )
    if tolerance < 0.0:
        raise ValueError("morph.attack_utility_tolerance must be non-negative")
    resource_weights = []
    for key in ("bandwidth_cost_weight", "fct_cost_weight"):
        weight = _finite_number(morph[key], f"morph.{key}")
        if weight < 0.0:
            raise ValueError(f"morph.{key} must be non-negative")
        resource_weights.append(weight)
    if not any(weight > 0.0 for weight in resource_weights):
        raise ValueError(
            "At least one of morph.bandwidth_cost_weight and "
            "morph.fct_cost_weight must be positive"
        )
    for key in ("maximum_padding_ratio", "maximum_fct_ratio"):
        if _finite_number(morph[key], f"morph.{key}") < 1.0:
            raise ValueError(f"morph.{key} cannot be below 1.0")
    minimum_size = _positive_integer(
        morph["minimum_packet_size"], "morph.minimum_packet_size"
    )
    maximum_size = _positive_integer(
        morph["maximum_packet_size"], "morph.maximum_packet_size"
    )
    if maximum_size > _INT32_MAX:
        raise ValueError(
            "morph.maximum_packet_size exceeds the supported int32 packet size"
        )
    if minimum_size > maximum_size:
        raise ValueError(
            "morph.minimum_packet_size cannot exceed morph.maximum_packet_size"
        )


def _resolve_paths(config: dict[str, Any], config_path: Path) -> None:
    base = config_path.parent
    for section, key in (("data", "root"), ("data", "cache"), ("output", "directory")):
        path = Path(config[section][key]).expanduser()
        if not path.is_absolute():
            path = base / path
        config[section][key] = str(path.resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a mapping")
    validate_config(config)
    _resolve_paths(config, config_path)
    return config
