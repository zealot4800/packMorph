"""Validate the shared capture, split, and CNN settings."""

from __future__ import annotations

import math
from pathlib import Path

DATA_KEYS = {"root", "environments", "categories", "max_sequence_length",
             "min_packets", "flow_idle_timeout_seconds", "max_windows_per_capture",
             "extraction_workers", "tshark_binary", "corrupt_capture_policy"}
MODEL_KEYS = {"device", "conv_filters", "kernel_sizes", "context_filters", "dilations",
              "dense_units", "dropout", "learning_rate", "batch_size", "epochs",
              "patience", "validation_fraction"}


def _positive(value, name: str, *, zero: bool = False) -> int:
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'} integer")
    return value


def _number(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _names(value, name: str) -> list[str]:
    if (not isinstance(value, list) or not value
            or any(not isinstance(item, str) or not item or Path(item).name != item
                   or item in {".", ".."} or "/" in item or "\\" in item for item in value)
            or len(set(item.casefold() for item in value)) != len(value)):
        raise ValueError(f"{name} must contain distinct directory names")
    return value


def validate_base_config(config: dict, path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not isinstance(config, dict) or set(config) != {"data", "split", "model"}:
        raise ValueError("Study config requires data, split, and model sections")
    data, split, model = config["data"], config["split"], config["model"]
    if not isinstance(data, dict) or set(data) != DATA_KEYS:
        raise ValueError("Invalid data settings")
    if not isinstance(split, dict) or set(split) != {"test_fraction"}:
        raise ValueError("Invalid split settings")
    if not isinstance(model, dict) or set(model) != MODEL_KEYS:
        raise ValueError("Invalid CNN settings")
    _names(data["environments"], "data.environments")
    categories = _names(data["categories"], "data.categories")
    if len(categories) < 2 or len(categories) > 32767 or len(data["environments"]) > 32767:
        raise ValueError("Invalid service or environment count")
    width = _positive(data["max_sequence_length"], "data.max_sequence_length")
    if width < _positive(data["min_packets"], "data.min_packets"):
        raise ValueError("Sequence width is smaller than min_packets")
    if _number(data["flow_idle_timeout_seconds"], "data.flow_idle_timeout_seconds") <= 0:
        raise ValueError("Flow idle timeout must be positive")
    if data["max_windows_per_capture"] is not None:
        _positive(data["max_windows_per_capture"], "data.max_windows_per_capture")
    _positive(data["extraction_workers"], "data.extraction_workers")
    if not isinstance(data["tshark_binary"], str) or not data["tshark_binary"]:
        raise ValueError("tshark_binary must be a command name")
    if (
        not isinstance(data["corrupt_capture_policy"], str)
        or data["corrupt_capture_policy"] not in {"exclude", "error"}
    ):
        raise ValueError("data.corrupt_capture_policy must be exclude or error")
    if not 0 < _number(split["test_fraction"], "split.test_fraction") < 1:
        raise ValueError("test_fraction must be between zero and one")
    if model["device"] not in {"cpu", "gpu", "auto"}:
        raise ValueError("model.device must be cpu, gpu, or auto")
    filters, kernels = model["conv_filters"], model["kernel_sizes"]
    if not isinstance(filters, list) or not filters or not isinstance(kernels, list) or len(filters) != len(kernels):
        raise ValueError("CNN filters and kernels must have matching nonempty lists")
    for index, (channels, kernel) in enumerate(zip(filters, kernels)):
        _positive(channels, f"conv_filters[{index}]")
        if _positive(kernel, f"kernel_sizes[{index}]") > width:
            raise ValueError("CNN kernel exceeds sequence width")
    dilations = model["dilations"]
    if not isinstance(dilations, list) or not dilations or len(set(dilations)) != len(dilations):
        raise ValueError("CNN dilations must be a nonempty unique list")
    for index, value in enumerate(dilations):
        if _positive(value, f"dilations[{index}]") > width:
            raise ValueError("CNN dilation exceeds sequence width")
    for key in ("context_filters", "dense_units", "batch_size", "epochs"):
        _positive(model[key], f"model.{key}")
    _positive(model["patience"], "model.patience", zero=True)
    if not 0 <= _number(model["dropout"], "model.dropout") < 1:
        raise ValueError("dropout must be in [0, 1)")
    if _number(model["learning_rate"], "model.learning_rate") <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 < _number(model["validation_fraction"], "model.validation_fraction") < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if not isinstance(data["root"], str) or not data["root"]:
        raise ValueError("data.root must be a path")
    root = Path(data["root"]).expanduser()
    data["root"] = str((path.parent / root).resolve() if not root.is_absolute() else root.resolve())
    return config
