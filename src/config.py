"""Settings for balanced-PKL training, flow transformations and comparisons."""

from __future__ import annotations

import math
import os
from pathlib import Path

import yaml

from .transform import METHODS, validate_parameters

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
DATA_KEYS = {"root", "environments", "categories"}
MODEL_KEYS = {"architecture", "device", "dropout", "learning_rate",
              "batch_size", "episodes", "queries_per_episode", "support_per_class",
              "calibration_support_per_class", "split_seed", "features"}


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


def _resolve_path(value: str, config_path: Path, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a path")
    location = Path(value).expanduser()
    return str((config_path.parent / location).resolve() if not location.is_absolute()
               else location.resolve())


def policy_parameters(config: dict, environment: str | None, method: str) -> dict:
    """Resolve optional environment-specific settings into a flat policy schema."""
    settings = config['obfuscation'][method]
    return ({key: value for key, value in settings.items() if key != 'environments'}
            | settings.get('environments', {}).get(environment, {}))


def validate_config(config: dict, path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not isinstance(config, dict) or set(config) != {"output_dir", "data", "model", "obfuscation"}:
        raise ValueError("Config requires output_dir, data, model, and obfuscation sections")
    data, model = config["data"], config["model"]
    if not isinstance(data, dict) or set(data) != DATA_KEYS:
        raise ValueError("Invalid data settings")
    if not isinstance(model, dict) or set(model) != MODEL_KEYS:
        raise ValueError("Invalid CICFlowMeter model settings")
    _names(data["environments"], "data.environments")
    categories = _names(data["categories"], "data.categories")
    feature_sets = model['features']
    if not isinstance(feature_sets, dict) or set(feature_sets) != set(data['environments']):
        raise ValueError('model.features must define a list for every configured environment')
    for environment, names in feature_sets.items():
        if (not isinstance(names, list) or len(names) != 16
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError(f'model.features.{environment} must contain exactly 16 distinct behavior features')

    if len(categories) < 2 or len(categories) > 32767 or len(data["environments"]) > 32767:
        raise ValueError("Invalid service or environment count")
    if model["architecture"] != "cic_prototypical":
        raise ValueError("model.architecture must be cic_prototypical")
    if model["device"] not in {"cpu", "gpu", "auto"}:
        raise ValueError("model.device must be cpu, gpu, or auto")
    for key in ("batch_size", "episodes", "queries_per_episode",
                "support_per_class", "calibration_support_per_class"):
        _positive(model[key], f"model.{key}")
    _positive(model["split_seed"], "model.split_seed", zero=True)
    if _number(model["dropout"], "model.dropout") != 0.25:
        raise ValueError("The classifier dropout must be 0.25")
    if _number(model["learning_rate"], "model.learning_rate") <= 0:
        raise ValueError("learning_rate must be positive")
    data["root"] = _resolve_path(data["root"], path, "data.root")
    config["output_dir"] = _resolve_path(config["output_dir"], path, "output_dir")
    obfuscation = config["obfuscation"]
    if (not isinstance(obfuscation, dict) or set(obfuscation) != set(METHODS)
            or any(not isinstance(parameters, dict) or not parameters
                   for parameters in obfuscation.values())):
        raise ValueError(f'obfuscation must define settings for {", ".join(METHODS)}')
    for method, parameters in obfuscation.items():
        overrides = parameters.get('environments', {})
        if (not isinstance(overrides, dict) or not set(overrides) <= set(data['environments'])
                or any(not isinstance(value, dict) for value in overrides.values())):
            raise ValueError(f'{method}.environments must map configured environments to parameter overrides')
        validate_parameters(method, policy_parameters(config, None, method), str(path))
        for environment in overrides:
            validate_parameters(method, policy_parameters(config, environment, method), str(path))
    return config


def load_config(source: str | Path = DEFAULT_CONFIG) -> dict:
    """Read classifier settings and transformation parameters."""
    path = Path(source).expanduser().resolve()
    try:
        settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {path}: {error}") from error
    if isinstance(settings, dict) and isinstance(settings.get("data"), dict):
        override = os.environ.get("PACKMORPH_FLOW_CSV_ROOT")
        if override:
            settings["data"]["root"] = str(Path(override).expanduser().resolve())
    if isinstance(settings, dict) and os.environ.get("PACKMORPH_OUTPUT_DIR"):
        settings["output_dir"] = str(Path(os.environ["PACKMORPH_OUTPUT_DIR"]).expanduser().resolve())
    return validate_config(settings, path)
