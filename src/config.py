"""Settings for balanced-PKL training, flow transformations and comparisons."""

from __future__ import annotations

import math
import os
from pathlib import Path

import yaml

from .transform import METHODS, validate_parameters

CONFIG_DIR = Path(__file__).resolve().parent.parent / 'configs'
DATASET_CONFIGS = {'vnat': CONFIG_DIR / 'vnat.yaml', 'cic-iscx': CONFIG_DIR / 'iscx.yaml'}
DATA_KEYS = {"root", "environments", "categories"}
MODEL_KEYS = {"architecture", "device", "dropout", "learning_rate",
              "batch_size", "episodes", "queries_per_episode", "support_per_class",
              "calibration_support_per_class", "split_seed", "features"}
TRAINING_DEFAULTS = {
    'validation_interval': 250, 'early_stopping_patience': 8, 'min_delta': 0.001,
    'lr_patience': 3, 'lr_factor': 0.5, 'min_learning_rate': 0.00001,
    'weight_decay': 0.0001, 'gradient_clipnorm': 1.0,
}


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
    if (not isinstance(data, dict) or not DATA_KEYS <= set(data)
            or set(data) - DATA_KEYS - {'schema'}):
        raise ValueError("Invalid data settings")
    if data.get('schema', 'cic-pkl-v1') != 'cic-pkl-v1':
        raise ValueError('Only prepared packet-feature PKLs (cic-pkl-v1) are supported')
    if (not isinstance(model, dict) or not MODEL_KEYS <= set(model)
            or set(model) - MODEL_KEYS - set(TRAINING_DEFAULTS)):
        raise ValueError("Invalid CICFlowMeter model settings")
    for key, value in TRAINING_DEFAULTS.items():
        model.setdefault(key, value)
    _names(data["environments"], "data.environments")
    categories = _names(data["categories"], "data.categories")
    feature_sets = model['features']
    if not isinstance(feature_sets, dict) or set(feature_sets) != set(data['environments']):
        raise ValueError('model.features must define a list for every configured environment')
    for environment, names in feature_sets.items():
        if (not isinstance(names, list) or not 1 <= len(names) <= 35
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError(f'model.features.{environment} must contain 1 to 35 distinct behavior features')

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
    if not 0 <= _number(model["dropout"], "model.dropout") < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if _number(model["learning_rate"], "model.learning_rate") <= 0:
        raise ValueError("learning_rate must be positive")
    for key in ('validation_interval', 'early_stopping_patience', 'lr_patience'):
        _positive(model[key], f'model.{key}')
    for key in ('min_delta', 'weight_decay'):
        if _number(model[key], f'model.{key}') < 0:
            raise ValueError(f'model.{key} must be nonnegative')
    if not 0 <= model['min_delta'] < 1:
        raise ValueError('model.min_delta must be in [0, 1)')
    if not 0 < _number(model['lr_factor'], 'model.lr_factor') < 1:
        raise ValueError('model.lr_factor must be between zero and one')
    if not 0 < _number(model['min_learning_rate'], 'model.min_learning_rate') <= model['learning_rate']:
        raise ValueError('min_learning_rate must be positive and no greater than learning_rate')
    if _number(model['gradient_clipnorm'], 'model.gradient_clipnorm') <= 0:
        raise ValueError('gradient_clipnorm must be positive')
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


def load_config(source: str | Path | None = None, *, dataset: str | None = None) -> dict:
    """Read a standalone config; default to VNAT when no dataset is selected."""
    if dataset is not None and dataset not in DATASET_CONFIGS:
        raise ValueError(f'Unknown dataset: {dataset!r}')
    path = Path(source if source is not None else DATASET_CONFIGS[dataset or 'vnat']).expanduser().resolve()
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


def add_dataset_arguments(parser) -> None:
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--config', type=Path, help='standalone YAML configuration')
    selection.add_argument('--dataset', choices=tuple(DATASET_CONFIGS),
                           help='vnat selects configs/vnat.yaml; cic-iscx selects configs/iscx.yaml (default: vnat)')


def dataset_config_path(args) -> Path:
    if args.config is not None:
        return Path(args.config)
    return DATASET_CONFIGS[args.dataset or 'vnat']
