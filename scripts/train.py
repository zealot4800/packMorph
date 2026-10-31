"""Train or load the classifier using only balanced train/val/test PKLs."""

from __future__ import annotations

import argparse
import json

from packmorph.baseline import ensure_baseline
from packmorph.config import DEFAULT_CONFIG, load_config
from packmorph.features import load_pickle_splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", nargs="?", choices=("all", "vpn", "non-vpn"),
                        default="all")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        environments = (config["data"]["environments"] if args.environment == "all"
                        else ("VPN",) if args.environment == "vpn" else ("Non VPN",))
        for environment in environments:
            splits = load_pickle_splits(config, environment)
            artifact = ensure_baseline(config, environment, splits)
            manifest = json.loads((artifact / 'manifest.json').read_text(encoding='utf-8'))
            print(f"{environment}: {artifact}")
            print(f"  {len(manifest['feature_names'])} configured features: {', '.join(manifest['feature_names'])}")
            for name in ('train', 'validation', 'test'):
                metrics = manifest[name + '_metrics']
                print(f"  {name}: {metrics['flows']} rows; accuracy={metrics['accuracy']:.4f}, macro F1={metrics['macro_f1']:.4f}")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Training stopped: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
