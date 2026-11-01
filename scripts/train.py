"""Train or load the classifier using dataset-specific train/val/test PKLs."""

from __future__ import annotations

import argparse
import json

from packmorph.baseline import ensure_baseline
from packmorph.config import add_dataset_arguments, dataset_config_path, load_config
from packmorph.features import load_pickle_splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", nargs="?", choices=("all", "vpn", "non-vpn"),
                        default="all")
    add_dataset_arguments(parser)
    args = parser.parse_args()
    args.config = dataset_config_path(args)
    try:
        config = load_config(args.config, dataset=args.dataset)
        environments = (config["data"]["environments"] if args.environment == "all"
                        else ("VPN",) if args.environment == "vpn" else ("Non VPN",))
        for environment in environments:
            splits = load_pickle_splits(config, environment)
            if not splits['train'].capture_disjoint_verified:
                print(f'{environment}: flow-level splits; capture separation is unverified.', flush=True)
            artifact = ensure_baseline(config, environment, splits)
            manifest = json.loads((artifact / 'manifest.json').read_text(encoding='utf-8'))
            print(f"{environment}: {artifact}")
            print(f"  {len(manifest['feature_names'])} configured features: {', '.join(manifest['feature_names'])}")
            checkpoint = manifest['selected_checkpoint']
            print(f"  Selected episode {checkpoint['episode']} after {checkpoint['episodes_run']} "
                  f"episodes ({checkpoint['stop_reason']}); history: {artifact / 'training_history.json'}")
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
