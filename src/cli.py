from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .dataset import SequenceDataset
from .experiment import run_experiment
from .extract import extract_dataset, extraction_fingerprint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="packmorph-sim",
        description="Simulation-only PackMorph evaluation with a CNN-BiGRU traffic classifier",
    )
    parser.add_argument("command", choices=("extract", "run", "all"))
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Rebuild the sequence cache when running the 'all' command",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.force_extract and args.command != "all":
        parser.error("--force-extract is only valid with the 'all' command")
    config = load_config(args.config)
    if args.command == "extract":
        extract_dataset(config)
        return

    if args.command == "all":
        cache = Path(config["data"]["cache"])
        fingerprint = extraction_fingerprint(config["data"])
        cache_is_current = (
            not args.force_extract
            and SequenceDataset.load_if_current(cache, fingerprint) is not None
        )
        if not cache_is_current:
            if not args.force_extract:
                print("Sequence cache is missing, stale, or malformed; rebuilding it")
            extract_dataset(config, fingerprint=fingerprint)
    run_experiment(config)
