from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .config import load_config
from .dataset import SequenceDataset
from .experiment import run_experiment
from .extract import extract_dataset, extraction_fingerprint


logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="packmorph-sim",
        description=(
            "Simulation-only PackMorph evaluation with a dilated temporal CNN "
            "traffic classifier"
        ),
    )
    parser.add_argument("command", choices=("extract", "run", "all"))
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Rebuild the sequence cache when running the 'all' command",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress and error context",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.force_extract and args.command != "all":
        parser.error("--force-extract is only valid with the 'all' command")
    level = logging.DEBUG if args.verbose else logging.INFO
    os.environ["PACKMORPH_LOG_LEVEL"] = logging.getLevelName(level)
    logging.basicConfig(level=level, format="%(message)s", force=True)

    try:
        config = load_config(args.config)
        if args.command == "extract":
            extract_dataset(config)
            return 0

        if args.command == "all":
            cache = Path(config["data"]["cache"])
            fingerprint = extraction_fingerprint(config["data"])
            cache_is_current = (
                not args.force_extract
                and SequenceDataset.load_if_current(cache, fingerprint) is not None
            )
            if not cache_is_current:
                if not args.force_extract:
                    logger.info(
                        "Sequence cache is missing, stale, or malformed; rebuilding it"
                    )
                extract_dataset(config, fingerprint=fingerprint)
        run_experiment(config)
        return 0
    except KeyboardInterrupt:
        logger.error("packmorph-sim: interrupted")
        return 130
    except Exception as error:
        if args.verbose:
            logger.exception("packmorph-sim failed")
        else:
            logger.error("packmorph-sim: error: %s", error)
        return 1
