"""Command-line interface for the FlowShift prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .policy import PolicyScheduler, load_config, load_document
from .receiver import receive_once
from .sender import send_file, send_video


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
FILE_CONFIG = CONFIG_DIR / "file-transfer-500mb.json"
STREAM_CONFIG = CONFIG_DIR / "streaming-60s-720p.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flowshift")
    subparsers = parser.add_subparsers(dest="command", required=True)

    schedule = subparsers.add_parser(
        "schedule", help="show PRG policy selections and delta times"
    )
    schedule.add_argument("--config", default=str(FILE_CONFIG))
    schedule.add_argument("--count", type=int, default=10)

    receive = subparsers.add_parser("receive", help="receive and restore one file")
    receive.add_argument("--bind", default="0.0.0.0")
    receive.add_argument("--port", type=int, default=9443)
    receive.add_argument("--output-dir", default="received")
    receive.add_argument("--play", action="store_true")

    send = subparsers.add_parser("send", help="transform and send a file")
    send.add_argument("file")
    send.add_argument("--host", required=True)
    send.add_argument("--port", type=int, default=9443)
    send.add_argument("--config", default=str(FILE_CONFIG))
    send.add_argument("--bind")

    stream = subparsers.add_parser(
        "stream-send", help="encode and transform an actual video in real time"
    )
    stream.add_argument("video")
    stream.add_argument("--host", required=True)
    stream.add_argument("--port", type=int, default=9443)
    stream.add_argument("--config", default=str(STREAM_CONFIG))
    stream.add_argument("--bind")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "schedule":
        if args.count < 1:
            raise ValueError("schedule count must be positive")
        config = load_config(args.config)
        scheduler = PolicyScheduler(config)
        print(
            json.dumps(
                [scheduler.plan(index).to_dict() for index in range(args.count)],
                indent=2,
            )
        )
        return
    if args.command == "receive":
        stats = receive_once(
            args.bind,
            args.port,
            args.output_dir,
            play=args.play,
        )
    elif args.command == "send":
        experiment = load_document(args.config)
        if experiment.get("workload_type") != "file-transfer":
            raise ValueError("configuration is not for file transfer")
        source = Path(args.file)
        expected_size = int(experiment["size_bytes"])
        if not source.is_file() or source.stat().st_size != expected_size:
            raise ValueError(
                f"configuration requires a {expected_size}-byte input file"
            )
        config = load_config(args.config)
        stats = send_file(
            args.host,
            args.port,
            args.file,
            config,
            bind_address=args.bind,
        )
    else:
        experiment = load_document(args.config)
        if experiment.get("workload_type") != "streaming":
            raise ValueError("configuration is not for streaming")
        config = load_config(args.config)
        stats = send_video(
            args.host,
            args.port,
            args.video,
            config,
            int(experiment["duration_seconds"]),
            int(experiment["width"]),
            int(experiment["height"]),
            int(experiment["video_bitrate_kbps"]),
            int(experiment["audio_bitrate_kbps"]),
            bind_address=args.bind,
        )
    print(json.dumps(stats.to_dict(), indent=2))


if __name__ == "__main__":
    main()
