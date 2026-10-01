"""Transform and send files or live video bytes."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from .planner import TransformationPlanner
from .policy import FlowConfig, PolicyScheduler
from .protocol import Frame, FrameType, ProtocolError, recv_frame, send_frame


@dataclass(slots=True)
class SendStats:
    original_bytes: int
    wire_bytes: int
    data_frames: int
    dummy_frames: int
    epochs: int
    duration_seconds: float
    sha256: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["byte_overhead_percent"] = (
            0.0
            if not self.original_bytes
            else (self.wire_bytes - self.original_bytes) / self.original_bytes * 100
        )
        return result


def _json_frame(kind: FrameType, value: dict[str, object], sequence: int = 0) -> Frame:
    payload = json.dumps(value, separators=(",", ":")).encode()
    return Frame(kind, sequence=sequence, valid_length=len(payload), payload=payload)


def _send_transformed(
    sock: socket.socket,
    filename: str,
    expected_size: int,
    config: FlowConfig,
    read: Callable[[], bytes | None],
    *,
    video: bool = False,
    finish: Callable[[], None] = lambda: None,
) -> SendStats:
    scheduler = PolicyScheduler(config)
    planner = TransformationPlanner(config)
    hello: dict[str, object] = {
        "filename": filename,
        "original_size": expected_size,
        "config": config.to_dict(),
    }
    if video:
        hello["mode"] = "video"

    wire_bytes = send_frame(sock, _json_frame(FrameType.HELLO, hello))
    pending: deque[Frame] = deque()
    digest = hashlib.sha256()
    sequence = 1
    object_id = data_frames = dummy_frames = original_bytes = epoch = 0
    eof = False
    started = time.monotonic()

    while not eof or pending:
        plan = scheduler.plan(epoch)
        epoch_started = time.monotonic()
        frame_count = plan.policy.target_frames_per_epoch

        for slot in range(frame_count):
            delay = plan.policy.inter_packet_ms[
                slot % len(plan.policy.inter_packet_ms)
            ]
            if delay:
                time.sleep(delay / 1000)

            while len(pending) < frame_count and not eof:
                chunk = read()
                if chunk is None:
                    break
                if not chunk:
                    eof = True
                    break
                digest.update(chunk)
                original_bytes += len(chunk)
                pending.extend(planner.transform(chunk, object_id, plan))
                object_id += 1

            if pending:
                frame = replace(pending.popleft(), sequence=sequence)
                data_frames += 1
            else:
                size = plan.policy.target_sizes[slot % len(plan.policy.target_sizes)]
                frame = replace(planner.dummy(plan, size), sequence=sequence)
                dummy_frames += 1
            wire_bytes += send_frame(sock, frame)
            sequence += 1

        if not (eof and not pending):
            remaining = plan.delta_time_ms / 1000 - (time.monotonic() - epoch_started)
            if remaining > 0:
                time.sleep(remaining)
        epoch += 1

    finish()
    checksum = digest.hexdigest()
    end = {
        "original_size": original_bytes,
        "sha256": checksum,
        "data_frames": data_frames,
        "dummy_frames": dummy_frames,
    }
    wire_bytes += send_frame(sock, _json_frame(FrameType.END, end, sequence))
    result = recv_frame(sock)
    if result.frame_type is not FrameType.RESULT:
        raise ProtocolError("receiver did not return a result frame")
    response = json.loads(result.payload[: result.valid_length])
    if not response.get("ok"):
        raise ProtocolError(f"receiver rejected transfer: {response.get('error')}")
    return SendStats(
        original_bytes,
        wire_bytes,
        data_frames,
        dummy_frames,
        epoch,
        time.monotonic() - started,
        checksum,
    )


def _connect(host: str, port: int, bind_address: str | None) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if bind_address:
        sock.bind((bind_address, 0))
    sock.connect((host, port))
    return sock


def send_file(
    host: str,
    port: int,
    path: str | Path,
    config: FlowConfig,
    *,
    bind_address: str | None = None,
) -> SendStats:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    sock = _connect(host, port, bind_address)
    try:
        with source.open("rb") as handle:
            return _send_transformed(
                sock,
                source.name,
                source.stat().st_size,
                config,
                lambda: handle.read(config.read_chunk_bytes),
            )
    finally:
        sock.close()


def send_video(
    host: str,
    port: int,
    path: str | Path,
    config: FlowConfig,
    duration_seconds: int,
    width: int,
    height: int,
    video_bitrate_kbps: int,
    audio_bitrate_kbps: int,
    *,
    bind_address: str | None = None,
) -> SendStats:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if (
        duration_seconds < 1
        or width < 16
        or height < 16
        or video_bitrate_kbps < 1
        or audio_bitrate_kbps < 1
    ):
        raise ValueError("invalid video settings")
    sock = _connect(host, port, bind_address)
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-stream_loop", "-1", "-re", "-i", str(source),
                "-t", str(duration_seconds), "-map", "0:v:0", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "veryfast",
                "-tune", "zerolatency",
                "-vf", (
                    f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
                ),
                "-b:v", f"{video_bitrate_kbps}k",
                "-maxrate", f"{video_bitrate_kbps}k",
                "-bufsize", f"{video_bitrate_kbps * 2}k",
                "-c:a", "aac", "-b:a", f"{audio_bitrate_kbps}k",
                "-f", "mpegts", "pipe:1",
            ],
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        if process.stdout is None:
            raise RuntimeError("FFmpeg output pipe was not created")
        os.set_blocking(process.stdout.fileno(), False)

        def read() -> bytes | None:
            try:
                return os.read(process.stdout.fileno(), config.read_chunk_bytes)
            except BlockingIOError:
                return None

        def finish() -> None:
            status = process.wait()
            if status:
                raise RuntimeError(f"FFmpeg exited with status {status}")

        return _send_transformed(
            sock,
            f"{source.stem}.ts",
            -1,
            config,
            read,
            video=True,
            finish=finish,
        )
    finally:
        sock.close()
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
