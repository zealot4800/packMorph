"""FlowShift receiver-side configuration and restoration manager."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

from .policy import FlowConfig, PolicyScheduler
from .protocol import Frame, FrameType, ProtocolError, recv_frame, send_frame


@dataclass(slots=True)
class ReceiveStats:
    output_path: str
    recovered_bytes: int
    data_frames: int
    dummy_frames: int
    duration_seconds: float
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _safe_filename(value: object) -> str:
    name = Path(str(value)).name
    if name in {"", ".", ".."}:
        raise ProtocolError("invalid output filename")
    return name


def _result(ok: bool, *, error: str | None = None) -> Frame:
    value: dict[str, object] = {"ok": ok}
    if error:
        value["error"] = error
    payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return Frame(
        frame_type=FrameType.RESULT,
        valid_length=len(payload),
        payload=payload,
    )


def receive_connection(
    sock: socket.socket,
    output_dir: str | Path,
    *,
    play: bool = False,
) -> ReceiveStats:
    hello = recv_frame(sock)
    if hello.frame_type is not FrameType.HELLO:
        raise ProtocolError("first frame must be HELLO")
    metadata = json.loads(hello.payload[: hello.valid_length])
    config = FlowConfig.from_dict(metadata["config"])
    scheduler = PolicyScheduler(config)
    filename = _safe_filename(metadata["filename"])
    expected_size = int(metadata["original_size"])
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    temporary = directory / f".{filename}.flowshift.part"

    expected_sequence = 1
    expected_object = 0
    active_fragments: list[bytes] = []
    active_fragment_count = 0
    digest = hashlib.sha256()
    recovered_bytes = 0
    data_frames = 0
    dummy_frames = 0
    started = time.monotonic()
    player: subprocess.Popen[bytes] | None = None
    player_pipe_open = False

    if play:
        if metadata.get("mode") != "video":
            raise ProtocolError("live playback requires a video stream")
        player = subprocess.Popen(
            [
                "ffplay",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-framedrop",
                "-autoexit",
                "-i",
                "pipe:0",
            ],
            stdin=subprocess.PIPE,
        )
        player_pipe_open = player.stdin is not None

    try:
        with temporary.open("wb") as output:
            while True:
                frame = recv_frame(sock)
                if frame.frame_type is FrameType.END:
                    if frame.sequence != expected_sequence:
                        raise ProtocolError("END frame sequence mismatch")
                    end = json.loads(frame.payload[: frame.valid_length])
                    if active_fragments:
                        raise ProtocolError("transfer ended during fragmented object")
                    if (
                        (expected_size >= 0 and recovered_bytes != expected_size)
                        or recovered_bytes != int(end["original_size"])
                    ):
                        raise ProtocolError("recovered file size mismatch")
                    if digest.hexdigest() != str(end["sha256"]):
                        raise ProtocolError("recovered file hash mismatch")
                    break
                if frame.frame_type not in {FrameType.DATA, FrameType.DUMMY}:
                    raise ProtocolError("unexpected frame in transfer")
                if frame.sequence != expected_sequence:
                    raise ProtocolError("non-contiguous frame sequence")
                expected_sequence += 1
                plan = scheduler.plan(frame.epoch)
                if frame.policy_index != plan.policy_index:
                    raise ProtocolError("sender and receiver policy schedules differ")
                if frame.frame_type is FrameType.DUMMY:
                    if frame.valid_length != 0:
                        raise ProtocolError("dummy frame contains valid application data")
                    dummy_frames += 1
                    continue

                data_frames += 1
                if frame.object_id != expected_object:
                    raise ProtocolError("application objects arrived out of order")
                if frame.fragment_count < 1:
                    raise ProtocolError("data object has no fragments")
                if not active_fragments:
                    active_fragment_count = frame.fragment_count
                if frame.fragment_count != active_fragment_count:
                    raise ProtocolError("fragment count changed within an object")
                if frame.fragment_index != len(active_fragments):
                    raise ProtocolError("fragment arrived out of order")
                active_fragments.append(frame.payload[: frame.valid_length])
                if len(active_fragments) == active_fragment_count:
                    restored = b"".join(active_fragments)
                    output.write(restored)
                    if player is not None and player_pipe_open and player.stdin:
                        try:
                            player.stdin.write(restored)
                            player.stdin.flush()
                        except BrokenPipeError:
                            player_pipe_open = False
                    digest.update(restored)
                    recovered_bytes += len(restored)
                    expected_object += 1
                    active_fragments.clear()
                    active_fragment_count = 0
            output.flush()
        if player is not None and player.stdin and player_pipe_open:
            player.stdin.close()
            player_pipe_open = False
        if player is not None:
            try:
                player.wait(timeout=10)
            except subprocess.TimeoutExpired:
                player.terminate()
                player.wait()
        temporary.replace(destination)
        stats = ReceiveStats(
            output_path=str(destination),
            recovered_bytes=recovered_bytes,
            data_frames=data_frames,
            dummy_frames=dummy_frames,
            duration_seconds=time.monotonic() - started,
            sha256=digest.hexdigest(),
        )
        send_frame(sock, _result(True))
        return stats
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if player is not None and player.poll() is None:
            player.terminate()
        try:
            send_frame(sock, _result(False, error=str(exc)))
        except OSError:
            pass
        raise


def receive_once(
    bind_address: str,
    port: int,
    output_dir: str | Path,
    *,
    play: bool = False,
) -> ReceiveStats:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((bind_address, port))
    listener.listen(1)
    try:
        connection, _peer = listener.accept()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            return receive_connection(connection, output_dir, play=play)
        finally:
            connection.close()
    finally:
        listener.close()
