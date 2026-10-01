from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock

import numpy as np

from .dataset import SequenceDataset, SequenceRecord, from_records


logger = logging.getLogger(__name__)

TSHARK_FIELDS = (
    "frame.time_epoch",
    "frame.len",
    "ip.src",
    "ipv6.src",
    "ip.dst",
    "ipv6.dst",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
)
_EXTRACTOR_VERSION = 7
_PCAP_SUFFIXES = {".pcap", ".pcapng"}
_MAX_FRAME_SIZE = int(np.iinfo(np.int32).max)
Endpoint = tuple[str, int]
FlowKey = tuple[Endpoint, Endpoint, int]
CaptureTask = tuple[str, str, Path]


@dataclass(slots=True)
class _Flow:
    origin: Endpoint
    started_at: float
    last_timestamp: float
    first_timestamp: float | None = None
    sizes: list[int] = field(default_factory=list)
    iats: list[float] = field(default_factory=list)
    chunk: int = 0


class _ExtractionCancelled(RuntimeError):
    pass


class CorruptCaptureError(RuntimeError):
    """A capture is damaged; any packets parsed before the error are unusable."""

    def __init__(self, path: Path, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"tshark found a corrupt capture {path}: {detail}")


class _ExtractionController:
    """Coordinate cancellation and child-process cleanup across capture workers."""

    def __init__(self) -> None:
        self._event = Event()
        self._lock = Lock()
        self._processes: set[subprocess.Popen] = set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def register(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.add(process)
            cancelled = self._event.is_set()
        if cancelled:
            self._terminate(process)

    def unregister(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.discard(process)

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            processes = tuple(self._processes)
        for process in processes:
            self._terminate(process)

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass


def discover_captures(config: dict) -> list[CaptureTask]:
    root = Path(config["root"])
    captures: list[CaptureTask] = []
    for environment in config["environments"]:
        for category in config["categories"]:
            directory = root / environment / category
            if not directory.is_dir():
                continue
            paths = sorted(
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in _PCAP_SUFFIXES
            )
            captures.extend((environment, category, path) for path in paths)
    return captures


def extraction_fingerprint(
    config: dict, captures: list[CaptureTask] | None = None
) -> str:
    root = Path(config["root"])
    extraction_settings = {
        key: config[key]
        for key in (
            "environments",
            "categories",
            "max_sequence_length",
            "min_packets",
            "flow_idle_timeout_seconds",
            "max_windows_per_capture",
            "tshark_binary",
            "corrupt_capture_policy",
        )
    }
    manifest = []
    capture_list = discover_captures(config) if captures is None else captures
    for _, _, path in capture_list:
        metadata = path.stat()
        manifest.append(
            {
                "path": str(path.relative_to(root)),
                "size": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
            }
        )
    payload = {
        "extractor_version": _EXTRACTOR_VERSION,
        "settings": extraction_settings,
        "pcaps": manifest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _packet_from_line(line: str) -> tuple[float, int, str, str, int, int, int] | None:
    fields = line.rstrip("\n").split("\t")
    if len(fields) < len(TSHARK_FIELDS):
        fields.extend([""] * (len(TSHARK_FIELDS) - len(fields)))
    try:
        timestamp = float(fields[0])
        size = int(fields[1])
        source = fields[2] or fields[3]
        destination = fields[4] or fields[5]
        if fields[6] and fields[7]:
            protocol = 6
            source_port, destination_port = int(fields[6]), int(fields[7])
        elif fields[8] and fields[9]:
            protocol = 17
            source_port, destination_port = int(fields[8]), int(fields[9])
        else:
            return None
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(timestamp)
        or not 0 < size <= _MAX_FRAME_SIZE
        or not source
        or not destination
    ):
        return None
    return timestamp, size, source, destination, protocol, source_port, destination_port


def _flow_key(
    source: str,
    destination: str,
    protocol: int,
    source_port: int,
    destination_port: int,
) -> FlowKey:
    first, second = sorted(((source, source_port), (destination, destination_port)))
    return first, second, protocol


def _record_from_flow(
    flow: _Flow,
    key: FlowKey,
    environment: str,
    category: str,
    group: str,
    min_packets: int,
) -> SequenceRecord | None:
    if len(flow.sizes) < min_packets:
        return None
    flow_group = (
        f"{key[0][0]}:{key[0][1]}-{key[1][0]}:{key[1][1]}-{key[2]}-"
        f"{flow.started_at:.9f}"
    )
    return SequenceRecord(
        sizes=np.asarray(flow.sizes, dtype=np.int32),
        iats=np.asarray(flow.iats, dtype=np.float32),
        environment=environment,
        category=category,
        group=group,
        flow_id=f"{flow_group}-{flow.chunk}",
        flow_group=flow_group,
        start_offset=float((flow.first_timestamp if flow.first_timestamp is not None
                            else flow.started_at) - flow.started_at),
        flow_start_timestamp=float(flow.started_at),
    )


def _wait_for_process(process: subprocess.Popen, terminate: bool) -> None:
    if process.poll() is None and terminate:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        process.wait()


def extract_capture(
    pcap_path: Path,
    dataset_root: Path,
    environment: str,
    category: str,
    config: dict,
    controller: _ExtractionController,
) -> list[SequenceRecord]:
    if controller.cancelled:
        raise _ExtractionCancelled("Capture extraction was cancelled")
    command = [
        config["tshark_binary"],
        "-n",
        "-r",
        str(pcap_path),
        "-Y",
        "tcp or udp",
        "-T",
        "fields",
        "-E",
        "occurrence=f",
    ]
    for field_name in TSHARK_FIELDS:
        command.extend(("-e", field_name))

    records: list[SequenceRecord] = []
    active: OrderedDict[FlowKey, _Flow] = OrderedDict()
    width = int(config["max_sequence_length"])
    minimum = int(config["min_packets"])
    timeout = float(config["flow_idle_timeout_seconds"])
    maximum_windows = config["max_windows_per_capture"]
    group = str(pcap_path.relative_to(dataset_root))

    def emit(flow: _Flow, key: FlowKey) -> None:
        record = _record_from_flow(flow, key, environment, category, group, minimum)
        if record is not None:
            records.append(record)

    stopped_early = False
    aborted = False
    timestamp_watermark: float | None = None
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as error_log:
        # System tshark must use system libraries; FPGA toolchain paths in the
        # parent shell can supply an incompatible libstdc++.
        tshark_environment = os.environ.copy()
        tshark_environment.pop("LD_LIBRARY_PATH", None)
        process = subprocess.Popen(
            command,
            env=tshark_environment,
            stdout=subprocess.PIPE,
            stderr=error_log,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        controller.register(process)
        assert process.stdout is not None
        try:
            for line in process.stdout:
                if controller.cancelled:
                    raise _ExtractionCancelled("Capture extraction was cancelled")
                packet = _packet_from_line(line)
                if packet is None:
                    continue
                (
                    timestamp,
                    size,
                    source,
                    destination,
                    protocol,
                    source_port,
                    destination_port,
                ) = packet
                # PCAP file order can contain small timestamp regressions. Clamp those
                # packets to the latest observed time so flow expiry remains monotonic
                # and their inter-arrival time is represented as zero.
                if timestamp_watermark is not None:
                    timestamp = max(timestamp, timestamp_watermark)
                timestamp_watermark = timestamp

                while active:
                    oldest_key = next(iter(active))
                    oldest_flow = active[oldest_key]
                    if timestamp - oldest_flow.last_timestamp <= timeout:
                        break
                    emit(active.pop(oldest_key), oldest_key)
                    if maximum_windows is not None and len(records) >= maximum_windows:
                        stopped_early = True
                        break
                if stopped_early:
                    break

                key = _flow_key(
                    source, destination, protocol, source_port, destination_port
                )
                endpoint = (source, source_port)
                flow = active.pop(key, None)
                if flow is None:
                    flow = _Flow(
                        origin=endpoint, started_at=timestamp, last_timestamp=timestamp
                    )
                if flow.first_timestamp is None:
                    flow.first_timestamp = timestamp
                iat = (
                    0.0 if not flow.sizes else max(timestamp - flow.last_timestamp, 0.0)
                )
                direction = 1 if endpoint == flow.origin else -1
                flow.sizes.append(direction * size)
                flow.iats.append(iat)
                flow.last_timestamp = timestamp

                if len(flow.sizes) == width:
                    emit(flow, key)
                    flow = _Flow(
                        origin=flow.origin,
                        started_at=flow.started_at,
                        last_timestamp=timestamp,
                        chunk=flow.chunk + 1,
                    )
                active[key] = flow
                if maximum_windows is not None and len(records) >= maximum_windows:
                    stopped_early = True
                    break
            if controller.cancelled:
                raise _ExtractionCancelled("Capture extraction was cancelled")
        except BaseException:
            aborted = True
            raise
        finally:
            process.stdout.close()
            _wait_for_process(process, terminate=stopped_early or aborted)
            controller.unregister(process)

        error_log.seek(0)
        stderr = error_log.read().strip()
        if not stopped_early and process.returncode != 0:
            detail = stderr[-2_000:] or "no error output"
            error_text = stderr.lower()
            if (
                "appears to be damaged or corrupt" in error_text
                or "appears to have been cut short" in error_text
            ):
                raise CorruptCaptureError(pcap_path, detail)
            raise RuntimeError(f"tshark failed for {pcap_path}: {detail}")

    if not stopped_early:
        for key, flow in active.items():
            emit(flow, key)
            if maximum_windows is not None and len(records) >= maximum_windows:
                break
    return records[:maximum_windows] if maximum_windows is not None else records


def extract_dataset(config: dict) -> SequenceDataset:
    data_config = config["data"]
    root = Path(data_config["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    tshark = data_config["tshark_binary"]
    if shutil.which(tshark) is None:
        raise RuntimeError(f"'{tshark}' is required to extract PCAP metadata")

    tasks = discover_captures(data_config)
    if not tasks:
        raise ValueError(f"No PCAP or PCAPNG files were found below {root}")
    expected_pairs = [
        (environment, category)
        for environment in data_config["environments"]
        for category in data_config["categories"]
    ]
    capture_counts = {pair: 0 for pair in expected_pairs}
    for environment, category, _ in tasks:
        capture_counts[(environment, category)] += 1
    insufficient = [
        (environment, category, capture_counts[(environment, category)])
        for environment, category in expected_pairs
        if capture_counts[(environment, category)] < 3
    ]
    if insufficient:
        details = ", ".join(
            f"{environment}/{category}={count}"
            for environment, category, count in insufficient
        )
        raise ValueError(
            "Every configured environment/class needs at least three capture groups "
            f"for leakage-free train/validation/test partitions; found {details}"
        )

    starting_fingerprint = extraction_fingerprint(data_config, captures=tasks)
    logger.info(
        f"Extracting {len(tasks)} captures with {data_config['extraction_workers']} workers"
    )

    controller = _ExtractionController()

    def run_task(task: CaptureTask) -> tuple[CaptureTask, list[SequenceRecord], str | None]:
        environment, category, path = task
        try:
            records = extract_capture(
                path, root, environment, category, data_config, controller=controller,
            )
        except CorruptCaptureError as error:
            if data_config["corrupt_capture_policy"] == "error":
                raise
            return task, [], error.detail
        return task, records, None

    executor = ThreadPoolExecutor(max_workers=int(data_config["extraction_workers"]))
    futures: list[Future[tuple[CaptureTask, list[SequenceRecord], str | None]]] = []
    try:
        futures = [executor.submit(run_task, task) for task in tasks]
        completed, _ = wait(futures, return_when=FIRST_EXCEPTION)
        failed_indices = [
            index
            for index, future in enumerate(futures)
            if future in completed
            and not future.cancelled()
            and future.exception() is not None
        ]
        if failed_indices:
            futures[min(failed_indices)].result()
        results = [future.result() for future in futures]
    except BaseException:
        controller.cancel()
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    records: list[SequenceRecord] = []
    excluded_captures: list[dict[str, str]] = []
    eligible_capture_counts = {pair: 0 for pair in expected_pairs}
    for task, capture_records, corruption_detail in results:
        if corruption_detail is not None:
            excluded_captures.append({
                "capture": str(task[2].relative_to(root)),
                "reason": corruption_detail,
            })
            continue
        records.extend(capture_records)
        if capture_records:
            eligible_capture_counts[(task[0], task[1])] += 1
        logger.debug(
            "%s/%s/%s: %d sequences",
            task[0],
            task[1],
            task[2].name,
            len(capture_records),
        )
    if excluded_captures:
        logger.warning(
            "Excluded %d corrupt capture(s): %s",
            len(excluded_captures),
            ", ".join(item["capture"] for item in excluded_captures),
        )

    insufficient_eligible = [
        (environment, category, eligible_capture_counts[(environment, category)])
        for environment, category in expected_pairs
        if eligible_capture_counts[(environment, category)] < 3
    ]
    if insufficient_eligible:
        details = ", ".join(
            f"{environment}/{category}={count}"
            for environment, category, count in insufficient_eligible
        )
        raise ValueError(
            "At least three captures per environment/class must produce eligible "
            f"sequences; found {details}. Excluded corrupt captures: "
            f"{', '.join(item['capture'] for item in excluded_captures) or 'none'}"
        )

    ending_fingerprint = extraction_fingerprint(data_config)
    if ending_fingerprint != starting_fingerprint:
        raise RuntimeError(
            "The PCAP manifest changed during extraction; retry extraction."
        )

    dataset = from_records(
        records,
        list(data_config["categories"]),
        list(data_config["environments"]),
        int(data_config["max_sequence_length"]),
        starting_fingerprint,
        excluded_captures=tuple(excluded_captures),
    )
    return dataset
