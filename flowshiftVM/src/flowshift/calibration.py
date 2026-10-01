"""Derive VM policy parameters from capture-level CICFlowMeter CSV summaries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .policy import FlowConfig
from .protocol import HEADER

QUANTILES = (0.10, 0.30, 0.70, 0.90)
STRENGTHS = (0.25, 0.50, 0.75, 1.00)
EPOCH_MS = 250
MODELED_HEADER_BYTES = HEADER.size


def quantile(ordered: list[float], q: float) -> float:
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def profile(csv_root: Path, environment: str, category: str,
            calibration_seed: int) -> dict:
    directory = csv_root / environment / category
    captures = []
    for path in sorted(directory.glob("*.csv")):
        sizes = []
        rates = []
        iats = []
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                try:
                    count = int(row["tot_fwd_pkts"]) + int(row["tot_bwd_pkts"])
                    duration = float(row["flow_duration"])
                    size = float(row["pkt_len_mean"])
                    rate = float(row["flow_byts_s"])
                    iat = float(row["flow_iat_mean"])
                except (KeyError, TypeError, ValueError):
                    continue
                if (count >= 32 and duration >= 0.1 and 64 <= size <= 1514
                        and math.isfinite(rate) and 0 < rate < 1_000_000_000
                        and math.isfinite(iat) and iat >= 0):
                    sizes.append(size)
                    rates.append(rate)
                    iats.append(iat)
        if sizes:
            captures.append({"path": path, "sizes": sorted(sizes),
                             "median_rate": statistics.median(rates),
                             "median_iat_ms": 1000 * statistics.median(iats)})
    if len(captures) < 3:
        raise ValueError(f"{directory}: need at least three eligible capture CSVs")
    captures.sort(key=lambda item: hashlib.sha256(
        f"{calibration_seed}:{item['path'].relative_to(csv_root)}".encode()).digest())
    kept = max(3, math.floor(0.8 * len(captures)))
    captures = captures[:kept]
    return {
        "category": category,
        "capture_count": len(captures),
        "flow_count": sum(len(item["sizes"]) for item in captures),
        "sizes": [round(statistics.median([quantile(item["sizes"], q) for item in captures]))
                  for q in QUANTILES],
        "rate_bytes_per_second": round(statistics.median([item["median_rate"] for item in captures])),
        "iat_ms": round(statistics.median([item["median_iat_ms"] for item in captures]), 6),
        "sources": [str(item["path"].relative_to(csv_root)) for item in captures],
    }


def size_variants(source: dict, target: dict) -> list[list[int]]:
    return [[max(64, round(a + strength * (b - a)))
             for a, b in zip(source["sizes"], target["sizes"])]
            for strength in STRENGTHS]


def make_policies(source: dict, target: dict, rate: float, chunk_size: int) -> list[dict]:
    policies = []
    for strength, sizes in zip(STRENGTHS, size_variants(source, target)):
        # The VM fragments application read chunks into fixed-size payloads.
        # Estimate those fragments directly, then leave 2% slot headroom.
        mean_fragments = sum(math.ceil(chunk_size / size) for size in sizes) / len(sizes)
        objects_per_epoch = rate * EPOCH_MS / 1000 / chunk_size
        frames = math.ceil(objects_per_epoch * mean_fragments * 1.02)
        frames = max(1, frames)
        delay = (0.0 if frames > 500 else
                 round(min(source["iat_ms"], EPOCH_MS * 0.5 / frames), 3))
        policies.append({
            "name": f"toward-{target['category'].lower().replace(' ', '-')}-{round(strength * 100)}",
            "target_sizes": sizes,
            "inter_packet_ms": [delay],
            "target_frames_per_epoch": frames,
        })
    return policies


def estimated_wire_ratio(source: dict, target: dict, chunk_size: int) -> float:
    """Conservative VM framing estimate across the four size variants."""
    return max(sum(math.ceil(chunk_size / size) * (size + MODELED_HEADER_BYTES)
                   for size in sizes) / (len(sizes) * chunk_size)
               for sizes in size_variants(source, target))


def derive(document: dict, source: dict, target: dict, environment: str,
           link_bps: int, digest: str, calibration_seed: int) -> dict:
    output = {key: value for key, value in document.items()
              if key not in {"pseudorandom_generator", "delta_time_ms", "policies", "derivation"}}
    is_stream = document["workload_type"] == "streaming"
    workload_rate = (round((document["video_bitrate_kbps"] + document["audio_bitrate_kbps"]) * 1000 / 8)
                     if is_stream else 0)
    chunk = int(document["read_chunk_bytes"])
    ratio = estimated_wire_ratio(source, target, chunk) * 1.02
    rate = (workload_rate if is_stream else
            min(source["rate_bytes_per_second"], link_bps / (8 * ratio)))
    rate = max(1, rate)
    seed_material = f"{digest}:{environment}:{document['workload_type']}:{document['profile']}"
    output.update({
        "pseudorandom_generator": {
            "algorithm": "hmac-sha256",
            "seed": hashlib.sha256(seed_material.encode()).hexdigest(),
        },
        "delta_time_ms": {"min": EPOCH_MS, "max": EPOCH_MS},
        "policies": make_policies(source, target, rate, chunk),
        "derivation": {
            "method": "capture-balanced-CICFlowMeter-v1",
            "environment": environment,
            "source_class": source["category"],
            "target_class": target["category"],
            "source_capture_count": source["capture_count"],
            "target_capture_count": target["capture_count"],
            "source_flow_count": source["flow_count"],
            "target_flow_count": target["flow_count"],
            "source_size_quantiles_bytes": source["sizes"],
            "target_size_quantiles_bytes": target["sizes"],
            "source_median_capture_rate_bytes_per_second": source["rate_bytes_per_second"],
            "source_median_capture_iat_ms": source["iat_ms"],
            "target_median_capture_iat_ms": target["iat_ms"],
            "assumed_application_rate_bytes_per_second": round(rate),
            "rate_basis": "configured media bitrate" if is_stream else "capture median capped by link",
            "link_capacity_bps": link_bps,
            "epoch_ms": EPOCH_MS,
            "modeled_frame_header_bytes": MODELED_HEADER_BYTES,
            "strengths": list(STRENGTHS),
            "size_quantiles": list(QUANTILES),
            "calibration_seed": calibration_seed,
            "estimated_max_wire_to_application_byte_ratio": round(ratio, 6),
            "frame_slot_headroom_fraction": 0.02,
            "csv_sha256": digest,
        },
    })
    FlowConfig.from_dict(output)
    return output


def generate(csv_root: Path, environment: str, link_bps: int, calibration_seed: int,
             config_dir: Path, output_dir: Path) -> list[Path]:
    """Validate and write all workload configurations from one CSV calibration."""
    source = profile(csv_root, environment, "File Transfer", calibration_seed)
    target = profile(csv_root, environment, "Streaming", calibration_seed)
    paths = sorted({csv_root / name for name in source["sources"] + target["sources"]})
    fingerprint = hashlib.sha256()
    for path in paths:
        fingerprint.update(str(path.relative_to(csv_root)).encode())
        fingerprint.update(hashlib.sha256(path.read_bytes()).digest())
    digest = fingerprint.hexdigest()
    generated = {}
    for path in sorted(config_dir.glob("*.json")):
        document = json.loads(path.read_text())
        if document.get("workload_type") not in {"file-transfer", "streaming"}:
            continue
        src, dst = (source, target) if document["workload_type"] == "file-transfer" else (target, source)
        generated[path.name] = derive(document, src, dst, environment,
                                      link_bps, digest, calibration_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, document in generated.items():
        path = output_dir / name
        path.write_text(json.dumps(document, indent=2) + "\n")
        written.append(path)
    return written
