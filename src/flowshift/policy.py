"""Policy loading and deterministic policy scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Policy:
    name: str
    target_sizes: tuple[int, ...]
    inter_packet_ms: tuple[float, ...]
    target_frames_per_epoch: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Policy":
        policy = cls(
            name=str(raw["name"]),
            target_sizes=tuple(int(value) for value in raw["target_sizes"]),
            inter_packet_ms=tuple(float(value) for value in raw["inter_packet_ms"]),
            target_frames_per_epoch=int(raw["target_frames_per_epoch"]),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if not self.name:
            raise ValueError("policy name cannot be empty")
        if not self.target_sizes or min(self.target_sizes) < 64:
            raise ValueError(f"{self.name}: target sizes must be at least 64 bytes")
        if max(self.target_sizes) > 1_000_000:
            raise ValueError(f"{self.name}: target size is unreasonably large")
        if not self.inter_packet_ms or min(self.inter_packet_ms) < 0:
            raise ValueError(f"{self.name}: delays must be non-negative")
        if self.target_frames_per_epoch < 1:
            raise ValueError(f"{self.name}: target_frames_per_epoch must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_sizes"] = list(self.target_sizes)
        result["inter_packet_ms"] = list(self.inter_packet_ms)
        return result


@dataclass(frozen=True, slots=True)
class FlowConfig:
    seed: str
    prg_algorithm: str
    delta_time_min_ms: int
    delta_time_max_ms: int
    read_chunk_bytes: int
    policies: tuple[Policy, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FlowConfig":
        prg = raw["pseudorandom_generator"]
        delta = raw["delta_time_ms"]
        config = cls(
            seed=str(prg["seed"]),
            prg_algorithm=str(prg.get("algorithm", "hmac-sha256")),
            delta_time_min_ms=int(delta["min"]),
            delta_time_max_ms=int(delta["max"]),
            read_chunk_bytes=int(raw.get("read_chunk_bytes", 64 * 1024)),
            policies=tuple(Policy.from_dict(item) for item in raw["policies"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.seed:
            raise ValueError("seed cannot be empty")
        if self.prg_algorithm != "hmac-sha256":
            raise ValueError("only hmac-sha256 is supported")
        if not self.policies:
            raise ValueError("at least one policy is required")
        if not 1 <= self.delta_time_min_ms <= self.delta_time_max_ms:
            raise ValueError("invalid delta-time range")
        if not 1 <= self.read_chunk_bytes <= 4 * 1024 * 1024:
            raise ValueError("read_chunk_bytes is outside the supported range")
        names = [policy.name for policy in self.policies]
        if len(names) != len(set(names)):
            raise ValueError("policy names must be unique")
        for policy in self.policies:
            scheduled_delay = sum(
                policy.inter_packet_ms[
                    index % len(policy.inter_packet_ms)
                ]
                for index in range(policy.target_frames_per_epoch)
            )
            if scheduled_delay > self.delta_time_min_ms:
                raise ValueError(
                    f"{policy.name}: packet-delay schedule ({scheduled_delay} ms) "
                    f"exceeds minimum delta time ({self.delta_time_min_ms} ms)"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pseudorandom_generator": {
                "algorithm": self.prg_algorithm,
                "seed": self.seed,
            },
            "delta_time_ms": {
                "min": self.delta_time_min_ms,
                "max": self.delta_time_max_ms,
            },
            "read_chunk_bytes": self.read_chunk_bytes,
            "policies": [policy.to_dict() for policy in self.policies],
        }


@dataclass(frozen=True, slots=True)
class PRGOutput:
    """The two deterministic values generated for interval k."""

    u: int
    v: float


@dataclass(frozen=True, slots=True)
class EpochPlan:
    index: int
    u: int
    v: float
    policy_index: int
    policy: Policy
    delta_time_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval": self.index,
            "u": self.u,
            "v": self.v,
            "policy_index": self.policy_index,
            "policy": self.policy.name,
            "delta_time_ms": self.delta_time_ms,
        }


class PseudorandomGenerator:
    """Counter-based PRG shared by the sender and receiver."""

    DOMAIN = b"flowshift-policy-schedule-v1"

    def __init__(self, seed: str):
        self.key = seed.encode("utf-8")

    def generate(self, interval: int) -> PRGOutput:
        if interval < 0:
            raise ValueError("interval cannot be negative")
        counter = interval.to_bytes(8, "big")
        digest = hmac.new(self.key, self.DOMAIN + counter, hashlib.sha256).digest()
        u = int.from_bytes(digest[:8], "big")
        v_integer = int.from_bytes(digest[8:16], "big")
        v = v_integer / (1 << 64)
        return PRGOutput(u=u, v=v)


class PolicyScheduler:
    """Reproduces the same policy sequence from a shared seed."""

    def __init__(self, config: FlowConfig):
        self.config = config
        self.prg = PseudorandomGenerator(config.seed)

    def plan(self, epoch: int) -> EpochPlan:
        generated = self.prg.generate(epoch)
        policy_index = generated.u % len(self.config.policies)
        span = self.config.delta_time_max_ms - self.config.delta_time_min_ms
        delta_time = self.config.delta_time_min_ms + int(generated.v * span)
        return EpochPlan(
            index=epoch,
            u=generated.u,
            v=generated.v,
            policy_index=policy_index,
            policy=self.config.policies[policy_index],
            delta_time_ms=delta_time,
        )


def load_document(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("configuration root must be an object")
    return document


def load_config(path: str | Path) -> FlowConfig:
    document = load_document(path)
    return FlowConfig.from_dict(document)
