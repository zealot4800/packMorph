"""FlowShift training profiles, policy-bank creation and seeded policy choice.

Profiles use unique training flows and equal weight per capture.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .features import FlowDataset
from .transform import FEATURE_NAMES, validate_policy_parameters

if TYPE_CHECKING:
    from .baseline import BehaviorScaler


@dataclass(slots=True)
class Profiles:
    """Capture-balanced targets fitted exclusively from unique training flows."""

    centers: np.ndarray
    class_names: tuple[str, ...]
    scaler: BehaviorScaler
    columns: tuple[str, ...]

    @classmethod
    def fit(cls, training: FlowDataset, rows: Iterable[Mapping]) -> "Profiles":
        from .baseline import BehaviorScaler

        _, first = np.unique(training.row_indices, return_index=True)
        selected = set(first)
        measured = {i: [float(row[n]) for n in FEATURE_NAMES]
                    for i, row in enumerate(rows) if i in selected}
        if len(measured) != len(first):
            raise ValueError("FlowShift context is missing training rows")
        matrix = np.asarray([measured[i] for i in first], dtype=np.float64)
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError('FlowShift training context must be finite and nonnegative')
        labels, captures = training.labels[first], training.captures[first]
        centers = []
        for label, name in enumerate(training.class_names):
            members = labels == label
            groups = np.unique(captures[members])
            if not len(groups):
                raise ValueError(f'No training flows for FlowShift class {name}')
            centers.append(np.median([np.median(matrix[members & (captures == group)], axis=0)
                                      for group in groups], axis=0))
        return cls(np.asarray(centers), training.class_names,
                   BehaviorScaler.fit(training.features[first]), training.feature_names)


@dataclass(frozen=True, slots=True)
class MorphingSpec:
    """Training-derived aggregate goals; not an executable VM wire policy."""

    name: str
    mode: str
    strength: float
    padding: tuple[float, float]
    dummy_sizes: tuple[int, int]
    volume_ratios: tuple[float, float]
    clock_ratio: float


@dataclass(slots=True)
class Candidate:
    policy: MorphingSpec
    row: dict[str, float]
    displacement: float
    byte_fraction: float
    fct_fraction: float


def pareto_front(candidates: list[Candidate]) -> list[Candidate]:
    """Maximize displacement and minimize both costs; retain objective ties."""
    if not candidates:
        return []
    objectives = np.array([[-c.displacement, c.byte_fraction, c.fct_fraction]
                           for c in candidates])
    dominates = (np.all(objectives[:, None] <= objectives[None, :], axis=2)
                 & np.any(objectives[:, None] < objectives[None, :], axis=2))
    return [candidate for candidate, dominated in zip(candidates, dominates.any(axis=0)) if not dominated]


class FlowShiftPolicy:
    """Build source-to-target banks and choose feasible policies deterministically."""

    name = 'flowshift'
    required = {'strengths': 'strengths', 'min_displacement': 'nonnegative',
                'max_fct_fraction': 'nonnegative', 'holding_time_ms': 'bounds'}

    @classmethod
    def validate(cls, parameters: dict, source: str) -> dict:
        return validate_policy_parameters(cls.name, parameters, cls.required, {}, source,
                                          byte_fraction_kind='nonnegative')

    def __init__(self, profiles: Profiles, source_class: str, settings: dict, seed: str, *,
                 capture: str | None = None):
        self.validate(settings, 'flowshift')
        self.profiles = profiles
        self.source = profiles.class_names.index(source_class)
        self.settings = settings
        self.seed = seed
        self.capture = capture
        self.flows = 0
        self.bank = self._build_bank()

    def _build_bank(self) -> list[MorphingSpec]:
        """Keep independent axes instead of rejecting an entire joint target."""
        source = dict(zip(FEATURE_NAMES, self.profiles.centers[self.source]))
        bank = []
        for label, center in enumerate(self.profiles.centers):
            if label == self.source:
                continue
            target = dict(zip(FEATURE_NAMES, center))
            padding = tuple(max(0., target[f'{d}_pkt_len_mean'] - source[f'{d}_pkt_len_mean'])
                            for d in ('fwd', 'bwd'))
            sizes = tuple(min(1500, max(64, round(target[f'{d}_pkt_len_mean'])))
                          for d in ('fwd', 'bwd'))
            volume = tuple(target[f'tot_{d}_pkts'] / max(1., source[f'tot_{d}_pkts'])
                           for d in ('fwd', 'bwd'))
            clock = target['flow_iat_mean'] / max(1e-12, source['flow_iat_mean'])
            for strength in self.settings['strengths']:
                for mode in ('size', 'timing', 'volume-fwd', 'volume-bwd', 'volume-both', 'balanced'):
                    bank.append(MorphingSpec(f'toward-{label}-{mode}-{strength:g}', mode,
                                             strength, padding, sizes, volume, clock))
        return bank

    @property
    def fingerprint(self) -> str:
        """Bind cached transformations to their training-derived profiles."""
        digest = hashlib.sha256()
        for value in (self.profiles.centers, self.profiles.scaler.mean, self.profiles.scaler.scale):
            digest.update(np.asarray(value, dtype='<f8').tobytes())
        digest.update(json.dumps([self.profiles.class_names, self.profiles.columns,
                                  self.settings, self.seed], sort_keys=True).encode())
        return digest.hexdigest()

    def _random(self, flow_key: str):
        from flowshift.policy import PseudorandomGenerator

        prefix = f'{self.seed}:{self.capture}' if self.capture else self.seed
        seed = hashlib.sha256(f'{prefix}:{flow_key}'.encode()).hexdigest()
        return PseudorandomGenerator(seed).generate(0)

    def holding_limit_s(self, flow_key: str) -> float:
        low, high = self.settings['holding_time_ms']
        return (low + int(self._random(flow_key).v * (high - low))) / 1000

    def choose(self, candidates: list[Candidate], flow_key: str) -> Candidate | None:
        """Apply the shared HMAC PRG to a deterministic non-dominated list."""
        selected = pareto_front(candidates)
        return selected[self._random(flow_key).u % len(selected)] if selected else None
