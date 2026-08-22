from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import SequenceDataset


@dataclass(slots=True)
class FeatureScaler:
    length_mean: float
    length_std: float
    iat_mean: float
    iat_std: float

    @classmethod
    def fit(cls, dataset: SequenceDataset) -> "FeatureScaler":
        mask = (
            np.arange(dataset.sizes.shape[1])[None, :] < dataset.packet_counts[:, None]
        )
        lengths = np.log1p(np.abs(dataset.sizes[mask]).astype(np.float64))
        iats = np.log1p(dataset.iats[mask].astype(np.float64) * 1000.0)
        return cls(
            length_mean=float(lengths.mean()),
            length_std=max(float(lengths.std()), 1e-6),
            iat_mean=float(iats.mean()),
            iat_std=max(float(iats.std()), 1e-6),
        )

    def transform(self, dataset: SequenceDataset) -> np.ndarray:
        mask = (
            np.arange(dataset.sizes.shape[1])[None, :] < dataset.packet_counts[:, None]
        )
        features = np.zeros((*dataset.sizes.shape, 4), dtype=np.float32)
        logged_lengths = np.log1p(np.abs(dataset.sizes).astype(np.float32))
        logged_iats = np.log1p(dataset.iats.astype(np.float32) * 1000.0)
        features[..., 0][mask] = (
            logged_lengths[mask] - self.length_mean
        ) / self.length_std
        features[..., 1][mask] = (logged_iats[mask] - self.iat_mean) / self.iat_std
        features[..., 2][mask] = np.sign(dataset.sizes[mask]).astype(np.float32)
        features[..., 3] = mask.astype(np.float32)
        return features
