from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FeatureLayout:
    hog: slice
    lbp: slice
    intensity: slice
    total_dimensions: int


class FaceFeatureExtractor:
    def __init__(self, face_size: int = 128) -> None:
        if face_size % 16 != 0:
            raise ValueError("face_size must be divisible by 16.")
        self.face_size = face_size
        self.hog = cv2.HOGDescriptor(
            _winSize=(face_size, face_size),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )
        probe = np.zeros((face_size, face_size), dtype=np.uint8)
        hog_size = int(self.hog.compute(probe).size)
        lbp_size = 4 * 4 * 16
        intensity_size = 32
        self.layout = FeatureLayout(
            hog=slice(0, hog_size),
            lbp=slice(hog_size, hog_size + lbp_size),
            intensity=slice(hog_size + lbp_size, hog_size + lbp_size + intensity_size),
            total_dimensions=hog_size + lbp_size + intensity_size,
        )

    def transform(self, faces: list[np.ndarray]) -> np.ndarray:
        if not faces:
            return np.empty((0, self.layout.total_dimensions), dtype=np.float32)
        return np.vstack([self.extract(face) for face in faces]).astype(np.float32)

    def extract(self, face: np.ndarray) -> np.ndarray:
        if face.shape != (self.face_size, self.face_size):
            raise ValueError(
                f"Expected normalized face shape {(self.face_size, self.face_size)}, "
                f"received {face.shape}."
            )
        hog_features = self.hog.compute(face).reshape(-1).astype(np.float32)
        lbp_features = self._spatial_lbp(face)
        intensity_hist = cv2.calcHist([face], [0], None, [32], [0, 256]).reshape(-1)
        intensity_hist = self._l1_normalize(intensity_hist)
        return np.concatenate([hog_features, lbp_features, intensity_hist])

    def family_slices(self) -> dict[str, slice]:
        return {
            "hog_shape": self.layout.hog,
            "lbp_texture": self.layout.lbp,
            "intensity": self.layout.intensity,
        }

    def _spatial_lbp(self, face: np.ndarray) -> np.ndarray:
        center = face[1:-1, 1:-1]
        neighbors = [
            face[:-2, :-2],
            face[:-2, 1:-1],
            face[:-2, 2:],
            face[1:-1, 2:],
            face[2:, 2:],
            face[2:, 1:-1],
            face[2:, :-2],
            face[1:-1, :-2],
        ]
        codes = np.zeros_like(center, dtype=np.uint8)
        for bit, neighbor in enumerate(neighbors):
            codes |= ((neighbor >= center).astype(np.uint8) << bit)

        features: list[np.ndarray] = []
        for row_chunk in np.array_split(codes, 4, axis=0):
            for cell in np.array_split(row_chunk, 4, axis=1):
                histogram, _ = np.histogram(cell // 16, bins=16, range=(0, 16))
                features.append(self._l1_normalize(histogram.astype(np.float32)))
        return np.concatenate(features)

    @staticmethod
    def _l1_normalize(values: np.ndarray) -> np.ndarray:
        denominator = float(values.sum())
        return values / denominator if denominator > 0 else values
