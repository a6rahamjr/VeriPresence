from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class EnrollmentSample:
    identity: str
    path: Path


class EnrollmentDataset:
    def __init__(self, root: str | Path, minimum_samples_per_identity: int = 2) -> None:
        self.root = Path(root)
        self.minimum_samples_per_identity = minimum_samples_per_identity

    def discover(self) -> list[EnrollmentSample]:
        if not self.root.exists():
            raise FileNotFoundError(f"Enrollment directory not found: {self.root}")

        samples = [
            EnrollmentSample(identity=identity_dir.name, path=image_path)
            for identity_dir in sorted(self.root.iterdir())
            if identity_dir.is_dir() and not identity_dir.name.startswith(".")
            for image_path in sorted(identity_dir.iterdir())
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not samples:
            raise ValueError(
                f"No enrollment images found under {self.root}. "
                "Expected data/raw/enrollment/<identity>/<image>."
            )

        counts = Counter(sample.identity for sample in samples)
        underfilled = {
            identity: count
            for identity, count in counts.items()
            if count < self.minimum_samples_per_identity
        }
        if underfilled:
            details = ", ".join(f"{name}={count}" for name, count in sorted(underfilled.items()))
            raise ValueError(
                "Each identity needs at least "
                f"{self.minimum_samples_per_identity} images; found {details}."
            )
        if len(counts) < 2:
            raise ValueError("At least two identities are required to train a classifier.")
        return samples

    @staticmethod
    def read_image(sample: EnrollmentSample) -> np.ndarray:
        image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode image: {sample.path}")
        return image

    @staticmethod
    def class_counts(samples: list[EnrollmentSample]) -> dict[str, int]:
        return dict(sorted(Counter(sample.identity for sample in samples).items()))

    @staticmethod
    def fingerprint(samples: list[EnrollmentSample]) -> str:
        digest = hashlib.sha256()
        for sample in sorted(samples, key=lambda item: str(item.path)):
            digest.update(sample.identity.encode("utf-8"))
            digest.update(sample.path.name.encode("utf-8"))
            with sample.path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
