from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from veripresence.config import PreprocessingConfig


@dataclass(frozen=True)
class FaceCrop:
    image: np.ndarray
    bbox: tuple[int, int, int, int]
    rotation_degrees: int = 0


class FacePreprocessor:
    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Could not load OpenCV face cascade: {cascade_path}")
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def extract_faces(self, image: np.ndarray) -> list[FaceCrop]:
        self._validate_image(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.config.detection_scale_factor,
            minNeighbors=self.config.detection_min_neighbors,
            minSize=(self.config.min_face_size, self.config.min_face_size),
        )
        crops = [
            FaceCrop(
                image=self.normalize_face(self._crop_with_margin(image, tuple(map(int, box)))),
                bbox=tuple(map(int, box)),
            )
            for box in boxes
        ]
        return sorted(crops, key=lambda item: item.bbox[2] * item.bbox[3], reverse=True)

    def extract_primary_face(
        self, image: np.ndarray, allow_full_frame_fallback: bool = True
    ) -> FaceCrop:
        self._validate_image(image)
        rotations: list[tuple[int, np.ndarray]] = [(0, image)]
        if self.config.try_rotations:
            rotations.extend(
                [
                    (90, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
                    (-90, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
                    (180, cv2.rotate(image, cv2.ROTATE_180)),
                ]
            )

        candidates: list[FaceCrop] = []
        for angle, rotated in rotations:
            for crop in self.extract_faces(rotated):
                candidates.append(
                    FaceCrop(image=crop.image, bbox=crop.bbox, rotation_degrees=angle)
                )
        if candidates:
            return max(candidates, key=lambda item: item.bbox[2] * item.bbox[3])

        if not allow_full_frame_fallback:
            raise ValueError("No face detected in image.")
        fallback = self._center_square(image)
        height, width = image.shape[:2]
        return FaceCrop(
            image=self.normalize_face(fallback),
            bbox=(0, 0, width, height),
            rotation_degrees=0,
        )

    def normalize_face(self, face: np.ndarray) -> np.ndarray:
        if face.ndim == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = self.clahe.apply(face)
        face = cv2.resize(
            face,
            (self.config.face_size, self.config.face_size),
            interpolation=cv2.INTER_AREA,
        )
        return face.astype(np.uint8)

    def augment(self, face: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        size = self.config.face_size
        angle = float(rng.uniform(-7.0, 7.0))
        scale = float(rng.uniform(0.97, 1.03))
        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
        augmented = cv2.warpAffine(
            face,
            matrix,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        alpha = float(rng.uniform(0.85, 1.15))
        beta = float(rng.uniform(-12.0, 12.0))
        augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
        if rng.random() < 0.5:
            augmented = cv2.flip(augmented, 1)
        return augmented

    def _crop_with_margin(
        self, image: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        x, y, width, height = bbox
        margin_x = int(width * self.config.crop_margin)
        margin_y = int(height * self.config.crop_margin)
        image_height, image_width = image.shape[:2]
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image_width, x + width + margin_x)
        y2 = min(image_height, y + height + margin_y)
        return image[y1:y2, x1:x2]

    @staticmethod
    def _center_square(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        side = min(height, width)
        x1 = (width - side) // 2
        y1 = (height - side) // 2
        return image[y1 : y1 + side, x1 : x1 + side]

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None or image.size == 0:
            raise ValueError("Image is empty.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a BGR image with three channels.")
