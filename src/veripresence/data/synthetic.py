from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_COUNTS = {"alex": 12, "blair": 12, "casey": 12}


def generate_synthetic_enrollment(
    destination: str | Path,
    counts: dict[str, int] | None = None,
    seed: int = 42,
    image_size: int = 160,
) -> dict[str, int]:
    destination = Path(destination)
    counts = counts or DEFAULT_COUNTS
    rng = np.random.default_rng(seed)

    for identity_index, (identity, count) in enumerate(sorted(counts.items())):
        identity_dir = destination / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        for sample_index in range(count):
            image = _draw_synthetic_face(
                identity_index=identity_index,
                sample_index=sample_index,
                size=image_size,
                rng=rng,
            )
            cv2.imwrite(str(identity_dir / f"{identity}_{sample_index + 1:03d}.png"), image)
    return dict(counts)


def _draw_synthetic_face(
    identity_index: int,
    sample_index: int,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    background = np.full((size, size, 3), 225, dtype=np.uint8)
    background = np.clip(
        background.astype(np.int16) + rng.normal(0, 5, background.shape),
        0,
        255,
    ).astype(np.uint8)

    center_jitter = rng.integers(-4, 5, size=2)
    center = (size // 2 + int(center_jitter[0]), size // 2 + int(center_jitter[1]))
    skin_palette = [(165, 190, 215), (125, 165, 205), (95, 135, 175)]
    skin = skin_palette[identity_index % len(skin_palette)]
    face_width = 48 + identity_index * 4
    face_height = 62 - identity_index * 3
    cv2.ellipse(background, center, (face_width, face_height), 0, 0, 360, skin, -1)

    eye_y = center[1] - 15 + identity_index * 2
    eye_gap = 21 + identity_index * 3
    eye_radius = 5 + identity_index
    for eye_x in (center[0] - eye_gap, center[0] + eye_gap):
        cv2.circle(background, (eye_x, eye_y), eye_radius + 2, (245, 245, 245), -1)
        cv2.circle(background, (eye_x, eye_y), eye_radius, (25, 25, 25), -1)

    nose_length = 14 + identity_index * 4
    cv2.line(
        background,
        (center[0], center[1] - 5),
        (center[0] - identity_index, center[1] + nose_length),
        (65, 85, 105),
        2,
    )
    mouth_width = 23 + identity_index * 5
    cv2.ellipse(
        background,
        (center[0], center[1] + 31),
        (mouth_width, 7 + identity_index),
        0,
        5,
        175,
        (45, 45, 120),
        3,
    )

    hairstyle = identity_index % 3
    if hairstyle == 0:
        cv2.ellipse(
            background,
            (center[0], center[1] - 45),
            (face_width, 24),
            0,
            180,
            360,
            (20, 20, 20),
            -1,
        )
    elif hairstyle == 1:
        for offset in range(-face_width, face_width, 8):
            cv2.line(
                background,
                (center[0] + offset, center[1] - 48),
                (center[0] + offset // 2, center[1] - 67),
                (30, 30, 30),
                4,
            )
    else:
        cv2.rectangle(
            background,
            (center[0] - face_width, center[1] - 65),
            (center[0] + face_width, center[1] - 48),
            (15, 15, 15),
            -1,
        )

    angle = float(rng.uniform(-8, 8))
    scale = float(rng.uniform(0.94, 1.04))
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    image = cv2.warpAffine(
        background,
        matrix,
        (size, size),
        borderMode=cv2.BORDER_REFLECT_101,
    )
    if sample_index % 5 == 0:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate privacy-safe demo enrollment data.")
    parser.add_argument(
        "--destination", type=Path, default=Path("data/raw/enrollment")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = generate_synthetic_enrollment(args.destination, seed=args.seed)
    print(f"Generated demo identities: {counts}")


if __name__ == "__main__":
    main()
