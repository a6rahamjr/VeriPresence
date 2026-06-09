from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from veripresence.data.dataset import IMAGE_SUFFIXES


def infer_identity(path: Path) -> str | None:
    match = re.match(r"([A-Za-z][A-Za-z _-]*?)(?:\d+)?$", path.stem.strip())
    if not match:
        return None
    identity = re.sub(r"[^A-Za-z]+", "-", match.group(1)).strip("-").lower()
    return identity or None


def import_flat_images(
    source_dir: str | Path,
    destination_dir: str | Path,
    minimum_images: int = 2,
) -> dict[str, object]:
    source = Path(source_dir)
    destination = Path(destination_dir)
    if not source.exists():
        raise FileNotFoundError(f"Source image directory not found: {source}")

    grouped: dict[str, list[Path]] = {}
    skipped: list[str] = []
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        identity = infer_identity(path)
        if identity is None:
            skipped.append(path.name)
            continue
        grouped.setdefault(identity, []).append(path)

    eligible = {
        identity: paths for identity, paths in grouped.items() if len(paths) >= minimum_images
    }
    excluded = {
        identity: len(paths) for identity, paths in grouped.items() if len(paths) < minimum_images
    }

    destination.mkdir(parents=True, exist_ok=True)
    copied = Counter()
    for identity, paths in sorted(eligible.items()):
        identity_dir = destination / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        for index, source_path in enumerate(paths, start=1):
            target = identity_dir / f"{identity}_{index:03d}{source_path.suffix.lower()}"
            shutil.copy2(source_path, target)
            copied[identity] += 1

    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "copied": dict(sorted(copied.items())),
        "excluded_groups": excluded,
        "skipped_files": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import filename-labeled face images.")
    parser.add_argument("source", type=Path, help="Directory containing source images.")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/enrollment"),
        help="Folder-based enrollment destination.",
    )
    parser.add_argument("--minimum-images", type=int, default=2)
    args = parser.parse_args()
    report = import_flat_images(args.source, args.destination, args.minimum_images)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
