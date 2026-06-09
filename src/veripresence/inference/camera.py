from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2

from veripresence.alerts.webhook import WebhookAlert
from veripresence.config import load_config
from veripresence.inference.engine import Recognition, RecognitionEngine
from veripresence.storage.repository import AttendanceRepository
from veripresence.utils.logging import configure_logging


def run_camera(
    config_path: str | Path,
    source: int | str | None = None,
    display: bool = True,
    record_path: str | Path | None = None,
    webhook_url: str | None = None,
) -> None:
    config = load_config(config_path)
    engine = RecognitionEngine(
        config.paths.model_path,
        config.preprocessing,
        config.quality,
        config.inference.unknown_label,
    )
    repository = AttendanceRepository(config.paths.database_path)
    alert = WebhookAlert(webhook_url)
    source = config.inference.camera_index if source is None else source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = _create_writer(capture, record_path)
    frame_number = 0
    latest: list[Recognition] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_number += 1
            if frame_number % config.inference.process_every_n_frames == 0:
                latest = engine.recognize_image(frame)
                for recognition in latest:
                    identity = recognition.identity if recognition.accepted else None
                    inserted = repository.record(
                        identity=identity,
                        accepted=recognition.accepted,
                        confidence=recognition.confidence,
                        margin=recognition.margin,
                        source=str(source),
                        cooldown_seconds=config.inference.attendance_cooldown_seconds,
                    )
                    if inserted and not recognition.accepted:
                        alert.send_unknown(recognition.confidence, str(source))

            _annotate(frame, latest)
            if writer is not None:
                writer.write(frame)
            if display:
                cv2.imshow("VeriPresence", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()


def _annotate(frame, recognitions: list[Recognition]) -> None:
    for recognition in recognitions:
        x, y, width, height = recognition.bbox
        color = (40, 190, 40) if recognition.accepted else (40, 40, 220)
        label = f"{recognition.identity} {recognition.confidence:.2f}"
        if recognition.reason == "poor_quality":
            label = f"{label} check image"
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            frame,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def _create_writer(capture, record_path: str | Path | None):
    if record_path is None:
        return None
    destination = Path(record_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 20.0
    return cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-time VeriPresence recognition.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--source",
        default=None,
        help="Camera index or video path. Defaults to the configured camera index.",
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    source: int | str | None = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    configure_logging()
    run_camera(
        config_path=args.config,
        source=source,
        display=not args.no_display,
        record_path=args.record,
        webhook_url=os.getenv("VERIPRESENCE_WEBHOOK_URL"),
    )


if __name__ == "__main__":
    main()
