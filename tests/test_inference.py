import cv2

from veripresence.inference.engine import RecognitionEngine
from veripresence.training.trainer import train


def test_inference_returns_a_known_identity_for_enrollment_image(synthetic_project):
    train(synthetic_project)
    engine = RecognitionEngine(
        synthetic_project.paths.model_path,
        synthetic_project.preprocessing,
        synthetic_project.quality,
        synthetic_project.inference.unknown_label,
    )
    sample_path = next(
        (synthetic_project.paths.enrollment_dir / "alex").glob("*.png")
    )
    image = cv2.imread(str(sample_path))

    recognitions = engine.recognize_image(image, allow_full_frame_fallback=True)

    assert len(recognitions) == 1
    assert recognitions[0].accepted
    assert recognitions[0].identity == "alex"
    assert recognitions[0].quality.acceptable


def test_inference_rejects_an_unusable_image(synthetic_project):
    train(synthetic_project)
    engine = RecognitionEngine(
        synthetic_project.paths.model_path,
        synthetic_project.preprocessing,
        synthetic_project.quality,
        synthetic_project.inference.unknown_label,
    )
    image = cv2.imread(
        str(next((synthetic_project.paths.enrollment_dir / "alex").glob("*.png")))
    )
    image[:] = 0

    recognition = engine.recognize_image(
        image, allow_full_frame_fallback=True
    )[0]

    assert not recognition.accepted
    assert recognition.reason == "poor_quality"
    assert {"too_dark", "low_contrast", "blurry"} <= set(
        recognition.quality.issues
    )
