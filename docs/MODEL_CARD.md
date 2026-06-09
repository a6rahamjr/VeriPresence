# VeriPresence Model Card

## Model Details

- Task: enrolled face identity classification with unknown rejection
- Detector: OpenCV frontal-face Haar cascade
- Features: HOG, spatial LBP, and grayscale intensity histogram
- Classifier: class-balanced linear support vector machine with probability estimates
- Artifact format: Joblib, schema version 1
- Input quality gate: brightness, contrast, and Laplacian sharpness checks

## Intended Use

The model is intended for consent-based attendance experiments in controlled indoor
environments. It can also serve as a lightweight baseline before adopting deep embeddings.

## Out-of-Scope Use

- Law enforcement, surveillance, or covert identification
- Authentication for high-value systems
- Demographic, emotion, health, or behavior inference
- Liveness or anti-spoofing decisions

## Training Data

The model is trained from local folder-based enrollment images. Data is not bundled in the
repository. A synthetic generator is provided for functional tests.

## Evaluation

Each run records a stratified holdout report, cross-validation score, confusion matrix,
thresholds, latency, and feature-family ablation. Small enrollment sets can produce
optimistic metrics when captures share lighting, device, or session conditions.

## Limitations and Risks

- Haar detection favors frontal, unobstructed faces.
- Probability scores are dataset-specific and require recalibration after enrollment changes.
- Quality thresholds are operational heuristics and should be tuned for each camera.
- Accuracy can vary across lighting, pose, camera quality, appearance, and demographic groups.
- Unknown rejection cannot be validated from known identities alone.
- Joblib artifacts must only be loaded from trusted sources.

## Privacy

Enrollment images, artifacts, and event databases are ignored by Git. Operators are still
responsible for encryption, access control, deletion, consent, and legal compliance.
