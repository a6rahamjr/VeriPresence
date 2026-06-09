from veripresence.models.artifact import ModelArtifact
from veripresence.training.trainer import train


def test_training_exports_versioned_artifact(synthetic_project):
    result = train(synthetic_project)
    artifact = ModelArtifact.load(result.model_path)

    assert result.model_path.exists()
    assert (result.run_dir / "metrics.json").exists()
    assert set(artifact.metadata["classes"]) == {"alex", "blair", "casey"}
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
