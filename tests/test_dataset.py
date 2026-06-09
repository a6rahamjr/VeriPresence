from veripresence.data.dataset import EnrollmentDataset


def test_dataset_discovers_folder_based_identities(synthetic_project):
    dataset = EnrollmentDataset(
        synthetic_project.paths.enrollment_dir,
        minimum_samples_per_identity=3,
    )
    samples = dataset.discover()

    assert len(samples) == 18
    assert dataset.class_counts(samples) == {"alex": 6, "blair": 6, "casey": 6}
    assert len(dataset.fingerprint(samples)) == 64
