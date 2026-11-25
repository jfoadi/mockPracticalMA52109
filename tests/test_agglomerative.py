import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster_maker import agglomerative_clustering, run_clustering


class TestAgglomerativeClustering(unittest.TestCase):
    def test_returns_labels_and_cluster_means(self):
        X = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ])

        labels, centroids = agglomerative_clustering(X, k=2)

        self.assertEqual(labels.shape, (4,))
        self.assertEqual(centroids.shape, (2, 2))
        self.assertEqual(len(np.unique(labels)), 2)
        np.testing.assert_allclose(
            np.sort(centroids[:, 0]),
            [0.05, 10.05],
        )

    def test_rejects_cluster_count_outside_sample_range(self):
        X = np.array([[0.0], [1.0]])

        with self.assertRaises(ValueError):
            agglomerative_clustering(X, k=3)

    def test_run_clustering_supports_agglomerative(self):
        data = pd.DataFrame({
            "x": [0.0, 0.1, 10.0, 10.1],
            "y": [0.0, 0.1, 10.0, 10.1],
        })
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "data.csv"
            data.to_csv(input_path, index=False)
            result = run_clustering(
                input_path=str(input_path),
                feature_cols=["x", "y"],
                algorithm="agglomerative",
                k=2,
                compute_elbow=True,
                elbow_k_values=[1, 2],
            )

        try:
            self.assertEqual(result["labels"].shape, (4,))
            self.assertEqual(set(result["elbow_inertias"]), {1, 2})
            self.assertIsNotNone(result["fig_elbow"])
        finally:
            plt.close(result["fig_cluster"])
            if result["fig_elbow"] is not None:
                plt.close(result["fig_elbow"])


if __name__ == "__main__":
    unittest.main()
