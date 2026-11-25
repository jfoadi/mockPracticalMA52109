import unittest
import numpy as np
import pandas as pd

from cluster_maker.preprocessing import (
    select_features,
    standardise_features,
)


class TestPreprocessing(unittest.TestCase):

    # Prevent clustering from proceeding when a requested feature
    # is missing from the input dataset.
    def test_missing_feature(self):
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6]
        })

        with self.assertRaises(KeyError):
            select_features(df, ["x", "z"])

    # Prevent non-numeric features from entering the clustering
    # pipeline, where numerical distance calculations would fail.
    def test_non_numeric_feature(self):
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "category": ["A", "B", "C"]
        })

        with self.assertRaises(TypeError):
            select_features(df, ["x", "category"])

    # Ensure features with different numerical scales are standardised
    # correctly so that larger-scale features do not dominate clustering.
    def test_standardise_features(self):
        X = np.array([
            [1.0, 100.0],
            [2.0, 500.0],
            [3.0, 900.0]
        ])

        X_scaled = standardise_features(X)

        np.testing.assert_allclose(
            X_scaled.mean(axis=0), [0.0, 0.0], atol=1e-8
        )

        np.testing.assert_allclose(
            X_scaled.std(axis=0), [1.0, 1.0], atol=1e-8
        )


if __name__ == "__main__":
    unittest.main()