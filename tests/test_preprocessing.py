
import unittest

import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):

    # Detect missing feature columns before they cause errors later
    # in the clustering pipeline.
    def test_missing_feature(self):
        df = pd.DataFrame({
            "height": [160, 170, 180],
            "weight": [55, 65, 75]
        })

        with self.assertRaises(KeyError):
            select_features(df, ["height", "age"])

    # Detect non-numeric features that cannot be used directly
    # in distance-based clustering algorithms.
    def test_non_numeric_feature(self):
        df = pd.DataFrame({
            "height": [160, 170, 180],
            "category": ["A", "B", "C"]
        })

        with self.assertRaises(TypeError):
            select_features(df, ["height", "category"])

    # Detect incorrect scaling that could cause features with
    # larger numerical ranges to dominate clustering distances.
    def test_standardisation(self):
        X = np.array([
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0]
        ])

        X_scaled = standardise_features(X)

        self.assertEqual(X_scaled.shape, X.shape)
        np.testing.assert_allclose(
            X_scaled.mean(axis=0),
            [0.0, 0.0],
            atol=1e-10
        )
        np.testing.assert_allclose(
            X_scaled.std(axis=0),
            [1.0, 1.0],
            atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
