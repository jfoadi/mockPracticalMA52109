###
## cluster_maker - test file for preprocessing.py
## MA52109 mock practical - Task 3
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):

    # Bug detected: bad input slipping silently into the pipeline.
    # A misspelled column name or a text column (e.g. a "city" label) must be
    # rejected immediately with a clear KeyError / TypeError. If these checks
    # are missing or raise the wrong exception, the problem only shows up later
    # as an obscure crash (or wrong clusters) deep inside k-means.
    def test_select_features_rejects_missing_and_non_numeric_columns(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "city": ["Bath", "Bristol", "Wells"],
        })

        with self.assertRaises(KeyError):
            select_features(df, ["x", "not_a_column"])

        with self.assertRaises(TypeError):
            select_features(df, ["x", "city"])

    # Bug detected: wrong columns, wrong order or altered values being passed
    # to the clustering step. Only the requested features may be returned
    # (a label column such as "true_cluster" must NOT leak into X), in the
    # order requested (plot_clusters_2d uses the first two columns as the axes
    # and centroids follow this order), with every row and value untouched.
    def test_select_features_returns_only_requested_columns_in_order(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "c": [0.5, 0.6, 0.7, 0.8],
            "true_cluster": [0, 0, 1, 1],
        })

        X_df = select_features(df, ["c", "a"])

        self.assertEqual(list(X_df.columns), ["c", "a"])
        self.assertEqual(len(X_df), len(df))
        self.assertTrue(np.array_equal(X_df["c"].to_numpy(), df["c"].to_numpy()))
        self.assertTrue(np.array_equal(X_df["a"].to_numpy(), df["a"].to_numpy()))

    # Bug detected: standardisation that fails to put features on a common
    # scale, or that produces NaN. Features on very different scales (here
    # ~1000 vs ~0.001) must each end up with mean 0 and std 1, otherwise the
    # large-scale feature dominates the Euclidean distances in k-means.
    # A constant column (zero variance) must not cause division by zero:
    # NaN/inf values would crash or silently corrupt the clustering.
    # The caller's array must also be left unmodified.
    def test_standardise_features_scales_columns_and_handles_constant_column(self):
        X = np.array([
            [1000.0, 0.001, 7.0],
            [2000.0, 0.002, 7.0],
            [3000.0, 0.003, 7.0],
            [4000.0, 0.004, 7.0],
        ])
        X_before = X.copy()

        X_scaled = standardise_features(X)

        self.assertEqual(X_scaled.shape, X.shape)
        self.assertTrue(np.all(np.isfinite(X_scaled)))

        # Every column is centred.
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-12)
        # The two varying columns have unit variance and are now identical
        # in scale, despite the 10^6 difference in the original units.
        np.testing.assert_allclose(X_scaled[:, :2].std(axis=0), 1.0, atol=1e-12)
        # The constant column becomes all zeros rather than NaN.
        np.testing.assert_allclose(X_scaled[:, 2], 0.0, atol=1e-12)

        # Input array not modified in place.
        self.assertTrue(np.array_equal(X, X_before))


if __name__ == "__main__":
    unittest.main()
