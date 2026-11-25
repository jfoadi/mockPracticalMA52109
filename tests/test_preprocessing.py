###
## cluster_maker - test file for preprocessing.py
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # A small table like the ones run_clustering receives: two numeric
        # features (x, y), a numeric column that is NOT a feature (the
        # true_cluster label added by simulate_data) and a text column.
        self.data = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [10.0, 20.0, 30.0, 40.0],
            "true_cluster": [0, 0, 1, 1],
            "name": ["a", "b", "c", "d"],
        })

    # Problem detected: select_features must give the pipeline exactly the
    # requested columns, in the requested order, and leave the user's table
    # alone. If extra columns leaked through (e.g. the true_cluster label),
    # k-means would silently cluster on the answer; if the order changed, the
    # centroids would be reported against the wrong features; if the original
    # table were altered, the labelled CSV exported later would be corrupted.
    def test_select_features_returns_only_requested_columns_in_order(self):
        original = self.data.copy()

        X_df = select_features(self.data, ["y", "x"])

        self.assertEqual(list(X_df.columns), ["y", "x"])
        self.assertEqual(X_df.shape, (4, 2))
        self.assertTrue(np.array_equal(X_df["y"].to_numpy(), original["y"].to_numpy()))
        self.assertTrue(self.data.equals(original))

    # Problem detected: a mistyped column name or a text column must fail at
    # once, with a clear message from select_features. Without these checks a
    # typo would surface as an obscure pandas error ("not in index") and a text
    # column would travel on to the scaler and crash deep inside scikit-learn.
    # The regex "missing" checks that it is OUR message, not pandas' own.
    def test_select_features_rejects_missing_and_non_numeric_columns(self):
        with self.assertRaisesRegex(KeyError, "missing"):
            select_features(self.data, ["x", "zzz"])

        with self.assertRaises(TypeError):
            select_features(self.data, ["x", "name"])

    # Problem detected: k-means works with distances, so a feature measured in
    # thousands would swamp one measured in tenths unless every feature is put
    # on the same scale. After standardising, each varying column must have
    # mean 0 and standard deviation 1. A constant column (zero variance) is a
    # real-world trap: it must become zeros, not NaN from dividing by a zero
    # standard deviation, because one NaN would make the clustering fail.
    # The caller's array must also be left unchanged.
    def test_standardise_features_gives_zero_mean_unit_std_and_keeps_input(self):
        X = np.array([
            [1000.0, 0.1, 5.0],
            [2000.0, 0.2, 5.0],
            [3000.0, 0.3, 5.0],
            [4000.0, 0.4, 5.0],
        ])
        X_before = X.copy()

        X_scaled = standardise_features(X)

        self.assertEqual(X_scaled.shape, X.shape)
        self.assertTrue(np.allclose(X_scaled[:, :2].mean(axis=0), 0.0))
        self.assertTrue(np.allclose(X_scaled[:, :2].std(axis=0), 1.0))
        self.assertFalse(np.isnan(X_scaled).any())
        self.assertTrue(np.allclose(X_scaled[:, 2], 0.0))
        self.assertTrue(np.array_equal(X, X_before))


if __name__ == "__main__":
    unittest.main()
