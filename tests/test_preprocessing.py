#Creating my own 3 unit tests verifying aspects of module 'preprocessing.py'

import unittest

import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):

    # If a feature name is misspelled or missing, select_features must raise
    # an error
    # Otherwise the clustering would silently run on the wrong
    # features and give misleading results.
    def test_missing_column(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        with self.assertRaises(KeyError):
            select_features(df, ["x", "z"]) #z is not a column of df, so this is as if the user made a typo

    # Clustering needs numbers to compute distances. A text column (e.g. a
    # name or ID) must be rejected here with a clear error as it is non-numeric
    def test_non_numeric_column(self):
        df = pd.DataFrame({"x": [1, 2, 3], "name": ["a", "b", "c"]})
        with self.assertRaises(TypeError):
            select_features(df, ["x", "name"])

    # Each column must be scaled separately to mean 0 and std 1. If not, a
    # feature with large values (like 1000s) would dominate the distances
    # and the small-valued feature would be ignored by k-means.
    def test_standardise_each_column(self):
        X = np.array([
            [1000.0, 0.1],
            [2000.0, 0.2],
            [3000.0, 0.3],
        ])
        X_scaled = standardise_features(X)

        np.testing.assert_allclose(X_scaled.mean(axis=0), [0, 0], atol=1e-10)
        np.testing.assert_allclose(X_scaled.std(axis=0), [1, 1], atol=1e-10)
        # axis=0 checks each column separately. atol allows for tiny
        # floating-point rounding errors (e.g. 1e-17 instead of exactly 0).


if __name__ == "__main__":
    unittest.main()