###
## cluster_maker - test file
## Francesco Gallo - University of Bath
## November 2026
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):


    # The bug it tests for: If the standardisation is done incorrectly
    # which could mean no scaling was done or scaling was done on the wrong axis
    def test_standardise_features_gives_zero_mean_unit_std_per_column(self):
        X = np.array([
            [1.0, 1000.0],
            [2.0, 3000.0],
            [3.0, 2000.0],
            [4.0, 5000.0],
            [5.0, 4000.0],
        ])

        X_scaled = standardise_features(X)

        self.assertEqual(X_scaled.shape, X.shape)
        np.testing.assert_allclose(X_scaled.mean(axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(X_scaled.std(axis=0), [1.0, 1.0], atol=1e-12)


    # The bug it tests for: non-numeric columns slipping 
    # through input validation unnoticed.
    def test_select_features_rejects_non_numeric_column(self):
        data = pd.DataFrame({
            "height": [1.2, 3.4, 5.6],
            "colour": ["red", "green", "blue"],
        })

        with self.assertRaises(TypeError) as context:
            select_features(data, ["height", "colour"])

        self.assertIn("colour", str(context.exception))

    # The bug it tests for: If the columns come back in a different order 
    # from the order requested 
    def test_select_features_keeps_requested_column_order(self):
        data = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "c": [7.0, 8.0, 9.0],
        })

        X_df = select_features(data, ["c", "a"])

        self.assertEqual(list(X_df.columns), ["c", "a"])
        np.testing.assert_array_equal(X_df.to_numpy()[:, 0], data["c"].to_numpy())

if __name__ == "__main__":
    unittest.main()