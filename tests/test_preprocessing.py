
import unittest #importing the built in unittest module for creating and running tests

import numpy as np
import pandas as pd

from cluster_maker.preprocessing import (
    select_features,
    standardise_features,
)

class TestPreprocessing(unittest.TestCase):

    # Selecting features should exclude unrelated columns from the dataset.
    def test_select_features_returns_only_requested_columns(self):
        data = pd.DataFrame({
            "age": [20, 30, 40],
            "salary": [2000, 3000, 4000],
            "name": ["Anna", "Ben", "Chris"],
        })

        result = select_features(data, ["age", "salary"])

        self.assertEqual(list(result.columns), ["age", "salary"])
        self.assertEqual(result.shape, (3, 2))

    # A missing feature column should be detected rather than silently ignored.
    def test_select_features_raises_error_for_missing_column(self):
        data = pd.DataFrame({
            "age": [20, 30, 40],
            "salary": [2000, 3000, 4000],
        })

        with self.assertRaises(KeyError):
            select_features(data, ["age", "height"])

    # Features on different scales should be standardised to mean zero and standard deviation one.
    def test_standardise_features_scales_data(self):
        X = np.array([
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
        ])

        result = standardise_features(X)

        np.testing.assert_allclose(result.mean(axis=0), [0.0, 0.0])
        np.testing.assert_allclose(result.std(axis=0), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
