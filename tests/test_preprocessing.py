# trying to use a similar approach to the test file for test_algorithms.py.

import unittest
import numpy as np
import pandas as pd
from cluster_maker.preprocessing import select_features, standardise_features



class TestStandardiseFeatures(unittest.TestCase):

    # Will pass OK if the input in standardise_features is a numpy array. Checking that error handling is done correctly.
    def test_standardise_features_correct_input(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = standardise_features(X)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, X.shape)

    # Will pass OK if the output of standardise_features is properly scaled to mean 0 and std 1. Checking that the standardisation is done correctly.
    def test_standardises_features(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        scaled = standardise_features(X)
        np.testing.assert_allclose(scaled.mean(axis=0), [0.0, 0.0], atol=1e-7) # checks if close to 0 with tolerance of 1e-7
        np.testing.assert_allclose(scaled.std(axis=0), [1.0, 1.0], atol=1e-7)

    # Detects selecting the wrong columns or returning them in the wrong order. If the wrong columns are selected then the pipeline will use wrong data
    def test_select_features_returns_correct_data(self):
        data = pd.DataFrame({
            "x1": [1.0, 2.0],
            "x2": [3.0, 4.0],
            "label": ["a", "b"],
        })

        selected = select_features(data, ["x2", "x1"])

        self.assertEqual(list(selected.columns), ["x2", "x1"]) # checks if the columns are in the correct order
        pd.testing.assert_frame_equal(selected, data[["x2", "x1"]]) # returns an error if the dataframes are non-equal

if __name__ == "__main__":
    unittest.main()