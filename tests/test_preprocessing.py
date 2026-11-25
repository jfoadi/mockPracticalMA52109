"""
Unit tests for the preprocessing module of cluster_maker.

This module verifies three essential aspects of data preprocessing:
    1. Correct feature standardisation.
    2. Rejection of non-numeric features.
    3. Appropriate handling of missing feature columns.

These tests help ensure that the clustering algorithms receive
correctly processed numerical data.
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cluster_maker.preprocessing import (
    select_features,
    standardise_features,
)


class TestPreprocessing(unittest.TestCase):
    """Test the correctness and reliability of data preprocessing."""

    # 1. Incorrect standardisation may cause features with larger numerical
    # scales to dominate Euclidean distance calculations in K-means.
    def test_standardise_features(self) -> None:
        """
        Verify that numerical features are standardised correctly.

        Compare the output against scikit-learn's StandardScaler and
        ensure that each feature has zero mean and unit variance.
        """

        # Create a dataset containing features with different numerical scales.
        X = np.array([
            [1.0, 100.0],
            [2.0, 300.0],
            [3.0, 200.0],
        ])

        # Calculate the reference result using scikit-learn.
        expected = StandardScaler().fit_transform(X)

        # Obtain the result from the package's preprocessing function.
        actual = standardise_features(X)

        # Compare actual and expected values within floating-point tolerance.
        np.testing.assert_allclose(actual, expected, atol=1e-10)

        # Verify that each standardised feature has zero mean.
        np.testing.assert_allclose(
            actual.mean(axis=0), 0.0, atol=1e-10
        )

        # Verify that each standardised feature has unit variance.
        np.testing.assert_allclose(
            actual.var(axis=0), 1.0, atol=1e-10
        )



    # 2.Accepting non-numeric features could cause clustering to fail
    # when calculating distances between observations and centroids.
    def test_non_numeric_features(self) -> None:
        """
        Verify that non-numerical features are rejected.

        Ensure that selecting a textual feature raises a TypeError
        before invalid data reaches the clustering algorithm.
        """

        # Create a dataset containing numerical and textual features.
        data = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": ["A", "B", "C"],
        })

        # Attempt to select both numerical and non-numerical columns.
        # The test passes only if the expected TypeError is raised.
        with self.assertRaises(TypeError):
            select_features(data, ["x", "y"])
            

    # 3. Missing features must not be silently ignored, as this could
    # result in clustering using an incomplete set of variables.
    def test_missing_features(self) -> None:
        """
        Verify that missing feature columns are handled correctly.

        Ensure that requesting a column that does not exist in the
        input dataset raises a KeyError rather than returning an
        incomplete set of features.
        """

        # Create a numerical dataset containing only columns x and y.
        data = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
        })

        # Request column z, which does not exist in the dataset.
        # The test passes only if the expected KeyError is raised.
        with self.assertRaises(KeyError):
            select_features(data, ["x", "z"])


if __name__ == "__main__":
    unittest.main()