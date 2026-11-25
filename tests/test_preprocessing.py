
import numpy as np
import pandas as pd
import pytest

from cluster_maker.preprocessing import (
    select_features,
    standardise_features
)


# Test 1: Detects a bug where feature selection accidentally includes
# unwanted columns or changes the original dataset.
def test_select_features_returns_only_requested_columns():
    data = pd.DataFrame({
        "age": [20, 30, 40],
        "salary": [2000, 3000, 4000],
        "name": ["A", "B", "C"]
    })

    result = select_features(data, ["age", "salary"])

    assert list(result.columns) == ["age", "salary"]
    assert result.shape == (3, 2)
    assert "name" not in result.columns
    assert "name" in data.columns


# Test 2: Detects missing or non-numeric features that could cause
# errors or unreliable results later in the clustering pipeline.
def test_select_features_rejects_invalid_columns():
    data = pd.DataFrame({
        "age": [20, 30, 40],
        "name": ["A", "B", "C"]
    })

    with pytest.raises(KeyError):
        select_features(data, ["salary"])

    with pytest.raises(TypeError):
        select_features(data, ["name"])


# Test 3: Detects incorrect standardisation that could allow features
# with larger scales to dominate the clustering algorithm.
def test_standardise_features_has_zero_mean_and_unit_variance():
    X = np.array([
        [10, 100],
        [20, 200],
        [30, 300],
        [40, 400]
    ])

    result = standardise_features(X)

    assert result.shape == X.shape
    assert np.allclose(result.mean(axis=0), [0, 0])
    assert np.allclose(result.std(axis=0), [1, 1])
