
"""
Agglomerative hierarchical clustering implementation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def agglomerative(
    X: np.ndarray,
    k: int,
    linkage: str = "ward",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform agglomerative hierarchical clustering.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Numerical data to cluster.
    k : int
        Number of clusters.
    linkage : str, default "ward"
        Method used to merge clusters.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster assignments.
    centroids : ndarray of shape (k, n_features)
        Mean feature values for each cluster.
    """

    # Check that the input is a NumPy array.
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    # Ensure the data has the expected two-dimensional shape.
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n_samples = X.shape[0]

    # Validate the requested number of clusters.
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be an integer.")

    if k < 1 or k > n_samples:
        raise ValueError("k must be between 1 and the number of samples.")

    # Check that all feature values are finite.
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("X must contain numeric values.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains missing or infinite values.")

    # Fit the hierarchical clustering model.
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage
    )

    labels = model.fit_predict(X)

    # Calculate cluster means to maintain compatibility
    # with the existing clustering functions.
    centroids = np.array([
        X[labels == i].mean(axis=0)
        for i in range(k)
    ])

    return labels, centroids
