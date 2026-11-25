###
## cluster_maker
## Agglomerative hierarchical clustering
###

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def agglomerative_clustering(
    X: np.ndarray,
    k: int,
    linkage: str = "ward",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform agglomerative hierarchical clustering.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    k : int
        Number of clusters.
    linkage : str, default "ward"
        Linkage method: "ward", "single", "complete" or "average".

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels assigned to each observation.
    centroids : ndarray of shape (k, n_features)
        Mean position of observations in each cluster.
    """
    # Validate the input data type.
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    # Ensure the input is a valid two-dimensional feature matrix.
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")

    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must contain at least two samples and one feature.")

    # Reject invalid cluster counts before fitting the model.
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise TypeError("k must be an integer.")

    if k <= 0 or k > X.shape[0]:
        raise ValueError(
            "k must be between 1 and the number of samples."
        )

    # Validate the requested linkage method.
    valid_linkages = ("ward", "single", "complete", "average")

    if linkage not in valid_linkages:
        raise ValueError(f"linkage must be one of {valid_linkages}.")

    # Fit the hierarchical clustering model and obtain cluster labels.
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage,
    )

    labels = model.fit_predict(X)

    # Agglomerative clustering does not directly return centroids.
    # Calculate cluster means for compatibility with existing package tools.
    centroids = np.array([
        X[labels == cluster_id].mean(axis=0)
        for cluster_id in range(k)
    ])

    return labels, centroids