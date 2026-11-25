"""
Agglomerative hierarchical clustering module for cluster_maker.

This module provides an implementation of bottom-up hierarchical
clustering using scikit-learn's AgglomerativeClustering class.

The algorithm initially treats each observation as an individual
cluster and progressively merges clusters according to a specified
linkage criterion until the desired number of clusters is reached.

Four linkage methods are supported:

    - Ward: minimises the increase in within-cluster variance.
    - Single: uses the minimum distance between observations
      belonging to different clusters.
    - Complete: uses the maximum distance between observations
      belonging to different clusters.
    - Average: uses the mean pairwise distance between observations
      belonging to different clusters.

Unlike K-means, agglomerative clustering does not require random
centroid initialisation and performs clustering through hierarchical
merging rather than iterative centroid updates.

The module returns cluster labels and cluster mean coordinates,
maintaining compatibility with the existing evaluation, plotting
and interface components of the cluster_maker package. These mean
coordinates are calculated after clustering and are not used to
determine cluster membership.

Input validation and exception handling are included to ensure
reliable operation when encountering invalid datasets, inappropriate
cluster counts, unsupported linkage methods or missing dependencies.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def agglomerative_clustering(
    X: np.ndarray,
    k: int,
    linkage: str = "ward",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge observations hierarchically until k clusters remain.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Finite numerical observations.
    k : int
        Number of clusters to form (between 1 and n_samples).
    linkage : str, default="ward"
        Merge criterion: ward, complete, average or single. Euclidean
        distance is used by default.

    Returns
    -------
    labels : np.ndarray of shape (n_samples,)
        Cluster assignment for each observation.
    centroids : np.ndarray of shape (k, n_features)
        Cluster means calculated after fitting for package compatibility.
        They are descriptive and are not used to assign observations.

    Raises
    ------
    TypeError
        If X is not a NumPy array or k is not an integer.
    ValueError
        If the data, cluster count or linkage is invalid, or fitting fails.
    ImportError
        If scikit-learn cannot be imported.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] == 0:
        raise ValueError("X must be 2D with at least two samples and one feature.")
    if X.dtype.kind not in "iuf" or not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite real numerical values.")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer.")
    if not 1 <= k <= X.shape[0]:
        raise ValueError("k must be between 1 and the number of samples.")
    if linkage not in ("ward", "complete", "average", "single"):
        raise ValueError("linkage must be ward, complete, average or single.")

    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:
        raise ImportError("Agglomerative clustering requires scikit-learn.") from exc

    try:
        # Start with individual samples and merge according to linkage.
        model = AgglomerativeClustering(n_clusters=int(k), linkage=linkage)
        labels = model.fit_predict(X)

        # Compute cluster means only after the hierarchy has been cut.
        centroids = np.vstack([
            X[labels == i].mean(axis=0) for i in range(k)
        ])
        return labels, centroids
    except (ValueError, TypeError) as e:
        raise ValueError(f"Agglomerative clustering failed: {e}") from e
