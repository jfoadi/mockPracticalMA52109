###
## cluster_maker
## Francesco Gallo - University of Bath
## November 2026
###

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering

VALID_LINKAGES = ("ward", "complete", "average", "single")


def compute_cluster_means(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute the mean point of each cluster.

    Agglomerative clustering does not produce centroids itself; the cluster
    means are returned so that the output can be used with the rest of the
    package (e.g. compute_inertia and plot_clusters_2d).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    labels : ndarray of shape (n_samples,)
        Cluster labels numbered 0..k-1.

    Returns
    -------
    means : ndarray of shape (k, n_features)
    """
    k = int(labels.max()) + 1
    return np.array([X[labels == cluster_id].mean(axis=0) for cluster_id in range(k)])


def agglomerative_clustering(
    X: np.ndarray,
    k: int,
    linkage: str = "ward",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hierarchical (agglomerative) clustering using scikit-learn's
    AgglomerativeClustering.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int
        Number of clusters.
    linkage : {"ward", "complete", "average", "single"}, default "ward"
        How the distance between two clusters is measured when deciding
        which clusters to merge.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
        Mean point of each cluster.

    Raises
    ------
    TypeError
        If X is not a NumPy array.
    ValueError
        If X is not 2D or contains NaN/inf, if k is not between 1 and
        n_samples, or if linkage is not recognised.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must not contain NaN or infinite values.")
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > X.shape[0]:
        raise ValueError("k cannot be larger than the number of samples.")
    if linkage not in VALID_LINKAGES:
        raise ValueError(f"linkage must be one of {VALID_LINKAGES}, got '{linkage}'.")

    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X)
    centroids = compute_cluster_means(X, labels)
    return labels, centroids