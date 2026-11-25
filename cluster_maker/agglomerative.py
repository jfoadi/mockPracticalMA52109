#agglomerative clustering, also known as 'robust hierarchical clustering', is when k-means starts with k centroids and moves them around
#Start with every point as its own cluster
#Find the two closest clusters and merge them into one.
#Repeat until only k clusters are left.



## Agglomerative (hierarchical) clustering
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
    Hierarchical clustering using scikit-learn's AgglomerativeClustering.

    Every point starts as its own cluster; the two closest clusters are
    merged repeatedly until k clusters are left.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int (integers)
        Number of clusters.
    linkage : {"ward", "complete", "average", "single"}, default "ward"
        How the distance between two clusters is measured when deciding
        which clusters to merge.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
        Mean of the points in each cluster. Agglomerative clustering does not
        use centroids itself, but returning them gives the same output as
        kmeans and sklearn_kmeans, so the result works with compute_inertia
        and plot_clusters_2d.
    """
    # Input checks
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if np.isnan(X).any():
        raise ValueError("X contains NaN values.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > X.shape[0]:
        raise ValueError("k cannot be larger than the number of samples.")
    if linkage not in ("ward", "complete", "average", "single"):
        raise ValueError(f"Unknown linkage '{linkage}'.")

    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X)

    # Centre of each cluster = mean of its points
    centroids = np.array([X[labels == c].mean(axis=0) for c in range(k)])
    return labels, centroids