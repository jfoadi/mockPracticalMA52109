
from __future__ import annotations


import numpy as np
from sklearn.cluster import AgglomerativeClustering


def agglomerative_clustering(
    X: np.ndarray,
    k: int,
    linkage: str = "ward",
) -> np.ndarray:
    """
    Perform agglomerative hierarchical clustering.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Dataset to cluster.
    k : int
        Number of clusters.
    linkage : str, default "ward"
        Linkage criterion: "ward", "complete", "average", or "single".

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster label assigned to each sample.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n_samples = X.shape[0]

    if k <= 0 or k > n_samples:
        raise ValueError("k must be between 1 and the number of samples.")

    valid_linkages = ("ward", "complete", "average", "single")
    if linkage not in valid_linkages:
        raise ValueError(
            f"linkage must be one of {valid_linkages}."
        )

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage,
    )

    labels = model.fit_predict(X)

    return labels #agglomerative clustering doesnt calculate centroids, so this function only returns labels
