## agglomerative.py will return labels and cluster means for agglomerative clustering. It will also check for invalid inputs and raise appropriate errors.

from __future__ import annotations

from inspect import signature
from typing import Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def agglomerative_clustering(
    X: np.ndarray,
    k: int,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster samples with hierarchical agglomerative clustering.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int
        Number of clusters.
    linkage : {"ward", "complete", "average", "single"}, default "ward"
    metric : str, default "euclidean"
        Distance metric used to measure distances between samples.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
        Mean feature values for each cluster.

    Raises
    ------
    TypeError
        If X is not a numeric NumPy array or k is not an integer.
    ValueError
        If the input shape, cluster count, linkage, or metric is invalid.
    """
    # Error handling
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array.")
    if not np.issubdtype(X.dtype, np.number) or not np.isrealobj(X):
        raise TypeError("X must contain real numeric values.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values.")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer.")
    if k < 1 or k > X.shape[0]:
        raise ValueError("k must be between 1 and the number of samples.")

    if not isinstance(linkage, str):
        raise TypeError("linkage must be a string.")
    valid_linkages = {"ward", "complete", "average", "single"}
    if linkage not in valid_linkages:
        raise ValueError(f"linkage must be one of {sorted(valid_linkages)}.")
    if not isinstance(metric, str):
        raise TypeError("metric must be a string.")
    if metric == "precomputed":
        raise ValueError("metric='precomputed' is not supported for feature data.")
    if linkage == "ward" and metric != "euclidean":
        raise ValueError("The 'ward' linkage requires the Euclidean metric.")

    model_parameters = signature(AgglomerativeClustering).parameters
    if "metric" in model_parameters:
        model = AgglomerativeClustering(
            n_clusters=int(k),
            linkage=linkage,
            metric=metric,
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=int(k),
            linkage=linkage,
            affinity=metric,
        )

    labels = model.fit_predict(X) # get the cluster labels for each sample
    centroids = np.vstack(
        [X[labels == cluster_id].mean(axis=0) for cluster_id in range(k)]
    )
    return labels, centroids
