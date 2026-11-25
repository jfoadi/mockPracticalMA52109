

###
## cluster_maker
## Agglomerative hierarchical clustering
###

from __future__ import annotations

import inspect

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def agglomerative_clustering(
    X: np.ndarray,
    n_clusters: int = 2,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> np.ndarray:
    """Cluster observations with scikit-learn's agglomerative algorithm.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Finite numeric feature matrix.
    n_clusters : int, default 2
        Number of clusters to form.
    linkage : {"ward", "complete", "average", "single"}, default "ward"
        Linkage criterion used to merge clusters. Ward linkage requires the
        Euclidean metric.
    metric : str, default "euclidean"
        Distance metric used to compute linkage. Ward only supports Euclidean.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Integer cluster label for each observation.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    n_samples = X.shape[0]
    if n_samples < 2 or X.shape[1] < 1:
        raise ValueError("X must contain at least two samples and one feature.")
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("X must contain numeric values.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must not contain missing or infinite values.")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)):
        raise TypeError("n_clusters must be an integer.")
    if not 2 <= n_clusters < n_samples:
        raise ValueError("n_clusters must be between 2 and n_samples - 1.")
    if linkage not in {"ward", "complete", "average", "single"}:
        raise ValueError("linkage must be 'ward', 'complete', 'average', or 'single'.")
    if linkage == "ward" and metric != "euclidean":
        raise ValueError("linkage='ward' requires metric='euclidean'.")

    parameters = {"n_clusters": int(n_clusters), "linkage": linkage}
    # sklearn renamed ``affinity`` to ``metric`` in version 1.2.
    metric_parameter = "metric" if "metric" in inspect.signature(
        AgglomerativeClustering
    ).parameters else "affinity"
    parameters[metric_parameter] = metric
    model = AgglomerativeClustering(**parameters)
    return model.fit_predict(X)