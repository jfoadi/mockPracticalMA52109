###
## Agglomerative Data
## Katherine Hall - University of Bath
## November 25 2026
###

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from .algorithms import update_centroids
from .evaluation import silhouette_score_sklearn

VALID_LINKAGES = ("ward", "complete", "average", "single")


def _validate_inputs(
    X: np.ndarray,
    k: Optional[int],
    linkage: str,
    metric: str,
    n_neighbors: Optional[int],
) -> None:
    """
    Validate the arguments shared by the public functions in this module.

    Raises
    ------
    TypeError
        If X is not a NumPy array or k is not an integer.
    ValueError
        If X is not a finite 2D array, k is out of range, the linkage is
        unknown, Ward linkage is combined with a non-Euclidean metric, or
        n_neighbors is out of range.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must not contain NaN or infinite values.")

    n_samples = X.shape[0]
    if k is not None:
        if not isinstance(k, (int, np.integer)):
            raise TypeError("k must be an integer.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if k > n_samples:
            raise ValueError("k cannot be larger than the number of samples.")

    if linkage not in VALID_LINKAGES:
        raise ValueError(
            f"Unknown linkage '{linkage}'. Use one of {list(VALID_LINKAGES)}."
        )
    if linkage == "ward" and metric != "euclidean":
        raise ValueError("Ward linkage requires metric='euclidean'.")

    if n_neighbors is not None and not (1 <= n_neighbors < n_samples):
        raise ValueError("n_neighbors must satisfy 1 <= n_neighbors < n_samples.")


def _build_connectivity(X: np.ndarray, n_neighbors: Optional[int]):
    """
    Build a symmetric k-nearest-neighbour connectivity graph, or None.

    Restricting merges to neighbouring points makes hierarchical clustering
    follow the local shape of the data (e.g. curved or ring-shaped clusters).
    """
    if n_neighbors is None:
        return None
    graph = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=False)
    return 0.5 * (graph + graph.T)


def _fit_agglomerative(
    X: np.ndarray,
    k: Optional[int],
    linkage: str,
    metric: str,
    n_neighbors: Optional[int],
    **extra,
) -> AgglomerativeClustering:
    """
    Fit scikit-learn's AgglomerativeClustering.

    If the connectivity graph has several connected components, scikit-learn
    completes it automatically and emits a warning. That behaviour is what we
    want here, so only that specific warning is silenced.
    """
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage,
        metric=metric,
        connectivity=_build_connectivity(X, n_neighbors),
        **extra,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="the number of connected components"
        )
        model.fit(X)
    return model


def _detect_outliers(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    """
    Flag points that are unusually far from their neighbours.

    A point is an outlier if its distance to its n_neighbors-th nearest
    neighbour is above Q3 + 1.5 * IQR of those distances (Tukey's rule).

    Returns
    -------
    is_outlier : boolean ndarray of shape (n_samples,)
    """
    n_neighbors = min(n_neighbors, X.shape[0] - 1)
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X).kneighbors(X)
    score = distances[:, -1]
    q1, q3 = np.percentile(score, [25, 75])
    return score > q3 + 1.5 * (q3 - q1)


def agglomerative_clustering(
    X: np.ndarray,
    k: int,
    linkage: str = "ward",
    metric: str = "euclidean",
    n_neighbors: Optional[int] = None,
    handle_outliers: bool = False,
    outlier_neighbors: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hierarchical clustering using scikit-learn's AgglomerativeClustering.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int
        Number of clusters.
    linkage : {"ward", "complete", "average", "single"}, default "ward"
    metric : str, default "euclidean"
        Distance metric. Ward linkage only supports "euclidean".
    n_neighbors : int or None, default None
        If given, merges are restricted to a symmetric k-nearest-neighbour
        graph, which helps with non-convex clusters.
    handle_outliers : bool, default False
        If True, isolated points are set aside before clustering (see
        `_detect_outliers`) and afterwards given the label of their nearest
        clustered point. Without this, linkages such as "single" and
        "average" tend to waste clusters on individual noise points.
    outlier_neighbors : int, default 10
        Neighbourhood size used to detect outliers.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
        Mean of the points in each cluster. Agglomerative clustering has no
        native centroids; they are computed so that the output has the same
        form as `kmeans` and `sklearn_kmeans`.
    """
    _validate_inputs(X, k, linkage, metric, n_neighbors)

    if not handle_outliers:
        labels = _fit_agglomerative(X, k, linkage, metric, n_neighbors).labels_
        return labels, update_centroids(X, labels, k)

    if outlier_neighbors < 1:
        raise ValueError("outlier_neighbors must be a positive integer.")

    is_outlier = _detect_outliers(X, outlier_neighbors)
    X_core = X[~is_outlier]
    if X_core.shape[0] < max(k, 2):
        raise ValueError(
            "Too few non-outlier points to form k clusters; "
            "use handle_outliers=False or a smaller k."
        )
    if n_neighbors is not None and n_neighbors >= X_core.shape[0]:
        raise ValueError("n_neighbors is too large once outliers are removed.")

    labels = np.empty(X.shape[0], dtype=int)
    labels[~is_outlier] = _fit_agglomerative(
        X_core, k, linkage, metric, n_neighbors
    ).labels_

    if np.any(is_outlier):
        nearest = NearestNeighbors(n_neighbors=1).fit(X_core)
        _, idx = nearest.kneighbors(X[is_outlier])
        labels[is_outlier] = labels[~is_outlier][idx[:, 0]]

    return labels, update_centroids(X, labels, k)


def merge_distances(
    X: np.ndarray,
    linkage: str = "ward",
    metric: str = "euclidean",
    n_neighbors: Optional[int] = None,
) -> np.ndarray:
    """
    Return the distances at which clusters are merged, in increasing order.

    The last values are the merges between the largest groups. A large jump
    near the end suggests that two well-separated groups are being forced
    together, which is useful evidence when choosing k (plot these values).

    Returns
    -------
    distances : ndarray of shape (n_samples - 1,)
    """
    _validate_inputs(X, None, linkage, metric, n_neighbors)
    if X.shape[0] < 2:
        raise ValueError("At least 2 samples are needed.")
    model = _fit_agglomerative(
        X, None, linkage, metric, n_neighbors,
        distance_threshold=0, compute_distances=True,
    )
    return np.sort(model.distances_)


def compare_linkages(
    X: np.ndarray,
    k: int,
    linkages: Optional[List[str]] = None,
    metric: str = "euclidean",
    n_neighbors: Optional[int] = None,
    handle_outliers: bool = False,
    outlier_neighbors: int = 10,
) -> Dict[str, Dict[str, object]]:
    """
    Run agglomerative clustering with several linkage criteria.

    Returns
    -------
    results : dict
        Maps each linkage name to a dict with keys "labels", "centroids",
        "silhouette" (float, or None if it cannot be computed) and "sizes"
        (number of points in each cluster).
    """
    if linkages is None:
        linkages = list(VALID_LINKAGES)

    results: Dict[str, Dict[str, object]] = {}
    for linkage in linkages:
        labels, centroids = agglomerative_clustering(
            X, k, linkage=linkage, metric=metric, n_neighbors=n_neighbors,
            handle_outliers=handle_outliers, outlier_neighbors=outlier_neighbors,
        )
        try:
            sil: Optional[float] = silhouette_score_sklearn(X, labels)
        except ValueError:
            sil = None
        results[linkage] = {
            "labels": labels,
            "centroids": centroids,
            "silhouette": sil,
            "sizes": np.bincount(labels, minlength=k),
        }
    return results