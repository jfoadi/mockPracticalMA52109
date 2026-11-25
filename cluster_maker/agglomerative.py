###
## cluster_maker
## Agglomerative (hierarchical) clustering
###

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .algorithms import update_centroids
from .evaluation import silhouette_score_sklearn


# Linkage criteria accepted by this module. All of them work with Euclidean
# distances, which is what the rest of the package uses.
LINKAGES = ("ward", "complete", "average", "single")


def agglomerative_clustering(
    X: np.ndarray,
    k: int,
    linkage: str = "ward",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hierarchical (agglomerative) clustering with scikit-learn's
    AgglomerativeClustering.

    Every sample starts as a cluster of its own, and the two closest clusters
    are merged, step after step, until k clusters are left. How "close" two
    clusters are is decided by the linkage criterion.

    Unlike k-means, the algorithm uses no random numbers: the same data always
    give the same result.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int
        Number of clusters, between 1 and n_samples.
    linkage : {"ward", "complete", "average", "single"}, default "ward"
        Criterion to decide which clusters are merged.
        - "ward" merges the pair that increases the within-cluster variance the
          least. It gives compact clusters of similar size and is a safe default.
        - "complete" and "average" use the largest / the mean distance between
          the points of two clusters.
        - "single" uses the smallest distance. It can follow elongated shapes,
          but a few points between two groups are enough to chain them together,
          so it is sensitive to noise.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
        Mean of the points of each cluster. The algorithm itself has no
        centroids; they are computed so that the output has the same form as
        kmeans() and sklearn_kmeans(), and works with compute_inertia(),
        plot_clusters_2d() and the other functions of the package.

    Raises
    ------
    TypeError
        If X is not a NumPy array or k is not an integer.
    ValueError
        If X is not 2D, contains NaN or infinite values, k is not between 1
        and the number of samples, or the linkage is not recognised.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if not np.isfinite(X).all():
        raise ValueError("X must not contain NaN or infinite values.")
    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > X.shape[0]:
        raise ValueError("k cannot be larger than the number of samples.")
    if linkage not in LINKAGES:
        raise ValueError(f"linkage must be one of {LINKAGES}, not '{linkage}'.")

    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    model.fit(X)
    labels = model.labels_

    # Centroids = cluster means (no cluster is empty, so no random choice is made)
    centroids = update_centroids(X, labels, k)
    return labels, centroids


def compare_linkages(
    X: np.ndarray,
    k: int,
    linkages: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Silhouette score of the k-cluster solution for each linkage criterion.

    A higher score means better separated clusters, so the result helps to
    choose the linkage.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int
        Number of clusters, at least 2 (the silhouette needs 2 clusters).
    linkages : sequence of str or None, default None
        Linkage criteria to compare. If None, all of LINKAGES are used.

    Returns
    -------
    scores : dict
        Mapping from each linkage name to its silhouette score.
    """
    if k < 2:
        raise ValueError("k must be at least 2 to compute a silhouette score.")
    if linkages is None:
        linkages = LINKAGES

    scores: Dict[str, float] = {}
    for linkage in linkages:
        labels, _ = agglomerative_clustering(X, k, linkage=linkage)
        scores[linkage] = silhouette_score_sklearn(X, labels)
    return scores
