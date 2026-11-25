###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###


from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
from sklearn.metrics import silhouette_score

from .algorithms import kmeans, sklearn_kmeans


def compute_inertia(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> float:
    """
    Compute the within-cluster sum of squared distances (inertia).
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a NumPy array.")

    if not isinstance(centroids, np.ndarray):
        raise TypeError("centroids must be a NumPy array.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")

    if centroids.ndim != 2:
        raise ValueError("centroids must be a 2D array.")

    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of samples.")

    if X.shape[1] != centroids.shape[1]:
        raise ValueError("X and centroids must have the same number of features.")

    if len(labels) == 0:
        raise ValueError("X and labels cannot be empty.")

    if len(centroids) == 0:
        raise ValueError("At least one centroid is required.")

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integers.")

    if np.any(labels < 0) or np.any(labels >= len(centroids)):
        raise ValueError("labels contain invalid cluster indices.")

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(centroids)):
        raise ValueError("X and centroids must contain finite values.")

    distances = X - centroids[labels]

    sq_dist = np.sum(distances ** 2)

    return float(sq_dist)


def silhouette_score_sklearn(
    X: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute the silhouette score using scikit-learn.
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a NumPy array.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")

    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of samples.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite values.")

    n_samples = X.shape[0]
    n_clusters = len(np.unique(labels))

    if n_clusters < 2:
        raise ValueError("Silhouette score requires at least 2 clusters.")

    if n_clusters >= n_samples:
        raise ValueError(
            "Number of clusters must be smaller than number of samples."
        )

    return float(silhouette_score(X, labels))


def elbow_curve(
    X: np.ndarray,
    k_values: List[int],
    random_state: Optional[int] = None,
    use_sklearn: bool = True,
) -> Dict[int, float]:
    """
    Compute inertia values for multiple K values (elbow method).
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must contain samples and features.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite values.")

    if not isinstance(k_values, (list, tuple)) or len(k_values) == 0:
        raise ValueError("k_values must be a non-empty list or tuple.")

    inertia_dict: Dict[int, float] = {}

    for k in k_values:

        if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
            raise TypeError("All k values must be integers.")

        if k <= 0 or k > X.shape[0]:
            raise ValueError(
                "All k values must be between 1 and the number of samples."
            )

        if use_sklearn:
            labels, centroids = sklearn_kmeans(
                X,
                k,
                random_state=random_state
            )
        else:
            labels, centroids = kmeans(
                X,
                k,
                random_state=random_state
            )

        inertia = compute_inertia(X, labels, centroids)

        inertia_dict[k] = inertia

    return inertia_dict
