###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###
"""
Clustering algorithms for synthetic and numerical datasets.

This module provides a manual implementation of K-means clustering
and a wrapper around scikit-learn's KMeans algorithm.

The manual implementation separates centroid initialisation,
cluster assignment and centroid updates into reusable functions.

Both implementations return cluster labels and centroid coordinates,
which can be used for subsequent evaluation and visualisation.
"""


from __future__ import annotations

from typing import Tuple, Optional

import numpy as np




def init_centroids(
    X: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Initialise k centroids by sampling data points without replacement.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input dataset containing numerical observations.
    k : int
        Number of centroids to initialise.
    random_state : int or None, optional
        Random seed for reproducible centroid selection.

    Returns
    -------
    np.ndarray of shape (k, n_features)
        Initial centroid coordinates selected from the input dataset X.

    Raises
    ------
    TypeError
        If X is not a NumPy array or k is not an integer.
    ValueError
        If X is empty or not two-dimensional, contains invalid numerical
        values, k is non-positive or exceeds the number of samples,
        or random sampling fails.
    """

    # Ensure that X is a non-empty two-dimensional NumPy array.
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    if X.ndim != 2 or X.size == 0:
        raise ValueError("X must be a non-empty two-dimensional array.")

    # Ensure that the dataset contains finite real numerical values.
    if X.dtype.kind not in "iuf" or not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite real numerical values.")

    # Validate the requested number of clusters.
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be an integer.")

    if k <= 0:
        raise ValueError("k must be a positive integer.")

    n_samples = X.shape[0]

    if k > n_samples:
        raise ValueError("k cannot be larger than the number of samples.")

    try:
        # Initialise the random number generator for reproducible sampling.
        rng = np.random.RandomState(random_state)

        # Randomly select k distinct row indices without replacement.
        indices = rng.choice(n_samples, size=k, replace=False)

        # Return the selected observations as initial centroid coordinates.
        return X[indices]

    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to initialise centroids: {e}") from e


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each sample to its nearest centroid using Euclidean distance.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input dataset containing numerical observations.
    centroids : np.ndarray of shape (k, n_features)
        Coordinates of the k cluster centroids.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Integer cluster labels representing the index of the nearest
        centroid for each observation in X.

    Raises
    ------
    TypeError
        If X or centroids are not NumPy arrays.
    ValueError
        If either array is empty, not two-dimensional, contains invalid
        numerical values, or has incompatible feature dimensions.
    """

    # Ensure that both inputs are NumPy arrays.
    if not isinstance(X, np.ndarray) or not isinstance(centroids, np.ndarray):
        raise TypeError("X and centroids must be NumPy arrays.")

    # Validate the dimensions and size of both arrays.
    if X.ndim != 2 or centroids.ndim != 2:
        raise ValueError("X and centroids must be two-dimensional arrays.")

    if X.size == 0 or centroids.size == 0:
        raise ValueError("X and centroids must not be empty.")

    # Ensure that observations and centroids have matching feature dimensions.
    if X.shape[1] != centroids.shape[1]:
        raise ValueError("X and centroids must have the same number of features.")

    # Ensure that both arrays contain finite real numerical values.
    if X.dtype.kind not in "iuf" or centroids.dtype.kind not in "iuf":
        raise ValueError("X and centroids must contain real numerical values.")

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(centroids)):
        raise ValueError("X and centroids must not contain NaN or infinite values.")

    try:
        # Convert to floating-point arrays to avoid integer arithmetic overflow.
        X = X.astype(float, copy=False)
        centroids = centroids.astype(float, copy=False)

        # Use broadcasting to calculate feature-wise differences
        # between every sample and every centroid.
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]

        # Calculate Euclidean distances along the feature axis.
        distances = np.linalg.norm(diff, axis=2)

        # Identify the index of the nearest centroid for each observation.
        labels = np.argmin(distances, axis=1)

        # Return one cluster label per observation.
        return labels

    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to assign clusters: {e}") from e

def update_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Update centroids by calculating the mean of observations in each cluster.

    If a cluster becomes empty, its centroid is re-initialised by
    randomly selecting an observation from the input dataset.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input dataset containing numerical observations.
    labels : np.ndarray of shape (n_samples,)
        Integer cluster assignment for each observation.
    k : int
        Number of clusters.
    random_state : int or None, optional
        Random seed for reproducible centroid re-initialisation.

    Returns
    -------
    np.ndarray of shape (k, n_features)
        Updated centroid coordinates for all k clusters.

    Raises
    ------
    TypeError
        If X or labels are not NumPy arrays, k is not an integer,
        or labels contain non-integer values.
    ValueError
        If X contains invalid numerical values, the input dimensions
        are incorrect, k is outside the valid range, cluster labels
        are invalid, or centroid updating fails.
    """

    # Ensure that both inputs are NumPy arrays.
    if not isinstance(X, np.ndarray) or not isinstance(labels, np.ndarray):
        raise TypeError("X and labels must be NumPy arrays.")

    # Validate the dataset dimensions and numerical values.
    if X.ndim != 2 or X.size == 0:
        raise ValueError("X must be a non-empty two-dimensional array.")

    if X.dtype.kind not in "iuf" or not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite real numerical values.")

    # Validate the requested number of clusters.
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be an integer.")

    if k <= 0 or k > X.shape[0]:
        raise ValueError("k must be between 1 and the number of samples.")

    # Ensure that each observation has exactly one cluster label.
    if labels.ndim != 1 or len(labels) != X.shape[0]:
        raise ValueError("There must be exactly one cluster label per sample.")

    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Cluster labels must be integers.")

    if np.any(labels < 0) or np.any(labels >= k):
        raise ValueError(f"Cluster labels must be between 0 and {k - 1}.")

    try:
        # Allocate an array to store the updated centroid coordinates.
        n_features = X.shape[1]
        new_centroids = np.zeros((k, n_features), dtype=float)

        rng = np.random.RandomState(random_state)

        for cluster_id in range(k):

            # Identify observations belonging to the current cluster.
            mask = labels == cluster_id

            if not np.any(mask):
                # Recover from an empty cluster by selecting a random sample.
                idx = rng.randint(0, X.shape[0])
                new_centroids[cluster_id] = X[idx]

            else:
                # Calculate the mean of the observations along each feature.
                new_centroids[cluster_id] = X[mask].mean(axis=0)

        # Return the updated centroid coordinates for all clusters.
        return new_centroids

    except (ValueError, TypeError, IndexError) as e:
        raise ValueError(f"Failed to update centroids: {e}") from e

def kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform K-means clustering using a manual NumPy implementation.

    The algorithm repeatedly assigns observations to their nearest
    centroids and updates the centroid coordinates. Iteration stops
    when centroid movement falls below the specified tolerance or
    the maximum number of iterations is reached.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input dataset containing numerical observations.
    k : int
        Number of clusters.
    max_iter : int, default=300
        Maximum number of clustering iterations.
    tol : float, default=1e-4
        Convergence tolerance for changes in centroid coordinates.
    random_state : int or None, optional
        Random seed for reproducible centroid initialisation
        and empty-cluster handling.

    Returns
    -------
    labels : np.ndarray of shape (n_samples,)
        Final cluster assignment for each observation.
    centroids : np.ndarray of shape (k, n_features)
        Final centroid coordinates for all k clusters.

    Raises
    ------
    TypeError
        If X is not a NumPy array, max_iter is not an integer,
        or tol is not numerical.
    ValueError
        If max_iter or tol is invalid, the dataset or number of
        clusters is invalid, or the clustering process fails.
    """

    # Validate the input dataset type.
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    # Ensure that the maximum iteration count is a positive integer.
    if not isinstance(max_iter, (int, np.integer)) or isinstance(
        max_iter, (bool, np.bool_)
    ):
        raise TypeError("max_iter must be an integer.")

    if max_iter <= 0:
        raise ValueError("max_iter must be greater than zero.")

    # Validate the convergence tolerance.
    if not isinstance(tol, (int, float, np.integer, np.floating)) or isinstance(
        tol, (bool, np.bool_)
    ):
        raise TypeError("tol must be a numerical value.")

    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and greater than zero.")

    try:
        # Initialise centroids using randomly selected observations.
        centroids = init_centroids(X, k, random_state=random_state)

        for _ in range(max_iter):

            # Assign each observation to its nearest centroid.
            labels = assign_clusters(X, centroids)

            # Recalculate centroid positions using cluster means.
            new_centroids = update_centroids(
                X, labels, k, random_state=random_state
            )

            # Measure the overall movement of the centroids.
            shift = np.linalg.norm(new_centroids - centroids)

            centroids = new_centroids

            # Stop iterating once the convergence tolerance is satisfied.
            if shift < tol:
                break

        # Recalculate cluster assignments using the final centroids.
        labels = assign_clusters(X, centroids)

        # Return the final cluster assignments and centroid coordinates.
        return labels, centroids

    except (ValueError, TypeError) as e:
        raise ValueError(f"Manual K-means clustering failed: {e}") from e


def sklearn_kmeans(
    X: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform K-means clustering using scikit-learn's KMeans estimator.

    This wrapper provides a consistent interface with the manual
    K-means implementation. Ten independent initialisations are
    performed to reduce sensitivity to initial centroid positions.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input dataset containing numerical observations.
    k : int
        Number of clusters.
    random_state : int or None, optional
        Random seed for reproducible clustering results.

    Returns
    -------
    labels : np.ndarray of shape (n_samples,)
        Cluster assignment for each observation.
    centroids : np.ndarray of shape (k, n_features)
        Final centroid coordinates for all k clusters.

    Raises
    ------
    TypeError
        If X is not a NumPy array or k is not an integer.
    ValueError
        If the dataset contains invalid numerical values,
        k is outside the valid range, or clustering fails.
    ImportError
        If the scikit-learn dependency is unavailable.
    """

    # Ensure that X is a non-empty two-dimensional NumPy array.
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    if X.ndim != 2 or X.size == 0:
        raise ValueError("X must be a non-empty two-dimensional array.")

    # Ensure that the dataset contains finite real numerical values.
    if X.dtype.kind not in "iuf" or not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite real numerical values.")

    # Validate the requested number of clusters.
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be an integer.")

    if k <= 0 or k > X.shape[0]:
        raise ValueError("k must be between 1 and the number of samples.")

    try:
        # Import scikit-learn only when this function is called.
        from sklearn.cluster import KMeans

    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for sklearn_kmeans(). "
            "Install it using: python -m pip install scikit-learn"
        ) from exc

    try:
        # Configure K-means with ten centroid initialisations.
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )

        # Fit the clustering model to the input dataset.
        model.fit(X)

        # Extract the cluster assignments and final centroid coordinates.
        labels = model.labels_
        centroids = model.cluster_centers_

        return labels, centroids

    except (ValueError, TypeError) as e:
        raise ValueError(f"Scikit-learn K-means clustering failed: {e}") from e