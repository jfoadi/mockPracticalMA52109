###
## cluster_maker: cluster analysis of simulated_data.csv
## Francesco Gallo - University of Bath
## November 2026
##
## This script uses the cluster_maker package to find a number
## of clusters in 'data/simulated_data.csv'. It compares k = 2..10 using
## the elbow method and the silhouette score, clusters the data with the
## best k, and saves plots, in demo_output subdirectory, that show why that choice is appropriate.
##
## Usage (from the repository root):
##     python demo/simulated_clustering.py
###

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import (
    calculate_descriptive_statistics,
    select_features,
    standardise_features,
    sklearn_kmeans,
    elbow_curve,
    silhouette_score_sklearn,
    plot_elbow,
    plot_clusters_2d,
    export_to_csv,
)

DATA_PATH = os.path.join("data", "simulated_data.csv")
OUTPUT_DIR = os.path.join("demo_output", "simulated_clustering")
K_VALUES = list(range(2, 11))
RANDOM_STATE = 42


def plot_feature_pairs(
    X: np.ndarray,
    feature_names: List[str],
    labels: Optional[np.ndarray],
    title: str,
    save_path: str,
) -> None:
    """
    Save a grid of scatter plots, one for every pair of features.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    feature_names : list of str
        Names used to label the axes.
    labels : ndarray of shape (n_samples,) or None
        Cluster labels used to colour the points. If None, all points
        are drawn in one colour.
    title : str
    save_path : str
        File path for the saved figure.
    """
    pairs = [
        (i, j)
        for i in range(len(feature_names))
        for j in range(i + 1, len(feature_names))
    ]
    n_cols = 3
    n_rows = int(np.ceil(len(pairs) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (i, j) in zip(axes, pairs):
        if labels is None:
            ax.scatter(X[:, i], X[:, j], s=10, color="grey", alpha=0.7)
        else:
            ax.scatter(X[:, i], X[:, j], s=10, c=labels, cmap="tab10", alpha=0.8)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])

    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_silhouettes(X: np.ndarray, k_values: List[int]) -> Dict[int, float]:
    """
    Cluster X with scikit-learn k-means for each k and return the
    silhouette score of each clustering.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k_values : list of int
        Values of k to try (each must be at least 2).

    Returns
    -------
    silhouettes : dict
        Mapping from k to silhouette score.
    """
    silhouettes: Dict[int, float] = {}
    for k in k_values:
        labels, _ = sklearn_kmeans(X, k, random_state=RANDOM_STATE)
        silhouettes[k] = silhouette_score_sklearn(X, labels)
    return silhouettes


def plot_silhouettes(silhouettes: Dict[int, float], best_k: int, save_path: str) -> None:
    """
    Save a bar chart of silhouette score against k, highlighting best_k.
    """
    k_values = list(silhouettes.keys())
    scores = list(silhouettes.values())
    colours = ["tab:orange" if k == best_k else "tab:blue" for k in k_values]

    fig, ax = plt.subplots()
    ax.bar(k_values, scores, color=colours)
    ax.set_xticks(k_values)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title(f"Silhouette score for different k (best: k = {best_k})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Error: data file not found: {DATA_PATH}")
        print("Run this script from the repository root:")
        print("    python demo/simulated_clustering.py")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: load and describe the data
    df = pd.read_csv(DATA_PATH)
    feature_cols = list(df.columns)
    print(f"Loaded {DATA_PATH}: {df.shape[0]} rows, {df.shape[1]} features")
    print("\nDescriptive statistics:")
    print(calculate_descriptive_statistics(df).round(2))

    X_raw = select_features(df, feature_cols).to_numpy(dtype=float)
    X = standardise_features(X_raw)

    # Step 2: look at the raw structure before any clustering
    plot_feature_pairs(
        X_raw,
        feature_cols,
        labels=None,
        title="Raw data: every pair of features",
        save_path=os.path.join(OUTPUT_DIR, "1_raw_feature_pairs.png"),
    )

    # Step 3: compare k using the elbow method and the silhouette score.
    # sklearn_kmeans is used because it tries 10 initialisations and keeps
    # the best, whereas the manual kmeans uses one and can get stuck in a
    # poor solution on this dataset.
    inertias = elbow_curve(X, [1] + K_VALUES, random_state=RANDOM_STATE, use_sklearn=True)
    silhouettes = compute_silhouettes(X, K_VALUES)
    best_k = max(silhouettes, key=silhouettes.get)

    print("\n  k   inertia   silhouette")
    for k in K_VALUES:
        marker = "  <- best" if k == best_k else ""
        print(f"{k:3d} {inertias[k]:9.1f} {silhouettes[k]:12.3f}{marker}")

    fig_elbow, _ = plot_elbow(
        [1] + K_VALUES,
        [inertias[k] for k in [1] + K_VALUES],
        title="Elbow curve (standardised features)",
    )
    fig_elbow.savefig(os.path.join(OUTPUT_DIR, "2_elbow_curve.png"), dpi=150)
    plt.close(fig_elbow)

    plot_silhouettes(silhouettes, best_k, os.path.join(OUTPUT_DIR, "3_silhouette_scores.png"))

    # Step 4: final clustering with the chosen k
    labels, centroids = sklearn_kmeans(X, best_k, random_state=RANDOM_STATE)

    fig_clusters, _ = plot_clusters_2d(
        X, labels, centroids=centroids,
        title=f"k-means clusters, k = {best_k} (first two standardised features)",
    )
    fig_clusters.savefig(os.path.join(OUTPUT_DIR, "4_clusters_2d.png"), dpi=150)
    plt.close(fig_clusters)

    plot_feature_pairs(
        X_raw,
        feature_cols,
        labels=labels,
        title=f"Raw data coloured by cluster (k = {best_k})",
        save_path=os.path.join(OUTPUT_DIR, "5_clustered_feature_pairs.png"),
    )

    df_out = df.copy()
    df_out["cluster"] = labels
    export_to_csv(df_out, os.path.join(OUTPUT_DIR, "simulated_data_clustered.csv"))

    # Step 5: summary
    print(f"\nChosen number of clusters: k = {best_k}")
    print(f"  Silhouette score: {silhouettes[best_k]:.3f} (highest of k = 2..10)")
    print(f"  Inertia falls from {inertias[best_k - 1]:.1f} at k = {best_k - 1} "
          f"to {inertias[best_k]:.1f} at k = {best_k}, "
          f"then only to {inertias[best_k + 1]:.1f} at k = {best_k + 1} (elbow)")
    print("\nCluster sizes:")
    print(pd.Series(labels).value_counts().sort_index().to_string())
    print("\nCluster centres (original units):")
    print(df_out.groupby("cluster")[feature_cols].mean().round(2))
    print(f"\nPlots and labelled data saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()