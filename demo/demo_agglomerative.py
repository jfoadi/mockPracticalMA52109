###
## cluster_maker: demo of agglomerative clustering
## Francesco Gallo - University of Bath
## November 2026
##
## This script shows the agglomerative module working on
## 'data/difficult_dataset.csv', which contains four noisy concentric rings.
## It first shows that clustering the raw (x, y) coordinates fails, both
## for k-means and for every agglomerative linkage, and then that
## agglomerative clustering on each point's distance from the centre
## separates the four rings. Plots are saved in demo_output/agglomerative/.
##
## Usage (from the repository root):
##     python demo/demo_agglomerative.py
###

from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import (
    calculate_descriptive_statistics,
    select_features,
    standardise_features,
    sklearn_kmeans,
    agglomerative_clustering,
    silhouette_score_sklearn,
    plot_clusters_2d,
    export_to_csv,
)

DATA_PATH = os.path.join("data", "difficult_dataset.csv")
OUTPUT_DIR = os.path.join("demo_output", "agglomerative")
K_VALUES = list(range(2, 9))
LINKAGES = ["ward", "complete", "average", "single"]
N_CLUSTERS_RAW = 4  # number of rings visible in the raw scatter plot
RANDOM_STATE = 42


def radial_distance(X: np.ndarray) -> np.ndarray:
    """
    Distance of each point from the centre (mean) of the data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)

    Returns
    -------
    r : ndarray of shape (n_samples, 1)
        Kept 2D so it can be passed straight to the clustering functions.
    """
    centre = X.mean(axis=0)
    return np.linalg.norm(X - centre, axis=1).reshape(-1, 1)


def order_labels_by_radius(labels: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Renumber cluster labels so that cluster 0 is the innermost ring,
    cluster 1 the next one out, and so on.
    """
    cluster_ids = np.unique(labels)
    mean_radius = [r[labels == c].mean() for c in cluster_ids]
    new_id = {old: new for new, old in enumerate(cluster_ids[np.argsort(mean_radius)])}
    return np.array([new_id[label] for label in labels])


def print_cluster_summary(name: str, labels: np.ndarray, r: np.ndarray) -> None:
    """
    Print the size and radius range of each cluster. A cluster that is one
    ring covers a narrow range of radii; a cluster that mixes rings covers
    almost the full range.
    """
    print(f"\n{name}")
    print("  cluster   size   radius range")
    for c in np.unique(labels):
        r_c = r[labels == c].ravel()
        print(f"  {c:7d} {len(r_c):6d}   {r_c.min():.2f} - {r_c.max():.2f}")


def plot_raw_data(X: np.ndarray, r: np.ndarray, save_path: str) -> None:
    """
    Save a scatter plot of the raw data next to a histogram of the
    distance of each point from the centre.
    """
    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(11, 5))

    ax_scatter.scatter(X[:, 0], X[:, 1], s=6, color="grey")
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xlabel("x")
    ax_scatter.set_ylabel("y")
    ax_scatter.set_title("Raw data")

    ax_hist.hist(r.ravel(), bins=80, color="grey")
    ax_hist.set_xlabel("Distance from centre")
    ax_hist.set_ylabel("Number of points")
    ax_hist.set_title("Distance from centre: one peak per ring")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_linkage_comparison(
    X: np.ndarray,
    results: Dict[str, np.ndarray],
    save_path: str,
) -> None:
    """
    Save a 2x2 grid of scatter plots, one per linkage, coloured by the
    clusters found on the raw (x, y) coordinates.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (linkage, labels) in zip(axes.ravel(), results.items()):
        ax.scatter(X[:, 0], X[:, 1], s=6, c=labels, cmap="tab10")
        ax.set_aspect("equal")
        ax.set_title(f"Agglomerative on (x, y), linkage = '{linkage}'")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle("Clustering the raw coordinates does not separate the rings")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_silhouettes(silhouettes: Dict[int, float], best_k: int, save_path: str) -> None:
    """
    Save a bar chart of silhouette score against k, highlighting best_k.
    """
    k_values = list(silhouettes.keys())
    colours = ["tab:orange" if k == best_k else "tab:blue" for k in k_values]

    fig, ax = plt.subplots()
    ax.bar(k_values, list(silhouettes.values()), color=colours)
    ax.set_xticks(k_values)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title(f"Agglomerative on distance from centre (best: k = {best_k})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_radius_by_cluster(r: np.ndarray, labels: np.ndarray, save_path: str) -> None:
    """
    Save a histogram of distance from centre, with one colour per cluster.
    """
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab10")
    for c in np.unique(labels):
        ax.hist(r[labels == c].ravel(), bins=30, color=cmap(c), label=f"Cluster {c}")
    ax.set_xlabel("Distance from centre")
    ax.set_ylabel("Number of points")
    ax.set_title("Each cluster covers one ring")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """
    Run the agglomerative clustering demo on difficult_dataset.csv and
    save all outputs.
    """
    if not os.path.exists(DATA_PATH):
        print(f"Error: data file not found: {DATA_PATH}")
        print("Run this script from the repository root:")
        print("    python demo/demo_agglomerative.py")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: load the data and look at its shape
    df = pd.read_csv(DATA_PATH)
    feature_cols = list(df.columns)
    X = select_features(df, feature_cols).to_numpy(dtype=float)
    r = radial_distance(X)

    print(f"Loaded {DATA_PATH}: {df.shape[0]} rows, {df.shape[1]} features")
    print("\nDescriptive statistics:")
    print(calculate_descriptive_statistics(df).round(2))
    plot_raw_data(X, r, os.path.join(OUTPUT_DIR, "1_raw_data.png"))
    print("\nThe scatter plot shows four concentric rings, and the histogram of")
    print("distance from the centre has four separate peaks.")

    # Step 2: cluster the raw (x, y) coordinates. The gap between two
    # neighbouring rings is smaller than the gaps between points along the
    # same (noisy) ring, so methods based on (x, y) distances mix the rings.
    X_std = standardise_features(X)
    print("\n--- Step 2: clustering the raw (x, y) coordinates ---")

    km_labels, km_centroids = sklearn_kmeans(X_std, N_CLUSTERS_RAW, random_state=RANDOM_STATE)
    fig_km, _ = plot_clusters_2d(
        X_std, km_labels, centroids=km_centroids,
        title="k-means on (x, y), k = 4",
    )
    fig_km.savefig(os.path.join(OUTPUT_DIR, "2_kmeans_raw.png"), dpi=150)
    plt.close(fig_km)
    print_cluster_summary("k-means on (x, y):", km_labels, r)

    raw_results: Dict[str, np.ndarray] = {}
    for linkage in LINKAGES:
        labels, _ = agglomerative_clustering(X_std, N_CLUSTERS_RAW, linkage=linkage)
        raw_results[linkage] = labels
        print_cluster_summary(f"Agglomerative on (x, y), linkage = '{linkage}':", labels, r)
    plot_linkage_comparison(X, raw_results, os.path.join(OUTPUT_DIR, "3_agglomerative_raw.png"))
    print("\nEvery method above produces clusters that span several rings")
    print("(wide radius ranges), so none of them separates the rings.")

    # Step 3: cluster on distance from the centre, choosing k by silhouette
    print("\n--- Step 3: clustering on distance from the centre ---")
    silhouettes: Dict[int, float] = {}
    for k in K_VALUES:
        labels, _ = agglomerative_clustering(r, k, linkage="ward")
        silhouettes[k] = silhouette_score_sklearn(r, labels)
    best_k = max(silhouettes, key=silhouettes.get)

    print("  k   silhouette")
    for k in K_VALUES:
        marker = "  <- best" if k == best_k else ""
        print(f"{k:3d} {silhouettes[k]:12.3f}{marker}")
    plot_silhouettes(silhouettes, best_k, os.path.join(OUTPUT_DIR, "4_silhouette_scores.png"))

    # Step 4: final clustering with the chosen k
    labels, _ = agglomerative_clustering(r, best_k, linkage="ward")
    labels = order_labels_by_radius(labels, r)

    fig_final, _ = plot_clusters_2d(
        X, labels,
        title=f"Agglomerative (ward) on distance from centre, k = {best_k}",
    )
    fig_final.savefig(os.path.join(OUTPUT_DIR, "5_agglomerative_rings.png"), dpi=150)
    plt.close(fig_final)
    plot_radius_by_cluster(r, labels, os.path.join(OUTPUT_DIR, "6_radius_by_cluster.png"))

    df_out = df.copy()
    df_out["cluster"] = labels
    export_to_csv(df_out, os.path.join(OUTPUT_DIR, "difficult_dataset_clustered.csv"))

    # Step 5: summary
    print(f"\nChosen number of clusters: k = {best_k} "
          f"(silhouette {silhouettes[best_k]:.3f})")
    print_cluster_summary("Agglomerative (ward) on distance from centre:", labels, r)
    print("\nEach cluster covers its own narrow, non-overlapping range of")
    print("radii, so each cluster is exactly one ring.")
    print(f"\nPlots and labelled data saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()