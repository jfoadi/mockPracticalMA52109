###
## cluster_maker: demo for the dataset 'simulated_data.csv'
##
## Goal: find a plausible separation into clusters of the data in
## 'data/simulated_data.csv', using only the clustering tools of the
## cluster_maker package (matplotlib is used for extra figures), and make
## clear WHY the chosen number of clusters is appropriate.
##
## Run from the project root with:
##     python demo/simulated_clustering.py
## (an optional argument gives a different input csv file).
##
## Outline:
##  1. Load and inspect the data
##  2. Look at the raw data, before any clustering
##  3. Choose k: elbow curve and silhouette score for several values of k
##  4. Check that the result is reliable (many random starts)
##  5. Final clustering: sizes, cluster means and plots
##  6. Summary: why this clustering is appropriate
##
## The analysis is printed first. The figures open together at the end.
###

from __future__ import annotations

import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import (
    calculate_correlation,
    calculate_descriptive_statistics,
    compute_inertia,
    elbow_curve,
    kmeans,
    plot_clusters_2d,
    plot_elbow,
    select_features,
    silhouette_score_sklearn,
    sklearn_kmeans,
    standardise_features,
)

INPUT_PATH = "data/simulated_data.csv"
RANDOM_STATE = 42
K_ELBOW = list(range(1, 9))        # k = 1, ..., 8 for the elbow curve
K_SILHOUETTE = list(range(2, 9))   # the silhouette needs at least 2 clusters
N_STARTS = 10                      # random starts for the reliability check


def plot_raw_pairs(df: pd.DataFrame) -> None:
    """Scatter plot of every pair of features (histograms on the diagonal)."""
    cols = list(df.columns)
    n = len(cols)
    plt.figure(figsize=(10, 10))
    for i in range(n):
        for j in range(n):
            ax = plt.subplot(n, n, i * n + j + 1)
            if i == j:
                ax.hist(df[cols[j]], bins=20, color="grey", edgecolor="white")
            else:
                ax.scatter(df[cols[j]], df[cols[i]], s=6, alpha=0.6)
            if i == n - 1:
                ax.set_xlabel(cols[j])
            if j == 0:
                ax.set_ylabel(cols[i])
    plt.tight_layout()


def plot_silhouette(silhouettes: dict, best_k: int) -> None:
    """Bar chart of the silhouette score for each k, best k in red."""
    ks = list(silhouettes.keys())
    colours = ["red" if k == best_k else "blue" for k in ks]
    plt.figure(figsize=(8, 5))
    plt.bar(ks, [silhouettes[k] for k in ks], color=colours)
    plt.title(f"Silhouette score for different k (best: k = {best_k})")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.grid(True)


def plot_all_pairs(df: pd.DataFrame, labels: np.ndarray) -> None:
    """All six pairs of the four features, coloured by cluster."""
    cols = list(df.columns)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    plt.figure(figsize=(13, 8))
    for p in range(len(pairs)):
        i, j = pairs[p]
        ax = plt.subplot(2, 3, p + 1)
        ax.scatter(df[cols[i]], df[cols[j]], c=labels, cmap="tab10",
                   s=12, alpha=0.8)
        ax.set_xlabel(cols[i])
        ax.set_ylabel(cols[j])
        ax.set_title(f"{cols[j]} against {cols[i]}")
        ax.grid(True)
    plt.tight_layout()


def cluster_sizes(labels: np.ndarray) -> List[int]:
    """Number of points in each cluster."""
    return [int(np.sum(labels == c)) for c in np.unique(labels)]


def main(args: List[str]) -> None:
    input_path = args[0] if args else INPUT_PATH

    # ------------------------------------------------------------------
    # 1. Load and inspect the data
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: file not found: {input_path}")
        print("Run the script from the project root folder.")
        sys.exit(1)

    feature_cols = list(df.columns)
    X_df = select_features(df, feature_cols)    # also checks they are numeric
    print(f"Data: {df.shape[0]} rows and {df.shape[1]} features: {feature_cols}")

    print("\nDescriptive statistics:")
    print(calculate_descriptive_statistics(df).round(2))
    print("\nCorrelations between features:")
    print(calculate_correlation(df).round(2))

    # The features have different scales: standardise before clustering,
    # otherwise the features with the largest spread would dominate.
    X = standardise_features(X_df.to_numpy(dtype=float))

    # ------------------------------------------------------------------
    # 2. The raw data
    # ------------------------------------------------------------------
    plot_raw_pairs(df)

    # ------------------------------------------------------------------
    # 3. Choose k: elbow curve and silhouette score
    # ------------------------------------------------------------------
    # scikit-learn k-means (several starts) is used for the decision; step 4
    # shows why it is preferred to the manual k-means of the package.
    inertias = elbow_curve(X, K_ELBOW, random_state=RANDOM_STATE, use_sklearn=True)
    plot_elbow(
        K_ELBOW,
        [inertias[k] for k in K_ELBOW],
        title="Elbow curve (standardised data)",
    )

    silhouettes = {}
    for k in K_SILHOUETTE:
        labels_k, _ = sklearn_kmeans(X, k, random_state=RANDOM_STATE)
        silhouettes[k] = silhouette_score_sklearn(X, labels_k)

    # The best k is the one with the highest silhouette score
    best_k = K_SILHOUETTE[0]
    for k in K_SILHOUETTE:
        if silhouettes[k] > silhouettes[best_k]:
            best_k = k
    plot_silhouette(silhouettes, best_k)

    print("\nChoosing the number of clusters:")
    for k in K_ELBOW:
        line = f"  k = {k}: inertia = {inertias[k]:.1f}"
        if k > 1:
            drop = 100 * (inertias[k - 1] - inertias[k]) / inertias[k - 1]
            line = line + f" (down {drop:.0f}% from k = {k - 1})"
        if k in silhouettes:
            line = line + f", silhouette = {silhouettes[k]:.3f}"
        print(line)

    # ------------------------------------------------------------------
    # 4. Is the result reliable? Repeat with different random starts
    # ------------------------------------------------------------------
    sk_inertias = []
    sk_silhouettes = []
    manual_inertias = []
    for seed in range(N_STARTS):
        labels_s, centroids_s = sklearn_kmeans(X, best_k, random_state=seed)
        sk_inertias.append(compute_inertia(X, labels_s, centroids_s))
        sk_silhouettes.append(silhouette_score_sklearn(X, labels_s))
        labels_m, centroids_m = kmeans(X, best_k, random_state=seed)
        manual_inertias.append(compute_inertia(X, labels_m, centroids_m))

    best_inertia = min(sk_inertias)
    n_reached = len([i for i in manual_inertias if i <= 1.01 * best_inertia])
    print(f"\nReliability of k = {best_k} over {N_STARTS} random starts:")
    print(f"  scikit-learn k-means: inertia {min(sk_inertias):.1f} to "
          f"{max(sk_inertias):.1f}, silhouette {min(sk_silhouettes):.3f} to "
          f"{max(sk_silhouettes):.3f}")
    print(f"  manual k-means: inertia {min(manual_inertias):.1f} to "
          f"{max(manual_inertias):.1f}; it reaches the best solution in "
          f"{n_reached} of {N_STARTS} starts")

    # ------------------------------------------------------------------
    # 5. Final clustering
    # ------------------------------------------------------------------
    labels, centroids = sklearn_kmeans(X, best_k, random_state=RANDOM_STATE)
    sizes = cluster_sizes(labels)
    print(f"\nFinal clustering with k = {best_k}:")
    print(f"  cluster sizes: {sizes}")
    print(f"  inertia: {compute_inertia(X, labels, centroids):.1f}")
    print(f"  silhouette score: {silhouette_score_sklearn(X, labels):.3f}")

    # Cluster means in the original units
    df_out = df.copy()
    df_out["cluster"] = labels
    means = df_out.groupby("cluster").mean().round(2)
    means["size"] = sizes
    print("\nCluster means (original units):")
    print(means)

    # What happens with one cluster fewer and one cluster more?
    print("\nCluster sizes with one cluster fewer or more:")
    for k in (best_k - 1, best_k + 1):
        labels_k, _ = sklearn_kmeans(X, k, random_state=RANDOM_STATE)
        print(f"  k = {k}: {cluster_sizes(labels_k)}")

    # Cluster plot from the package: the first two features
    fig, ax = plot_clusters_2d(
        X[:, :2],
        labels,
        centroids=centroids[:, :2],
        title=f"k-means with k = {best_k}",
    )
    ax.set_xlabel(f"{feature_cols[0]} (standardised)")
    ax.set_ylabel(f"{feature_cols[1]} (standardised)")

    # Supporting figure: the clusters seen in every pair of features
    plot_all_pairs(df, labels)

    # ------------------------------------------------------------------
    # 6. Summary: why this clustering is appropriate
    # ------------------------------------------------------------------
    print(f"\nSummary: the data separate into {best_k} clusters, because")
    print(f"  1. the silhouette score is highest for k = {best_k} "
          f"({silhouettes[best_k]:.3f});")
    drop_to_best = 100 * (inertias[best_k - 1] - inertias[best_k]) / inertias[best_k - 1]
    if best_k + 1 in inertias:
        drop_after = 100 * (inertias[best_k] - inertias[best_k + 1]) / inertias[best_k]
        print(f"  2. the inertia falls by {drop_to_best:.0f}% when going to "
              f"k = {best_k}, but only by {drop_after:.0f}% beyond it (elbow);")
    print(f"  3. the clusters have sizes {sizes}, and with one cluster fewer or "
          f"more the groups are merged or split (see above);")
    if max(sk_inertias) <= 1.01 * min(sk_inertias):
        print(f"  4. the solution is stable: all {N_STARTS} random starts of "
              f"scikit-learn k-means give the same inertia.")
    else:
        print(f"  4. the solution changes with the random start, so it should "
              f"be treated with caution.")

    # Display all the figures (close the windows to finish)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
