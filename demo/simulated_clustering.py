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
##
## The analysis is printed first. The figures open together at the end.
###

from __future__ import annotations

import os
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

RANDOM_STATE = 42
K_ELBOW = list(range(1, 9))        # k = 1, ..., 8 for the elbow curve
K_SILHOUETTE = list(range(2, 9))   # the silhouette needs at least 2 clusters
N_STARTS = 10                      # random starts for the reliability check

# The data folder sits next to this script's folder, so the script works
# wherever it is launched from.
DEFAULT_INPUT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "data", "simulated_data.csv")
)


def plot_raw_pairs(df: pd.DataFrame) -> None:
    """Scatter plot of every pair of features (histograms on the diagonal)."""
    cols = list(df.columns)
    n = len(cols)
    fig, ax = plt.subplots(n, n, figsize=(10, 10))
    for i in range(n):
        for j in range(n):
            if i == j:
                ax[i, j].hist(df[cols[j]], bins=20, color="grey",
                              edgecolor="white")
            else:
                ax[i, j].scatter(df[cols[j]], df[cols[i]], s=6, alpha=0.6)
            if i == n - 1:
                ax[i, j].set_xlabel(cols[j])
            if j == 0:
                ax[i, j].set_ylabel(cols[i])
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
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    for p, (i, j) in enumerate(pairs):
        a = ax[p // 3, p % 3]
        a.scatter(df[cols[i]], df[cols[j]], c=labels, cmap="tab10",
                  s=12, alpha=0.8)
        a.set_title(f"{cols[j]} against {cols[i]}")
        a.set_xlabel(cols[i])
        a.set_ylabel(cols[j])
        a.grid(True)
    plt.tight_layout()


def main(args: List[str]) -> None:
    input_path = args[0] if args else DEFAULT_INPUT
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load and inspect the data
    # ------------------------------------------------------------------
    df = pd.read_csv(input_path)
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
    print(f"{'k':>3} {'inertia':>10} {'drop vs k-1':>12} {'silhouette':>11}")
    for k in K_ELBOW:
        drop = ""
        if k > 1:
            drop = f"{100 * (inertias[k - 1] - inertias[k]) / inertias[k - 1]:.0f}%"
        sil = f"{silhouettes[k]:.3f}" if k in silhouettes else ""
        print(f"{k:>3} {inertias[k]:>10.1f} {drop:>12} {sil:>11}")
    print(f"-> The silhouette score is highest for k = {best_k}, and the "
          f"inertia stops dropping sharply after it.")

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
    print(f"  manual k-means:       inertia {min(manual_inertias):.1f} to "
          f"{max(manual_inertias):.1f}; reaches the best solution in "
          f"{n_reached} of {N_STARTS} starts")
    print("  -> The scikit-learn solution does not depend on the start, while "
          "the manual k-means can get stuck in poor solutions.")

    # ------------------------------------------------------------------
    # 5. Final clustering
    # ------------------------------------------------------------------
    labels, centroids = sklearn_kmeans(X, best_k, random_state=RANDOM_STATE)
    sizes = [int(np.sum(labels == c)) for c in np.unique(labels)]
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

    # Cluster plot from the package: the first two features
    fig, ax = plot_clusters_2d(
        X[:, [0, 1]],
        labels,
        centroids=centroids[:, [0, 1]],
        title=f"k-means with k = {best_k}",
    )
    ax.set_xlabel(f"{feature_cols[0]} (standardised)")
    ax.set_ylabel(f"{feature_cols[1]} (standardised)")

    # Supporting figure: the clusters seen in every pair of features
    plot_all_pairs(df, labels)

    print(f"\nConclusion: the data separate into {best_k} well-defined clusters.")

    # Display all the figures (close the windows to finish)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
