###
## cluster_maker: demo of the agglomerative module on 'difficult_dataset.csv'
##
## Goal: show that the new module 'cluster_maker/agglomerative.py' gives a
## sensible clustering of the data in 'data/difficult_dataset.csv', a dataset
## on which the usual methods fail.
##
## Run from the project root with:
##     python demo/demo_agglomerative.py
## (an optional argument gives a different input csv file).
##
## Outline:
##  1. Load and look at the data
##  2. The usual methods fail on the raw coordinates
##  3. A better description of the data: distance from the centre
##  4. Choose the number of clusters and the linkage with the new module
##  5. Final clustering and summary
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
    agglomerative_clustering,
    calculate_descriptive_statistics,
    compare_linkages,
    plot_clusters_2d,
    select_features,
    silhouette_score_sklearn,
    sklearn_kmeans,
    standardise_features,
)

INPUT_PATH = "data/difficult_dataset.csv"
RANDOM_STATE = 42
K_TRY = 4                          # the number of groups seen in the raw plot
K_VALUES = list(range(2, 9))       # k = 2, ..., 8 when choosing k
LINKAGES = ["ward", "complete", "average", "single"]


def cluster_sizes(labels: np.ndarray) -> List[int]:
    """Number of points in each cluster."""
    return [int(np.sum(labels == c)) for c in np.unique(labels)]


def ring_spread(distance: np.ndarray, labels: np.ndarray) -> float:
    """
    Average spread of the distance from the centre inside each cluster.
    A small value means that each cluster is a ring around the centre.
    """
    spreads = [np.std(distance[labels == c]) for c in np.unique(labels)]
    return float(np.mean(spreads))


def draw_clusters(ax, X: np.ndarray, labels: np.ndarray, title: str) -> None:
    """Scatter plot of the points of X coloured by cluster, on the axes ax."""
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=5)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def main(args: List[str]) -> None:
    input_path = args[0] if args else INPUT_PATH

    # ------------------------------------------------------------------
    # 1. Load and look at the data
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: file not found: {input_path}")
        print("Run the script from the project root folder.")
        sys.exit(1)

    X_df = select_features(df, list(df.columns))    # also checks they are numeric
    X = X_df.to_numpy(dtype=float)
    print(f"Data: {df.shape[0]} points with features {list(df.columns)}")
    print("\nDescriptive statistics:")
    print(calculate_descriptive_statistics(df).round(2))

    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], s=5)
    plt.title("The raw data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    print("\nThe plot shows points arranged in concentric rings, so we expect "
          "one cluster per ring.")

    # ------------------------------------------------------------------
    # 2. The usual methods fail on the raw coordinates
    # ------------------------------------------------------------------
    # We give every method the number of groups seen in the plot (k = 4).
    # 'ring spread' measures how far from a ring each cluster is: the spread
    # of the distance from the centre inside the cluster. A cluster that is a
    # ring has a spread close to 0.
    centre = X.mean(axis=0)
    distance = np.linalg.norm(X - centre, axis=1)
    Xs = standardise_features(X)

    methods = {}
    methods["k-means"], _ = sklearn_kmeans(Xs, K_TRY, random_state=RANDOM_STATE)
    for linkage in LINKAGES:
        methods[f"agglomerative ({linkage})"], _ = agglomerative_clustering(
            Xs, K_TRY, linkage=linkage
        )

    print(f"\nClustering the raw coordinates with k = {K_TRY}:")
    spreads = []
    for name in methods:
        labels_m = methods[name]
        spreads.append(ring_spread(distance, labels_m))
        print(f"  {name}: sizes {cluster_sizes(labels_m)}, "
              f"silhouette {silhouette_score_sklearn(Xs, labels_m):.3f}, "
              f"ring spread {spreads[-1]:.2f}")
    print("  -> None of them finds four rings. k-means and Ward cut across all "
          "the rings, complete and average mix them,")
    print("     and single linkage puts almost every point in one cluster "
          "(its small spread is only a side effect of that).")
    print("     The silhouette score prefers compact blobs, so it cannot be "
          "trusted for rings.")

    plt.figure(figsize=(15, 9))
    p = 1
    for name in methods:
        ax = plt.subplot(2, 3, p)
        draw_clusters(ax, X, methods[name], name)
        p = p + 1
    plt.tight_layout()

    # ------------------------------------------------------------------
    # 3. A better description of the data: distance from the centre
    # ------------------------------------------------------------------
    # Points on the same ring have (almost) the same distance from the centre,
    # and different rings have different distances. The centre is estimated
    # by the mean of the data, which is reasonable for rings around a common
    # centre. The new description has a single feature.
    D = distance.reshape(-1, 1)

    plt.figure(figsize=(8, 5))
    plt.hist(distance, bins=40, color="grey", edgecolor="white")
    plt.title("Distance of each point from the centre")
    plt.xlabel("Distance from the centre")
    plt.ylabel("Frequency")
    plt.grid(True)

    print(f"\nCentre of the data: ({centre[0]:.2f}, {centre[1]:.2f}). The histogram "
          "of the distances from the centre shows separate bands.")

    # ------------------------------------------------------------------
    # 4. Choose the number of clusters and the linkage
    # ------------------------------------------------------------------
    silhouettes = {}
    for k in K_VALUES:
        labels_k, _ = agglomerative_clustering(D, k, linkage="ward")
        silhouettes[k] = silhouette_score_sklearn(D, labels_k)

    # The best k is the one with the highest silhouette score
    best_k = K_VALUES[0]
    for k in K_VALUES:
        if silhouettes[k] > silhouettes[best_k]:
            best_k = k

    plt.figure(figsize=(8, 5))
    colours = ["red" if k == best_k else "blue" for k in K_VALUES]
    plt.bar(K_VALUES, [silhouettes[k] for k in K_VALUES], color=colours)
    plt.title(f"Silhouette score of the agglomerative clustering (best: k = {best_k})")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.grid(True)

    print("\nChoosing k (Ward linkage on the distance from the centre):")
    for k in K_VALUES:
        print(f"  k = {k}: silhouette = {silhouettes[k]:.3f}")

    print(f"\nComparing the linkages for k = {best_k}:")
    linkage_scores = compare_linkages(D, best_k)
    for linkage in linkage_scores:
        print(f"  {linkage}: silhouette = {linkage_scores[linkage]:.3f}")
    print("  -> 'single' is clearly worse: a few points between two rings are "
          "enough to chain them together.")
    print("     Ward gives a top score and is the default of the module.")

    # ------------------------------------------------------------------
    # 5. Final clustering and summary
    # ------------------------------------------------------------------
    labels, centroids = agglomerative_clustering(D, best_k, linkage="ward")
    sizes = cluster_sizes(labels)
    radii = sorted([round(float(c), 2) for c in centroids[:, 0]])
    spread_final = ring_spread(distance, labels)
    spread_kmeans = ring_spread(distance, methods["k-means"])

    print(f"\nFinal clustering with k = {best_k} (Ward):")
    print(f"  cluster sizes: {sizes}")
    print(f"  mean distance from the centre of each cluster (ring radius): {radii}")
    print(f"  silhouette score: {silhouettes[best_k]:.3f}")
    print(f"  ring spread: {spread_final:.2f} (k-means on the raw coordinates: "
          f"{spread_kmeans:.2f})")

    # Cluster plot from the package, on the original coordinates
    fig, ax = plot_clusters_2d(X, labels, title=f"Agglomerative clustering, k = {best_k}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    print(f"\nSummary: the data are {best_k} concentric rings, and the new "
          "agglomerative module recovers them, because")
    print(f"  1. k-means and the four linkages fail on the raw coordinates "
          f"(ring spreads from {min(spreads):.2f} to {max(spreads):.2f});")
    print(f"  2. on the distance from the centre the silhouette score is "
          f"highest for k = {best_k} ({silhouettes[best_k]:.3f});")
    print(f"  3. the clusters have similar sizes {sizes} and ring spread "
          f"{spread_final:.2f}: each cluster is one ring;")
    print("  4. the module needs no random start, so the result is always the "
          "same.")
    print("Limit: this relies on the rings sharing a common centre, estimated "
          "by the mean of the data.")

    # Display all the figures (close the windows to finish)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
