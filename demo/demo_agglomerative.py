import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import (
    select_features,
    sklearn_kmeans,
    agglomerative_clustering,
    compute_inertia,
    silhouette_score_sklearn,
    plot_clusters_2d,
)


def main():

    # Load the difficult dataset.
    df = pd.read_csv("data/difficult_dataset.csv")
    X = select_features(df, ["x", "y"]).to_numpy(dtype=float)

    # Four concentric rings are visible in the original dataset.
    k = 4

    print("Dataset shape:", X.shape)
    print("Number of clusters:", k)

    # Visualise the original data before clustering.
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], s=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Original difficult dataset")
    ax.set_aspect("equal")

    # Use K-means as a baseline for comparison.
    labels, centroids = sklearn_kmeans(
        X, k=k, random_state=42
    )

    print("\nK-means:")
    print(f"Inertia: {compute_inertia(X, labels, centroids):.4f}")
    print(f"Silhouette: {silhouette_score_sklearn(X, labels):.4f}")

    fig, ax = plot_clusters_2d(
        X,
        labels,
        centroids=centroids,
        title="K-means clustering (k = 4)",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # Compare all four hierarchical linkage methods.
    linkages = ["ward", "complete", "average", "single"]

    for linkage in linkages:

        labels, centroids = agglomerative_clustering(
            X, k=k, linkage=linkage
        )

        print(f"\nAgglomerative clustering ({linkage}):")
        print(f"Inertia: {compute_inertia(X, labels, centroids):.4f}")
        print(f"Silhouette: {silhouette_score_sklearn(X, labels):.4f}")

        print("Cluster sizes:")
        print(pd.Series(labels).value_counts().sort_index())

        # Centroids are omitted because they are not representative
        # of non-convex clusters such as concentric rings.
        fig, ax = plot_clusters_2d(
            X,
            labels,
            title=f"Agglomerative clustering: {linkage} linkage",
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    # Transform Cartesian coordinates into radial distances.
    # Points on the same concentric ring have similar distances
    # from the origin, making their underlying structure clearer.
    X_radial = np.sqrt(
        X[:, 0] ** 2 + X[:, 1] ** 2
    ).reshape(-1, 1)

    # Apply hierarchical clustering to the transformed feature.
    labels_radial, centroids_radial = agglomerative_clustering(
        X_radial, k=k, linkage="ward"
    )

    print("\nAgglomerative clustering with radial transformation:")

    print(
        f"Inertia: "
        f"{compute_inertia(X_radial, labels_radial, centroids_radial):.4f}"
    )

    print(
        f"Silhouette: "
        f"{silhouette_score_sklearn(X_radial, labels_radial):.4f}"
    )

    print("Cluster sizes:")
    print(pd.Series(labels_radial).value_counts().sort_index())

    # Visualise the new cluster assignments using the original
    # coordinates to assess whether the four rings are recovered.
    fig, ax = plot_clusters_2d(
        X,
        labels_radial,
        title="Agglomerative clustering with radial transformation",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # Explain the effectiveness of hierarchical clustering
    # when combined with an appropriate feature representation.
    print("\nConclusion:")

    print(
        "The original dataset contains four concentric rings."
    )

    print(
        "K-means fails to recover the rings because it assigns "
        "observations according to their distance from centroids."
    )

    print(
        "Applying hierarchical clustering directly to Cartesian "
        "coordinates also fails to fully recover the ring structure."
    )

    print(
        "Single linkage suffers from the chaining effect, "
        "merging most observations into one large cluster."
    )

    print(
        "Transforming the coordinates into radial distances "
        "allows Ward linkage to identify four distinct rings."
    )

    print(
        "The results demonstrate the importance of both algorithm "
        "selection and feature representation in clustering."
    )

    plt.show()


if __name__ == "__main__":
    main()