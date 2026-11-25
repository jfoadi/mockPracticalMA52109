from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make the project root available for package imports
PROJECT_DIR = (
    Path(globals()["__file__"]).resolve().parent.parent
    if "__file__" in globals()
    else Path.cwd().resolve()
)
sys.path.insert(0, str(PROJECT_DIR))

# Import clustering tools from cluster_maker
from cluster_maker.preprocessing import (
    select_features,
    standardise_features
)

from cluster_maker.algorithms import sklearn_kmeans

from cluster_maker.evaluation import (
    compute_inertia,
    silhouette_score_sklearn,
    elbow_curve
)

from cluster_maker.plotting_clustered import (
    plot_clusters_2d,
    plot_elbow
)


def main():

    # --------------------------------------------------
    # 1. Load the dataset
    # --------------------------------------------------

    data_path = PROJECT_DIR / "data" / "simulated_data.csv"

    df = pd.read_csv(data_path)

    print("Dataset loaded successfully")
    print("Dataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())

    # --------------------------------------------------
    # 2. Select numerical features
    # --------------------------------------------------

    feature_cols = [
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4"
    ]

    X_df = select_features(df, feature_cols)
    X = X_df.to_numpy(dtype=float)

    if not np.all(np.isfinite(X)):
        raise ValueError("Dataset contains missing or infinite values.")

    print("\nSelected features:", feature_cols)

    # --------------------------------------------------
    # 3. Standardise the features
    # --------------------------------------------------

    X_scaled = standardise_features(X)

    print("\nFeatures standardised successfully")

    # --------------------------------------------------
    # 4. Calculate the elbow curve
    # --------------------------------------------------

    k_values = list(range(1, 11))

    inertias = elbow_curve(
        X_scaled,
        k_values=k_values,
        random_state=42,
        use_sklearn=True
    )

    print("\nElbow method results:")

    for k in k_values:
        print(f"k = {k}: Inertia = {inertias[k]:.3f}")

    # Use cluster_maker plotting function
    fig_elbow, ax_elbow = plot_elbow(
        k_values,
        [inertias[k] for k in k_values],
        title="Elbow Method - Simulated Dataset"
    )

    # --------------------------------------------------
    # 5. Calculate silhouette scores
    # --------------------------------------------------

    silhouette_values = {}

    silhouette_k_values = range(2, 11)

    for k in silhouette_k_values:

        labels, centroids = sklearn_kmeans(
            X_scaled,
            k=k,
            random_state=42
        )

        score = silhouette_score_sklearn(
            X_scaled,
            labels
        )

        silhouette_values[k] = score

    print("\nSilhouette scores:")

    for k, score in silhouette_values.items():
        print(f"k = {k}: Silhouette score = {score:.4f}")

    # Select the solution with the highest silhouette score
    best_k = max(
        silhouette_values,
        key=silhouette_values.get
    )

    print("\nRecommended number of clusters:", best_k)

    # --------------------------------------------------
    # 6. Fit the final clustering model
    # --------------------------------------------------

    labels, centroids = sklearn_kmeans(
        X_scaled,
        k=best_k,
        random_state=42
    )

    inertia = compute_inertia(
        X_scaled,
        labels,
        centroids
    )

    silhouette = silhouette_score_sklearn(
        X_scaled,
        labels
    )

    print("\nFINAL CLUSTERING RESULTS")
    print("------------------------")
    print("Number of clusters:", best_k)
    print("Inertia:", round(inertia, 4))
    print("Silhouette score:", round(silhouette, 4))

    print("\nNumber of observations in each cluster:")

    unique_labels, counts = np.unique(
        labels,
        return_counts=True
    )

    for cluster, count in zip(unique_labels, counts):
        print(f"Cluster {cluster}: {count} observations")

    # --------------------------------------------------
    # 7. Visualise the final clusters
    # --------------------------------------------------

    fig_clusters, ax_clusters = plot_clusters_2d(
        X_scaled,
        labels,
        centroids=centroids,
        title=f"Final K-means Clustering (k={best_k})"
    )

    # --------------------------------------------------
    # 8. Supporting visualisation: silhouette scores
    # --------------------------------------------------

    fig_silhouette, ax_silhouette = plt.subplots(
        figsize=(8, 5)
    )

    ax_silhouette.plot(
        list(silhouette_values.keys()),
        list(silhouette_values.values()),
        marker="o"
    )

    ax_silhouette.axvline(
        best_k,
        linestyle="--",
        label=f"Selected k = {best_k}"
    )

    ax_silhouette.set_xlabel("Number of clusters (k)")
    ax_silhouette.set_ylabel("Silhouette score")
    ax_silhouette.set_title("Silhouette Score Comparison")
    ax_silhouette.set_xticks(list(silhouette_values.keys()))
    ax_silhouette.grid(True)
    ax_silhouette.legend()

    # --------------------------------------------------
    # 9. Supporting visualisation: feature relationships
    # --------------------------------------------------

    fig_pairs, axes = plt.subplots(
        2, 3,
        figsize=(14, 9)
    )

    feature_pairs = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3)
    ]

    for ax, (i, j) in zip(axes.flat, feature_pairs):

        ax.scatter(
            X_scaled[:, i],
            X_scaled[:, j],
            c=labels,
            cmap="tab10",
            alpha=0.8
        )

        ax.set_xlabel(feature_cols[i])
        ax.set_ylabel(feature_cols[j])
        ax.set_title(
            f"{feature_cols[i]} vs {feature_cols[j]}"
        )

    fig_pairs.suptitle(
        f"Pairwise Feature Relationships (k={best_k})"
    )

    fig_pairs.tight_layout()

    # --------------------------------------------------
    # 10. Save results and plots
    # --------------------------------------------------

    output_dir = PROJECT_DIR / "demo" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save cluster assignments
    result_df = df.copy()
    result_df["cluster"] = labels

    result_df.to_csv(
        output_dir / "clustered_data.csv",
        index=False
    )

    # Save plots
    fig_elbow.savefig(
        output_dir / "elbow_plot.png",
        dpi=300
    )

    fig_clusters.savefig(
        output_dir / "cluster_plot.png",
        dpi=300
    )

    fig_silhouette.savefig(
        output_dir / "silhouette_plot.png",
        dpi=300
    )

    fig_pairs.savefig(
        output_dir / "feature_relationships.png",
        dpi=300
    )

    print("\nPlots and results saved in:", output_dir)

    plt.show()


if __name__ == "__main__":
    main()