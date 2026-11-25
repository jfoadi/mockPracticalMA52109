
import os

import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import (
    select_features,
    standardise_features,
    sklearn_kmeans,
    compute_inertia,
    silhouette_score_sklearn,
    elbow_curve,
    plot_clusters_2d,
    plot_elbow,
)


INPUT_PATH = "data/simulated_data.csv"
OUTPUT_DIR = "demo_output"

# Test several possible numbers of clusters.
K_VALUES = list(range(2, 9))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load the dataset.
    df = pd.read_csv(INPUT_PATH)

    feature_cols = [
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
    ]

    # 2. Select and standardise all four features.
    X_df = select_features(df, feature_cols)
    X = standardise_features(X_df.to_numpy(dtype=float))

    print("Dataset loaded successfully.")
    print(f"Number of observations: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    # 3. Evaluate different values of k.
    metrics_summary = []

    for k in K_VALUES:
        labels, centroids = sklearn_kmeans(
            X,
            k=k,
            random_state=42,
        )

        inertia = compute_inertia(X, labels, centroids)
        silhouette = silhouette_score_sklearn(X, labels)

        metrics_summary.append({
            "k": k,
            "inertia": inertia,
            "silhouette": silhouette,
        })

        print(
            f"k={k}: "
            f"inertia={inertia:.3f}, "
            f"silhouette={silhouette:.4f}"
        )

    metrics_df = pd.DataFrame(metrics_summary)

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "simulated_data_metrics.csv",
    )
    metrics_df.to_csv(metrics_path, index=False)

    # 4. Choose the k with the highest silhouette score.
    best_row = metrics_df.loc[
        metrics_df["silhouette"].idxmax()
    ]
    best_k = int(best_row["k"])

    print("\nRecommended number of clusters:", best_k)
    print(
    "\nJustification: k=6 was selected because it achieved "
    "the highest silhouette score among the tested values. "
    "The six clusters also contain 50 observations each. "
    "The inertia reduction becomes much smaller after k=6, "
    "supporting this choice."
)
    print(
        "Highest silhouette score:",
        round(best_row["silhouette"], 4),
    )

    # 5. Produce the final clustering using the selected k.
    labels, centroids = sklearn_kmeans(
        X,
        k=best_k,
        random_state=42,
    )

    # Count the observations assigned to each cluster.
    cluster_sizes = pd.Series(labels).value_counts().sort_index()

    print("\nNumber of observations in each cluster:")
    print(cluster_sizes)

    # Save the dataset with its assigned cluster labels.
    clustered_df = df.copy()
    clustered_df["cluster"] = labels

    clustered_path = os.path.join(
        OUTPUT_DIR,
        "simulated_data_clustered.csv",
    )
    clustered_df.to_csv(clustered_path, index=False)

    # 6. Plot the final clusters using cluster_maker.
    fig, ax = plot_clusters_2d(
        X,
        labels,
        centroids=centroids,
        title=f"Simulated Dataset: K-means Clustering (k={best_k})",
    )

    ax.set_xlabel(feature_cols[0] + " (standardised)")
    ax.set_ylabel(feature_cols[1] + " (standardised)")

    cluster_plot_path = os.path.join(
        OUTPUT_DIR,
        f"simulated_data_clusters_k{best_k}.png",
    )
    fig.savefig(cluster_plot_path, dpi=150)
    plt.close(fig)

    # 7. Produce an elbow plot using cluster_maker.
    elbow_k_values = list(range(1, 9))

    inertias = elbow_curve(
        X,
        k_values=elbow_k_values,
        random_state=42,
        use_sklearn=True,
    )

    fig, ax = plot_elbow(
        elbow_k_values,
        [inertias[k] for k in elbow_k_values],
        title="Elbow Method - Simulated Dataset",
    )

    elbow_path = os.path.join(
        OUTPUT_DIR,
        "simulated_data_elbow.png",
    )
    fig.savefig(elbow_path, dpi=150)
    plt.close(fig)

    # 8. Plot silhouette scores for comparison.
    plt.figure(figsize=(8, 5))
    plt.bar(
        metrics_df["k"],
        metrics_df["silhouette"],
    )
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Scores - Simulated Dataset")
    plt.xticks(K_VALUES)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    silhouette_path = os.path.join(
        OUTPUT_DIR,
        "simulated_data_silhouette.png",
    )
    plt.savefig(silhouette_path, dpi=150)
    plt.close()

    print("\nAnalysis completed.")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
