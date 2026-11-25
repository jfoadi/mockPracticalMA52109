
import os

import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import (
    select_features,
    agglomerative_clustering,
    silhouette_score_sklearn,
    plot_clusters_2d,
)


INPUT_PATH = "data/difficult_dataset.csv"
OUTPUT_DIR = "demo_output"

K_VALUES = range(2, 9)
LINKAGE_METHODS = ["ward", "complete", "average", "single"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load the dataset and select the two coordinate columns.
    df = pd.read_csv(INPUT_PATH)

    X_df = select_features(df, ["x", "y"])
    X = X_df.to_numpy(dtype=float)

    print("Difficult dataset loaded successfully.")
    print(f"Number of observations: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    # 2. Compare different linkage methods and cluster counts.
    results = []

    for linkage in LINKAGE_METHODS:
        for k in K_VALUES:
            labels = agglomerative_clustering(
                X,
                k=k,
                linkage=linkage,
            )

            silhouette = silhouette_score_sklearn(X, labels)

            results.append({
                "linkage": linkage,
                "k": k,
                "silhouette": silhouette,
            })

            print(
                f"Linkage={linkage:8s} "
                f"k={k}: silhouette={silhouette:.4f}"
            )

    # 3. Save the comparison results.
    results_df = pd.DataFrame(results)

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "agglomerative_metrics.csv",
    )
    results_df.to_csv(metrics_path, index=False)

    # 4. Select the combination with the highest silhouette score.
    best_row = results_df.loc[
        results_df["silhouette"].idxmax()
    ]

    best_linkage = best_row["linkage"]
    best_k = int(best_row["k"])
    best_score = best_row["silhouette"]

    print("\nRecommended clustering:")
    print("Linkage method:", best_linkage)
    print("Number of clusters:", best_k)
    print("Silhouette score:", round(best_score, 4))

    # 5. Run the final clustering with the selected settings.
    labels = agglomerative_clustering(
        X,
        k=best_k,
        linkage=best_linkage,
    )

    # Show the number of observations in each cluster.
    cluster_sizes = pd.Series(labels).value_counts().sort_index()

    print("\nNumber of observations in each cluster:")
    print(cluster_sizes)

    # Save the original data with its cluster labels.
    clustered_df = df.copy()
    clustered_df["cluster"] = labels

    clustered_path = os.path.join(
        OUTPUT_DIR,
        "difficult_dataset_clustered.csv",
    )
    clustered_df.to_csv(clustered_path, index=False)

    # 6. Plot the final clusters using cluster_maker.
    fig, ax = plot_clusters_2d(
        X,
        labels,
        centroids=None,
        title=(
            f"Agglomerative Clustering "
            f"(k={best_k}, linkage={best_linkage})"
        ),
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    cluster_plot_path = os.path.join(
        OUTPUT_DIR,
        "difficult_dataset_agglomerative.png",
    )
    fig.savefig(cluster_plot_path, dpi=150)
    plt.close(fig)

    # 7. Create a comparison plot for the linkage methods.
    comparison_df = results_df.pivot(
        index="k",
        columns="linkage",
        values="silhouette",
    )

    ax = comparison_df.plot(
        marker="o",
        figsize=(9, 5),
    )

    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Agglomerative Clustering: Linkage Comparison")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    comparison_path = os.path.join(
        OUTPUT_DIR,
        "agglomerative_linkage_comparison.png",
    )
    plt.savefig(comparison_path, dpi=150)
    plt.close()

    print("\nAnalysis completed.")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

