
"""
Demonstrate agglomerative clustering on difficult_dataset.csv.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker.agglomerative import agglomerative
from cluster_maker.preprocessing import select_features, standardise_features
from cluster_maker.evaluation import silhouette_score_sklearn
from cluster_maker.plotting_clustered import plot_clusters_2d


INPUT_PATH = "data/difficult_dataset.csv"
OUTPUT_DIR = "demo_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and prepare the dataset using cluster_maker tools.
df = pd.read_csv(INPUT_PATH)
X_df = select_features(df, ["x", "y"])
X = standardise_features(X_df.to_numpy(dtype=float))

results = []

# Compare different numbers of clusters.
for k in range(2, 9):
    labels, centroids = agglomerative(X, k=k)

    score = silhouette_score_sklearn(X, labels)

    results.append({
        "k": k,
        "silhouette": score
    })

    # Use the package's plotting function.
    fig, _ = plot_clusters_2d(
        X,
        labels,
        centroids=centroids,
        title=f"Agglomerative clustering (k={k})"
    )

    fig.savefig(
        f"{OUTPUT_DIR}/agglomerative_k{k}.png",
        dpi=150
    )
    plt.close(fig)

# Compare silhouette scores.
metrics_df = pd.DataFrame(results)

print("\nAgglomerative clustering comparison:")
print(metrics_df.to_string(index=False))

best_k = int(
    metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"]
)

print(f"\nBest k by silhouette score: {best_k}")

# Plot silhouette scores across different k.
plt.figure(figsize=(8, 5))
plt.plot(
    metrics_df["k"],
    metrics_df["silhouette"],
    marker="o"
)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Agglomerative clustering comparison")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/agglomerative_comparison.png",
    dpi=150
)
plt.close()

# Save the final labelled dataset.
labels, centroids = agglomerative(X, k=best_k)

output_df = df.copy()
output_df["cluster"] = labels
output_df.to_csv(
    f"{OUTPUT_DIR}/difficult_clustered.csv",
    index=False
)


# Compare Ward linkage with single linkage on the ring-shaped data.
# Single linkage can identify non-convex clusters more effectively.

single_labels, single_centroids = agglomerative(
    X,
    k=4,
    linkage="single"
)

single_score = silhouette_score_sklearn(X, single_labels)

print(f"\nSingle linkage (k=4) silhouette: {single_score:.3f}")

fig, _ = plot_clusters_2d(
    X,
    single_labels,
    centroids=single_centroids,
    title="Agglomerative clustering - single linkage (k=4)"
)

fig.savefig(
    f"{OUTPUT_DIR}/agglomerative_single_k4.png",
    dpi=150
)
plt.close(fig)


# Compare alternative linkage methods for four clusters.
# The dataset contains concentric rings, which can be difficult
# for methods that favour compact clusters.

for method in ["complete", "average"]:

    labels, centroids = agglomerative(
        X,
        k=4,
        linkage=method
    )

    score = silhouette_score_sklearn(X, labels)

    print(f"{method.capitalize()} linkage (k=4) silhouette: {score:.3f}")

    fig, _ = plot_clusters_2d(
        X,
        labels,
        centroids=centroids,
        title=f"Agglomerative clustering - {method} linkage (k=4)"
    )

    fig.savefig(
        f"{OUTPUT_DIR}/agglomerative_{method}_k4.png",
        dpi=150
    )
    plt.close(fig)

# Different linkage methods were compared on the concentric-ring dataset.
# Single linkage showed a chaining effect, producing one dominant cluster.
# Average and complete linkage divided the rings into several sections
# rather than identifying each concentric ring separately.
# This illustrates the sensitivity of hierarchical clustering to
# linkage choice and the geometry of the dataset.

print("\nAnalysis completed.")
print(f"Outputs saved in: {OUTPUT_DIR}")
