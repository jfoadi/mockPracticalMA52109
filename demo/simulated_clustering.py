
"""
Analyse simulated data using cluster_maker.
Compare different values of k using inertia and silhouette scores.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import run_clustering


INPUT_PATH = "data/simulated_data.csv"
OUTPUT_DIR = "demo_output"
FEATURES = ["feature_1", "feature_2", "feature_3", "feature_4"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []

# Compare several possible numbers of clusters.
for k in range(2, 9):
    result = run_clustering(
        input_path=INPUT_PATH,
        feature_cols=FEATURES,
        algorithm="kmeans",
        k=k,
        standardise=True,
        random_state=42
    )

    metrics = result["metrics"]

    results.append({
        "k": k,
        "inertia": metrics["inertia"],
        "silhouette": metrics["silhouette"]
    })

    # Save the cluster_maker visualisation for each k.
    result["fig_cluster"].savefig(
        f"{OUTPUT_DIR}/simulated_k{k}.png",
        dpi=150
    )
    plt.close(result["fig_cluster"])


metrics_df = pd.DataFrame(results)
print("\nClustering comparison:")
print(metrics_df.to_string(index=False))

# Plot inertia and silhouette scores for comparison.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(metrics_df["k"], metrics_df["inertia"], marker="o")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow method")

axes[1].plot(metrics_df["k"], metrics_df["silhouette"], marker="o")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Cluster separation")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/simulated_comparison.png", dpi=150)
plt.close()

# Select the solution with the highest silhouette score.
best_k = int(
    metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"]
)

print(f"\nBest k by silhouette score: {best_k}")

# Run the selected clustering again and save labelled data.
best_result = run_clustering(
    input_path=INPUT_PATH,
    feature_cols=FEATURES,
    algorithm="kmeans",
    k=best_k,
    standardise=True,
    random_state=42,
    output_path=f"{OUTPUT_DIR}/simulated_clustered.csv"
)

# Show the chosen clusters using the first two features.
data = best_result["data"]

plt.figure(figsize=(8, 6))
plt.scatter(
    data["feature_1"],
    data["feature_2"],
    c=data["cluster"],
    cmap="viridis",
    alpha=0.7
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"Selected clustering (k={best_k})")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/simulated_best_clusters.png", dpi=150)
plt.close()

plt.close(best_result["fig_cluster"])


# Interpretation:
# k=4 achieves the highest silhouette score (0.562) among
# the tested values, indicating relatively well-separated clusters.
# Inertia also decreases substantially between k=3 and k=4.
#
# The 2D visualisation shows several distinct groups, some of
# which share the same cluster label. This is not necessarily
# incorrect because clustering uses all four features, whereas
# the plot displays only the first two.
#
# Inertia decreases sharply again at k=8, suggesting that
# alternative clustering solutions may deserve investigation.
# Nevertheless, k=4 is a reasonable choice based on the
# highest silhouette score among the tested solutions.



print("\nAnalysis completed.")
print(f"Outputs saved in: {OUTPUT_DIR}")
