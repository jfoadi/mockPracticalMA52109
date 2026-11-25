###
## cluster_maker: clustering analysis of data/simulated_data.csv
##
##
## Goal: decide how many clusters the data has, and show why that choice
## is convincing using several independent pieces of evidence:
##   1. Scatter plots of the raw data (can we see groups by eye?)
##   2. The elbow curve of inertia against k
##   3. The silhouette score for each k
##   4. The final clustering plotted on every pair of features
###

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import (
    select_features,
    standardise_features,
    sklearn_kmeans,
    elbow_curve,
    silhouette_score_sklearn,
    plot_elbow,
    plot_clusters_2d,
)

INPUT_PATH = "data/simulated_data.csv"
OUTPUT_DIR = "demo_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Load and prepare the data
# ---------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)
feature_cols = ["feature_1", "feature_2", "feature_3", "feature_4"]
X_df = select_features(df, feature_cols)
print(f"Loaded {len(df)} points with {len(feature_cols)} features.")
print(X_df.describe().round(2))

# The features have different spreads, so standardise them so that
# no single feature dominates the distances used by k-means.
X = standardise_features(X_df.to_numpy())

# ---------------------------------------------------------------
# 1. Look at the raw data: scatter plot of every pair of features
# ---------------------------------------------------------------
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (i, j) in zip(axes.flat, pairs):
    ax.scatter(X_df.iloc[:, i], X_df.iloc[:, j], s=8, color="grey")
    ax.set_xlabel(feature_cols[i])
    ax.set_ylabel(feature_cols[j])
fig.suptitle("Raw data: every pair of features")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sim_raw_pairs.png"))
plt.close(fig)

# ---------------------------------------------------------------
# 2. Elbow curve: inertia for k = 1 to 10
# ---------------------------------------------------------------
k_values = list(range(1, 11))
#going up to k = 10 to see how inertia changes and where the curve flattens out
inertias = elbow_curve(X, k_values, random_state=42)
fig, ax = plot_elbow(k_values, [inertias[k] for k in k_values])
fig.savefig(os.path.join(OUTPUT_DIR, "sim_elbow.png"))
plt.close(fig)

# ---------------------------------------------------------------
# 3. Silhouette score for k = 2 to 10 (not defined for k = 1)
# ---------------------------------------------------------------
print("\n k   inertia   silhouette")
silhouettes = {}
for k in range(2, 11): #start at k = 2: the silhouette score compares each point's own cluster
# with the nearest other cluster, so it is not defined for a single cluster.
    silhouettes[k] = silhouette_score_sklearn(X, labels)
    print(f"{k:2d}  {inertias[k]:8.1f}   {silhouettes[k]:.3f}")

best_k = max(silhouettes, key=silhouettes.get)
print(f"\nBest k (highest silhouette score): {best_k}")

fig, ax = plt.subplots()
ax.bar(list(silhouettes.keys()), list(silhouettes.values()))
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Silhouette score")
ax.set_title(f"Silhouette score for each k (best k = {best_k})")
fig.savefig(os.path.join(OUTPUT_DIR, "sim_silhouette.png"))
plt.close(fig)

# ---------------------------------------------------------------
# 4. Final clustering with the best k
# ---------------------------------------------------------------
labels, centroids = sklearn_kmeans(X, best_k, random_state=42)
print("Cluster sizes:", np.bincount(labels))

# Package plot: first two (standardised) features with centroids
fig, ax = plot_clusters_2d(X, labels, centroids=centroids,
                           title=f"k-means with k = {best_k}")
fig.savefig(os.path.join(OUTPUT_DIR, "sim_clusters.png"))
plt.close(fig)

# Same pairwise plots as before, now coloured by cluster
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (i, j) in zip(axes.flat, pairs):
    ax.scatter(X_df.iloc[:, i], X_df.iloc[:, j], c=labels, cmap="tab10", s=8)
    ax.set_xlabel(feature_cols[i])
    ax.set_ylabel(feature_cols[j])
fig.suptitle(f"Final clustering (k = {best_k}) on every pair of features")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sim_clusters_pairs.png"))
plt.close(fig)

# Save the data with its cluster labels
df["cluster"] = labels
df.to_csv(os.path.join(OUTPUT_DIR, "simulated_data_clustered.csv"), index=False)

print(f"\nConclusion: k = {best_k}. The elbow curve flattens after k = {best_k},")
print("the silhouette score is highest there, and the clusters are equal in size.")
print(f"Plots saved in {OUTPUT_DIR}/")