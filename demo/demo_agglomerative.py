###
## cluster_maker: demo of agglomerative clustering
##
## data/difficult_dataset.csv contains four rings, one inside the other.
## Clustering the raw (x, y) values cuts the rings into wedges, so instead
## each point is described by its distance from the centre (radius).
## Agglomerative clustering on the radius separates the four rings.
###

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import (
    agglomerative_clustering,
    silhouette_score_sklearn,
    plot_clusters_2d,
)

INPUT_PATH = "data/difficult_dataset.csv"
OUTPUT_DIR = "demo_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)
X = df[["x", "y"]].to_numpy()

# Naive attempt on raw (x, y): fails, the rings are cut into wedges
labels_naive, _ = agglomerative_clustering(X, 4)
fig, ax = plot_clusters_2d(X, labels_naive, title="Raw (x, y): rings cut into wedges")
fig.savefig(os.path.join(OUTPUT_DIR, "agglo_naive.png"))
plt.close(fig)

# Better feature: distance from the centre (same for all points on a ring)
radius = np.sqrt(((X - X.mean(axis=0)) ** 2).sum(axis=1)).reshape(-1, 1)

# Histogram of the radius: four separate peaks show that the radius
# splits the data into four groups, one per ring
fig, ax = plt.subplots()
ax.hist(radius[:, 0], bins=60, color="grey")
ax.set_xlabel("Distance from centre")
ax.set_ylabel("Number of points")
ax.set_title("Histogram of radius: four clear peaks")
fig.savefig(os.path.join(OUTPUT_DIR, "agglo_radius_hist.png"))
plt.close(fig)

# Choose k with the silhouette score (not defined for k = 1)
silhouettes = {}
for k in range(2, 9):
    labels, _ = agglomerative_clustering(radius, k)
    silhouettes[k] = silhouette_score_sklearn(radius, labels)
    print(f"k = {k}: silhouette = {silhouettes[k]:.3f}")
best_k = max(silhouettes, key=silhouettes.get)

# Final clustering, plotted in the original (x, y) space
labels, _ = agglomerative_clustering(radius, best_k)
fig, ax = plot_clusters_2d(X, labels, title=f"Clustering on radius, k = {best_k}")
fig.savefig(os.path.join(OUTPUT_DIR, "agglo_final.png"))
plt.close(fig)

print(f"\nConclusion: the data contains {best_k} rings. The radius histogram")
print(f"has {best_k} clear peaks, the silhouette score is highest at k = {best_k},")
print(f"and the clusters have sizes {np.bincount(labels).tolist()}: one per ring.")
print(f"Plots saved in {OUTPUT_DIR}/")