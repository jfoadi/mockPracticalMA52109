import numpy as np

from cluster_maker.agglomerative import agglomerative_clustering

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [10, 11],
    [11, 12],
    [12, 13]
])

labels, model = agglomerative_clustering(
    X,
    n_clusters=2,
    linkage="ward"
)

print("Cluster labels:", labels)
print("Number of clusters:", len(np.unique(labels)))