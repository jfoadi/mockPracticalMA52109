###
## cluster_maker: agglomerative clustering demo
##
## The difficult dataset forms concentric rings. This script clusters points
## by their distance from the estimated centre, compares k = 2 through 5, and
## saves the selected labels, metrics, and visualisations.
###

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster_maker import (
    agglomerative_clustering,
    plot_clusters_2d,
    silhouette_score_sklearn,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "difficult_dataset.csv"
OUTPUT_DIR = ROOT_DIR / "demo_output"
K_VALUES = range(2, 6)  # Compare k = 2 through 5


def main() -> None:
    # Load the two-dimensional dataset used for the cluster plots.
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    feature_cols = list(data.select_dtypes(include="number").columns)
    if len(feature_cols) != 2:
        raise ValueError("The difficult dataset must contain exactly two numeric features.")

    X = data[feature_cols].to_numpy(dtype=float)
    center = X.mean(axis=0)
    centered = X - center
    # Radial distance separates concentric rings more clearly than x and y.
    radii = np.linalg.norm(centered, axis=1).reshape(-1, 1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    metrics_summary = []
    clusterings = {}
    # Compare cluster counts using silhouette scores.
    for k in K_VALUES:
        labels, _ = agglomerative_clustering(radii, k=k, linkage="complete")
        score = silhouette_score_sklearn(radii, labels)
        metrics_summary.append({"k": k, "silhouette": score})
        clusterings[k] = labels

    metrics = pd.DataFrame(metrics_summary)
    selected_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    labels = clusterings[selected_k]
    best_score = metrics.loc[metrics["k"] == selected_k, "silhouette"].iloc[0]

    # Save the selected labels with the original coordinates and radii.
    output_data = data.copy()
    output_data["radius"] = radii[:, 0]
    output_data["cluster"] = labels
    output_data.to_csv(OUTPUT_DIR / "difficult_dataset_clustered.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "difficult_dataset_metrics.csv", index=False)

    # Show the selected assignments in the original x/y coordinate space.
    fig, _ = plot_clusters_2d(
        X,
        labels,
        title=f"Agglomerative clustering of the original data (k={selected_k})",
    )
    fig.savefig(OUTPUT_DIR / "difficult_dataset_clusters.png", dpi=150)
    plt.close(fig)

    # Plot radius against angle to make the ring-shaped groups visible.
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        angles,
        radii[:, 0],
        c=labels,
        cmap="tab10",
        alpha=0.75,
        s=18,
    )
    ax.set_xlabel("Angle from estimated centre (radians)")
    ax.set_ylabel("Distance from estimated centre")
    ax.set_title(f"Radial structure of the data (k={selected_k})")
    fig.colorbar(scatter, ax=ax, label="Cluster label")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "difficult_dataset_radial_structure.png", dpi=150)
    plt.close(fig)

    # Compare silhouette scores to explain the selected number of clusters.
    fig, ax = plt.subplots()
    ax.plot(metrics["k"], metrics["silhouette"], marker="o")
    ax.axvline(
        selected_k,
        color="black",
        linestyle="--",
        label=f"Selected k={selected_k}",
    )
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score on radius")
    ax.set_title("Agglomerative cluster-count comparison")
    ax.set_xticks(metrics["k"])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "difficult_dataset_silhouette.png", dpi=150)
    plt.close(fig)

    print(f"Used features: {', '.join(feature_cols)}")
    print("Clustering is performed on each point's distance from the dataset centre.")
    print("\nSilhouette scores:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSelected k={selected_k} (silhouette score {best_score:.3f}).")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
