"""Compare agglomerative clustering choices on difficult_dataset.csv."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from cluster_maker.agglomerative import agglomerative_clustering
from cluster_maker.preprocessing import select_features, standardise_features
from cluster_maker.plotting_clustered import plot_clusters_2d


def main():
    data_path = PROJECT_DIR / "data" / "difficult_dataset.csv"
    df = pd.read_csv(data_path)
    feature_cols = list(df.select_dtypes(include=[np.number]).columns)
    if len(feature_cols) < 2:
        raise ValueError("The dataset must contain at least two numeric features.")

    X = select_features(df, feature_cols).to_numpy(dtype=float)
    if not np.all(np.isfinite(X)):
        raise ValueError("Dataset contains missing or infinite feature values.")
    X_scaled = standardise_features(X)

    # Evaluate several linkage rules and cluster counts. A fixed silhouette
    # sample keeps this comparison fast and repeatable on larger datasets.
    linkage_options = ("ward", "complete", "average", "single")
    cluster_counts = range(2, 9)
    candidates = []
    print(f"Loaded {len(df)} observations with features {feature_cols}.")
    print("Silhouette comparison (standardised features):")
    for linkage in linkage_options:
        for k in cluster_counts:
            labels = agglomerative_clustering(X_scaled, n_clusters=k, linkage=linkage)
            score = silhouette_score(
                X_scaled, labels, sample_size=min(500, len(X_scaled)), random_state=42
            )
            candidates.append((score, linkage, k, labels))
            print(f"  {linkage:8s} k={k}: silhouette={score:.3f}")

    best_score, best_linkage, best_k, best_labels = max(candidates, key=lambda item: item[0])
    counts = np.bincount(best_labels)
    print(
        f"\nSelected {best_linkage} linkage with k={best_k} "
        f"(silhouette={best_score:.3f})."
    )
    print("Cluster sizes:", ", ".join(f"{i}: {n}" for i, n in enumerate(counts)))

    result_df = df.copy()
    result_df["cluster"] = best_labels
    output_dir = PROJECT_DIR / "demo" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_dir / "agglomerative_clustered_data.csv", index=False)

    fig_clusters, _ = plot_clusters_2d(
        X_scaled, best_labels,
        title=f"Agglomerative clustering ({best_linkage}, k={best_k})",
    )
    fig_clusters.savefig(output_dir / "agglomerative_clusters.png", dpi=200)

    fig_scores, ax = plt.subplots(figsize=(10, 5))
    for linkage in linkage_options:
        rows = [item for item in candidates if item[1] == linkage]
        ax.plot([item[2] for item in rows], [item[0] for item in rows], marker="o", label=linkage)
    ax.set(xlabel="Number of clusters", ylabel="Sampled silhouette score",
           title="Agglomerative model comparison")
    ax.set_xticks(list(cluster_counts))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Linkage")
    fig_scores.tight_layout()
    fig_scores.savefig(output_dir / "agglomerative_model_comparison.png", dpi=200)

    print("Saved labelled data and plots to:", output_dir)
    plt.show()


if __name__ == "__main__":
    main()
