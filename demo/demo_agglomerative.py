"""
Demonstration and evaluation of agglomerative hierarchical clustering.

This script demonstrates the effectiveness of the agglomerative
clustering module on the difficult_dataset.csv dataset.

The dataset exhibits a concentric-ring structure, presenting a
challenge for conventional centroid-based algorithms such as K-means,
which generally favour compact clusters.

The analysis consists of the following stages:

    1. Load and inspect the original two-dimensional dataset.
    2. Visualise the underlying structure using scatter plots.
    3. Calculate radial distances to investigate concentric groups.
    4. Compare candidate cluster counts using silhouette scores.
    5. Compare K-means, single-linkage hierarchical clustering,
       Cartesian Ward clustering and radial Ward clustering.
    6. Visualise the resulting cluster assignments.
    7. Analyse cluster sizes and radial distance statistics.
    8. Export the selected clustering, evaluation metrics and plots.

The analysis investigates how hierarchical clustering combined with
appropriate feature engineering can identify structures that may not
be captured effectively by conventional K-means clustering.

All clustering operations use the functions provided by the
cluster_maker package. Supporting visualisations are produced
using Matplotlib.

Input:
    data/difficult_dataset.csv

Output directory:
    demo_output/agglomerative/

Execution from the project root:
    python demo/demo_agglomerative.py

The script includes input validation, informative error messages
and figure resource management to ensure reliable execution.
"""

from pathlib import Path
import sys

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from cluster_maker import (
        agglomerative,
        select_features,
        sklearn_kmeans,
        silhouette_score_sklearn,
        plot_clusters_2d,
        export_to_csv,
    )
except ImportError as exc:
    raise SystemExit(f"Required dependency unavailable: {exc}") from exc

INPUT_PATH = Path("data/difficult_dataset.csv")
OUTPUT_DIR = Path("demo_output/agglomerative")
K = 4


def save_plot(fig: plt.Figure, filename: str) -> None:
    """Save a figure and release its resources even if saving fails."""
    try:
        fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


def main() -> None:
    """Compare clustering methods, inspect radial structure and export labels."""
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_PATH}")
    data = pd.read_csv(INPUT_PATH)
    X = select_features(data, ["x", "y"]).to_numpy(dtype=float)
    if len(X) <= K or len(np.unique(X, axis=0)) < K:
        raise ValueError("At least five observations and four distinct points are required.")
    if not np.all(np.isfinite(X)):
        raise ValueError("The dataset contains NaN or infinite values.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Equal axis scaling preserves the circular geometry of the observations.
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.7)
    ax.set(xlabel="x", ylabel="y", title="Original difficult dataset", aspect="equal")
    save_plot(fig, "original_data.png")

    # Distance from the origin distinguishes rings with the same centre.
    # Keep x and y in their original common scale to preserve geometry.
    radius = np.linalg.norm(X, axis=1).reshape(-1, 1)
    fig, ax = plt.subplots()
    ax.hist(radius[:, 0], bins=50, edgecolor="white")
    ax.set(xlabel="Distance from origin", ylabel="Frequency", title="Radial distribution")
    save_plot(fig, "radial_distribution.png")

    # Compare plausible radial cluster counts, without relying on the
    # silhouette score in Cartesian space to identify concentric groups.
    radial_scores = []
    max_k = min(8, len(X) - 1, len(np.unique(radius)))
    for k in range(2, max_k + 1):
        try:
            labels, _ = agglomerative.agglomerative_clustering(radius, k, linkage="ward")
            score = silhouette_score_sklearn(radius, labels)
        except ValueError as exc:
            print(f"Skipping radial k={k}: {exc}", file=sys.stderr)
            continue
        radial_scores.append({"k": k, "silhouette": score})
    if not radial_scores:
        raise ValueError("No valid radial clustering could be evaluated.")
    scores = pd.DataFrame(radial_scores)
    export_to_csv(scores, str(OUTPUT_DIR / "radial_scores.csv"))
    fig, ax = plt.subplots()
    ax.plot(scores["k"], scores["silhouette"], marker="o")
    ax.set(xlabel="Number of clusters", ylabel="Radial silhouette score", title="Radial cluster comparison")
    ax.set_xticks(scores["k"])
    save_plot(fig, "radial_scores.png")

    # Compare methods at k=4, motivated by the four apparent radial bands.
    km_labels, _ = sklearn_kmeans(X, K, random_state=42)
    single_labels, _ = agglomerative.agglomerative_clustering(X, K, linkage="single")
    ward_labels, _ = agglomerative.agglomerative_clustering(X, K, linkage="ward")
    radial_labels, _ = agglomerative.agglomerative_clustering(radius, K, linkage="ward")

    methods = {
        "kmeans": km_labels,
        "single_linkage": single_labels,
        "cartesian_ward": ward_labels,
        "radial_ward": radial_labels,
    }
    print("=== Agglomerative clustering demonstration ===")
    print(f"Observations: {len(X)} | Features: x, y | Candidate k: {K}")
    print("\nRadial silhouette scores:")
    print(scores.round(4).to_string(index=False))

    # Use the package plotting function for direct visual comparisons.
    # Centroids are omitted because ring centres do not describe ring shape.
    for name, labels in methods.items():
        fig, ax = plot_clusters_2d(X, labels, title=f"{name.replace('_', ' ').title()} (k={K})")
        ax.set(xlabel="x", ylabel="y", aspect="equal")
        save_plot(fig, f"{name}.png")
        print(f"\n{name}: cluster sizes {np.bincount(labels).tolist()}")

    # Export the radial Ward solution with original coordinates retained.
    labelled = data.copy()
    labelled["radius"] = radius[:, 0]
    labelled["cluster"] = radial_labels
    export_to_csv(labelled, str(OUTPUT_DIR / "clustered_data.csv"))
    profile = labelled.groupby("cluster")["radius"].agg(["count", "mean", "std", "min", "max"])
    profile = profile.sort_values("mean")
    export_to_csv(profile.reset_index(), str(OUTPUT_DIR / "cluster_profiles.csv"))

    print("\nRadial cluster profiles:")
    print(profile.round(3).to_string())
    best = scores.loc[scores["silhouette"].idxmax()]
    print(f"\nHighest radial silhouette: k={int(best['k'])}, score={best['silhouette']:.4f}")
    print("\nInterpretation: K-means favours compact groups, whereas the data")
    print("appear to contain four rings with a shared centre. Single linkage")
    print("can follow curved groups but may merge rings through noise bridges.")
    print("Ward clustering on radius separates observations by ring distance.")
    print("Confirm k=4 using the histogram, radial scores and cluster plots;")
    print("the radial silhouette is not directly comparable to Cartesian scores.")
    print(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError) as e:
        print(f"Clustering demonstration failed: {e}", file=sys.stderr)
        sys.exit(1)

