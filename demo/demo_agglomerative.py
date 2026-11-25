###
## Demo agglomerative
## Katherine Hall - University of Bath
## November 25 2026
###

"""
Demo: agglomerative (hierarchical) clustering on a difficult dataset.

Usage (from the project root):
    python demo/demo_agglomerative.py
    python demo/demo_agglomerative.py data/difficult_dataset.csv --k 4
    python demo/demo_agglomerative.py --linkage average --n-neighbors 10

The script
  1. plots the raw data and the hierarchical merge distances (evidence for k),
  2. shows that k-means, which assumes compact round clusters, struggles,
  3. compares every linkage with and without outlier handling,
  4. plots and saves the final agglomerative clustering.

All figures and the labelled data are written to demo_output/.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster_maker import (
    select_features,
    standardise_features,
    calculate_descriptive_statistics,
    kmeans,
    silhouette_score_sklearn,
    plot_clusters_2d,
    export_to_csv,
    agglomerative_clustering,
    compare_linkages,
    merge_distances,
)
from cluster_maker.agglomerative import VALID_LINKAGES

OUTPUT_DIR = "demo_output"

#look at the raw-data plot (figure 1) first, then set the number of clusters you can see. 
#it can also be overridden with --k on the command line.
DEFAULT_K = 3


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "input_path", nargs="?", default=os.path.join("data", "difficult_dataset.csv")
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="number of clusters")
    parser.add_argument(
        "--linkage", choices=VALID_LINKAGES, default="single",
        help="linkage for the final clustering (default: single)",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=None,
        help="restrict merges to a k-nearest-neighbour graph (default: off)",
    )
    parser.add_argument(
        "--no-outlier-handling", action="store_true",
        help="do not set aside isolated points before clustering",
    )
    parser.add_argument(
        "--features", nargs=2, default=None, metavar=("X", "Y"),
        help="names of the two columns to cluster (default: first two numeric)",
    )
    return parser.parse_args(argv)


def save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if not os.path.exists(args.input_path):
        print(f"Error: input file '{args.input_path}' not found.")
        sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f" Agglomerative clustering demo: {args.input_path}")
    print("=" * 60)

#loading data 

    df = pd.read_csv(args.input_path)
    if args.features is not None:
        feature_cols = list(args.features)
    else:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            print("Error: the CSV must have at least two numeric columns.")
            sys.exit(1)
        feature_cols = numeric_cols[:2]
    print(f"Features: {feature_cols}   k = {args.k}")

    print("\n--- Descriptive statistics ---")
    print(calculate_descriptive_statistics(df[feature_cols]))

    X_raw = select_features(df, feature_cols).to_numpy(dtype=float)
    X = standardise_features(X_raw)


#figure 1: raw data and merge distances
   
    fig, (ax_raw, ax_merge) = plt.subplots(1, 2, figsize=(12, 5))
    ax_raw.scatter(X[:, 0], X[:, 1], s=12, color="grey", alpha=0.7)
    ax_raw.set_title("Raw data (standardised)")
    ax_raw.set_xlabel(feature_cols[0])
    ax_raw.set_ylabel(feature_cols[1])

    n_last = 12
    for linkage in ("single", "average", "ward"):
        heights = merge_distances(X, linkage=linkage)
        last = heights[-n_last:]
        ax_merge.plot(range(n_last + 1, 1, -1), last, marker="o", label=linkage)
    ax_merge.invert_xaxis()
    ax_merge.set_yscale("log")
    ax_merge.set_xlabel("Number of clusters remaining after the merge")
    ax_merge.set_ylabel("Merge distance (log scale)")
    ax_merge.set_title("Last merges: a jump suggests a good k")
    ax_merge.legend()
    ax_merge.grid(alpha=0.3)
    save(fig, "01_raw_data_and_merge_distances.png")

   
#figure 2: k-means baseline
   
    km_labels, km_centroids = kmeans(X, k=args.k, random_state=42)
    fig_km, _ = plot_clusters_2d(
        X, km_labels, centroids=km_centroids, title=f"Baseline: k-means (k={args.k})"
    )
    save(fig_km, "02_kmeans_baseline.png")

    
#figure 3: every linkage, with and without outlier handling
    
    print("\n--- Linkage comparison ---")
    linkages = list(VALID_LINKAGES)
    fig, axes = plt.subplots(2, len(linkages), figsize=(4 * len(linkages), 8))
    for row, handle in enumerate((False, True)):
        results = compare_linkages(
            X, args.k, linkages=linkages,
            n_neighbors=args.n_neighbors, handle_outliers=handle,
        )
        for col, linkage in enumerate(linkages):
            res = results[linkage]
            sil = res["silhouette"]
            sil_txt = "n/a" if sil is None else f"{sil:.3f}"
            axes[row, col].scatter(
                X[:, 0], X[:, 1], c=res["labels"], cmap="tab10", s=10
            )
            axes[row, col].set_title(
                f"{linkage}, outliers {'handled' if handle else 'ignored'}\n"
                f"silhouette {sil_txt}, smallest cluster {int(res['sizes'].min())}",
                fontsize=9,
            )
            print(
                f"{linkage:9s} outliers {'handled' if handle else 'ignored':7s} | "
                f"silhouette {sil_txt:>6s} | sizes {res['sizes'].tolist()}"
            )
    fig.suptitle(
        "Agglomerative clustering: linkage and outlier handling "
        "(silhouette favours compact clusters, so judge by the picture too)"
    )
    save(fig, "03_linkage_comparison.png")

## final clustering

    handle = not args.no_outlier_handling
    labels, centroids = agglomerative_clustering(
        X, args.k, linkage=args.linkage,
        n_neighbors=args.n_neighbors, handle_outliers=handle,
    )
    fig_final, _ = plot_clusters_2d(
        X, labels, centroids=centroids,
        title=f"Agglomerative ({args.linkage} linkage, k={args.k})",
    )
    save(fig_final, "04_agglomerative_final.png")

    out_df = df.copy()
    out_df["cluster"] = labels
    out_path = os.path.join(OUTPUT_DIR, "difficult_dataset_agglomerative.csv")
    export_to_csv(out_df, out_path)
    print(f"Saved: {out_path}")

    print("\n--- Final result ---")
    print(f"Linkage: {args.linkage} | outlier handling: {handle} | "
          f"n_neighbors: {args.n_neighbors}")
    print(f"Cluster sizes: {np.bincount(labels).tolist()}")
    try:
        print(f"Silhouette (final):   {silhouette_score_sklearn(X, labels):.3f}")
        print(f"Silhouette (k-means): {silhouette_score_sklearn(X, km_labels):.3f}")
    except ValueError:
        pass
    print(
        "Note: the silhouette score rewards compact, round clusters, so it can be "
        "lower for a clustering that correctly follows curved or ring-shaped "
        "structure. Compare 02_kmeans_baseline.png with 04_agglomerative_final.png."
    )


if __name__ == "__main__":
    main(sys.argv[1:])