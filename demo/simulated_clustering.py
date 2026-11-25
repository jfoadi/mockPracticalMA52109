###
## Simulating clustering
## Katherine Hall - University of Bath
## November 25 2026
###

from __future__ import annotations

import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing all the clustering tools 
from cluster_maker import (
    select_features,
    standardise_features,
    calculate_descriptive_statistics,
    calculate_correlation,
    kmeans,
    compute_inertia,
    silhouette_score_sklearn,
    plot_clusters_2d,
    plot_elbow,
    export_to_csv,
    run_clustering,
)

OUTPUT_DIR = "demo_output"


def main(args: List[str] = None) -> None:

    input_path = args[0] if args else os.path.join("data", "simulated_data.csv")

    if not os.path.exists(input_path):
        print(f"Error: input file '{input_path}' not found.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))

    print("==================================================")
    print(f" Cluster Analysis: {input_path}")
    print("==================================================")

###
##Data loading and exploratory data analysis
###

    df = pd.read_csv(input_path)

    #automatically identify numeric columns
    numeric_cols = [
        col for col in df.columns 
        if pd.api.types.is_numeric_dtype(df[col])
        ]
    if len(numeric_cols) < 2:
        print("Error: Input CSV must have at least two numeric feature columns.")
        sys.exit(1)

    feature_cols = numeric_cols[:2]  #selects the first two numerical columns
    print(f"Selected Feature Columns: {feature_cols}") #prints these columns

    #computes descriptive statistics
    stats_df = calculate_descriptive_statistics(df[feature_cols])
    print("\n--- Descriptive Statistics ---")
    print(stats_df)

    #computes the correlation matrix 
    corr_df = calculate_correlation(df[feature_cols])
    print("\n--- Feature Correlation Matrix ---")
    print(corr_df)

    #extracting the features column
    X_df = select_features(df, feature_cols)
    X_raw = X_df.to_numpy(dtype=float)

    #standardising features to zero mean and unit variance 
    X_scaled = standardise_features(X_raw)

### 
## Evaluating cluster quality for k=2,3,4,5,6,7
###

    k_range = list(range(2, 8))
    inertia_scores: Dict[int, float] = {}
    silhouette_scores: Dict[int, float] = {}

    print("\n--- Evaluating Cluster Quality Across Range k = 2 to 7 ---")
    for k in k_range:
        #runing the K-means algorithm from cluster_maker.algorithms
        labels, centroids = kmeans(X_scaled, k=k, random_state=42)

        #calculating within-cluster sum of squares (inertia)
        inertia = compute_inertia(X_scaled, labels, centroids)
        inertia_scores[k] = inertia

        #calculatig the cluster separation score (silhouette score)
        try:
            sil = silhouette_score_sklearn(X_scaled, labels)
        except ValueError:
            sil = 0.0
        silhouette_scores[k] = sil

        print(f"k = {k:2d} | Inertia: {inertia:10.4f} | Silhouette Score: {sil:.4f}")

    #determining the optimal k based on maximum silhouette score
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print("\n--------------------------------------------------")
    print(f" Optimal Cluster Count: k = {best_k} (Peak Silhouette: {silhouette_scores[best_k]:.4f})")
    print("--------------------------------------------------")

### 
## Plotting the elbow curve
###

    fig_elbow, _ = plot_elbow(
        k_values=k_range,
        inertias=[inertia_scores[k] for k in k_range],
        title=f"Inertia Elbow Curve ({base})",
    )
    elbow_plot_path = os.path.join(OUTPUT_DIR, f"{base}_elbow_curve.png")
    fig_elbow.savefig(elbow_plot_path, dpi=150)
    plt.close(fig_elbow)
    print(f"Saved Elbow Plot: {elbow_plot_path}")


###
## Plotting the silhouette score comparison
###

    fig_sil, ax_sil = plt.subplots(figsize=(7, 4.5))
    bars = ax_sil.bar(list(silhouette_scores.keys()), list(silhouette_scores.values()), color="skyblue", edgecolor="navy")

    #highlighting the optimal k bar
    best_idx = list(silhouette_scores.keys()).index(best_k)
    bars[best_idx].set_color("dodgerblue")
    bars[best_idx].set_edgecolor("darkblue")
    bars[best_idx].set_linewidth(2)

    ax_sil.set_xlabel("Number of Clusters (k)")
    ax_sil.set_ylabel("Silhouette Score")
    ax_sil.set_title(f"Silhouette Score Evaluation (Optimal k = {best_k})")
    ax_sil.grid(axis="y", linestyle="--", alpha=0.7)

    sil_plot_path = os.path.join(OUTPUT_DIR, f"{base}_silhouette_scores.png")
    fig_sil.tight_layout()
    fig_sil.savefig(sil_plot_path, dpi=150)
    plt.close(fig_sil)
    print(f"Saved Silhouette Plot: {sil_plot_path}")



###
## Plotting the optimal cluster scatter plot
###

    clustering_result = run_clustering(
        input_path=input_path,
        feature_cols=feature_cols,
        algorithm="kmeans",
        k=best_k,
        standardise=True,
        output_path=os.path.join(OUTPUT_DIR, f"{base}_clustered_optimal_k{best_k}.csv"),
        random_state=42,
    )

    #saving the 2D cluster plot
    fig_cluster = clustering_result["fig_cluster"]
    cluster_plot_path = os.path.join(OUTPUT_DIR, f"{base}_optimal_clusters_k{best_k}.png")
    fig_cluster.savefig(cluster_plot_path, dpi=150)
    plt.close(fig_cluster)
    print(f"Saved Cluster Scatter Plot: {cluster_plot_path}")

###
##Summary of the data
### 

    summary_df = pd.DataFrame({
        "k": k_range,
        "inertia": [inertia_scores[k] for k in k_range],
        "silhouette": [silhouette_scores[k] for k in k_range],
    })
    metrics_summary_path = os.path.join(OUTPUT_DIR, f"{base}_metrics_summary.csv")
    export_to_csv(summary_df, metrics_summary_path)
    print(f"Saved Summary CSV: {metrics_summary_path}")

    print("\n==================================================")
    print(" Analysis Completed Successfully!")
    print(f" Outputs saved in directory: {OUTPUT_DIR}/")
    print("==================================================")


if __name__ == "__main__":
    main(sys.argv[1:])