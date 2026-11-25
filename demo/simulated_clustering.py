###
## cluster_maker: simulated data clustering demo
##
## This script compares K-means clusterings for k = 2 through 5 using all
## numeric features in simulated_data.csv. It selects the k with the highest
## silhouette score and saves cluster plots, metrics, and labelled data.
###

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cluster_maker import run_clustering

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data" / "simulated_data.csv"
OUTPUT_DIR = ROOT_DIR / "demo_output"
K_VALUES = (2, 3, 4, 5)
RANDOM_STATE = 42


def main() -> None:
    # Load the dataset and identify numeric features for the analysis.
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    data = pd.read_csv(INPUT_PATH)
    feature_cols = list(data.select_dtypes(include="number").columns)
    if len(feature_cols) < 2:
        raise ValueError("The input CSV must have at least two numeric features.")
    if len(data) < max(K_VALUES):
        raise ValueError("The dataset has too few rows for the candidate k values.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    metrics_summary = []

    # Compare candidate cluster counts, saving each labelled dataset and plot.
    for k in K_VALUES:
        print(f"\n=== Running K-means with k = {k} ===")
        result = run_clustering(
            input_path=str(INPUT_PATH),
            feature_cols=feature_cols,
            algorithm="sklearn_kmeans",
            k=k,
            standardise=True,
            output_path=str(OUTPUT_DIR / f"simulated_data_clustered_k{k}.csv"),
            random_state=RANDOM_STATE,
        )

        plot_path = OUTPUT_DIR / f"simulated_data_k{k}.png"
        result["fig_cluster"].savefig(plot_path, dpi=150)
        plt.close(result["fig_cluster"])

        metrics = {"k": k}
        metrics.update(result["metrics"])
        metrics_summary.append(metrics)
        print("Metrics:")
        for key, value in result["metrics"].items():
            print(f"  {key}: {value}")

    # Select the best-scoring k, then save its cluster plot and elbow curve.
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv(OUTPUT_DIR / "simulated_data_metrics.csv", index=False)
    selected_k = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])

    selected_result = run_clustering(
        input_path=str(INPUT_PATH),
        feature_cols=feature_cols,
        algorithm="sklearn_kmeans",
        k=selected_k,
        standardise=True,
        output_path=str(OUTPUT_DIR / "simulated_data_clustered.csv"),
        random_state=RANDOM_STATE,
        compute_elbow=True,
        elbow_k_values=list(range(1, max(K_VALUES) + 1)),
    )
    selected_result["fig_cluster"].savefig(
        OUTPUT_DIR / "simulated_data_selected_clusters.png", dpi=150
    )
    plt.close(selected_result["fig_cluster"])
    selected_result["fig_elbow"].savefig(
        OUTPUT_DIR / "simulated_data_elbow.png", dpi=150
    )
    plt.close(selected_result["fig_elbow"])

    # Plot silhouette scores to show how the candidate clusterings compare.
    if "silhouette" in metrics_df.columns:
        fig, ax = plt.subplots()
        ax.bar(metrics_df["k"], metrics_df["silhouette"])
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Silhouette scores for candidate cluster counts")
        ax.set_xticks(metrics_df["k"])
        ax.axvline(
            selected_k,
            color="black",
            linestyle="--",
            label=f"Selected k={selected_k}",
        )
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "simulated_data_silhouette.png", dpi=150)
        plt.close(fig)

    best_silhouette = metrics_df.loc[
        metrics_df["k"] == selected_k, "silhouette"
    ].iloc[0]
    print("\nMetrics across candidate cluster counts:")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(
        f"\nSelected k={selected_k}: it has the highest silhouette score "
        f"({best_silhouette:.3f}) among the candidates. The elbow plot provides "
        "a second view of how much inertia improves as k increases."
    )
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
