# Explanation of the Clustering Demo

## 1. Errors in the original script

Two errors were identified and corrected in `demo/cluster_plot.py`.

**Incorrect number of clusters**

The original script used `k=min(k, 3)` when calling `run_clustering()`. This restricted the number of clusters to a maximum of three, even when the loop requested four or five clusters.

This was corrected to `k=k`, ensuring that the script performs clustering with the intended values of 2, 3, 4 and 5.

**Incorrect silhouette metric name**

The original script checked whether `silhouette_score` was present in the metrics DataFrame. However, the clustering function returns this metric under the name `silhouette`.

Consequently, the condition was never satisfied and the silhouette comparison plot was not generated.

The condition and plotting code were updated to reference `silhouette`, allowing the plot to be produced correctly.

## 2. What the corrected script does

The script reads the input CSV file and selects its first two numeric columns for clustering.

It then performs k-means clustering for k = 2, 3, 4 and 5, with standardisation enabled.

For each value of k, the script:

- Saves a CSV file containing the clustering results.
- Generates a plot showing the clusters.
- Calculates and reports clustering metrics, including inertia and silhouette score.

The script also saves a combined metrics CSV file and a bar chart comparing silhouette scores across the four values of k.

For the supplied `demo_data.csv`, k = 3 produces the highest silhouette score (approximately 0.725), suggesting that three clusters provide the strongest separation among the tested options.

## 3. Overview of the cluster_maker package

The `cluster_maker` package provides tools for preparing data, performing clustering analysis, evaluating results and producing visualisations.

Its main components are:

- **`preprocessing.py`:** Handles data preparation, including standardisation of features before clustering.
- **`algorithms.py`:** Contains clustering algorithms, including k-means, centroid initialisation and assignment of observations to clusters.
- **`evaluation.py`:** Provides metrics for assessing clustering quality, including inertia and silhouette scores.
- **`plotting_clustered.py`:** Provides functionality for visualising clustering results.
- **`interface.py`:** Provides the main interface for running the clustering pipeline, connecting the different components.
- **`data_analyser.py`, `dataframe_builder.py` and `data_exporter.py`:** Support data analysis, construction of output DataFrames and exporting results.

Together, these modules allow users to perform clustering analysis through a reusable interface rather than implementing each step separately.