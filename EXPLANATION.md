# Explanation of the Clustering Demo

## 1. Original Errors and Corrections

Two logical errors were identified in the original `cluster_plot.py` script.

### Incorrect number of clusters

The argument `k=min(k, 3)` restricted the maximum number of clusters to three. Consequently, the iterations for k = 3, 4 and 5 produced identical clustering results, inertia values and silhouette scores.

This was corrected by replacing `k=min(k, 3)` with `k=k`, ensuring that the clustering algorithm uses the intended number of clusters in each iteration.

### Incorrect silhouette column name

The script searched for a column named `silhouette_score` in the metrics DataFrame. However, the `run_clustering()` function returns this metric under the name `silhouette`.

This prevented the silhouette plot from being generated. Both references to `silhouette_score` were replaced with `silhouette` to correct this issue.

## 2. Functionality of the Corrected Demo

The corrected script reads the input CSV file and selects its first two numeric columns as clustering features.

The features are standardised before applying the manual K-means algorithm for k = 2, 3, 4 and 5. A fixed random state ensures reproducibility.

For each value of k, the script:

- Performs K-means clustering.
- Calculates inertia and the silhouette score.
- Exports the dataset with its assigned cluster labels.
- Generates a two-dimensional cluster plot showing the data points and centroids.

Finally, the script exports a summary of the evaluation metrics and generates a bar chart comparing silhouette scores.

The corrected results show that inertia decreases as k increases. The highest silhouette score is approximately 0.725 at k = 3, suggesting that three clusters provide the most appropriate separation among the tested values.

## 3. Overview of the cluster_maker Package

The `cluster_maker` package provides tools for generating, preprocessing, analysing, clustering and visualising data.

Its main components are:

- **preprocessing.py:** Selects numeric features, validates requested columns and standardises data using `StandardScaler` so that differences in feature scales do not disproportionately influence clustering.

- **algorithms.py:** Implements manual K-means clustering, including centroid initialisation, cluster assignment and centroid updates. It also provides a scikit-learn implementation of K-means.

- **evaluation.py:** Evaluates clustering performance using inertia and silhouette scores. It also supports the elbow method for comparing different numbers of clusters.

- **plotting_clustered.py:** Generates two-dimensional cluster plots, optionally displaying centroids, and elbow plots for visualising inertia against the number of clusters.

- **interface.py:** Provides the high-level `run_clustering()` function, which combines data loading, preprocessing, clustering, evaluation, plotting and optional CSV export into a single workflow.

- **dataframe_builder.py:** Provides functions for defining data structures and generating synthetic data.

- **data_analyser.py:** Provides functions for calculating descriptive statistics and correlations.

- **data_exporter.py:** Provides functions for exporting data.

- **\_\_init\_\_.py:** Exposes the package's public functions, allowing users to import them directly from `cluster_maker`.

Together, these modules provide a structured and reusable framework for performing clustering analysis.