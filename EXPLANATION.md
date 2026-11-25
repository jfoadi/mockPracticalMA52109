# Explanation of the cluster_maker Package

## 1. Errors in the Original Demo Script and Fixes

The original `cluster_plot.py` script contained several logical errors.

- The number of clusters was incorrectly limited using `min(k, 3)`. This meant that when the script attempted to run clustering with 4 and 5 clusters, it actually used 3 clusters. This was corrected by passing `k=k` directly to the clustering function.
- The script checked for a metric named `silhouette_score`, but the `cluster_maker` interface stores the metric under the name `silhouette`. The condition was corrected.
- The silhouette plot also attempted to access the incorrect `silhouette_score` column. This was changed to `silhouette`.

These corrections allow the script to test all four requested cluster counts and generate the silhouette comparison plot.

## 2. What the Corrected Demo Script Does

The corrected `cluster_plot.py` script accepts a CSV file as a command-line argument and analyses its numerical data.

It selects the first two numerical columns as clustering features and runs K-means clustering for k = 2, 3, 4 and 5.

For each clustering experiment, the script:

- Runs the clustering algorithm.
- Calculates inertia and silhouette score.
- Saves the dataset with the assigned cluster labels.
- Generates and saves a two-dimensional cluster plot.

It also creates a metrics CSV file and a bar chart comparing silhouette scores for the different values of k.

The script helps compare different clustering solutions and assess which number of clusters provides the best separation.

## 3. Overview of the cluster_maker Package

The `cluster_maker` package provides tools for generating, preprocessing, clustering, evaluating and visualising datasets.

### Data generation and analysis

- `dataframe_builder.py`: Creates DataFrames describing cluster centres and generates synthetic clustered datasets.
- `data_analyser.py`: Calculates descriptive statistics and correlations between numerical variables.
- `data_exporter.py`: Exports DataFrames to CSV files or formatted text.

### Preprocessing

- `preprocessing.py`: Selects numerical feature columns and standardises data using StandardScaler. This prepares data for clustering.

### Clustering algorithms

- `algorithms.py`: Implements manual K-means clustering and provides a wrapper around scikit-learn's KMeans.

### Evaluation

- `evaluation.py`: Calculates clustering inertia and silhouette scores, and provides an elbow method for comparing different numbers of clusters.

### Plotting

- `plotting_clustered.py`: Provides functions to visualise clusters in two dimensions and plot elbow curves.

### High-level interface

- `interface.py`: Connects the different package components into a complete clustering workflow, including loading data, preprocessing, clustering, evaluation, plotting and optional CSV export.

Overall, the package provides a modular set of tools for performing and analysing clustering experiments in Python.
### Results and Interpretation

The experiment produced the highest silhouette score for k = 3 (approximately 0.725). This suggests that three clusters provide the best separation among the four tested configurations. Although inertia decreases as the number of clusters increases, the silhouette results indicate that adding more clusters does not necessarily improve cluster separation.