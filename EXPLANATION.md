
# Task 2 - Clustering Demo Explanation

## 1. Problems in the Original Script and Corrections

The original `cluster_plot.py` script was designed to perform K-means clustering for k = 2, 3, 4 and 5. However, the following argument was passed to the `run_clustering()` function:

`k=min(k, 3)`

This restricted the maximum number of clusters to three. Consequently, the iterations for k = 4 and k = 5 incorrectly performed clustering with k = 3, producing misleading plots (only 3 clusters for both k = 4 and 5), exported datasets and evaluation metrics (again, returning same results as k = 3 for k = 4 and 5).

This was corrected by replacing `k=min(k, 3)` with `k=k`, ensuring that each iteration uses the intended number of clusters.

Another logical error was the inconsistent naming of the silhouette metric. The `interface.py` module returns the silhouette score using the key `silhouette`, whereas the demo script attempted to access `silhouette_score`. This prevented the silhouette comparison graph from being generated. The error was corrected by using `silhouette` consistently in both the conditional statement and the plotting command.

Additional improvements include:

- Removing the unused NumPy import to reduce unnecessary code.
- Improving command-line usage instructions and providing an example command.
- Adding a meaningful docstring to the `main()` function describing its purpose, parameters and outputs.

These changes improve the correctness, readability, usability and maintainability of the script.

## 2. Functionality of the Corrected Script

The corrected script reads an input CSV file, selects its first two numerical columns and standardises the selected features.

It then performs manual K-means clustering for k = 2, 3, 4 and 5 using a fixed random seed to ensure reproducibility.

For each value of k, the script:

- Performs K-means clustering and assigns observations to their nearest centroids.
- Exports the original dataset with its predicted cluster labels to a CSV file.
- Generates a two-dimensional scatter plot displaying the clusters and their centroids.
- Calculates and displays evaluation metrics, including inertia and the silhouette score where defined.

After completing all four clustering configurations, the script exports a combined metrics summary as a CSV file and generates a bar chart comparing the silhouette scores for different values of k.

All generated datasets, plots and evaluation results are saved in the `demo_output/` directory. Figures are closed after saving so that the script can continue without interruption.

The script can be executed from the project root using:

```bash
python demo/cluster_plot.py data/demo_data.csv
```

## 3. Overview of the cluster_maker Package

The `cluster_maker` package is an educational Python package designed to generate synthetic clustered data, preprocess datasets, perform clustering, evaluate clustering performance and visualise results.

The package follows a modular structure in which different components are responsible for specific stages of the clustering workflow.

It has the following modules - 

'dataframe_builder.py` - Defines cluster centres and generates synthetic clustered datasets using Gaussian noise.

`data_analyser.py` - Calculates descriptive statistics and correlation matrices to support exploratory data analysis.

`preprocessing.py` - Selects numerical feature columns and standardises features using scikit-learn's StandardScaler.

`algorithms.py` - Provides manual NumPy-based and scikit-learn implementations of K-means, including centroid initialisation, cluster assignment and centroid updates

`evaluation.py` -  Evaluates clustering performance using inertia and silhouette scores. It also calculates inertia for multiple values of k for the elbow method.

`plotting_clustered.py` - Generates two-dimensional cluster visualisations, displays centroid positions and produces elbow plots using Matplotlib.

`data_exporter.py` - Exports DataFrames as CSV files or formatted text tables.

`interface.py` - Coordinates the complete clustering workflow through the run_clustering() function, integrating data loading, preprocessing, clustering, evaluation, visualisation and data export.

`__init__.py` - Exposes the public functions of the package, allowing users to access its functionality through a consistent import interface. 

The modular organisation of the package separates computation, data processing, visualisation and user interaction. This improves readability, maintainability, reusability and ease of testing.
