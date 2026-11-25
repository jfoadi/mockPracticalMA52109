# Explanation for Task 2


## Part 1 - Explanation of why the code was wrong
In line 62 of cluster_plot.py in a for loop which was iterating k from 2 to 5 this line was there `k = min(k, 3)`. As you can see from the code k would equal the smallest of the two options which are k or 3 meaning when in the loop k = 4 or 5 the line would change k to equal 3. This would mean when it was meant to test k = 4 or 5 it was just testing k = 3 again 2 more times. The fix used was to make `k=k` so that k could equal to 4 and 5.

In line 89 and 91 "silhouette_score" was used as shown below.
```python
if "silhouette_score" in metrics_df.columns:
    ...
    plt.bar(metrics_df["k"], metrics_df["silhouette_score"])
```

The problem with this is that the score returned by the function `run_clustering` is not called "silhouette_score". This meant that the plt.bar was never running because of the if statement.
The fix that was used to solve this was to change "silhouette_score" to the actual name of the key returned from `run_clustering` which is "silhouette"

After changing these fixes I checked the output of the code and saw the inertia changing and that demo_data_silhouette.png would now be created.

## Part 2 - What the corrected script now does
### How the code is run
The code is run from the repository root `python demo/cluster_plot.py data/demo_data.csv`.
As shown it takes one argument which is the path of a csv file.

### Input Checks (line 27-47)
Firstly the python file checks that the inputs all make sense. It first makes sure there is exactly one argument. If the argument or lack of argument is wrong it prints a usage message and exits. If the right argument is inputted but the file does not exist then it prints an error and exits. If the file finds the csv file it looks for the first 2 numeric columns and ignores the rest. If the file contains less than 2 numeric columns it prints an error and exits. If the subdirectory demo_output/ does not exist then one is created. 

### The main loop (line 55-82)
This part runs this for loop where k = 2,3,4,5. For each k value the function `run_clustering` is run in which it uses the package's manual k-means implementation. As in the code it says `standardise=True` all the features are scaled to mean 0 and variance 1 before the clustering and a random_state of 42 is used so the results are reproducible. At the end of the loop it prints the inertia and the silhouette score for that k that was just completed.

### Outputs
The file will create 4 types of outputs but 10 files in total. There's one CSV and one PNG per k (8 files), plus the metrics CSV and the silhouette chart. These are created into the subdirectory called demo_output. The first output is demo_data_clustered_k{k}.csv which contains the original data plus a cluster label column. The second output is demo_data_k{k}.png which contains a 2D scatter of the clusters with centroids. The third output is demo_data_metrics.csv which contains the inertia and silhouette for every k. Finally the last output contains demo_data_silhouette.png which is a picture of the bar chart of silhouette score vs k.

## Part 3 - Overview of the cluster_maker package
The package cluster_maker is used to generate synthetic clustered data, preprocessing it, running k-means clustering, evaluating the results and plotting them.

| Module | Key Functions | Purpose |
|---|---|---|
|`dataframe_builder.py`|`define_dataframe_structure`, `simulate_data`|Defines cluster centres and then generates Gaussian noisy points around them with a true_cluster column|
|`data_analyser.py`|`calculate_descriptive_statistics`, `calculate_correlation`|Creates summary statistics and a correlation matrix for a DataFrame|
|`data_exporter.py`|`export_to_csv`, `export_formatted`|Save DataFrames as CSV or as a formatted text table|
|`preprocessing.py`|`select_features`, `standardise_features`|Picks feature columns with an error if any are missing or non-numeric and then scales them to mean 0, variance 1|
|`algorithms.py`|`init_centroids`, `assign_clusters`, `update_centroids`, `kmeans`, `sklearn_kmeans`|a Manual k-means is built from three steps, plus a wrapper around scikit-learn's KMeans|
|`evaluation.py`|`compute_inertia`, `silhouette_score_sklearn`, `elbow_curve`|Measures clustering quality and computes inertia across several values of k|
|`plotting_clustered.py`|`plot_clusters_2d`, `plot_elbow`|Plots a 2D scatter graph of clusters with centroids, and the elbow plot|
|`interface.py`|`run_clustering`|A high-level function that chains the full pipeline: load CSV → select and standardise → cluster → metrics → plots → optional export|
|`__init__.py`|	(none)|Exposes all the public functions, so users can write from cluster_maker import run_clustering|