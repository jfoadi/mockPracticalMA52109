# Explanation: Task 2 (demo/cluster_plot.py)

## 1. What was wrong with the original script, and how I fixed it

The script ran without crashing, but it did not do what its header says (run
k-means for k = 2, 3, 4, 5 and report the results). There were two mistakes,
both in `demo/cluster_plot.py`.

**Mistake 1: the number of clusters was capped at 3 (line 62).**
The call to `run_clustering` used `k = min(k, 3)`. The loop went through
k = 2, 3, 4, 5, but the clustering itself never used more than 3 clusters.
Evidence from the original run:

- k = 3, 4 and 5 produced identical inertia (43.27) and identical silhouette
  score (0.72);
- the files labelled `..._k4` and `..._k5` contained only 3 distinct clusters,
  although their names (and the metrics table) claimed 4 and 5.

*Fix:* I replaced `k = min(k, 3)` with `k=k`, so each pass of the loop clusters
with the value of k it announces.

**Mistake 2: the silhouette chart was never drawn (lines 89 and 91).**
The script only draws the silhouette bar chart `if "silhouette_score" in
metrics_df.columns`. But `run_clustering` stores this score under the key
`"silhouette"`, so the column is called `silhouette`, the condition was always
false, and the chart was silently skipped (no `..._silhouette.png` was created).

*Fix:* I changed `"silhouette_score"` to `"silhouette"` in the condition and in
the `plt.bar(...)` call. I changed the demo and not the package, because
`"silhouette"` is the name the package returns.

After the fixes, the script was re-run. The results now differ for each k
(inertia 185.3, 43.3, 26.1, 21.3 for k = 2, 3, 4, 5; silhouette 0.61, 0.72,
0.67, 0.61), the labelled files contain 2, 3, 4 and 5 clusters, and the
silhouette chart is produced. The silhouette score is highest for k = 3 and the
inertia drops sharply up to k = 3, then flattens, which points to three clusters
in `demo_data.csv`. The unit tests in `tests/` still pass.

## 2. What the corrected demo script does

Run as `python demo/cluster_plot.py data/demo_data.csv`, the script:

1. checks that exactly one argument (the input CSV) was given and that the file
   exists, and stops with a usage message otherwise;
2. creates an output folder called `demo_output`;
3. reads the CSV, finds its numeric columns and uses the first two of them as
   the features (it needs at least two);
4. for each k in 2, 3, 4, 5, calls `run_clustering` with the k-means algorithm,
   standardised features and a fixed `random_state=42` (so results are
   reproducible), saving the data with its cluster labels to a CSV file, and
   the 2D cluster plot to a PNG file;
5. collects the inertia and the silhouette score for each k into a table, saved
   as `<name>_metrics.csv`;
6. draws a bar chart of the silhouette score against k and saves it as
   `<name>_silhouette.png`, to help judge which k fits the data best.

## 3. Overview of the `cluster_maker` package

`cluster_maker` is a small package for generating clustered data, grouping it
with k-means, judging the result and plotting it. Its main components are:

| Module | Purpose |
|---|---|
| `dataframe_builder.py` | Creates test data. `define_dataframe_structure` builds a table of cluster centres; `simulate_data` scatters points around them with Gaussian noise and records the true cluster. |
| `data_analyser.py` | Quick summaries of a DataFrame: descriptive statistics (`describe`) and the correlation matrix. |
| `data_exporter.py` | Saves results: `export_to_csv` writes a CSV, `export_formatted` writes a readable text table. |
| `preprocessing.py` | Prepares the features. `select_features` picks the chosen columns and checks they exist and are numeric; `standardise_features` rescales them to mean 0 and standard deviation 1 (so no feature dominates the distances). |
| `algorithms.py` | The clustering itself: a hand-written k-means (`init_centroids`, `assign_clusters`, `update_centroids`, `kmeans`) and `sklearn_kmeans`, a thin wrapper around scikit-learn's `KMeans`. |
| `evaluation.py` | Judging the clusters: `compute_inertia` (within-cluster sum of squares), `silhouette_score_sklearn`, and `elbow_curve` (inertia for several k, for the elbow method). |
| `plotting_clustered.py` | Figures: `plot_clusters_2d` (scatter plot coloured by cluster, with centroids) and `plot_elbow` (inertia against k). |
| `interface.py` | `run_clustering`, the high-level function that chains everything: load the CSV, select and standardise the features, run the chosen algorithm, compute the metrics, make the plots and optionally save the labelled data. It returns the data, labels, centroids, metrics and figures in a dictionary. |

The demo scripts call only `run_clustering`, so one function call covers the
whole workflow.
