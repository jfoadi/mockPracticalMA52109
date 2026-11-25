Explanation
1. What was wrong with demo/cluster_plot.py

The script ran without crashing but did not do what it was designed to do: run k-means for k = 2, 3, 4, 5 and compare the results.

k = min(k, 3): this capped k at 3, so the k = 4 and k = 5 runs were really k = 3 and gave identical metrics. Fix: use k=k.
"silhouette_score": run_clustering stores this metric as "silhouette", so the check was always false and the silhouette bar chart was never saved. Fix: use "silhouette" (two places).

After the fix each k gives different results (silhouette: 0.606, 0.725, 0.672, 0.611 for k = 2 to 5), with the best value at k = 3.

2. What the corrected script does

For the first two numeric columns of the input CSV, it runs k-means for k = 2, 3, 4, 5 (standardised features, random_state=42) and saves in demo_output/ a clustered CSV and a scatter plot for each k, a metrics table (inertia, silhouette), and a bar chart of silhouette score against k.

3. Overview of cluster_maker
dataframe_builder: defines cluster centres and simulates labelled data.
data_analyser / data_exporter: descriptive statistics, correlations, and export to CSV or text.
preprocessing: selects numeric feature columns and standardises them.
algorithms: manual k-means and a scikit-learn k-means wrapper.
evaluation: inertia, silhouette score and the elbow curve.
plotting_clustered: cluster scatter plots and elbow plots.
interface: run_clustering runs the whole workflow from CSV to plots.