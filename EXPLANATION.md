# Explanation of Corrections Made to `cluster_plot.py`

## Problematic Lines and Corrections

- Line **62**: `k = min(k, 3)` changed to `k = min(k, 5)`
- Line **89**: `if "silhouette_score" in metrics_df.columns:` changed to `if "silhouette" in metrics_df.columns:`
- Line **91**: `plt.bar(metrics_df["k"], metrics_df["silhouette_score"])` changed to `plt.bar(metrics_df["k"], metrics_df["silhouette"])`

## Summary of Corrected Script

When the command "python demo/cluster_plot.py data/demo_data.csv" is run, the script executes the following functionality:

1. Checks the provided file path of the data.
2. Validates that there is at least the correct number of columns (two).
3. Extracts these first two columns.
4. Runs the clustering algorithm for different values of k.
5. Export clustering data (as CSV files) for each value of k.
6. Produce clustering plots for each value of k.
7. Produce a sillhoute comparison bar chart.
