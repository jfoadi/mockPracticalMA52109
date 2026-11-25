###
##
# EXPLANATION OF TASK 2

# What was wrong and how I fixed it
The script was meant to run k-means for k = 2,3,4, and 5 on the first two numeric columns of the CSV file called 'demo_data.csv' by first standardising each column and then running the k-means seperately for each k. For each k it saves the labelled data and a 2D-cluster plot and it records the inertia (the total squared distance from each point to its cluster centre) and the silhouette score(betwenn -1 and 1 and it measures how well each point sits in its own cluster compared with the nearest other cluster).

The two bugs were:
1) k was capped at 3 ~ The call to 'run_clustering' used 'k = min(k, 3)`,
   so k = 4 and k = 5 actually ran k-means with k = 3. Their output files were labelled k4 and k5 but gave identical results to k = 3 (inertia 43.271705304894496,
   silhouette 0.7248810719786047). 
   
   
   **Fix:** replace 'k = min(k, 3)` with 'k=k' so that each run uses the k it is labelled with .
    
2) The silhouette plot was never made
The script looked for a column
called "silhouette_score" , but 'run_clustering' stores the metric under the key "silhouette". Hence 'silhouette_score' never existed.
 The 'if' condition was always False, so the plot was skipped without any error. 
 
 **Fix:** use '"silhouette"` in both the
   'if' check and 'plt.bar(metrics_df["k"], metrics_df["silhouette"])`.


After the fixes, each k now produces a differenet result for the inertia and silhouette score: 
| k | inertia | silhouette |
|---|---------|------------|
| 2 | 185.32  | 0.606      |
| 3 | 43.27   | 0.725      |
| 4 | 26.09   | 0.672      |
| 5 | 21.30   | 0.611      |

k=3 has the highest silhouette score, therefore three clusters fit this data best.


# Summary of what corrected script does: 
1. Checks that one CSV file is given and that it exists.
2. Reads the CSV and uses the first two numeric columns as features.
3. For each k = 2, 3, 4, 5 it calls `run_clustering` (standardised features,
   manual k-means, random_state = 42). It saves the labelled data
   (`<name>_clustered_k<k>.csv`) and a cluster plot (`<name>_k<k>.png`), and
   prints the inertia and silhouette score.
4. Saves a summary table of all metrics (`<name>_metrics.csv`).
5. Draws a bar chart of silhouette score against k (`<name>_silhouette.png`)
   to show which k fits best. (the value of k with the highest silhouette score)

All outputs go to the `demo_output/` folder.



# OVERVIEW of cluster_maker package
`cluster_maker` is a small Python package that takes you through the whole clustering process, from getting the data ready to choosing the number of clusters and plotting the results. Each module handles one part of that process:

- **dataframe_builder.py** lets you choose where cluster centres should be and then generates random data scattered around them by distributing points as evenly possible across clusters. This is useful for testing, because you know the true clusters in advance.
- **data_analyser.py** gives a quick summary of each numeric column of a dataset: means, standard
  deviations and computes the correlations (correlation matrix) between columns in the dataframe.
- **data_exporter.py** saves results, either as a CSV file or as a neatly formatted text table.
- **preprocessing.py** gets the data ready for clustering. 
`select_features`
  picks out the columns to use and stops with a clear error if a column is missing or isn't numeric. 
`standardise_features` rescales each column to
  have mean 0 and standard deviation 1. 


- **algorithms.py** contains the clustering itself. It has a k-means written
  from scratch, broken into steps (choose starting centroids (these are the centre point of a cluster), assigning each point to its nearest centroid. It then move each centroid to the mean of its points and repeats until the centeroids converge), plus a simpler version that calls scikit-learn's KMeans.


- **evaluation.py** measures how good a clustering is. 
Inertia is how tightly the points sit around their centres. It decreases as k increases, as more centeroids means every point has one nearer to it.
The silhouette score is how well separated the clusters are, and the silhouette score always lies between -1 and 1, and the elbow curve shows how inertia changes as k increases.


- **plotting_clustered.py** draws the results: a scatter plot of the points coloured by cluster with the centroids marked, and an elbow plot. It checks that there are atleast 2 feautures before creating the figure and plotting


- **interface.py** ties everything together. Its `run_clustering` function runs the full pipeline in one call: it loads the CSV, selects and standardises the features, runs the chosen algorithm, computes the metrics/scores, adds labels to the data frame, plots the clusters, and saves the labelled data if asked.