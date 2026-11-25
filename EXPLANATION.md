Task 2: Explanation for the issues in cluster_plot.py

There were two issues with the cluster_plot.py file.

The first of these was in line 62 where it wrote 'k = min(k, 3),'.
The second is in lines 90 'if "silhouette_score" in metrics_df.columns:' and 93 'plt.bar(metrics_df["k"], metrics_df["silhouette_score"])' 

After running 'py -m demo.cluster_plot data/demo_data.csv' this is what was produced in the terminal window:

=== Running k-means with k = 2 ===
Metrics:
  inertia: 185.32044176044144
  silhouette: 0.6064840027444722

=== Running k-means with k = 3 ===
Metrics:
  inertia: 43.27170530489448
  silhouette: 0.7248810719786047

=== Running k-means with k = 4 ===
Metrics:
  inertia: 43.27170530489448
  silhouette: 0.7248810719786047

=== Running k-means with k = 5 ===
Metrics:
  inertia: 43.27170530489448
  silhouette: 0.7248810719786047

Demo completed.
Outputs saved in: demo_output

As we can see the final values for k=3,4,5 all give the same values for the inertia and the silhouette however this is not correct. Instead we change this line to 'k=min(k,5)' to inclue the values for k=4,5. 

We have also created a folder called 'demo_output' with various files inside. One file that is missing is 'demo_data_silhouette.png' as we have the wrong column names in the original line. In the 'interface.py' file the column is called 'silhouette'. Instead we change these lines to 'if "silhouette" in metrics_df.columns:' and 'plt.bar(metrics_df["k"], metrics_df["silhouette"])' to ensure we create these files.

Again running 'py -m demo.cluster_plot data/demo_data.csv' we get the following:

=== Running k-means with k = 2 ===
Metrics:
  inertia: 185.32044176044144
  silhouette: 0.6064840027444722

=== Running k-means with k = 3 ===
Metrics:
  inertia: 43.27170530489448
  silhouette: 0.7248810719786047

=== Running k-means with k = 4 ===
Metrics:
  inertia: 26.089147599849774
  silhouette: 0.6715383522567352

=== Running k-means with k = 5 ===
Metrics:
  inertia: 21.303166341609
  silhouette: 0.6113547187834774

Demo completed.
Outputs saved in: demo_output

which is correct and also produces the 'demo_data_silhouette.png' file.


The cluster_maker package is designed to generate data points, grouping them together based on similarities and then creating visual plots of this. 

The 'preprocessing.py' file takes the specific columns we want to use fomr the datasets and scales them.

The 'algorithms.py' file uses KMeans to group points together based on how far they are from the other points. It places them randomly on a dataset, measuring the distance between these and colours/groups the ones that are closer together to the same colour. It then calculates the average position of these dots and creates a middle pin. It then keeps doing this until the groups are fully formed and the middle pin stops moving. 

The 'evaluation.py' file then evaluates how close these groups are to the middle pin and to each other group. It also does this for various values of pins.

The 'plotting_clustered.py' file then plots this data so we can see the scatter plots of the data for various values of k and the groups, with their middle pins. It also saves these as files to view as the user.

The 'interface.py' file does all of this together, inputting how many values of k (the number of pins) and runs the KMeans algorithm to then calculate the score and saves these graphs to the directory.