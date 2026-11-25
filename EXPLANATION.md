## File to explain what cluster_plot.py was doing, trying to achieve and how it was fixed


## Explanation of what was wrong and fixes:

File was trying to run k clustering for k = 2,3,4,5.
However it was setting 'k = min(k,3)', running k = 4 and 5 as 3 instead. Therefore repeating clustering for k = 3 (the plots produced for k=3,4,5 are identical). 
This was changed to 'k = k'.

The silhouette scores bar plot is not being created. 
Upon inspecting the function run_clustering() in cluster_maker/interface, in the 'metrics' dictionary inside the dictionary 'result' returned by the function, there is a key called 'silhouette'. 
The file cluster_plot.py was trying to access 'silhouette_score' which does not exist. 
This was changed to 'silhouette' in both the if statement and when plotting the bar chart.

## Summary of what script does

cluster_plot.py takes a CSV file and selects its first two columns. 
It will then run K-means clustering for k = 2, 3, 4 and 5. On each value of K, it will save a plot and a CSV file of the points in a folder named demo_output. 
It also saves a metrics csv file that has cluster count, inertia and silhouette score for each value of k. 
Lastly another plot is saved, this one is a bar chart comparing the silhouette score across the values of k.

## Overview of what cluster_maker package does

cluster_maker is an educational Python package for generating synthetic clustered data, running clustering algorithms, evaluating results, and producing user-friendly plots.

The main components and what they do are:

Preprocessing : Selects feature columns (ensuring they're numeric) and can standardise them.
Algorithms: Provides functions to be able to carry out k-means clustering. This can be done via its own functions or scikit-learn.
Plotting_clustered: Used to create cluster plots in 2D and plot inertia vs k (elbow method).
Evaluation: Calculates inertia and silhouette scores.
Interface: Runs the full clustering workflow; returns a dictionary involving outputs from above components (such as inertia and silhouette score).
Dataframe_builder: Generates synthetic (generated) data.
Data_analyser: Computes descriptive statistics (e.g correlations) for each numeric column in the provided DataFrame.
Data_exporter: Exports a DataFrame to a CSV.
