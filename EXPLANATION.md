#Issues: 
#The first mistake correct was k=min(k,3) to k=k because this caused k=3,4,5 to all identical, so when I changed it allows for the correct number of clusters.
#The next mistake was silhouette_score to silhouette. Since silhouette_score didn't exist it won't generate the plot but silhouette exist so had to change.

#Correct Demo: 
#The correct Demo reads the CSV file inputted and uses the first 2 columns. Uses clustering for k=2,3,4,5. It generates a bar chart for each of the k. The output is in demo_output

#Overview:
#Cluster_maker provides the tools for preparing, analysing, clustering and creating visuals for data. 

#Purpose: Main components 
    #preprocessing: chooses the numerical features and standardises the data before everything
    #algorithms: contains the implementations of K clustering 
    #evaluation:calculates silhouette and other values
    #plotting_clustered: creates cluster plots
    #interface: connects the other components together
    #dataframe_builder: creates dataframe
    #data_analyser: analysising and calculates the statistics and correlations
    #data_exporter: exports data in a CSV. 

