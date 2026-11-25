ERRORS:
siloute_score changed to siloute as siloute_score was not a column of the metrics dataframe - this was causing the bar graph for te siloutes not to print 

Changed k=min(k,3) to be k=k - This was causing only the clustering for k=2 and k=3 to be done correctly as min(3,4) and min(3,5) are both. Therefore the programme was displaying that it worked for k=4,5 but in fact using k=3 for both of these 

EXPLANATION:
This code takes a dataset (in our case demo_data.csv) and first checks that only 1 dataset is provided and that it exists. It also uses an output directory (or creates one if it doesnt exist), it then checks the dataset has at least 2 numerical columns and runs k-means on it for k=2,3,4,5. It then creates and saves cluster plots for each k as well as some metrics and finally a bar plot of the silhoute for each k.