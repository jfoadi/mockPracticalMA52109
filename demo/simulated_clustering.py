

#Simulated clustering demo ### AI assisted ###

import pandas as pd
import matplotlib.pyplot as plt
from cluster_maker import run_clustering

input_path = "data/simulated_data.csv"
features = ["feature_1", "feature_2", "feature_3", "feature_4"]

# Compare candidate cluster counts and print their metrics.
scores = []
for k in [2, 3, 4, 5, 6, 7]:
    result = run_clustering(
        input_path=input_path,
        feature_cols=features,
        algorithm="sklearn_kmeans",
        k=k,
        standardise=True,
        random_state=42,
    )
    scores.append({
    "k": k,
    "inertia": result["metrics"]["inertia"],
    "silhouette": result["metrics"]["silhouette"],
})
    plt.close(result["fig_cluster"])

scores_df = pd.DataFrame(scores)
print(scores_df.to_string(index=False))
scores_df.to_csv("simulated_metrics.csv", index=False)

# Choose k based on the printed metrics;
chosen_k = 7
final = run_clustering(
    input_path=input_path,
    feature_cols=features,
    algorithm="sklearn_kmeans",
    k=chosen_k,
    standardise=True,
    random_state=42,
    compute_elbow=True,
    elbow_k_values=list(range(1, 9)),
    output_path="simulated_clustered.csv",
)

final["fig_cluster"].savefig("simulated_clusters.png", dpi=150)
final["fig_elbow"].savefig("simulated_elbow.png", dpi=150)
plt.close(final["fig_cluster"])
plt.close(final["fig_elbow"])

# Supporting Matplotlib view of another feature pair.
data = pd.read_csv(input_path)
plt.scatter(
    data["feature_3"],
    data["feature_4"],
    c=final["labels"],
    cmap="tab10",
)
plt.xlabel("feature_3")
plt.ylabel("feature_4")
plt.title(f"Clusters on feature_3 vs feature_4 (k={chosen_k})")
plt.savefig("simulated_feature_pair.png", dpi=150)
plt.close()



