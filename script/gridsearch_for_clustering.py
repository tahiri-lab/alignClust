import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from script.compute_similarity_matrix import dissimilarity_matrix, get_dissimilarity_matrix


def grid_search_cluster(filepath
                        , score_metric="silhouette"
                        , method="KMeans"
                        , min_clusters=2, max_clusters=5
                        , affinity="euclidean", linkage="complete"):

    matrix = get_dissimilarity_matrix(filepath)
    matrix = pd.DataFrame(matrix, columns=range(0, len(matrix)))

    # Dictionary to store for each silhouette score found, the number of clusters and the predicted labels of clusters
    silhouette_scores = {}

    # Iterate over cluster
    for num_clusters in range(min_clusters, max_clusters):
        if method == "KMeans":
            cluster = KMeans(n_clusters=num_clusters)
        if method == "AgglomerativeClustering":
            cluster = AgglomerativeClustering(n_clusters=num_clusters,
                                              affinity=affinity,
                                              linkage=linkage)

        # Predict clustering based on dissimilarity matrix
        cluster.fit(matrix)
        predicted_labels = cluster.predict(matrix)

        # Calculate the silhouette score
        score = 0
        if score_metric == "silhouette":
            score = silhouette_score(matrix, predicted_labels)

        silhouette_scores[score] = [num_clusters, method, predicted_labels]

    return silhouette_scores


# Testing grid_search_cluster() function
results_dictionary = grid_search_cluster('input\\2_aligned\\2\'-5\'_RNA_ligase.fasta'
                              , method="KMeans"
                              , min_clusters=2
                              , max_clusters=5
                              )

best_score = max(results_dictionary.keys())
print("\nBest score: ", best_score,
      "\nBest number of clusters : ", results_dictionary[best_score][0],
      "\nClustering method : ", results_dictionary[best_score][1]
      , "\nPredictions of clusters :", results_dictionary[best_score][2]
      )


