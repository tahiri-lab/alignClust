import itertools

import pandas as pd
from script.compute_similarity_matrix import get_dissimilarity_matrix
from settings import grid_search_parameters

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from pyclustering.cluster.kmedoids import kmedoids

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def get_clustering_results(model, data, score_metric):
    # Predict clustering based on dissimilarity matrix
    model.fit(data)
    try:        # in the case of KMeans,
        predicted_labels = model.predict(data)
    except:     # in the case of DBSCAN, AgglomerativeClustering,
        predicted_labels = model.labels_

    # Calculate the silhouette score
    score = 0
    if len(set(predicted_labels)) == 1:     # if only one cluster is found in prediction, the score cannot be calculated
        pass
    elif score_metric == "silhouette":
        score = silhouette_score(data, predicted_labels)

    return score, predicted_labels


def get_grid_search_clustering_results(filepath
                                       , parameters
                                       , score_metric="silhouette"):

    # data
    matrix = get_dissimilarity_matrix(filepath)
    matrix_length = len(matrix)
    matrix = pd.DataFrame(matrix, columns=range(0, matrix_length))

    # models to use for clustering
    methods = parameters.keys()

    # Dictionary to store for each silhouette score found, the number of clusters and the predicted labels of clusters
    silhouette_scores = {}

    for method in methods :
        if method == "KMeans":
            # hyperparameters of the method
            hyperparameters = parameters[method]
            for num_clusters, n_init, max_iter in list(itertools.product(
                    range(hyperparameters["min_clusters"], hyperparameters["max_clusters"])
                    , hyperparameters["n_init"]
                    , hyperparameters["max_iter"])):

                # defining the clustering model
                cluster = KMeans(n_clusters=num_clusters, n_init=n_init, max_iter=max_iter)

                # getting the results of the clustering
                score, predicted_labels = get_clustering_results(cluster, matrix, score_metric)

                # storing the results with corresponding actual parameters
                best_hyperparameters = {"num_clusters": num_clusters, "n_init": n_init, "max_iter": max_iter}
                silhouette_scores[score] = [method, num_clusters, best_hyperparameters, predicted_labels]

        elif method == "DBSCAN":
            # hyperparameters of the method
            hyperparameters = parameters[method]

            for eps, min_samples, metric in list(itertools.product(
                    hyperparameters["eps"]
                    , hyperparameters["min_samples"]
                    , hyperparameters["metric"])):

                # defining the clustering model
                cluster = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

                # getting the results of the clustering
                score, predicted_labels = get_clustering_results(cluster, matrix, score_metric)

                # storing the results with corresponding actual parameters
                best_hyperparameters = {"eps": eps, "min_samples": min_samples, "metric": metric}
                num_clusters = len(set(predicted_labels))

                # Defining the dictionary of the results of gridsearch
                silhouette_scores[score] = [method, num_clusters, best_hyperparameters, predicted_labels]

        elif method == "AgglomerativeClustering":
            # hyperparameters of the method
            hyperparameters = parameters[method]
            for num_clusters, affinity, linkage in list(itertools.product(
                    range(hyperparameters["min_clusters"], hyperparameters["max_clusters"])
                    , hyperparameters["affinity"]
                    , hyperparameters["linkage"])):

                print("num_clusters:", num_clusters)
                print("affinity:", affinity)
                print("linkage:", linkage)

                # defining the clustering model
                cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity=affinity, linkage=linkage)

                # getting the results of the clustering
                score, predicted_labels = get_clustering_results(cluster, matrix, score_metric)

                # storing the results with corresponding actual parameters
                best_hyperparameters = {"num_clusters": num_clusters, "min_samples": min_samples, "metric": metric}
                num_clusters = len(set(predicted_labels))

                # Defining the dictionary of the results of gridsearch
                silhouette_scores[score] = [method, num_clusters, best_hyperparameters, predicted_labels]

    return silhouette_scores


def get_grid_search_best_model_report(filepath, verbose=True, save_pdf=True):

    # All results dataframe
    results_dictionary = get_grid_search_clustering_results(filepath, parameters=grid_search_parameters)
    results_df = pd.DataFrame(results_dictionary.values(), columns=[
        "Clustering method"
        , "Number of clusters"
        , "Clustering hyper-parameters"
        , "Predictions of clusters"])
    results_df.insert(0, "Score", results_dictionary.keys(), True)
    print(results_df.to_string())

    # Best model dataframe
    best_score = max(results_dictionary.keys())
    best_model_dictionary = {
        "Best score": best_score
        , "Clustering method": results_dictionary[best_score][0]
        , "Best number of clusters": results_dictionary[best_score][1]
        , "Clustering hyper-parameters": results_dictionary[best_score][2]
        , "Predictions of clusters": results_dictionary[best_score][3]
        }
    best_model_df = pd.DataFrame.from_dict(best_model_dictionary, orient='index').T

    if verbose:
        print(results_df.to_string())
        print("\nBest score: ", best_score
              , "\nClustering method : ", results_dictionary[best_score][0]
              , "\nBest number of clusters : ", results_dictionary[best_score][1]
              , "\nClustering hyper-parameters : ", results_dictionary[best_score][2]
              , "\nPredictions of clusters :", results_dictionary[best_score][3]
              )

    if save_pdf:
        save_pdf_report(best_model_df, results_df)

    return results_df, best_model_df


def save_pdf_report(results_df, best_model_df):

    plt.rcParams.update({'figure.figsize': (30, 40), 'figure.dpi': 250})

    fig, ax = plt.subplots(2, 1, figsize=(30, 40))
    # ax.axis('tight')
    # ax.axis('off')

    ax[0].table(cellText=results_df.values, colLabels=results_df.columns, loc='center')
    ax[1].table(cellText=best_model_df.values, colLabels=results_df.columns, loc='center')

    # plt.savefig(table_name)
    # plt.show()

    pp = PdfPages("report.pdf")
    pp.savefig(fig, bbox_inches='tight')
    pp.close()





# Testing grid_search_cluster() function
get_grid_search_best_model_report('input\\2_aligned\\2\'-5\'_RNA_ligase.fasta')

