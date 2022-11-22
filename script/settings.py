# Choosing the method of dealing with gaps in computing dissimilarity matrix
gap_penalty_type = "Linear"  # ["Constant", "Linear", "Affine"]

# Defining parameters for the grid search
grid_search_parameters = {
    "KMeans": {
        "min_clusters": 2
        , "max_clusters": 8
        , "init": ['k-means++']
        , "n_init": [5, 10, 15]
        , "max_iter": [200, 250, 300, 350]
        , "tol": [0.0001]
        , "algorithm:": ['auto']
    },
    "DBSCAN": {
        "eps": [0.5, 1, 1.5]
        , "min_samples": [5, 10, 15]
        , "metric": ['euclidean']
        , "algorithm": ['auto']
        , "leaf_size": [30]
    },
    "AgglomerativeClustering": {
        "min_clusters": 2
        , "max_clusters": 8
        , "affinity": ["euclidean"]
        , "linkage": ["ward", "complete", "average", "single"]
    },
    "KMedoids": {
        "min_clusters": 2
        , "max_clusters": 8
        , "metric": ['euclidean']
        , "method": ['alternate']
        , "init": ['heuristic']
        , "max_iter": [300]
    }
}
