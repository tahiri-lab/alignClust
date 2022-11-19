from sklearn.cluster import AgglomerativeClustering
from compute_similarity_matrix import get_dissimilarity_matrix


def clustering_fasta_file(filepath, clustering_method="AgglomerativeClustering"):

    dissimilarity_matrix = get_dissimilarity_matrix(filepath)

    # if clustering_method == "AgglomerativeClustering":
        # for k in range(4, 5):
    clustering_model = AgglomerativeClustering(
        affinity='precomputed',
        n_clusters=4,
        linkage='complete'
    )

    clustering_model.fit(dissimilarity_matrix)

    # print(model.labels_)
    return clustering_model


# Testing clustering of species from a Fasta file
model = clustering_fasta_file('input\\2_aligned\\2\'-5\'_RNA_ligase.fasta')
print("%s: \n %s" % (model, model.labels_))
