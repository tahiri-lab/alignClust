import pandas as pd
import numpy as np
import re


def read_fasta(file_path, columns):

    from Bio.SeqIO.FastaIO import SimpleFastaParser
    import os
    from pathlib import Path

    root_folder = Path(__file__).parents[1]
    file_path = os.path.join(root_folder, 'input\\2_aligned\\2\'-5\'_RNA_ligase.fasta')

    # with open("Proof.txt") as fasta_file:
    with open(file_path) as fasta_file:
        records = []  # create empty list
        for title, sequence in SimpleFastaParser(
                fasta_file):  # SimpleFastaParser Iterate over Fasta records as string tuples. For each record a tuple of two strings is returned, the FASTA title line (without the leading ‘>’ character), and the sequence (with any whitespace removed).
            record = []
            title_splits = re.findall(r"[\w']+", title)  # Data cleaning is needed

            record.append(title_splits[0])  # First values are ID (Append adds element to a list)
            record.append(len(sequence))  # Second values are sequences lengths
            sequence = " ".join(sequence)  # It converts into one line
            record.append(sequence)  # Third values are sequences

            records.append(record)

        # print(records)
    return pd.DataFrame(records, columns=columns)  # We have created a function that returns a dataframe


# Now let's use this function by inserting in the first argument the file name (or file path if your working directory is different from where the fasta file is)
# And in the second one the names of columns
data = read_fasta("Proof.txt", columns=["id", "sequence_length", "sequence"])

print(data)

# Similarity matrix
from difflib import SequenceMatcher
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

sequences_number = data.shape[0]
matrix = np.empty([sequences_number, sequences_number])

for i in range(0, sequences_number):
    for j in range(0, sequences_number):
        matrix[i][j] = similar(data.iloc[i]['sequence'], data.iloc[j]['sequence'])
print(matrix)

# Clusteting

from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(
  affinity='precomputed',
  n_clusters=4,
  linkage='complete'
).fit(matrix)
print(model.labels_)







# sequence1 = data.iloc[0]['sequence']
# sequence2 = data.iloc[1]['sequence']

# print(similar(sequence1, sequence2))

#print(data.iloc[1]['id'])
#print(data.iloc[2]['id'])