# Importing libraries
import os       # For finding absolute path of the Fasta file
from pathlib import Path        # For finding absolute path of the Fasta file
from Bio.SeqIO.FastaIO import SimpleFastaParser     # For iterating over Fasta file
import re       # For cleaning the Fasta file sequence data
import pandas as pd     # For creating a dataframe representing the Fasta file


def read_fasta(file_path):
    # Getting the absolute path to the Fasta file
    root_folder = Path(__file__).parents[1]
    absolute_path = os.path.join(root_folder, file_path)

    # Creating dataframe from the Fasta file
    records = []
    with open(absolute_path) as fasta_file:

        # Iterate over records of the Fasta file
        for title, sequence in SimpleFastaParser(
                fasta_file):
            record = []
            # The name of the specie
            title_splits = re.findall(r"[\w']+", title)  # Data cleaning is needed
            record.append(title_splits[0])

            # The genome sequence of the specie
            sequence = " ".join(sequence)  # It converts into one line
            record.append(sequence)  # Third values are sequences

            records.append(record)

    # Return a dataframe of the records of the Fasta file
    return pd.DataFrame(records, columns=["id", "sequence"])


# Testing the read_fasta() function
data = read_fasta('input\\2_aligned\\2\'-5\'_RNA_ligase.fasta')
print(data)








# sequence1 = data.iloc[0]['sequence']
# sequence2 = data.iloc[1]['sequence']

# print(similar(sequence1, sequence2))

#print(data.iloc[1]['id'])
#print(data.iloc[2]['id'])