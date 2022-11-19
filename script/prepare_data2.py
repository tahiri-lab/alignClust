import pandas as pd
import re


def read_fasta(file_path, columns):

    from Bio.SeqIO.FastaIO import SimpleFastaParser
    with open("Proof.txt") as fasta_file:
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