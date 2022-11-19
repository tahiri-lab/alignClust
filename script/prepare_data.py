import pandas as pd
import re


# a function to return the count number of animals in the fasta in a dictionary
def count_animals(file_path, separator, animal_index):
    animal_dictionary = {}
    with open("Proof.txt") as fasta_file:
        fasta_contents = fasta_file.readlines()
        for line in fasta_contents:
            if line.startswith(">"):
                animal = line.split(separator)[animal_index]
                if animal not in animal_dictionary:
                    animal_dictionary[animal] = 1
                else:
                    animal_dictionary[animal] = animal_dictionary[animal] + 1
    return animal_dictionary


def read_fasta(file_path, columns):
    counter = 0  # counter used to print no more than number of times 'Dog'
    animal_d = count_animals(file_path, '|', 1)  # execute function
    max_animal = max(animal_d.values())  # get max number of animals
    min_animal = min(animal_d.values())  # get min number of animals

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