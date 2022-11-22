# Importing libraries
import numpy as np
import argparse
from prepare_data import read_fasta


def get_dissimilarity_matrix(filepath):

    # Getting the dataframe from the Fasta file
    data = read_fasta(filepath)

    # Initialization of the dissimilarity matrix
    sequences_number = data.shape[0]
    matrix = np.empty([sequences_number, sequences_number])

    # Computing the dissimilarity matrix by looping over sequences two by two
    for i in range(0, sequences_number):
        for j in range(0, sequences_number):
            matrix[i][j] = dissimilarity_score_between_two_sequences(data.iloc[i]['sequence'], data.iloc[j]['sequence']
                                                                     , 'Constant')
    return matrix
def dissimilarity_score_between_two_sequences(a, b, gap_penalty_type):
    parser = argparse.ArgumentParser()
    parser.add_argument('gap_penalty_type',
                        metavar='gap_penalty_type',
                        type=str,
                        choices=["Constant", "Linear", "Affine"],
                        help='Type of the gap penalty')

    score = 0
    for i in range(len(a)):
        if a[i] != "-" and b[i] != "-":
            if a[i] == b[i]:    # comme 'G' et 'G'
                score += 0
            else:
                score += 1      # comme 'G' et 'C'

        elif a[i] == "-" and b[i] == "-":       # càd '-' et '-'
            score += 0

        else:      # comme 'G' et '-' (ou '-' et 'G')
            if gap_penalty_type == 'Linear':
                score += 2
            if gap_penalty_type == 'Constant':
                if i ==0 :  # First gap, then it's considered a new gap
                    score += 2
                else :
                    if a[i-1] == '-' or b[i-1] == '-':  # It's a constant gap, we'll not consider a new gap penalty
                        score += 1
                    else:
                        score += 2      # New gap, so a adding the gap penalty

    return score


# Testing the matrix_of_dissimilarity() function
'''
dissimilarity_matrix = get_dissimilarity_matrix('input\\2_aligned\\2\'-5\'_RNA_ligase.fasta')
print(dissimilarity_matrix)
'''