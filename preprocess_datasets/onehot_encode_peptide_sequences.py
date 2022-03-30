####################################
# Script for onehot-encoding peptide sequences
# Author: Ayse Dincer

# The script takes as input_filename which contains the peptide level quants, runs, i.e., the number of runs (samples), and output_filename to record the results

# The input file should be tab seperated and should consist of the following columns:
# 'Peptide' that contains the peptide sequences, the peptide sequences should not contain any characters beside the amino acids
# 'Protein' that contains the protein each peptide occurs in 
# The last r columns should correspond to different runs containing peptide quants

# The original peptide sequences is replaced with the onehot-encoded peptide sequences and the results are recorded to the output_filename 

####################################

import numpy as np
import pandas as pd
import sys

#Function to encode sequences to int
def sequence_to_int(sequence):
    integer_encoded = []
    
    for char in sequence:
        if char in alphabet:
            integer_encoded.append(char_to_int[char] + 1)
        else:
            integer_encoded.append(0)
    return integer_encoded

#Function to encode sequences to one hot
def sequence_to_onehot(sequence):

    onehot_encoded = list()
    
    #First convert to int
    integer_encoded = sequence_to_int(sequence)
    
    #Then, one hot encode each int
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        if value != 0:
            letter[value - 1] = 1
            onehot_encoded.append(letter)
        else:
            onehot_encoded.append(letter) 
    #Flatten onehot_encoded
    onehot_encoded = np.array(onehot_encoded).flatten()
    
    #Convert to dataframe
    indices = []
    for s in range(len(sequence)):
        sub_indices = [(str(s) + '_' + c) for c in alphabet]
        indices.extend(sub_indices)
    onehot_encoded_df = pd.DataFrame(onehot_encoded, index = indices)
    
    return onehot_encoded_df

####################################

# Read dataset
print("Reading input file...")
input_filename = sys.argv[1]
runs = int(sys.argv[2])
output_filename = sys.argv[3]
print("Running with input file ", input_filename, "containing ", runs, " runs")

data_df = pd.read_csv(input_filename, sep = '\t', index_col = 0)
print("Dataset ", data_df.shape)

# Get peptide sequences
peptide_sequences = data_df['Peptide'].values
print("Peptide sequences ", peptide_sequences[:10])

#Define the amino acid alphabet
alphabet = 'ACDEFGHIKLMNPQRSTVWY'

#Define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))

print("Making sequence lengths equal...")
#Make sure all sequences have the same length
imputed_peptide_sequences = peptide_sequences
print("Maximum sequence length ", np.max([len(s) for s in imputed_peptide_sequences]))

max_length = 60

for i in range(len(imputed_peptide_sequences)):
    s = imputed_peptide_sequences[i]
    new_s = s + ' ' * (max_length - len(s))
    imputed_peptide_sequences[i] = new_s
    
#Now encode each sequence
print("Encoding sequences...")
peptide_sequences_onehot = []
for sequence in imputed_peptide_sequences:
    onehot = sequence_to_onehot(sequence)
    peptide_sequences_onehot.append(onehot)
    
onehot_encoded_df = pd.concat(peptide_sequences_onehot, axis = 1).T
onehot_encoded_df.index = data_df.index
print("onehot_encoded_df ", onehot_encoded_df.shape)
print("onehot_encoded_df ", onehot_encoded_df.head())

#Record the results
print("Recording the results...")
onehot_encoded_df = pd.concat([onehot_encoded_df, data_df['Protein'], data_df.iloc[:, -(runs + 6):]], axis = 1) #6 columns correspond to charge states
onehot_encoded_df.index = data_df.index
onehot_encoded_df.to_csv(output_filename, sep = '\t', index = True)
print("Final df ", onehot_encoded_df.shape)
print("Final df ", onehot_encoded_df.head())
