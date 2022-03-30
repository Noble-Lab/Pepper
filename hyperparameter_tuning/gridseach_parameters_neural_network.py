####################################
# Script for training Pepper model with random hyperparameters selected from the grid
# Records the scores on the validaiton set
# Author: Ayse Dincer

####################################

import numpy as np
import pandas as pd

import random
import copy
import re
import argparse, sys

import matplotlib.pyplot as plt
import seaborn as sns

import random
import os

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Dense, Input, Dropout, Conv2D, BatchNormalization, Activation, Flatten, Concatenate, MaxPooling2D
from keras.wrappers.scikit_learn import KerasRegressor
import keras.backend as K
from keras.layers import Lambda
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from tensorflow.keras import initializers
from tensorflow.keras.constraints import max_norm

from scipy import stats
import scipy

print("Tensorflow version: ", tf.__version__)
print("Keras version: ", keras.__version__)

#Parser for all arguments
parser = argparse.ArgumentParser()
parser.add_argument('--peptide_file', help='File of peptide-level quants', required = True)
parser.add_argument('--n_runs', help='Number of runs', required = True)
parser.add_argument('--seq_length', help='Length of the sequence', required = True)
parser.add_argument('--output_file', help='Output file to record all predicted coefficients', required = True)

args = parser.parse_args()
print(args)

#First define the entire grid
n_filters_list = [5, 10, 20, 40]
filter_size_list = [3, 4, 5, 6, 7]
n_layers_list = [1, 2, 3, 4, 5]
n_conv_layers_list = [1]
n_nodes_list = [10, 20, 40, 80]
dropout_rate_list = [0.0, 0.25, 0.5, 0.75]
batch_size_list = [250, 500]#[500, 1000]
learning_rate_list = [1e-4, 5e-3]

np.random.seed(12345)
tf.random.set_seed(12345)

while True:
    #Randomly sample from the grid
    n_filters = random.sample(n_filters_list, 1)[0]
    filter_size = random.sample(filter_size_list, 1)[0]
    n_layers = random.sample(n_layers_list, 1)[0]
    n_nodes = random.sample(n_nodes_list, 1)[0]
    dropout_rate = random.sample(dropout_rate_list, 1)[0]
    batch_size = random.sample(batch_size_list, 1)[0]
    learning_rate = random.sample(learning_rate_list, 1)[0]
    n_conv_layers = random.sample(n_conv_layers_list, 1)[0]

    #Check if file exists
    filepath = 'tuning_results/' + str(args.output_file) + '/Gridsearch_Results_' + \
                'n_conv_layers' + str(n_conv_layers) + \
                '_n_filters_' + str(n_filters) + \
                '_filter_size_' + str(filter_size) + \
                '_n_layers_' + str(n_layers) + \
                '_n_nodes_' + str(n_nodes) + \
                '_dropout_rate_' + str(dropout_rate) + \
                '_batch_size_' + str(batch_size) + \
                '_learning_rate_' + str(learning_rate) + '.tsv'

    if not os.path.isfile(filepath):
        print("File does not exist")
        break


print()
print("Running with n_conv_layers " + str(n_conv_layers))
print("Running with n_filters " + str(n_filters))
print("Running with filter_size " + str(filter_size))
print("Running with n_layers " + str(n_layers))
print("Running with n_nodes " + str(n_nodes))
print("Running with dropout_rate " + str(dropout_rate))
print("Running with batch_size " + str(batch_size))
print("Running with learning_rate " + str(learning_rate))


####################################
print("Step 1: Reading datasets...")

#Read the onehot encoded matrix
data_df = pd.read_csv(args.peptide_file, sep = '\t', index_col = 0)
print("Peptide df ", data_df.shape)
print("Peptide df ", data_df.head())


####################################
print("Step 2: Defining inputs...")

seq_length = int(args.seq_length)
n_runs = int(args.n_runs)
print("Sequence length: ", seq_length, " No of runs: ", n_runs)

#Input 1: intensity measurements
q_df = data_df.iloc[:, -n_runs:]

#Input 2: protein labels
#Convert protein labels to int values
protein_labels = data_df['Protein'].values
unique_proteins = np.unique(protein_labels)
n_proteins = len(unique_proteins)
int_protein_labels = [np.where(protein_labels[i] == unique_proteins)[0][0] for i in range(protein_labels.shape[0])]
int_protein_labels = np.asarray(int_protein_labels)
print("Number of unique proteins ", n_proteins)
print("Protein labels ", int_protein_labels)

#Input 3: peptide sequences
peptide_sequences = data_df.iloc[:, :(seq_length * 20)]
print("peptide_sequences ", peptide_sequences.shape)
print("peptide_sequences ", peptide_sequences)

#Input 4: charge states
charge_column_names = ['Charge 1', 'Charge 2', 'Charge 3', 'Charge 4', 'Charge 5', 'Charge 6']
peptide_charges = data_df[charge_column_names]
print("peptide_charges ", peptide_charges.shape)
print("peptide_charges ", peptide_charges)

#Convert nan values to 0
q_df[np.isnan(q_df )] = 0.0

#Normalize the intensities such that the sum of elements in each column is equal
X = q_df.values
print("Quants before normalization ", X.sum(axis = 0))
X = (X / X.sum(axis=0, keepdims=1)) * X.shape[0]
print("Quants after normalization ", X.sum(axis = 0))

q_df = pd.DataFrame(X, index = q_df.index, columns = q_df.columns)
peptide_intensities = q_df
n_peptides = data_df.shape[0]

print("No of peptides: ", n_peptides)
print("No of proteins: ", n_proteins)
print("No of runs: ", n_runs)


####################################
print("Step 3: Defining train/test splits...")

#Split the proteins into train/test sets
train_proteins, test_proteins = train_test_split((np.arange(len(np.unique(protein_labels)))), 
                                   test_size=0.2, random_state=12345)
train_proteins, val_proteins = train_test_split(train_proteins, test_size=0.2, random_state=12345)

#Define train/validation/test peptide pairs
train_peptides = np.concatenate([list(np.where(protein_labels == np.unique(protein_labels)[p])[0]) for p in train_proteins])
val_peptides = np.concatenate([list(np.where(protein_labels == np.unique(protein_labels)[p])[0]) for p in val_proteins])
test_peptides = np.concatenate([list(np.where(protein_labels == np.unique(protein_labels)[p])[0]) for p in test_proteins])
print("No of train/val/test proteins: %d/%d/%d" % (len(train_proteins), len(val_peptides), len(test_proteins)))
print("No of train/val/test peptides: %d/%d/%d" % (len(train_peptides), len(val_peptides), len(test_peptides)))

#Splitting runs into train/test sets
train_runs, test_runs = train_test_split((np.arange(q_df.shape[1] / 2)), 
                                   test_size=0.2, random_state=12345)
train_runs = np.array([[2*i, 2*i+1] for i in train_runs]).astype(int).ravel()
test_runs = np.array([[2*i, 2*i+1] for i in test_runs]).astype(int).ravel()
print("No of train runs ", len(train_runs))
print("No of test runs ", len(test_runs))
print("Train runs ", q_df.columns[train_runs])
print("Test runs ", q_df.columns[test_runs])


#Define X, Q, C, and P
#X is the peptide sequence
#Q is the matrix of intensities
#C is the one-hot encoded matrix of charge states
#P is the protein labels

X_train = peptide_sequences.iloc[train_peptides]
X_val = peptide_sequences.iloc[val_peptides]

Q_train = peptide_intensities.iloc[train_peptides, train_runs]
Q_val = peptide_intensities.iloc[val_peptides, train_runs]

C_train = peptide_charges.iloc[train_peptides]
C_val = peptide_charges.iloc[val_peptides]

P_train = int_protein_labels[train_peptides]
P_train = [np.where(np.unique(P_train) == p)[0][0] for p in P_train]
P_val = int_protein_labels[val_peptides]
P_val = [np.where(np.unique(P_val) == p)[0][0] for p in P_val]

print()
print("X train: ", X_train.shape)
print("X val: ", X_val.shape)
print()
print("Q train: ", Q_train.shape)
print("Q val: ", Q_val.shape)
print()
print("C train: ", C_train.shape)
print("C val: ", C_val.shape)
print()
print("P train: ", len(P_train))
print("P val: ", len(P_val))



####################################
print("Step 4: Reshaping sequences for CNN...")

#Reshape sequences for CNN

X_train = X_train.values.reshape((X_train.shape[0], seq_length, 20))
X_train = np.expand_dims(X_train, axis=3)

X_val = X_val.values.reshape((X_val.shape[0], seq_length, 20))
X_val = np.expand_dims(X_val, axis=3)


print()
print("X train: ", X_train.shape)
print("X val: ", X_val.shape)



####################################
print("Step 5: Initializing alpha matrices...")

Q_train[Q_train == 0.0] = np.nan
Q_val[Q_val == 0.0] = np.nan

#Define alpha matrices to calculate loss
alpha_initial_train = np.zeros((len(train_proteins), len(train_runs)))
for i in range(len(np.unique(P_train))):
    all_sub_peptides = np.where(P_train ==  np.unique(P_train)[i])[0]
    mean_abundances = np.nanmedian(Q_train.iloc[all_sub_peptides], axis = 0)
    alpha_initial_train[i, :] = mean_abundances

print("alpha_initial_train ", alpha_initial_train.shape)
print("alpha_initial_train ", alpha_initial_train)


alpha_initial_val = np.zeros((len(val_proteins), len(train_runs)))
for i in range(len(np.unique(P_val))):
    all_sub_peptides = np.where(P_val ==  np.unique(P_val)[i])[0]
    mean_abundances = np.nanmedian(Q_val.iloc[all_sub_peptides], axis = 0)
    alpha_initial_val[i, :] = mean_abundances

print("alpha_initial_val ", alpha_initial_val.shape)
print("alpha_initial_val ", alpha_initial_val)


alpha_initial_train[np.isnan(alpha_initial_train)] = 0.0
alpha_initial_val[np.isnan(alpha_initial_val)] = 0.0

Q_train[np.isnan(Q_train)] = 0.0
Q_val[np.isnan(Q_val)] = 0.0

Q_train[np.isinf(Q_train)] = 0.0
Q_val[np.isinf(Q_val)] = 0.0


####################################
print("Step 6: Defining loss functions and the model...")

#Define custom layer that calculates the peptide loss
class CustomLossLayer(layers.Layer):
    def __init__(self):
        super(CustomLossLayer, self).__init__()

        #Define alpha variables that are trainable
        self.alphas = tf.Variable(alpha_initial_train, 
                                  trainable = True, 
                                  dtype = 'float32')

    def get_vars(self):
        return self.alphas

    def peptide_loss(self, y_true, y_pred):

        #Define all inputs
        c_pred = K.abs(y_pred)
        c_pred = tf.reshape(c_pred,[-1]) #this is very important for correctness of calculation
        q_input = y_true[:, :-1] #dimension (batch_size, K)
        label_input = y_true[:, -1] #dimension (batch_size, 1)
        label_input = tf.cast(label_input, tf.int32)

        #Exclude missing intensities in pairwise distance calculation
        zero_peptides = K.not_equal(q_input, K.constant(0))
        zero_peptides = K.cast(zero_peptides, K.floatx())

        #Exclude peptides with 0 coefficients in pairwise distance calculation
        zero_coeffs = K.not_equal(c_pred, K.constant(0))
        zero_coeffs = K.cast(zero_coeffs, K.floatx())
        zero_coeffs = tf.expand_dims(zero_coeffs, 1)

        #Find the corresponding alpha value for each peptide
        corresponding_protein_abundances = tf.gather(K.abs(self.alphas), label_input, axis = 0)

        #Calculate adjusted abundances
        c_pred = tf.expand_dims(c_pred, 1)
        adjusted_abundances = c_pred * corresponding_protein_abundances

        #Calculate the difference values
        differences = q_input - adjusted_abundances
        differences = differences * zero_peptides * zero_coeffs
        differences = K.square(differences)

        #Return the mean loss
        total_loss = K.sum(differences)
        all_runs = K.sum(zero_peptides * zero_coeffs)
        return total_loss / all_runs

    #We add the loss to the final model loss
    def call(self, y_true, y_pred):
        self.add_loss(self.peptide_loss(y_true, y_pred))
        return y_pred


#Define alpha-based peptide loss function
def peptide_loss(x, c_pred, alphas):

    c_pred = K.abs(c_pred)

    #q_input is the intensity values from the experiment
    q_input = x[:, :-1] #dimension (batch_size, K)
    #label_input is the protein labels for each peptide
    label_input = x[:, -1] #dimension (batch_size, 1)
    label_input = tf.cast(label_input, tf.int32)

    c_pred = tf.reshape(c_pred,[-1]) #this is very important for correctness of calculation

    #Exclude missing intensities in pairwise distance calculation
    zero_peptides = K.not_equal(q_input, K.constant(0))
    zero_peptides = K.cast(zero_peptides, K.floatx())

    #Exclude peptides with 0 coefficients in pairwise distance calculation
    zero_coeffs = K.not_equal(c_pred, K.constant(0))
    zero_coeffs = K.cast(zero_coeffs, K.floatx())
    zero_coeffs = tf.expand_dims(zero_coeffs, 1)

    #Find the corresponding alpha value for each peptide
    corresponding_protein_abundances = tf.gather(K.abs(alphas), label_input, axis = 0)

    #Calculate the differences
    c_pred = tf.expand_dims(c_pred, 1)
    adjusted_abundances = c_pred * corresponding_protein_abundances
    differences = q_input - adjusted_abundances
    differences = differences * zero_peptides * zero_coeffs
    differences = K.square(differences)

    #Record final average loss
    total_loss = K.sum(differences)
    all_runs = K.sum(zero_peptides * zero_coeffs)
    return total_loss / all_runs


#Define model
def define_model():

    #Define custom absolute valued activation function
    def absActivation(x) :
        activated_x = K.abs(x)
        return activated_x

    #Define network
    inputs =  Input(shape=(seq_length, 20, 1), name = 'sequence')
    inputs_charge = Input(shape=(6,), name = 'charge')
    inputs_label = Input(shape=(len(train_runs) + 1,), name = 'y_true')

    #Define the first convolutional layer
    x = Conv2D(n_filters, kernel_size=(filter_size, 20), activation="relu", input_shape=(seq_length, 20, 1))(inputs)
    x = MaxPooling2D(pool_size=(2, 1))(x)

    #Define other convolutional layers
    input_shape = seq_length
    for n in range(n_conv_layers - 1):

        x = Dropout(dropout_rate)(x) 
        input_shape = (input_shape - filter_size + 1) / 2

        x = Conv2D(n_filters, kernel_size=(filter_size, 1), activation="relu", input_shape=(input_shape, 1, n_filters))(x)
        x = MaxPooling2D(pool_size=(2, 1))(x)

    #Final joined layer
    sequence_representation = Flatten()(x)
    sequence_representation = Dropout(dropout_rate)(sequence_representation)

    #Second input is the one-hot encoded charge states
    concatenated_representation = Concatenate()([sequence_representation, inputs_charge])
    x = concatenated_representation

    #Define dense layers
    for n in range(n_layers):
        x = Dense(n_nodes, activation="relu")(x)
        x = Dropout(dropout_rate)(x)

    output = Dense(1, activation=absActivation)(x) #predict peptide coefficients

    #Define model with custom layer
    my_custom_layer = CustomLossLayer()(inputs_label, output) # here can also initialize those var1, var2
    model = Model(inputs = [inputs, inputs_charge, inputs_label], outputs = my_custom_layer)
    model.summary()

    #Compile the model
    opt = tf.optimizers.Adam(learning_rate = learning_rate, clipnorm = 1)
    model.compile(optimizer=opt)

    return model


####################################
print("Step 7: Model training...")

#Define joined intensities
joined_intensities = np.column_stack((Q_train, P_train)).astype(np.float32)
print("Train joined intensities ", joined_intensities.shape)

joined_intensities_val = np.column_stack((Q_val, P_val)).astype(np.float32)
print("Val joined intensities ", joined_intensities_val.shape)

#Calculate default losses
default_train_peptide_loss = K.eval(peptide_loss(K.constant(joined_intensities),
                                                  K.constant(np.ones(joined_intensities.shape[0])),
                                                  K.constant(alpha_initial_train)))
default_val_peptide_loss = K.eval(peptide_loss(K.constant(joined_intensities_val),
                                             K.constant(np.ones(joined_intensities_val.shape[0])),
                                             K.constant(alpha_initial_val)))

print("Default train new peptide loss: ", default_train_peptide_loss)
print("Default val new peptide loss: ", default_val_peptide_loss)

#Define model
model = define_model()

#Train model using early stopping manually
val_losses = []
best_val_loss = 1e20
best_model = None
best_epoch = 0

min_delta = 0.01
patience_threshold = 100
patience_count = 0

for epoch in range(1000):

    history = model.fit(x = [X_train, C_train,  joined_intensities], 
                    y = None, 
                    epochs=1, batch_size = batch_size)    

    #Make predictions and calculate loss
    predicted_val_coeffs = model.predict([X_val, C_val, joined_intensities_val]).ravel()


    val_peptide_loss = K.eval(peptide_loss(K.constant(joined_intensities_val),
                                                  K.constant(predicted_val_coeffs),
                                                  K.constant(alpha_initial_val)))
    print("Epoch " + str(epoch) + " validation loss: " + str(val_peptide_loss))

    val_losses.append(val_peptide_loss)

    if (best_val_loss - val_losses[-1]) > min_delta:
        best_val_loss = val_losses[-1]
        best_model = model.get_weights()
        best_epoch = epoch + 1
        patience_count = 0

    else:
        patience_count += 1

    if patience_count == patience_threshold:
        break

#Recover the best model
model.set_weights(best_model)
print("Best validation loss " + str(best_val_loss) + " at epoch ", str(best_epoch))

#Record predictions
predicted_train_coeffs = model.predict([X_train, C_train,
                                        np.ones(joined_intensities.shape)]).ravel()
predicted_val_coeffs = model.predict([X_val, C_val, np.ones((X_val.shape[0], joined_intensities.shape[1]))]).ravel()



####################################
print("Step 8: Calculate final losses...")

#Make predictions for all samples and record all coefficients
all_sequences = np.expand_dims(data_df.iloc[:, :(seq_length * 20)].values.reshape((data_df.iloc[:, :(seq_length * 20)].shape[0], seq_length, 20)), axis=3)
all_charges = data_df[charge_column_names].values
predicted_coefficients = model.predict([all_sequences, 
                                       all_charges, 
                                       np.ones((all_sequences.shape[0], 
                                                joined_intensities.shape[1]))]).ravel()

predicted_coefficients = np.abs(predicted_coefficients)
predicted_coefficients = pd.DataFrame(predicted_coefficients, index = data_df.index)
print("Sum of coeffs: ", np.sum(predicted_coefficients))
print("Predicted coeffs: ", predicted_coefficients.sort_values(by = 0))


tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x

####################################
#Optimize the loss to get the final alpha values for the train set

#Convert matrices to tensors
joined_X = tf.convert_to_tensor(joined_intensities)
peptide_coefficients = tf.convert_to_tensor(predicted_train_coeffs)

#Define optimizer
op =   tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)

#Define minimization problem
alpha_final_train = alpha_initial_train.astype(np.float32)
alpha_final_train = tf.Variable(alpha_final_train)

loss_value = peptide_loss(joined_X, peptide_coefficients, alpha_final_train)
train_op = op.minimize(loss_value, 
             var_list = alpha_final_train)

#Optimization
init = tf.compat.v1.initialize_all_variables()
final_alpha_value = alpha_final_train
loss_scores_list = [0.0]
score_diff = 100

#Optimize the loss function
with tf.compat.v1.Session() as session:
    session.run(init)
    print("Starting at", "alphas:", session.run(alpha_final_train))
    print("Starting at", "Loss:", session.run(loss_value))

    step = 0
    #Train until convergence or max no of steps
    while score_diff > 0.001 and step < 10000:
        session.run(train_op)
        step = step + 1

        score_diff = np.abs(session.run(loss_value) - loss_scores_list[-1])
        loss_scores_list.append(session.run(loss_value))

        final_alpha_value = session.run(alpha_final_train)

    print("Done at", "Loss:", session.run(loss_value))

alpha_final_train = final_alpha_value


####################################
#Optimize the loss to get the final alpha values for the val set

#Convert matrices to tensors
joined_X = tf.convert_to_tensor(joined_intensities_val)
peptide_coefficients = tf.convert_to_tensor(predicted_val_coeffs)

#Define optimizer
op =  tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)

#Define minimization problem
alpha_final_val = alpha_initial_val.astype(np.float32)
alpha_final_val = tf.Variable(alpha_final_val)

loss_value = peptide_loss(joined_X, peptide_coefficients, alpha_final_val)
train_op = op.minimize(loss_value, 
             var_list = alpha_final_val)

#Record scores from all experiments
init = tf.compat.v1.initialize_all_variables()
final_alpha_value = alpha_final_val
loss_scores_list = [0.0]
score_diff = 100

#Optimize the loss function
with tf.compat.v1.Session() as session:
    session.run(init)
    print("Starting at", "alphas:", session.run(final_alpha_value))
    print("Starting at", "Loss:", session.run(loss_value))

    step = 0
    #Train until convergence or max no of steps
    while score_diff > 0.001 and step < 10000:

        session.run(train_op)
        step = step + 1

        score_diff = np.abs(session.run(loss_value) - loss_scores_list[-1])
        loss_scores_list.append(session.run(loss_value))

        final_alpha_value = session.run(alpha_final_val)

    print("Done at", "Loss:", session.run(loss_value))

alpha_final_val = final_alpha_value


####################################
#Record final losses

final_train_peptide_loss = K.eval(peptide_loss(K.constant(joined_intensities),
                                                  K.constant(predicted_train_coeffs),
                                                  K.constant(alpha_final_train)))

final_val_peptide_loss = K.eval(peptide_loss(K.constant(joined_intensities_val),
                                                  K.constant(predicted_val_coeffs),
                                                  alpha_final_val))

final_train_improvement = ((default_train_peptide_loss - final_train_peptide_loss) / default_train_peptide_loss) * 100
final_val_improvement = ((default_val_peptide_loss - final_val_peptide_loss) / default_val_peptide_loss) * 100

print()
print("Default train new peptide loss: ", default_train_peptide_loss)
print("Default val new peptide loss: ", default_val_peptide_loss)
print()
print("Final train peptide loss: ", final_train_peptide_loss)
print("Final val peptide loss: ", final_val_peptide_loss)
print()
print("Percent improvement in train: ", final_train_improvement)
print("Percent improvement in val: ", final_val_improvement)


#Record results
results = np.array([final_val_peptide_loss,
                    final_val_improvement,
                    model.count_params(),
                    best_epoch])

columns = ['Val peptide loss', 'Val percent improvement', 
           'No of parameters', 'No of epochs']

result_df = pd.DataFrame(results.reshape((1, len(columns))), columns = columns)

result_df.to_csv('tuning_results/' + str(args.output_file) + '/Gridsearch_Results_' + \
            'n_conv_layers' + str(n_conv_layers) + \
            '_n_filters_' + str(n_filters) + \
            '_filter_size_' + str(filter_size) + \
            '_n_layers_' + str(n_layers) + \
            '_n_nodes_' + str(n_nodes) + \
            '_dropout_rate_' + str(dropout_rate) + \
            '_batch_size_' + str(batch_size) + \
            '_learning_rate_' + str(learning_rate) + '.tsv', sep = '\t')
