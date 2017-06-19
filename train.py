"""
This file contains the code for training the omnidirectional collaborative filtering model 
using the movielens dataset

"""
from __future__ import division
from __future__ import print_function
import keras
from model import omni_model
#from ops import accurate_MAE, nMAE
import pandas as pd
import numpy as np
from keras import metrics
import tensorflow as tf


#Parameters:

num_epochs = 20
data_path = "/data1/jester/all_jester.csv"#'/data1/movielens/ml-20m'
num_items = 100#27000
num_users = 73421#138000
train_sparsity = 0.5 #Probability of a data point being treated as an input (lower numbers mean a sparser recommendation problem)
test_sparsity = 0.1 #0.0 Corresponds to the cold start problem
batch_size = 128 #Bigger batches appear to be very important in getting this to work well. I hypothesize that this is because the optimizer is not fighting itself when optimizing for different things across trials
rating_range = 20 #20 for jester
patience = 3

numlayers = 2
num_hidden_units = 256

aux_var_value = -1 #Zhouwen's suggestion NOT CURRENTLY IMPLEMENTED


#Load data
data = pd.read_csv(data_path)
all_data_array = data.as_matrix()
#Split the data into inputs and targets 
observed_vars = data.applymap(lambda x: 1 if x!=99 else 0).as_matrix() #Get the values that are actually missing...

#NEED TO IMPLEMENT TRAIN-TEST SPLIT

#Build model
omni_m = omni_model(numlayers, num_hidden_units, num_items, aux_var_value = aux_var_value)
m = omni_m.model


def accurate_MAE(y_true, y_pred):
	#num_predictions = [0 if y_true[i]==y_pred[i]==0 else 1 for i in range(num_items)]
	num_predictions = tf.count_nonzero(y_true+y_pred, dtype=tf.float32)
	MAE = metrics.mae(y_true, y_pred)
	return (MAE/num_predictions)*num_items*batch_size

def nMAE(y_true, y_pred):
	#num_predictions = [0 if y_true[i]==y_pred[i]==0 else 1 for i in range(num_items)]
	num_predictions = tf.count_nonzero(y_true+y_pred, dtype=tf.float32)
	MAE = metrics.mae(y_true, y_pred)
	return ((MAE/num_predictions)*num_items*batch_size)/rating_range


m.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['mae', accurate_MAE, nMAE])

callbax = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')]

for i in range(1):
	#print("\nPerforming randomized reciprocal IO dropout for epoch ", i+1)
	random_dropout_array = np.random.choice([0,1], size=all_data_array.shape, p=[1-train_sparsity, train_sparsity])
	input_masks = random_dropout_array*observed_vars
	output_masks = ((random_dropout_array-1)*-1)*observed_vars

	inputs = all_data_array*input_masks
	targets = all_data_array*output_masks
	#print("Inputs: ", inputs[0,:])
	#print("Targets: ", targets[0,:])
	#Train model
	m.fit([inputs, observed_vars, output_masks], targets, 
		batch_size = batch_size, validation_split=0.1, epochs=num_epochs, shuffle=True, callbacks=callbax) 
	#Unclear whether to use missing_data_mask or input_masks as the auxilliary variable.
	#Right now we use missing_data_mask to avoid having to build a model-internal dropout


#NEED TO IMPLEMENT TESTING

#MINIBATCH GRADIENT DESCENT FOR THE WIN!!!!!

