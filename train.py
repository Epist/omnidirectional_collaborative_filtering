#Training script for the movielens dataset

from __future__ import division
from __future__ import print_function
from data_reader import data_reader
import keras
from model import omni_model
#from ops import accurate_MAE, nMAE
import pandas as pd
import numpy as np
from keras import metrics
import tensorflow as tf
import datetime

#Parameters:

#Dataset parameters 
dataset = "amazon_moviesAndTv" # movielens, amazon_books, amazon_moviesAndTv, amazon_videoGames

#Training parameters
num_epochs = 20
train_sparsity = 0.5 #Probability of a data point being treated as an input (lower numbers mean a sparser recommendation problem)
test_sparsities = [0.0, 0.1, 0.4, 0.5, 0.6, 0.9] #0.0 Corresponds to the cold start problem
batch_size = 128 #Bigger batches appear to be very important in getting this to work well. I hypothesize that this is because the optimizer is not fighting itself when optimizing for different things across trials
patience = 2
shuffle_data_every_epoch = True
val_split = [0.8, 0.1, 0.1]
useJSON = True
early_stopping_metric = "val_accurate_RMSE" # "val_loss"

#Model parameters
numlayers = 3
num_hidden_units = 256
use_causal_info = True #Toggles whether or not the model incorporates the auxilliary info. Setting this to off and setting the auxilliary_mask_type to "zeros" have the same computational effect, however this one runs faster but causes some errors with model saving. It is recommended to keep this set to True
auxilliary_mask_type = "dropout" #Default is "dropout". Other options are "causal" and "zeros"
aux_var_value = -1 #-1 is Zhouwen's suggestion. Seems to work better than the default of 1.
model_save_path = "models/"
model_save_name = "0p5trainSparsity_128bs_3lay_256hu" #"noCausalInfo_0p5trainSparsity_128bs_3lay_256hu"



#Set dataset params
if dataset == "movielens":
	data_path = "./data/movielens/"#'/data1/movielens/ml-20m'
	num_items = 26744 #27000
	num_users = 138493 #138000
	rating_range = 4.5 #20 for jester, 4.5 for movielens (min rating is 0.5)
	nonsequentialusers = True #False
if dataset == "amazon_books":
	data_path = "./data/amazon_books/"
	num_items = 2330066 #9.659 ratings per item
	num_users = 8026324 #2.804 ratings per user
	rating_range = 4.0
	nonsequentialusers = True
if dataset == "amazon_moviesAndTv":
	data_path = "./data/amazon_moviesAndTv/"
	num_items = 200941 #22.927 ratings per item
	num_users = 2088620 #2.206 ratings per user
	rating_range = 4.0
	nonsequentialusers = True
if dataset == "amazon_videoGames":
	data_path = "./data/amazon_videoGames/"
	num_items = 50210 #26.384 ratings per item
	num_users = 826767 #1.602 ratings per user
	rating_range = 4.0
	nonsequentialusers = True

model_save_name += "_" + dataset + "_"
modelRunIdentifier = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
model_save_name += modelRunIdentifier #Append a unique identifier to the filename

print("Loading data for " + dataset)
data_reader = data_reader(num_items, num_users, data_path, nonsequentialusers = nonsequentialusers, use_json=useJSON)

data_reader.split_for_validation(val_split) #Create a train-valid-test split

#NEED TO IMPLEMENT TRAIN-TEST SPLIT

#Build model
omni_m = omni_model(numlayers, num_hidden_units, num_items, use_causal_info = use_causal_info)
m = omni_m.model


def accurate_MAE(y_true, y_pred):
	#num_predictions = [0 if y_true[i]==y_pred[i]==0 else 1 for i in range(num_items)]
	num_predictions = tf.count_nonzero(y_true+y_pred, dtype=tf.float32)
	MAE = metrics.mae(y_true, y_pred)
	return (MAE/num_predictions)*num_items*batch_size

def accurate_RMSE(y_true, y_pred):
	#num_predictions = [0 if y_true[i]==y_pred[i]==0 else 1 for i in range(num_items)]
	num_predictions = tf.count_nonzero(y_true+y_pred, dtype=tf.float32)
	MSE = metrics.mse(y_true, y_pred)
	return tf.sqrt((MSE/num_predictions)*num_items*batch_size)

def nMAE(y_true, y_pred):
	#num_predictions = [0 if y_true[i]==y_pred[i]==0 else 1 for i in range(num_items)]
	num_predictions = tf.count_nonzero(y_true+y_pred, dtype=tf.float32)
	MAE = metrics.mae(y_true, y_pred)
	return ((MAE/num_predictions)*num_items*batch_size)/rating_range


m.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['mae', accurate_MAE, nMAE, accurate_RMSE])

#callbax = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto'), 
#		keras.callbacks.ModelCheckpoint(model_save_path+model_save_name)]
#callbax = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')]
callbax = [keras.callbacks.ModelCheckpoint(model_save_path+model_save_name)]

max_loss = None
max_epoch = 0
val_history = []
for i in range(num_epochs):
	print("Starting epoch ", i+1)
	#Rebuild the generators for each epoch (the train-valid set assignments stay the same)
	train_gen = data_reader.data_gen(batch_size, train_sparsity, train_val_test = "train", shuffle=shuffle_data_every_epoch, auxilliary_mask_type = auxilliary_mask_type, aux_var_value = aux_var_value)
	valid_gen = data_reader.data_gen(batch_size, train_sparsity, train_val_test = "valid", shuffle=shuffle_data_every_epoch, auxilliary_mask_type = auxilliary_mask_type, aux_var_value = aux_var_value)
	
	#Train model
	history = m.fit_generator(train_gen, np.floor(data_reader.train_set_size/batch_size)-1, 
		callbacks=callbax, validation_data=valid_gen, validation_steps=np.floor(data_reader.val_set_size/batch_size)-1)
	
	#Early stopping code
	val_loss_list = history.history[early_stopping_metric]
	val_loss = val_loss_list[len(val_loss_list)-1]
	val_history.extend(val_loss_list)
	if max_loss == None:
		max_loss = val_loss
	elif max_loss<val_loss:
		max_loss = val_loss
		max_epoch = i
	elif i-max_epoch>patience:
		print("Stopping early at epoch ", i+1)
		print("Val history: ", val_history)
		break
	


#Testing

print("Testing model")
for i, test_sparsity in enumerate(test_sparsities):

	test_gen = data_reader.data_gen(batch_size, test_sparsity, train_val_test = "test", shuffle=shuffle_data_every_epoch, auxilliary_mask_type = auxilliary_mask_type, aux_var_value = aux_var_value)

	test_results = m.evaluate_generator(test_gen, np.floor(data_reader.test_set_size/batch_size)-1)

	print("\nTest results with sparsity: ", test_sparsity)
	print(test_results)
	for i in range(len(test_results)):
		print(m.metrics_names[i], " : ", test_results[i])



#TODO:

#SHOULD ADD MODEL SAVING


#Figure out how to make use of contextual inforamtion such as movie tags, timestamps, etc.
#Train an additional network to compute inter-movie similarity?
#Embed the movies based on tags and train a network to predict ratings based on input and output movie embeddings?
#Something closer to the current implementation?

#Try using gender and age side-information

#Need to implement custom early stopping routine

#Should try various training techniques to allow the model to handle various different sparsities 
#(such as training with various mixed i/o dropout frequencies or possibly even training with interior dropout)