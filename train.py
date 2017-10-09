#Training script for omnidirectional collaborative filtering models


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
dataset = "movielens20m" # movielens20m, amazon_books, amazon_moviesAndTv, amazon_videoGames, amazon_clothing, beeradvocate, yelp, netflix
useTimestamps = False
reverse_user_item_data = False

#Training parameters
max_epochs = 100
train_sparsity = [0.0, 1.0] #Probability of a data point being treated as an input (lower numbers mean a sparser recommendation problem)
test_sparsities = [0.0, 0.1, 0.4, 0.5, 0.6, 0.9] #0.0 Corresponds to the cold start problem. This is not used when eval_mode = "fixed_split"
batch_size = 256 #Bigger batches appear to be very important in getting this to work well. I hypothesize that this is because the optimizer is not fighting itself when optimizing for different things across trials
patience = 4
shuffle_data_every_epoch = True
val_split = [0.8, 0.1, 0.1]
useJSON = True
early_stopping_metric = "val_accurate_MSE" # "val_loss" #"val_accurate_RMSE"
eval_mode = "fixed_split" # "ablation" or "fixed_split" #Ablation is for splitting the datasets by user and predicting ablated ratings within a user. This is a natural metric because we want to be able to predict unobserved user ratings from observed user ratings
#Fixed split is for splitting the datasets by rating. This is the standard evaluation procedure in the literature. 
l2_weight_regulatization = None #0.01 #The parameter value for the l2 weight regularization. Use None for no regularization.

#Model parameters
numlayers = 3
num_hidden_units = 512
use_causal_info = True #Toggles whether or not the model incorporates the auxilliary info. Setting this to off and setting the auxilliary_mask_type to "zeros" have the same computational effect, however this one runs faster but causes some errors with model saving. It is recommended to keep this set to True
auxilliary_mask_type = "dropout" #Default is "dropout". Other options are "causal", "zeros", and "both" which uses both the causal and the dropout masks.
aux_var_value = -1 #-1 is Zhouwen's suggestion. Seems to work better than the default of 1.
model_save_path = "models/"
model_loss = 'mean_squared_error' # "mean_absolute_error" 'mean_squared_error'
optimizer = 'adagrad' #'rmsprop' 'adam' 'adagrad'
activation_type = 'tanh' #Try 'selu' or 'elu'
use_sparse_representation = False

model_save_name = "0p0-1p0trainSparsity_"+str(batch_size)+"bs_"+str(numlayers)+"lay_"+str(num_hidden_units)+"hu_" + str(l2_weight_regulatization) + "regul_" + optimizer#"noCausalInfo_0p5trainSparsity_128bs_3lay_256hu"

#Set dataset params
if dataset == "movielens20m":
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
if dataset == "amazon_videoGames": #1324753 ratings
	data_path = "./data/amazon_videoGames/"
	num_items = 50210 #26.384 ratings per item
	num_users = 826767 #1.602 ratings per user
	rating_range = 4.0
	nonsequentialusers = True
if dataset == "amazon_clothing": #Clothing, shoes, and jewlery
	data_path = "./data/amazon_clothing/"
	num_items = 1136004 
	num_users = 3117268 
	rating_range = 4.0
	nonsequentialusers = True
if dataset == "beeradvocate":
	data_path = "./data/beeradvocate/" #"/data1/beer/beeradvocate-crawler/ba_ratings.csv"
	num_items = 35611
	num_users = 84196
	rating_range = 4.0 #From 1.0 to 5.0
	nonsequentialusers = True
if dataset == "yelp":
	data_path = "./data/yelp/" 
	num_items = 156638
	num_users = 1183362
	rating_range = 4.0 #From 1.0 to 5.0
	nonsequentialusers = True
if dataset == "netflix":
	data_path = "./data/netflix/" 
	num_items = 17768
	num_users = 477412
	rating_range = 4.0 #From 1.0 to 5.0
	nonsequentialusers = True

if reverse_user_item_data:
	#data_path = data_path+"reverse_item-user/"
	num_items_temp = num_items
	num_items = num_users
	num_users = num_items_temp
	model_save_name += "_itemUserReverse"

model_save_name += "_" + dataset + "_"
modelRunIdentifier = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
model_save_name += modelRunIdentifier #Append a unique identifier to the filename

print("Loading data for " + dataset)
data_reader = data_reader(num_items, num_users, data_path, nonsequentialusers = nonsequentialusers, use_json=useJSON, eval_mode=eval_mode, useTimestamps=useTimestamps, reverse_user_item_data = reverse_user_item_data)

if eval_mode == "ablation":
	data_reader.split_for_validation(val_split) #Create a train-valid-test split
	#If the eval mode is "fixed_split" then the data is aldready split

#NEED TO IMPLEMENT TRAIN-TEST SPLIT

#Build model
if auxilliary_mask_type=='both':
	use_both_masks=True
else:
	use_both_masks=False
omni_m = omni_model(numlayers, num_hidden_units, num_items, dense_activation = activation_type, use_causal_info = use_causal_info, use_timestamps = useTimestamps, use_both_masks = use_both_masks, l2_weight_regulatization=l2_weight_regulatization, sparse_representation=use_sparse_representation)
m = omni_m.model


def accurate_MAE(y_true, y_pred):
	#num_predictions = [0 if y_true[i]==y_pred[i]==0 else 1 for i in range(num_items)]
	num_predictions = tf.count_nonzero(y_true+y_pred, dtype=tf.float32)#Count ratings that are non-zero in both the prediction and the targets (the predictions are zeroed explicitly for missing ratings.)
	MAE = metrics.mae(y_true, y_pred)
	return (MAE*num_items*batch_size)/num_predictions #Normalize to count only the ratings that are present.
	#return (MAE/num_predictions)*num_items*batch_size #Normalize to count only the ratings that are present.

def accurate_RMSE(y_true, y_pred):
	#num_predictions = [0 if y_true[i]==y_pred[i]==0 else 1 for i in range(num_items)]
	num_predictions = tf.count_nonzero(y_true+y_pred, dtype=tf.float32)#Count ratings that are non-zero in both the prediction and the targets (the predictions are zeroed explicitly for missing ratings.)
	MSE = metrics.mse(y_true, y_pred)
	return tf.sqrt((MSE*num_items*batch_size)/num_predictions) #Normalize to count only the ratings that are present. Then take the square root for RMSE.

def accurate_RMSE_test(y_true, y_pred):
	#num_predictions = [0 if y_true[i]==y_pred[i]==0 else 1 for i in range(num_items)]
	num_predictions = tf.count_nonzero(y_true+y_pred, dtype=tf.float32)#Count ratings that are non-zero in both the prediction and the targets (the predictions are zeroed explicitly for missing ratings.)
	scale = tf.cast(tf.size(y_true, out_type=tf.int32), tf.float32)
	MSE = metrics.mse(y_true, y_pred)
	return tf.sqrt((MSE*scale)/num_predictions) #Normalize to count only the ratings that are present. Then take the square root for RMSE.

def accurate_RMSE_tensorflow(y_true, y_pred):
	#num_predictions = [0 if y_true[i]==y_pred[i]==0 else 1 for i in range(num_items)]
	num_predictions = tf.count_nonzero(y_true+y_pred, dtype=tf.float32)#Count ratings that are non-zero in both the prediction and the targets (the predictions are zeroed explicitly for missing ratings.)
	MSE = tf.metrics.mean_squared_error(y_true, y_pred)
	return tf.sqrt((MSE*num_items*batch_size)/num_predictions) #Normalize to count only the ratings that are present. Then take the square root for RMSE.

def accurate_MSE(y_true, y_pred):
	#num_predictions = [0 if y_true[i]==y_pred[i]==0 else 1 for i in range(num_items)]
	num_predictions = tf.count_nonzero(y_true+y_pred, dtype=tf.float32)#Count ratings that are non-zero in both the prediction and the targets (the predictions are zeroed explicitly for missing ratings.)
	MSE = metrics.mse(y_true, y_pred)
	return (MSE*num_items*batch_size)/num_predictions #Normalize to count only the ratings that are present.

def nMAE(y_true, y_pred):
	#num_predictions = [0 if y_true[i]==y_pred[i]==0 else 1 for i in range(num_items)]
	num_predictions = tf.count_nonzero(y_true+y_pred, dtype=tf.float32)#Count ratings that are non-zero in both the prediction and the targets (the predictions are zeroed explicitly for missing ratings.)
	MAE = metrics.mae(y_true, y_pred)
	return ((MAE*num_items*batch_size)/num_predictions)/rating_range #Normalize to count only the ratings that are present. Then normalize by the rating range.


m.compile(optimizer=optimizer,
              loss=model_loss,
              metrics=['mae', accurate_MAE, nMAE, accurate_RMSE, accurate_MSE])

#callbax = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto'), 
#		keras.callbacks.ModelCheckpoint(model_save_path+model_save_name)]
#callbax = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')]


min_loss = None
best_epoch = 0
val_history = []
for i in range(max_epochs):
	print("Starting epoch ", i+1)
	#Rebuild the generators for each epoch (the train-valid set assignments stay the same)
	train_gen = data_reader.data_gen(batch_size, train_sparsity, train_val_test = "train", shuffle=shuffle_data_every_epoch, auxilliary_mask_type = auxilliary_mask_type, aux_var_value = aux_var_value)
	valid_gen = data_reader.data_gen(batch_size, train_sparsity, train_val_test = "valid", shuffle=shuffle_data_every_epoch, auxilliary_mask_type = auxilliary_mask_type, aux_var_value = aux_var_value)
	
	#Train model
	callbax = [keras.callbacks.ModelCheckpoint(model_save_path+model_save_name+"_epoch_"+str(i+1))] #Could also set save_weights_only=True
	history = m.fit_generator(train_gen, np.floor(data_reader.train_set_size/batch_size)-1, 
		callbacks=callbax, validation_data=valid_gen, validation_steps=np.floor(data_reader.val_set_size/batch_size)-1)
	
	#Early stopping code
	val_loss_list = history.history[early_stopping_metric]
	val_loss = val_loss_list[len(val_loss_list)-1]
	val_history.extend(val_loss_list)
	if min_loss == None:
		min_loss = val_loss
	elif min_loss>val_loss:
		min_loss = val_loss
		best_epoch = i
	elif i-best_epoch>patience:
		print("Stopping early at epoch ", i+1)
		print("Best epoch was ", best_epoch+1)
		print("Val history: ", val_history)
		break
	


#Testing
try:
	best_m = keras.models.load_model(model_save_path+model_save_name+"_epoch_"+str(best_epoch+1), 
		custom_objects={'accurate_MAE': accurate_MAE, 'accurate_RMSE': accurate_RMSE, 'nMAE': nMAE, 'accurate_MSE': accurate_MSE})
	best_m.save(model_save_path+model_save_name+"_bestValidScore") #resave the best one so it can be found later
	test_epoch = best_epoch+1
except:
	print("FAILED TO LOAD BEST MODEL. TESTING WITH MOST RECENT MODEL.")
	best_m = m
	test_epoch = i+1

print("Testing model from epoch: ", test_epoch)

if eval_mode == "ablation":
	print("\nEvaluating model with ablations")
	for i, test_sparsity in enumerate(test_sparsities):

		test_gen = data_reader.data_gen(batch_size, test_sparsity, train_val_test = "test", shuffle=shuffle_data_every_epoch, auxilliary_mask_type = auxilliary_mask_type, aux_var_value = aux_var_value)

		test_results = best_m.evaluate_generator(test_gen, np.floor(data_reader.test_set_size/batch_size)-1)

		print("\nTest results with sparsity: ", test_sparsity)
		print(test_results)
		for i in range(len(test_results)):
			print(m.metrics_names[i], " : ", test_results[i])

elif eval_mode == "fixed_split":
	print("\nEvaluating model with fixed split")
	test_gen = data_reader.data_gen(batch_size, None, train_val_test = "test", shuffle=shuffle_data_every_epoch, auxilliary_mask_type = auxilliary_mask_type, aux_var_value = aux_var_value)
	test_results = best_m.evaluate_generator(test_gen, np.floor(data_reader.test_set_size/batch_size)-1)
	print("Test results with fixed split")
	#print(test_results)
	for i in range(len(test_results)):
		print(m.metrics_names[i], " : ", test_results[i])


	print("Testing manually")
	test_gen_manual = data_reader.data_gen(batch_size, None, train_val_test = "test", shuffle=shuffle_data_every_epoch, auxilliary_mask_type = auxilliary_mask_type, aux_var_value = aux_var_value, return_target_count=True)
	
	predictions = []
	targets = []
	ratings_count = 0
	print("Predicting")
	for i in range(int(np.floor(data_reader.test_set_size/batch_size))):
		current_data = test_gen_manual.next()
		input_list = current_data[0]
		current_targets = current_data[1]
		cur_ratings_count = current_data[2]
		targets.append(current_targets)
		ratings_count += cur_ratings_count
		current_preds = best_m.predict(input_list, batch_size=batch_size, verbose=0)
		predictions.append(current_preds)

	print("Computing error")
	def compute_full_RMSE(predictions, targets, ratings_count):
		sum_squared_error = 0
		for i in range(len(predictions)):
			cur_preds = predictions[i]
			cur_tars = targets[i]
			error_contribution = np.sum(np.square(np.subtract(cur_preds, cur_tars)))
			sum_squared_error += error_contribution
		MSE = sum_squared_error/ratings_count
		RMSE = np.sqrt(MSE)
		return RMSE

	RMSE = compute_full_RMSE(predictions, targets, ratings_count)
	print("Manual test RMSE is ", RMSE)


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
