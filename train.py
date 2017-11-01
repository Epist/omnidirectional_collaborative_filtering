#Training script for omnidirectional collaborative filtering models
from __future__ import division
from __future__ import print_function
from data_reader import data_reader
import keras
from model import omni_model
import pandas as pd
import numpy as np
from keras import metrics
import tensorflow as tf
import datetime
from keras.optimizers import Adagrad, RMSprop
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import h5py
import copy
import json

#Parameters:

#Dataset parameters 
dataset = "ml1m" # movielens20m, amazon_books, amazon_moviesAndTv, amazon_videoGames, amazon_clothing, beeradvocate, yelp, netflix, ml1m
useTimestamps = False
reverse_user_item_data = True

#Training parameters
max_epochs = 500
train_sparsity = [1.0, 1.0] #Probability of a data point being treated as an input (lower numbers mean a sparser recommendation problem)
test_sparsities = [0.0, 0.1, 0.4, 0.5, 0.6, 0.9] #0.0 Corresponds to the cold start problem. This is not used when eval_mode = "fixed_split"
batch_size = 128 #Bigger batches appear to be very important in getting this to work well. I hypothesize that this is because the optimizer is not fighting itself when optimizing for different things across trials
patience = 0
shuffle_data_every_epoch = True
val_split = [0.8, 0.1, 0.1]
useJSON = True
early_stopping_metric = "val_accurate_MSE" # "val_loss" #"val_accurate_RMSE"
eval_mode = "fixed_split" # "ablation" or "fixed_split" #Ablation is for splitting the datasets by user and predicting ablated ratings within a user. This is a natural metric because we want to be able to predict unobserved user ratings from observed user ratings
#Fixed split is for splitting the datasets by rating. This is the standard evaluation procedure in the literature. 
l2_weight_regulatization = None #0 #0.01 #The parameter value for the l2 weight regularization. Use None for no regularization.
pass_through_input_training = True #Turns the model into a denosing autoencoder...
dropout_probability = 0.2

#Model parameters
numlayers = 1
num_hidden_units = 512
use_causal_info = False #Toggles whether or not the model incorporates the auxilliary info. Setting this to off and setting the auxilliary_mask_type to "zeros" have the same computational effect, however this one runs faster but causes some errors with model saving. It is recommended to keep this set to True
auxilliary_mask_type = None#"zeros" #Default is "dropout". Other options are "causal", "zeros", and "both" which uses both the causal and the dropout masks.
aux_var_value = -1 #-1 is Zhouwen's suggestion. Seems to work better than the default of 1.
model_save_path = "models/"
model_loss = 'mean_squared_error' # "mean_absolute_error" 'mean_squared_error'
learning_rate = 0.005
optimizer = Adagrad(lr=learning_rate, epsilon=1e-08, decay=0.0) #'rmsprop' 'adam' 'adagrad'
activation_type = 'sigmoid' #Try 'selu' or 'elu' or 'softplus' or 'sigmoid'
use_sparse_representation = False #Works, but requires a change in keras backend (at least if using Keras (2.0.4) )
use_experimental_sparse_masking_layer = False

load_weights_from = None#"stackedDenoising_NOfinetuning_[0.5, 0.5]trainSparsity_128bs_2lay_512hu_0.005lr_Noneregul_None_sigmoid_itemUserReverse_movielens20m_11_12AM_October_28_2017_bestValidScore" #"0p5trainSparsity_256bs_3lay_512hu_1.0regul_netflix_10_11AM_October_03_2017_bestValidScore" # Model to load weights from for transfer learning experiments
perform_finetuning = False #Set to False if you want to fix the weights
#start_core_training_epoch = 1000
#layers_to_replace = [True, True, False] #Which layers to load weights for and freeze
#perform_finetuning = False

model_save_name = "stackedDenoising_WITHfinetuning_" + str(train_sparsity) +"trainSparsity_"+str(batch_size)+"bs_"+str(numlayers)+"lay_"+str(num_hidden_units)+"hu_" + str(learning_rate) + "lr_" + str(l2_weight_regulatization) + "regul_" + str(auxilliary_mask_type) + "_" + str(activation_type)#"noCausalInfo_0p5trainSparsity_128bs_3lay_256hu"

#Set dataset params
with open("./datasets_metadata.json", "r") as f:
	metadata = json.load(f)
dataset_dict = metadata[dataset]
data_path = dataset_dict["data_path"]
num_items = dataset_dict["num_items"]
num_users = dataset_dict["num_users"]
rating_range = dataset_dict["rating_range"]
nonsequentialusers = dataset_dict["nonsequentialusers"]

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
omni_m = omni_model(numlayers, num_hidden_units, num_items, batch_size, dense_activation = activation_type, use_causal_info = use_causal_info, 
	use_timestamps = useTimestamps, use_both_masks = use_both_masks, l2_weight_regulatization=l2_weight_regulatization, sparse_representation=use_sparse_representation, 
	dropout_probability = dropout_probability, use_sparse_masking_layer = use_experimental_sparse_masking_layer)
m = omni_m.model


def accurate_MAE(y_true, y_pred):
	num_predictions = tf.count_nonzero(y_true+y_pred, dtype=tf.float32)#Count ratings that are non-zero in both the prediction and the targets (the predictions are zeroed explicitly for missing ratings.)
	MAE = metrics.mae(y_true, y_pred)
	return (MAE*num_items*batch_size)/num_predictions #Normalize to count only the ratings that are present.
	#return (MAE/num_predictions)*num_items*batch_size #Normalize to count only the ratings that are present.

def accurate_RMSE(y_true, y_pred):
	num_predictions = tf.count_nonzero(y_true+y_pred, dtype=tf.float32)#Count ratings that are non-zero in both the prediction and the targets (the predictions are zeroed explicitly for missing ratings.)
	MSE = metrics.mse(y_true, y_pred)
	return tf.sqrt((MSE*num_items*batch_size)/num_predictions) #Normalize to count only the ratings that are present. Then take the square root for RMSE.

def accurate_MSE(y_true, y_pred):
	num_predictions = tf.count_nonzero(y_true+y_pred, dtype=tf.float32)#Count ratings that are non-zero in both the prediction and the targets (the predictions are zeroed explicitly for missing ratings.)
	MSE = metrics.mse(y_true, y_pred)
	return (MSE*num_items*batch_size)/num_predictions #Normalize to count only the ratings that are present.

def nMAE(y_true, y_pred):
	num_predictions = tf.count_nonzero(y_true+y_pred, dtype=tf.float32)#Count ratings that are non-zero in both the prediction and the targets (the predictions are zeroed explicitly for missing ratings.)
	MAE = metrics.mae(y_true, y_pred)
	return ((MAE*num_items*batch_size)/num_predictions)/rating_range #Normalize to count only the ratings that are present. Then normalize by the rating range.



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

m.compile(optimizer=optimizer,
              loss=model_loss,
              metrics=['mae', accurate_MAE, nMAE, accurate_RMSE, accurate_MSE])


if load_weights_from is not None:
	print("Loading weights from ", load_weights_from)
	#Set the weights of the dense layers of the model to weights from a pretrained model (for domain adaptation experiments)
	donor_model = keras.models.load_model(model_save_path+load_weights_from, 
		custom_objects={'accurate_MAE': accurate_MAE, 'accurate_RMSE': accurate_RMSE, 'nMAE': nMAE, 'accurate_MSE': accurate_MSE})
	if perform_finetuning:
		print("Fine tuning")
		omni_m.manually_load_all_weights(donor_model)
	else:
		omni_m.load_and_fix_for_denoising_autoencoders(donor_model)

min_loss = None
best_epoch = 0
val_history = []
for i in range(max_epochs):
	print("Starting epoch ", i+1)

	#Rebuild the generators for each epoch (the train-valid set assignments stay the same)
	train_gen = data_reader.data_gen(batch_size, train_sparsity, train_val_test = "train", shuffle=shuffle_data_every_epoch, auxilliary_mask_type = auxilliary_mask_type, aux_var_value = aux_var_value, sparse_representation = use_sparse_representation, pass_through_input_training = pass_through_input_training)
	valid_gen = data_reader.data_gen(batch_size, train_sparsity, train_val_test = "valid", shuffle=shuffle_data_every_epoch, auxilliary_mask_type = auxilliary_mask_type, aux_var_value = aux_var_value, sparse_representation = use_sparse_representation)

	history = m.fit_generator(train_gen, np.floor(data_reader.train_set_size/batch_size)-1, 
		validation_data=valid_gen, validation_steps=np.floor(data_reader.val_set_size/batch_size)-1) #callbacks=callbax
	
	#Early stopping code
	val_loss_list = history.history[early_stopping_metric]
	val_loss = val_loss_list[len(val_loss_list)-1]
	val_history.extend(val_loss_list)
	if min_loss == None:
		min_loss = val_loss
	elif min_loss>val_loss:
		min_loss = val_loss
		best_epoch = i
		m.save(model_save_path+model_save_name+"_epoch_"+str(i+1)+"_bestValidScore") #Only save if it is the best model (will save a lot of time and disk space...)

	elif i-best_epoch>patience:

		print("Stopping early at epoch ", i+1)
		print("Best epoch was ", best_epoch+1)
		print("Val history: ", val_history)

		break
	


#Testing
best_model_fn = model_save_path+model_save_name+"_epoch_"+str(best_epoch+1)+"_bestValidScore"
try: #Delete optimizer if it exists
	print("Deleting optimizer weights for model at ", best_model_fn)
	f = h5py.File(best_model_fn, 'r+')
	del f['optimizer_weights']
	f.close()
except:
	print("Could not delete optimizer weights. They probably weren't saved with the model.")
try:
	best_m = keras.models.load_model(best_model_fn, 
		custom_objects={'accurate_MAE': accurate_MAE, 'accurate_RMSE': accurate_RMSE, 'nMAE': nMAE, 'accurate_MSE': accurate_MSE})
	best_m.save(model_save_path+model_save_name+"_bestValidScore") #resave the best one so it can be found later
	test_epoch = best_epoch+1
except:
	print("FAILED TO LOAD BEST MODEL. TESTING WITH MOST RECENT MODEL.")
	best_m = m
	test_epoch = i+1
test_epoch = best_epoch+1
print("Testing model from epoch: ", test_epoch)

if eval_mode == "ablation":
	print("\nEvaluating model with ablations")
	for i, test_sparsity in enumerate(test_sparsities):

		test_gen = data_reader.data_gen(batch_size, test_sparsity, train_val_test = "test", shuffle=shuffle_data_every_epoch, auxilliary_mask_type = auxilliary_mask_type, aux_var_value = aux_var_value, sparse_representation = use_sparse_representation)

		test_results = best_m.evaluate_generator(test_gen, np.floor(data_reader.test_set_size/batch_size)-1)

		print("\nTest results with sparsity: ", test_sparsity)
		print(test_results)
		for i in range(len(test_results)):
			print(m.metrics_names[i], " : ", test_results[i])

elif eval_mode == "fixed_split":
	print("\nEvaluating model with fixed split")
	test_gen = data_reader.data_gen(batch_size, None, train_val_test = "test", shuffle=shuffle_data_every_epoch, auxilliary_mask_type = auxilliary_mask_type, aux_var_value = aux_var_value, sparse_representation = use_sparse_representation)
	test_results = best_m.evaluate_generator(test_gen, np.floor(data_reader.test_set_size/batch_size)-1)
	print("Test results with fixed split")
	#print(test_results)
	for i in range(len(test_results)):
		print(m.metrics_names[i], " : ", test_results[i])


	print("Testing manually")
	test_gen_manual = data_reader.data_gen(batch_size, None, train_val_test = "test", shuffle=shuffle_data_every_epoch, auxilliary_mask_type = auxilliary_mask_type, aux_var_value = aux_var_value, return_target_count=True, sparse_representation = use_sparse_representation)
	
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
	print("Load this model at: ", model_save_path+model_save_name+"_bestValidScore")