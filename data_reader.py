#Movie lens data reader
from __future__ import division
from __future__ import print_function
import pickle
import json
import numpy as np

class data_reader(object):
	def __init__(self, num_items, num_users, filepath, nonsequentialusers=False, use_json=False, eval_mode = "ablation"):
		self.num_items = num_items
		self.num_users = num_users
		self.filepath = filepath
		self.nonsequentialusers = nonsequentialusers
		self.eval_mode = eval_mode

		self.unique_items = self.load_data(self.filepath, "unique_items_list", use_json)

		self.items_to_densevec = {}
		self.densevec_to_items = {}
		for i, item in enumerate(self.unique_items):
		    self.items_to_densevec[item] = i
		    self.densevec_to_items[i] = item #Inverse mapping to get original item ID back

		if nonsequentialusers == True:
			self.unique_users = self.load_data(self.filepath, "unique_users_list", use_json)

			self.users_to_densevec = {}
			self.densevec_to_users = {}
			for i, user in enumerate(self.unique_users):
			    self.users_to_densevec[user] = i
			    self.densevec_to_users[i] = user #Inverse mapping to get original item ID back
		else:
			self.densevec_to_users = {} #Only use this if the users are sequentially ordered with no gaps and zero-based indexing
			for i in range(num_users):
				self.densevec_to_users[i] = i #make a faux userId mapping

		if self.eval_mode == "ablation":
			self.user_dict = self.load_data(self.filepath, "ratingsByUser_dict", use_json)

		elif self.eval_mode == "fixed_split":
			#Load the seperate train, valid, and test files
			self.user_dicts_train = self.load_data(self.filepath, "ratingsByUser_dicts_train", use_json)
			self.user_dict = self.user_dicts_train #Useful for method reuse since the training is handled via ablations
			self.user_dicts_valid =self.load_data(self.filepath, "ratingsByUser_dicts_valid", use_json)
			self.user_dicts_test = self.load_data(self.filepath, "ratingsByUser_dicts_test", use_json)

			#Create the set sizes
			self.train_set_size = len(self.user_dicts_train.keys())
			self.val_set_size = len(self.user_dicts_valid[1].keys())
			self.test_set_size = len(self.user_dicts_test[1].keys())

			#Create the set user orders
			self.train_set = self.user_dicts_train.keys()
			self.val_set = self.user_dicts_valid[1].keys()
			self.test_set = self.user_dicts_test[1].keys()

		print("Finished loading data")


	def load_data(self, filepath, filename, use_json):
		if use_json:
				with open(filepath+filename+".json", "r") as f:
					file_data = json.load(f)
		else:
			with open(filepath+filename+".p", "rb") as f:
				file_data = pickle.load(f)
		return file_data


	def build_sparse_batch(self, user_order, batch_size, start_index, end_index):
		mask_batch = np.zeros([batch_size, self.num_items])
		ratings_batch = np.zeros([batch_size, self.num_items])
		batch_element=0 #For indexing the rows within a batch (since the index i is global)
		for i in range(start_index, end_index):
			user_id = user_order[i]
			if self.eval_mode == "ablation":
				user_id_raw = self.densevec_to_users[user_id] #Get the user_id used in the dataset (user_id_raw)
			elif self.eval_mode == "fixed_split":
				user_id_raw = user_id
			item_rating_list = self.user_dict[user_id_raw]
			
			for item_rating in item_rating_list:
				item_id = self.items_to_densevec[item_rating[0]]###CHANGED
				rating = item_rating[1]
				mask_batch[batch_element, item_id] = 1
				ratings_batch[batch_element, item_id] = rating
			batch_element+=1

		return (mask_batch, ratings_batch)

	def build_sparse_batch_fixed_split(self, input_dict, target_dict, user_order, batch_size, start_index, end_index, aux_var_value): 
    	#This is a batch generator and goes in the data_reader file
		mask_batch_input = np.zeros([batch_size, self.num_items])
		mask_batch_target = np.zeros([batch_size, self.num_items])

		ratings_batch_input = np.zeros([batch_size, self.num_items])
		ratings_batch_target = np.zeros([batch_size, self.num_items])

		batch_element=0 #For indexing the rows within a batch (since the index i is global)
		#found = 0
		#total = 0
		for i in range(start_index, end_index):
			user_id_raw = user_order[i]
			item_rating_list_input = input_dict[user_id_raw]
			item_rating_list_target = target_dict[user_id_raw]
			#total+=1

			if item_rating_list_input is not None:
				#found+=1
				for item_rating in item_rating_list_input:
					item_id = self.items_to_densevec[item_rating[0]]
					rating = item_rating[1]
					mask_batch_input[batch_element, item_id] = aux_var_value
					ratings_batch_input[batch_element, item_id] = rating
			else:

				pass #Keep the vector of all zeros

			for item_rating in item_rating_list_target:
				item_id = self.items_to_densevec[item_rating[0]]
				rating = item_rating[1]
				mask_batch_target[batch_element, item_id] = aux_var_value
				ratings_batch_target[batch_element, item_id] = rating

			batch_element+=1
		#print("Batch input density is ", found/total)

		#Aliasing for readability
		input_masks = mask_batch_input
		output_masks = mask_batch_target
		inputs = ratings_batch_input * mask_batch_input
		targets = ratings_batch_target * mask_batch_target

		return (input_masks, output_masks, inputs, targets)

	def split_for_validation(self, val_split, seed=None):
		self.val_split = val_split
		if seed is not None:
			np.random.seed(seed)
		#random_user_order = np.random.permutation(self.num_users) +1 #Add one to shift index from 0 start to 1 start
		random_user_order = np.random.permutation(self.num_users)

		self.train_set_size = int(self.num_users*val_split[0])
		self.val_set_size = int(self.num_users*val_split[1])
		self.test_set_size = int(self.num_users*val_split[2])

		self.train_set = random_user_order[0:self.train_set_size]
		self.val_set = random_user_order[self.train_set_size : self.train_set_size+self.val_set_size]
		self.test_set = random_user_order[self.train_set_size+self.val_set_size : ]

	def data_gen(self, batch_size, data_sparsity, train_val_test = "train", shuffle=True, auxilliary_mask_type = "dropout", aux_var_value = -1):

		if train_val_test=="train":
			user_order = self.train_set
			num_users_set = self.train_set_size
		elif train_val_test=="valid":
			user_order = self.val_set
			num_users_set = self.val_set_size
		elif train_val_test=="test":
			user_order = self.test_set
			num_users_set = self.test_set_size

		if shuffle:
			user_order = np.random.permutation(user_order)

		num_batches = int(np.floor(num_users_set/batch_size))
		
		if self.eval_mode == "ablation" or  train_val_test == "train":
			for i in range(num_batches):
				start_index = i*batch_size
				end_index = (i+1)*batch_size


				(missing_data_mask, ratings) = self.build_sparse_batch(user_order, batch_size, start_index, end_index)

				random_dropout_array = np.random.choice([0,1], size=ratings.shape, p=[1-data_sparsity, data_sparsity])

				input_masks = random_dropout_array*missing_data_mask*aux_var_value
				output_masks = ((random_dropout_array-1)*-1)*missing_data_mask*aux_var_value

				inputs = ratings*input_masks
				targets = ratings*output_masks

				if auxilliary_mask_type == "causal": #Use an auxilliary mask input that masks only the variables not present in the dataset
					yield([inputs, missing_data_mask, output_masks], targets)
				elif auxilliary_mask_type == "dropout": #Use an auxilliary mask input that masks both the dropped-out variables and the variables not present in the dataset
					yield([inputs, input_masks, output_masks], targets)
				elif auxilliary_mask_type == "zeros": #Use a mask that contains no additional information (this allows the model to include none of this info, but avoid hanging inputs which cause problems in the current version of Keras)
					zero_mask = np.zeros_like(input_masks)
					yield([inputs, zero_mask, output_masks], targets)

		elif self.eval_mode == "fixed_split":
			for i in range(num_batches):
				start_index = i*batch_size
				end_index = (i+1)*batch_size

				if train_val_test == "valid":
					input_dict = self.user_dicts_valid[0]
					target_dict = self.user_dicts_valid[1]
				elif train_val_test == "test":
					input_dict = self.user_dicts_test[0]
					target_dict = self.user_dicts_test[1]

				(input_masks, output_masks, inputs, targets) = self.build_sparse_batch_fixed_split(input_dict, target_dict, user_order, batch_size, start_index, end_index, aux_var_value)
				yield([inputs, input_masks, output_masks], targets)

		while True: #Makes it an infinite generator so it doesn't error in parallel... (There are other ways to handle this, but the proper way would be to edit Keras... so this will suffice)
			yield None
