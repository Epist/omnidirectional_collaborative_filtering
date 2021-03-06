#Movie lens data reader
from __future__ import division
from __future__ import print_function
import pickle
import json
import numpy as np
from tensorflow import SparseTensor
import scipy
import scipy.sparse

class data_reader(object):
	def __init__(self, num_items, num_users, filepath, nonsequentialusers=False, use_json=True, eval_mode = "ablation", useTimestamps=False, reverse_user_item_data = False):
		self.num_items = num_items
		self.num_users = num_users
		self.filepath = filepath
		self.nonsequentialusers = nonsequentialusers
		self.eval_mode = eval_mode
		self.useTimestamps = useTimestamps

		if reverse_user_item_data:
			self.unique_items = self.load_data(self.filepath, "unique_users_list", use_json)
		else:
			self.unique_items = self.load_data(self.filepath, "unique_items_list", use_json)
		self.items_to_densevec = {}
		self.densevec_to_items = {}
		for i, item in enumerate(self.unique_items):
		    self.items_to_densevec[item] = i
		    self.densevec_to_items[i] = item #Inverse mapping to get original item ID back

		if nonsequentialusers == True:
			if reverse_user_item_data:
				self.unique_users = self.load_data(self.filepath, "unique_items_list", use_json)
			else:
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

		if reverse_user_item_data:
			fn_base_string = "ratingsByItem"
		else:
			fn_base_string = "ratingsByUser"

		if self.eval_mode == "ablation":
			if self.useTimestamps:
				self.user_dict = self.load_data(self.filepath, fn_base_string+"_dict_timestamps", use_json)
			else:
				self.user_dict = self.load_data(self.filepath, fn_base_string+"_dict", use_json)

		elif self.eval_mode == "fixed_split":
			if self.useTimestamps:
				#Load the seperate train, valid, and test files
				self.user_dicts_train, self.user_dicts_train_timestamps = self.load_data(self.filepath, fn_base_string+"_dicts_withtimestamps_train", use_json)
				self.user_dict = self.user_dicts_train #Useful for method reuse since the training is handled via ablations
				self.user_dict_timestamps = self.user_dicts_train_timestamps #Useful for method reuse since the training is handled via ablations
				self.user_dicts_valid, self.user_dicts_valid_timestamps =self.load_data(self.filepath, fn_base_string+"_dicts_withtimestamps_valid", use_json)
				self.user_dicts_test, self.user_dicts_test_timestamps = self.load_data(self.filepath, fn_base_string+"_dicts_withtimestamps_test", use_json)
			else:
				#Load the seperate train, valid, and test files
				self.user_dicts_train = self.load_data(self.filepath, fn_base_string+"_dicts_train", use_json)
				self.user_dict = self.user_dicts_train #Useful for method reuse since the training is handled via ablations
				self.user_dicts_valid =self.load_data(self.filepath, fn_base_string+"_dicts_valid", use_json)
				self.user_dicts_test = self.load_data(self.filepath, fn_base_string+"_dicts_test", use_json)

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


	def build_sparse_batch(self, user_order, batch_size, start_index, end_index, data_sparsity, aux_var_value, sparse_representation = False, pass_through_input_training = False):

		#If using timestamps, do not mask the timestamps...
		if sparse_representation:
			mask_batch_inputs = ([], ([],[]))
			#mask_batch_targets = ([], [], [batch_size, self.num_items])#(indices, values, shape)
			ratings_batch_inputs = ([], ([],[]))
			#ratings_batch_targets = ([], [], [batch_size, self.num_items])#(indices, values, shape)

			missing_data_mask = ([], ([],[]))

			if self.useTimestamps:
				timestamps_batch = ([], ([],[]))
		else:
			mask_batch_inputs = np.zeros([batch_size, self.num_items])
			ratings_batch_inputs = np.zeros([batch_size, self.num_items])

			missing_data_mask = np.zeros([batch_size, self.num_items])

			if self.useTimestamps:
				timestamps_batch = np.zeros([batch_size, self.num_items])

		ratings_batch_targets = np.zeros([batch_size, self.num_items])
		mask_batch_targets = np.zeros([batch_size, self.num_items])

		batch_data_sparsities = np.random.uniform(low=data_sparsity[0], high=data_sparsity[1], size = batch_size)
		batch_element=0 #For indexing the rows within a batch (since the index i is global)
		for i in range(start_index, end_index):
			user_id = user_order[i]
			if self.eval_mode == "ablation":
				user_id_raw = self.densevec_to_users[user_id] #Get the user_id used in the dataset (user_id_raw)
			elif self.eval_mode == "fixed_split":
				user_id_raw = user_id
			item_rating_list = self.user_dict[user_id_raw]
			num_items_cur_user = len(item_rating_list)
			random_dropout_split = np.random.choice([0,1], size=num_items_cur_user, p=[1-batch_data_sparsities[batch_element], batch_data_sparsities[batch_element]])
			if self.useTimestamps:
				item_timestamp_list = self.user_dict[user_id_raw]
			
			for j, item_rating in enumerate(item_rating_list):
				item_id = self.items_to_densevec[item_rating[0]]###CHANGED
				rating = item_rating[1]
				if sparse_representation:
					if random_dropout_split[j] == 1:
						mask_batch_inputs[0].append(aux_var_value)
						mask_batch_inputs[1][0].append(batch_element)
						mask_batch_inputs[1][1].append(item_id)
						ratings_batch_inputs[0].append(rating)
						ratings_batch_inputs[1][0].append(batch_element)
						ratings_batch_inputs[1][1].append(item_id)
						if pass_through_input_training:
							mask_batch_targets[batch_element, item_id] = aux_var_value
							ratings_batch_targets[batch_element, item_id] = rating
					elif random_dropout_split[j] == 0:
						mask_batch_targets[batch_element, item_id] = aux_var_value
						ratings_batch_targets[batch_element, item_id] = rating
					else:
						raise(exception("Reciprocal dropout exception!"))
					missing_data_mask[0].append(aux_var_value)
					missing_data_mask[1][0].append(batch_element)
					missing_data_mask[1][1].append(item_id)					

				else:
					if random_dropout_split[j] == 1:
						mask_batch_inputs[batch_element, item_id] = aux_var_value
						ratings_batch_inputs[batch_element, item_id] = rating
						if pass_through_input_training:
							mask_batch_targets[batch_element, item_id] = aux_var_value
							ratings_batch_targets[batch_element, item_id] = rating
					elif random_dropout_split[j] == 0:
						mask_batch_targets[batch_element, item_id] = aux_var_value
						ratings_batch_targets[batch_element, item_id] = rating
					else:
						raise(exception("Reciprocal dropout exception!"))
					missing_data_mask[batch_element, item_id] = aux_var_value
					

			if self.useTimestamps:
				for item_timestamp in item_timestamp_list:
					item_id = self.items_to_densevec[item_timestamp[0]]
					timestamp = item_timestamp[1]
					if sparse_representation:
						timestamps_batch[0].append(timestamp)
						timestamps_batch[1][0].append(batch_element)
						timestamps_batch[1][1].append(item_id)
					else:
						timestamps_batch[batch_element, item_id] = timestamp
			batch_element+=1

		if sparse_representation:
			input_masks = scipy.sparse.coo_matrix(mask_batch_inputs, (batch_size, self.num_items))
			inputs = scipy.sparse.coo_matrix(ratings_batch_inputs, (batch_size, self.num_items))
			missing_data_mask = scipy.sparse.coo_matrix(missing_data_mask, (batch_size, self.num_items))
			if self.useTimestamps:
				timestamps_batch = scipy.sparse.coo_matrix(timestamps_batch, (batch_size, self.num_items))
		else:
			input_masks = mask_batch_inputs
			inputs = ratings_batch_inputs

		output_masks = mask_batch_targets
		targets = ratings_batch_targets

		if self.useTimestamps:
			return (input_masks, output_masks, inputs, targets, missing_data_mask, timestamps_batch)
		else:
			return (input_masks, output_masks, inputs, targets, missing_data_mask)

	def build_sparse_batch_fixed_split(self, input_dict, target_dict, user_order, batch_size, start_index, end_index, aux_var_value, sparse_representation, timestamps=None):

		#If using timestamps, do not mask the timestamps...
    	#This is a batch generator and goes in the data_reader file

		if sparse_representation:
			mask_batch_input = ([], ([],[]))
			ratings_batch_input = ([], ([],[]))
			missing_data_mask = ([], ([],[]))

			if self.useTimestamps:
				timestamps_batch = ([], ([],[]))
		else:
			mask_batch_input = np.zeros([batch_size, self.num_items])
			ratings_batch_input = np.zeros([batch_size, self.num_items])
			missing_data_mask = np.zeros([batch_size, self.num_items])

			if self.useTimestamps:
				timestamps_batch = np.zeros([batch_size, self.num_items])
		ratings_batch_target = np.zeros([batch_size, self.num_items])
		mask_batch_target = np.zeros([batch_size, self.num_items])

		batch_element=0 #For indexing the rows within a batch (since the index i is global)
		target_count = 0
		for i in range(start_index, end_index):
			user_id_raw = user_order[i]
			item_rating_list_input = input_dict[user_id_raw]
			item_rating_list_target = target_dict[user_id_raw]
			if self.useTimestamps:
				item_timestamp_list = timestamps[user_id_raw]
			#total+=1

			if item_rating_list_input is not None:
				#found+=1
				for item_rating in item_rating_list_input:
					item_id = self.items_to_densevec[item_rating[0]]
					rating = item_rating[1]
					if sparse_representation:
						mask_batch_input[0].append(aux_var_value)
						mask_batch_input[1][0].append(batch_element)
						mask_batch_input[1][1].append(item_id)
						ratings_batch_input[0].append(rating)
						ratings_batch_input[1][0].append(batch_element)
						ratings_batch_input[1][1].append(item_id)
						missing_data_mask[0].append(aux_var_value)
						missing_data_mask[1][0].append(batch_element)
						missing_data_mask[1][1].append(item_id)
					else:
						mask_batch_input[batch_element, item_id] = aux_var_value
						ratings_batch_input[batch_element, item_id] = rating
						missing_data_mask[batch_element, item_id] = aux_var_value
			else:
				pass #Keep the vector of all zeros

			for item_rating in item_rating_list_target:
				item_id = self.items_to_densevec[item_rating[0]]
				rating = item_rating[1]
				mask_batch_target[batch_element, item_id] = aux_var_value
				ratings_batch_target[batch_element, item_id] = rating
				if sparse_representation:
					missing_data_mask[0].append(aux_var_value)
					missing_data_mask[1][0].append(batch_element)
					missing_data_mask[1][1].append(item_id)
				else:
					missing_data_mask[batch_element, item_id] = aux_var_value

				target_count += 1

			if self.useTimestamps:
				for item_timestamp in item_timestamp_list:
					item_id = self.items_to_densevec[item_timestamp[0]]
					timestamp = item_timestamp[1]
					if sparse_representation:
						timestamps_batch[0].append(timestamp)
						timestamps_batch[1][0].append(batch_element)
						timestamps_batch[1][1].append(item_id)
					else:
						timestamps_batch[batch_element, item_id] = timestamp

			batch_element+=1

		if sparse_representation:
			input_masks = scipy.sparse.coo_matrix(mask_batch_input, (batch_size, self.num_items))
			inputs = scipy.sparse.coo_matrix(ratings_batch_input, (batch_size, self.num_items))
			missing_data_mask = scipy.sparse.coo_matrix(missing_data_mask, (batch_size, self.num_items))
			if self.useTimestamps:
				timestamps_batch = scipy.sparse.coo_matrix(timestamps_batch, (batch_size, self.num_items))
		else:		
			input_masks = mask_batch_input
			inputs = ratings_batch_input #* mask_batch_input
		output_masks = mask_batch_target
		targets = ratings_batch_target #* mask_batch_target

		if self.useTimestamps:
			return (input_masks, output_masks, inputs, targets, missing_data_mask, timestamps_batch, target_count)
		else:
			return (input_masks, output_masks, inputs, targets, missing_data_mask, target_count)

	def split_for_validation(self, val_split, seed=None):
		self.val_split = val_split
		if seed is not None:
			np.random.seed(seed)
		random_user_order = np.random.permutation(self.num_users)

		self.train_set_size = int(self.num_users*val_split[0])
		self.val_set_size = int(self.num_users*val_split[1])
		self.test_set_size = int(self.num_users*val_split[2])

		self.train_set = random_user_order[0:self.train_set_size]
		self.val_set = random_user_order[self.train_set_size : self.train_set_size+self.val_set_size]
		self.test_set = random_user_order[self.train_set_size+self.val_set_size : ]

	def data_gen(self, batch_size, data_sparsity, train_val_test = "train", shuffle=True, auxilliary_mask_type = "dropout", aux_var_value = -1, return_target_count=False, sparse_representation = False, pass_through_input_training = False):

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

				if self.useTimestamps:
					(input_masks, output_masks, inputs, targets, missing_data_mask, timestamps_batch) = self.build_sparse_batch(user_order, batch_size, start_index, end_index, data_sparsity, aux_var_value, sparse_representation=sparse_representation, pass_through_input_training = pass_through_input_training)
				else:
					(input_masks, output_masks, inputs, targets, missing_data_mask) = self.build_sparse_batch(user_order, batch_size, start_index, end_index, data_sparsity, aux_var_value, sparse_representation=sparse_representation, pass_through_input_training = pass_through_input_training)

				if auxilliary_mask_type == "causal": #Use an auxilliary mask input that masks only the variables not present in the dataset
					mask_to_feed = missing_data_mask
				elif auxilliary_mask_type == "dropout": #Use an auxilliary mask input that masks both the dropped-out variables and the variables not present in the dataset
					mask_to_feed = input_masks
				elif auxilliary_mask_type == "zeros": #Use a mask that contains no additional information (this allows the model to include none of this info, but avoid hanging inputs which cause problems in the current version of Keras)
					zero_mask = np.zeros_like(input_masks)
					mask_to_feed = zero_mask
				elif auxilliary_mask_type == "both": #Use both the causal and dropout masks
					mask_to_feed = input_masks #The first mask
				elif auxilliary_mask_type is None:
					pass
				else:
					print("Auxilliary mask type ", auxilliary_mask_type, " doesn't exist")
				if auxilliary_mask_type is None:
					input_list = [inputs, output_masks]
				else:
					input_list = [inputs, mask_to_feed, output_masks]
				if self.useTimestamps:
					input_list.append(timestamps)
				if auxilliary_mask_type=="both":
					input_list.append(missing_data_mask)

				yield(input_list, targets)


		elif self.eval_mode == "fixed_split":
			for i in range(num_batches):
				start_index = i*batch_size
				end_index = (i+1)*batch_size

				if train_val_test == "valid":
					input_dict = self.user_dicts_valid[0]
					target_dict = self.user_dicts_valid[1]
					if self.useTimestamps:
						timestamps = self.user_dicts_valid_timestamps
				elif train_val_test == "test":
					input_dict = self.user_dicts_test[0]
					target_dict = self.user_dicts_test[1]
					if self.useTimestamps:
						timestamps = self.user_dicts_test_timestamps

				if self.useTimestamps:
					(input_masks, output_masks, inputs, targets, missing_data_mask, timestamps_batch, target_count) = self.build_sparse_batch_fixed_split(input_dict, target_dict, user_order, batch_size, start_index, end_index, aux_var_value, sparse_representation, timestamps=timestamps)
					#input_list = [inputs, input_masks, output_masks, timestamps_batch]
				else:
					(input_masks, output_masks, inputs, targets, missing_data_mask, target_count) = self.build_sparse_batch_fixed_split(input_dict, target_dict, user_order, batch_size, start_index, end_index, aux_var_value, sparse_representation)
					#input_list = [inputs, input_masks, output_masks]
				
				if auxilliary_mask_type =="causal":
					mask_to_feed = missing_data_mask
				elif auxilliary_mask_type == "dropout":
					mask_to_feed = input_masks
				elif auxilliary_mask_type == "zeros": #Use a mask that contains no additional information (this allows the model to include none of this info, but avoid hanging inputs which cause problems in the current version of Keras)
					zero_mask = np.zeros_like(input_masks)
					mask_to_feed = zero_mask
				elif auxilliary_mask_type =="both":
					mask_to_feed = input_masks #The first mask
					#input_list.append(second_mask)
				elif auxilliary_mask_type is None:
					pass
				else:
					print("Auxilliary mask type ", auxilliary_mask_type, " doesn't exist")
				
				if auxilliary_mask_type is None:
					input_list = [inputs, output_masks]
				else:
					input_list = [inputs, mask_to_feed, output_masks]
				if self.useTimestamps:
					input_list.append(timestamps)
				if auxilliary_mask_type=="both":
					input_list.append(missing_data_mask)

				if return_target_count:
					yield(input_list, targets, target_count)
				else:
					yield(input_list, targets)

		while True: #Makes it an infinite generator so it doesn't error in parallel... (There are other ways to handle this, but the proper way would be to edit Keras... so this will suffice)
			yield None
