#Movie lens data reader
from __future__ import division
from __future__ import print_function
import pickle
import numpy as np

class movielens_reader(object):
	def __init__(self, num_items, num_users, filepath):
		self.num_items = num_items
		self.num_users = num_users
		self.filepath = filepath
		with open(self.filepath+"user_dict.p", "rb") as f:
			self.user_dict = pickle.load(f)

		with open(self.filepath+"unique_movies_list.p", "rb") as f:
			self.unique_movies = pickle.load(f)

		movie_index = 0
		self.movies_to_densevec = {}
		self.densevec_to_movies = {}
		for i, movie in enumerate(self.unique_movies):
		    self.movies_to_densevec[movie] = i
		    self.densevec_to_movies[i] = movie #Inverse mapping to get original movie ID back
		print("Finished loading data")

		#print(self.user_dict[1])


	def build_sparse_batch(self, user_order, batch_size, start_index, end_index):
		mask_batch = np.zeros([batch_size, self.num_items])
		ratings_batch = np.zeros([batch_size, self.num_items])
		batch_element=0 #For indexing the rows within a batch (since the index i is global)
		for i in range(start_index, end_index):
			user_id = user_order[i]
			movie_rating_list = self.user_dict[user_id]
			
			for movie_rating in movie_rating_list:
				movie_id = self.movies_to_densevec[int(movie_rating[0])]
				rating = movie_rating[1]
				mask_batch[batch_element, movie_id] = 1
				ratings_batch[batch_element, movie_id] = rating
			batch_element+=1

		return (mask_batch, ratings_batch)

	def split_for_validation(self, val_split, seed=None):
		self.val_split = val_split
		if seed is not None:
			np.random.seed(seed)
		random_user_order = np.random.permutation(self.num_users) +1 #Add one to shit index from 0 start to 1 start

		self.train_set_size = int(self.num_users*val_split[0])
		self.val_set_size = int(self.num_users*val_split[1])
		self.test_set_size = int(self.num_users*val_split[2])

		self.train_set = random_user_order[0:self.train_set_size]
		self.val_set = random_user_order[self.train_set_size : self.train_set_size+self.val_set_size]
		self.test_set = random_user_order[self.train_set_size+self.val_set_size : ]

	def data_gen(self, batch_size, data_sparsity, train_val_test = "train", shuffle=True, auxilliary_mask_type = "dropout", aux_var_value = 1):

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


		while True: #Makes it an infinite generator so it doesn't error in parallel... (There are other ways to handle this, but the proper way would be to edit Keras... so this will suffice)
			yield None