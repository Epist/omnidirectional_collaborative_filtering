#Train-valid-test split
"""
This script splits a full data file randomly into training, validation, and test data files based on the desired ratio
It takes a csv file with one rating per row as an input
It also constructs and saves quivalent copies of these split data in a format suitable for mymedialite as well as in the format suitable for omnidirectional learning 
(one user per row with columns representing items and entries representing ratings)
It is also configured to output data with timestamp information for use in omnidirectional learning.
"""

from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import json


#Parameters
trainvalidtest_split = [.8, .1, .1]
full_data_filepath = "/data1/movielens/ml-1m/ratings.csv" #'/data1/amazon/productGraph/categoryFiles/ratings_Video_Games.csv' #"/data1/movielens/ml-20m/ratings.csv" "/data1/beer/beeradvocate-crawler/ba_ratings.csv" "/data1/yelp/yelp_ratings.csv" "/data1/netflix/netflix_ratings.csv"
output_filepath = "data/ml1m/" #"data/amazon_videoGames/" #"data/movielens/"
schema_type = "movielens" #"movielens", "amazon", "beeradvocate", "yelp"
build_data_for_omni = True
include_timestamps = True
save_users_and_items = False
reverse_user_item_data = False

if reverse_user_item_data:
	print("Generating reverse user-item data")
	output_filepath = output_filepath+"reverse_item-user/"

def split_data(save_users_and_items=False):
	#Load data file
	print("Loading CSV from ", full_data_filepath)
	ratings = pd.read_csv(full_data_filepath)
	num_ratings = len(ratings)
	

	#Noramlize schema
	if schema_type == "movielens":
		#ratings.rename(index=str, columns={"movieId": "itemId"})
		if reverse_user_item_data:
			ratings.columns=["itemId", "userId", "rating", "timestamp"]
		else:
			ratings.columns=["userId", "itemId", "rating", "timestamp"]
		cast_user_to_int = True
	elif schema_type == "amazon":
		if reverse_user_item_data:
			ratings.columns=["itemId", "userId", "rating", "timestamp"]
		else:
			ratings.columns=["userId", "itemId", "rating", "timestamp"]
		cast_user_to_int = False
	elif schema_type == "beeradvocate":
		if reverse_user_item_data:
			ratings.columns=["itemId", "userId", "rating", "timestamp"]
		else:
			ratings.columns=["userId", "itemId", "rating", "timestamp"]
		cast_user_to_int = False
	elif schema_type == "yelp":
		if reverse_user_item_data:
			ratings.columns=["itemId", "userId", "rating", "timestamp"]
		else:
			ratings.columns=["userId", "itemId", "rating", "timestamp"]
		cast_user_to_int = False
	elif schema_type == "netflix":
		if reverse_user_item_data:
			ratings.columns=["itemId", "userId", "rating"]
		else:
			ratings.columns=["userId", "itemId", "rating"]
		cast_user_to_int = False


	#Split it
	print("Splitting data")
	random_rating_order = np.random.permutation(num_ratings)

	train_set_size = int(num_ratings*trainvalidtest_split[0])
	val_set_size = int(num_ratings*trainvalidtest_split[1])
	test_set_size = int(num_ratings*trainvalidtest_split[2])

	train_set_ratings_list = random_rating_order[0:train_set_size]
	val_set_ratings_list = random_rating_order[train_set_size : train_set_size+val_set_size]
	test_set_ratings_list = random_rating_order[train_set_size+val_set_size : ]
	test_set_inputs_list = random_rating_order[0 : train_set_size+val_set_size]

	train_set_ratings = ratings.iloc[train_set_ratings_list]
	val_set_ratings = ratings.iloc[val_set_ratings_list]
	test_set_ratings = ratings.iloc[test_set_ratings_list]
	test_set_inputs_ratings = ratings.iloc[test_set_inputs_list]

	print("Saving splits")
	if include_timestamps:
		convert_and_save_mml(test_set_inputs_ratings, output_filepath+"train_data_mml_withtimestamps.csv") #This is both the train set and the valid set
		convert_and_save_mml(test_set_ratings, output_filepath+"test_data_mml_withtimestamps.csv")
	else:
		convert_and_save_mml(test_set_inputs_ratings, output_filepath+"train_data_mml.csv") #This is both the train set and the valid set
		convert_and_save_mml(test_set_ratings, output_filepath+"test_data_mml.csv")

	if build_data_for_omni:
		#Construct user-item matrix and save
		train_dict = build_and_save(train_set_ratings, "train")
		valid_dict = build_and_save(val_set_ratings, "valid", input_set_dicts = train_dict)
		#build_and_save(test_set_ratings, "test", input_set_dict = merge_data_sets(train_dict, valid_dict))
		build_and_save(test_set_ratings, "test", input_set_dicts = build_user_item_dict(test_set_inputs_ratings))

	if save_users_and_items:
		unique_items = list(ratings["itemId"].unique())

		unique_users_orig = list(ratings["userId"].unique())
		if cast_user_to_int:
			unique_users_str = [str(int(x)) for x in unique_users_orig]
		else:
			unique_users_str = [str(x) for x in unique_users_orig]

		with open(output_filepath + "unique_items_list" + ".json" , "w") as f:
			json.dump( unique_items, f)

		with open(output_filepath + "unique_users_list" + ".json" , "w") as f:
			json.dump( unique_users_str, f)


def build_user_item_dict(ratings):
	user_dict_ratings = {}
	user_dict_timestamps = {}
	for i in range(ratings.shape[0]):
		row = ratings.iloc[i]
		if schema_type == "movielens":
			user = str(int(row["userId"]))
		elif schema_type == "amazon":
			user = str(row["userId"])
		elif schema_type == "beeradvocate":
			user = str(row["userId"])
		elif schema_type == "yelp":
			user = str(row["userId"])
		elif schema_type == "netflix":
			user = str(row["userId"])

		item = row["itemId"]
		rating = row["rating"]
		if include_timestamps: #Get the real timestamp
			timestamp = row["timestamp"]
		else: #Get a dummy timestamp to make the rest of the code simpler (It will not be used) This is for datasets without timestamps...
			timestamp = None
		if user in user_dict_ratings:
			user_dict_ratings[user].append((item, rating))
			user_dict_timestamps[user].append((item, timestamp))
		else:
			user_dict_ratings[user] = [(item, rating)]
			user_dict_timestamps[user] = [(item, timestamp)]
	return (user_dict_ratings, user_dict_timestamps)

def save_files(user_dicts, trainvalidtest):
	if include_timestamps:
		with open(output_filepath + "ratingsByUser_dicts_withtimestamps_" + trainvalidtest + ".json" , "w") as f:
			json.dump( user_dicts, f)
	else:
		with open(output_filepath + "ratingsByUser_dicts_" + trainvalidtest + ".json" , "w") as f:
			json.dump( user_dicts, f)

def build_and_save(ratings, trainvalidtest, input_set_dicts = None):
	print("Building " + trainvalidtest + " set")
	output_dict_ratings, output_dict_timestamps = build_user_item_dict(ratings)
	
	if trainvalidtest=="train":
		if include_timestamps:
			user_dicts = (output_dict_ratings, output_dict_timestamps)
		else:
			user_dicts = output_dict_ratings
	elif trainvalidtest=="valid" or trainvalidtest=="test":
		input_set_dict_ratings = input_set_dicts[0]
		input_set_dict_timestamps = input_set_dicts[1]
		input_dict_ratings = map_inputs_to_targets(input_set_dict_ratings, output_dict_ratings)
		if include_timestamps:
			#input_dict_timestamps = map_inputs_to_targets(input_set_dict_timestamps, output_dict_timestamps)
			user_dicts = ((input_dict_ratings, output_dict_ratings), merge_timestamps(input_set_dict_timestamps, output_dict_timestamps))
		else:
			user_dicts = (input_dict_ratings, output_dict_ratings)

	print("Saving " + trainvalidtest + " set")
	save_files(user_dicts, trainvalidtest)

	return (output_dict_ratings, output_dict_timestamps)

def map_inputs_to_targets(input_set_dict, target_dict): #This goes in the TrainValidSplit file
	#Find the inputs that correspond to the targets and construct a paired dataset
	input_dict = {}
	for user in target_dict:
		#Try finding the corresponding row/user in the input mat
		if user in input_set_dict:
			#If it is present, add it to the corresponding new_input_mat row
			input_dict[user] = input_set_dict[user]
		else:
			#If it is not present, make the corresponding row into a zero vector
			input_dict[user] = None #This signals the creation of a zero vector later

	return input_dict

def merge_timestamps(input_dict_timestamps, output_dict_timestamps):
	input_keys = input_dict_timestamps.keys()
	output_keys = output_dict_timestamps.keys()
	timestamps = {}

	for user in input_keys:
		timestamps[user] = input_dict_timestamps[user]

	for user in output_keys:
		if user in timestamps:
			timestamps[user].extend(output_dict_timestamps[user])
		else:
			timestamps[user] = output_dict_timestamps[user]

	return timestamps

def convert_and_save_mml(ratings, filename):
	#Convert the rating splits to mymedialite format and save them.
	# userId::itemId::rating (or can also use a standard csv)
	if include_timestamps:
		ratings.to_csv(filename, columns = ["userId", "itemId", "rating", "timestamp"], header=False, index= False)
	else:
		ratings.to_csv(filename, columns = ["userId", "itemId", "rating"], header=False, index= False)


split_data(save_users_and_items)
