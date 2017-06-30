#Train-valid-test split
"""
This file splits a full data file randomly into training, validation, and test data files based on the desired ratio
"""

from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import json


#Parameters
trainvalidtest_split = [.8, .1, .1]
full_data_filepath = "/data1/amazon/productGraph/categoryFiles/ratings_Books.csv"#'/data1/amazon/productGraph/categoryFiles/ratings_Video_Games.csv' #"/data1/movielens/ml-20m/ratings.csv"
output_filepath = "data/amazon_books/" #"data/amazon_videoGames/" #"data/movielens/"
schema_type = "amazon" #"movielens", "amazon"


def split_data(save_users_and_items=False):
	#Load data file
	print("Loading CSV from ", full_data_filepath)
	ratings = pd.read_csv(full_data_filepath)
	num_ratings = len(ratings)

	#Noramlize schema
	if schema_type == "movielens":
		#ratings.rename(index=str, columns={"movieId": "itemId"})
		ratings.columns=["userId", "itemId", "rating", "timestamp"]
	elif schema_type == "amazon":
		ratings.columns=["userId", "itemId", "rating", "timestamp"]

	#Split it
	print("Splitting data")
	random_rating_order = np.random.permutation(num_ratings)

	train_set_size = int(num_ratings*trainvalidtest_split[0])
	val_set_size = int(num_ratings*trainvalidtest_split[1])
	test_set_size = int(num_ratings*trainvalidtest_split[2])

	train_set_ratings_list = random_rating_order[0:train_set_size]
	val_set_ratings_list = random_rating_order[train_set_size : train_set_size+val_set_size]
	test_set_ratings_list = random_rating_order[train_set_size+val_set_size : ]

	train_set_ratings = ratings.iloc[train_set_ratings_list]
	val_set_ratings = ratings.iloc[val_set_ratings_list]
	test_set_ratings = ratings.iloc[test_set_ratings_list]


	#Construct user-item matrix and save
	train_dict = build_and_save(train_set_ratings, "train")
	valid_dict = build_and_save(val_set_ratings, "valid", input_set_dict = train_dict)
	build_and_save(test_set_ratings, "test", input_set_dict = merge_data_sets(train_dict, valid_dict))

	if save_users_and_items:
		unique_items = list(ratings["itemId"].unique())
    
		unique_users_orig = list(ratings["userId"].unique())
		unique_users_str = [str(int(x)) for x in unique_users_orig]

		with open(output_filepath + "unique_items_list" + ".json" , "w") as f:
			json.dump( unique_items, f)
	        
		with open(output_filepath + "unique_users_list" + ".json" , "w") as f:
			json.dump( unique_users_str, f)


def build_user_item_dict(ratings):
    user_dict = {}
    for i in range(ratings.shape[0]):
        row = ratings.iloc[i]
        if schema_type == "movielens":
        	user = str(int(row["userId"]))
        elif schema_type == "amazon":
        	user = str(row["userId"])
        item = row["itemId"]
        rating = row["rating"]
        if user in user_dict:
            user_dict[user].append((item, rating))
        else:
            user_dict[user] = [(item, rating)]
    return user_dict

def save_files(user_dicts, trainvalidtest):
    with open(output_filepath + "ratingsByUser_dicts_" + trainvalidtest + ".json" , "w") as f:
        json.dump( user_dicts, f)
        
def build_and_save(ratings, trainvalidtest, input_set_dict = None):
	print("Building " + trainvalidtest + " set")
	output_dict = build_user_item_dict(ratings)
	
	if trainvalidtest=="train":
		user_dicts = output_dict
	elif trainvalidtest=="valid" or trainvalidtest=="test":
		input_dict = map_inputs_to_targets(input_set_dict, output_dict)
		user_dicts = (input_dict, output_dict)
    
	print("Saving " + trainvalidtest + " set")
	save_files(user_dicts, trainvalidtest)

	return output_dict

def map_inputs_to_targets(input_set_dict, target_dict): #This goes in the TrainValidSplit file
    #Find the inputs that correspond to the targets and construct a paired dataset
    input_dict = {}
    for key in target_dict:
        #Try finding the corresponding row/user in the input mat
        if key in input_set_dict:
            #If it is present, add it to the corresponding new_input_mat row
            input_dict[key] = input_set_dict[key]
        else:
            #If it is not present, make the corresponding row into a zero vector
            input_dict[key] = None #This signals the creation of a zero vector later
        
    return input_dict

def merge_data_sets(train, val):
    #Merge the dataset dictionaries together so that they can be used as input data for testing
    merged = train.copy()
    for key in val:
    	if key in merged:
    		merged[key] = merged[key].extend(val[key])
    	else:
    		merged[key] = val[key]

    #merged = train.copy()
    #merged.update(val)
    return merged

split_data()