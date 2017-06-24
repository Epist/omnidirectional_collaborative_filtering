#Train-valid-test split
"""
This file splits a full data file randomly into training, validation, and test data files based on the desired ratio
"""

from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import json



trainvalidtest_split = [.8, .1, .1]
full_data_filepath = "/data1/movielens/ml-20m/ratings.csv"

output_filepath = "data/movielens/"

def split_data():
	#Load data file
	print("Loading CSV")
	ratings = pd.read_csv(full_data_filepath)
	num_ratings = len(ratings)

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
	build_and_save(train_set_ratings, "train")
	build_and_save(val_set_ratings, "valid")
	build_and_save(test_set_ratings, "test")



def build_user_item_matrix(ratings):
    user_dict = {}
    for i in range(ratings.shape[0]):
        row = ratings.iloc[i]
        user = str(row["userId"])
        movie = row["movieId"]
        rating = row["rating"]
        if user in user_dict:
            user_dict[user].append((movie, rating))
        else:
            user_dict[user] = [(movie, rating)]
    return user_dict

def save_files(user_dict, unique_items, unique_users, trainvalidtest):
    with open(output_filepath + "ratingsByUser_dict_" + trainvalidtest + ".json" , "w") as f:
        json.dump( user_dict, f)
        
    with open(output_filepath + "unique_items_list_" + trainvalidtest + ".json" , "w") as f:
        json.dump( unique_items, f)
        
    with open(output_filepath + "unique_users_list_" + trainvalidtest + ".json" , "w") as f:
        json.dump( unique_users, f)
        
        
def build_and_save(ratings, trainvalidtest):
	print("Building " + trainvalidtest + " set")
	user_dict = build_user_item_matrix(ratings)
	unique_items = list(ratings["movieId"].unique())
    
	unique_users = list(ratings["userId"].unique())
	unique_users_str = [str(int(x)) for x in unique_users]
    
	print("Saving " + trainvalidtest + " set")
	save_files(user_dict, unique_items, unique_users_str, trainvalidtest)


split_data()