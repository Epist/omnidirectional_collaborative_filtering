# Omnidirectional-collaborative-filtering
Omnidirectional neural networks for collaborative filtering recommendation

This code can be used to implement a wide variety of autoencoder-based recommender systems including versions of AutoRec, omnidirecitonal collaborative filtering, and Nested Denosiing Autoencoders for collaborative filtering

Raw data is expected in a cannonical row-wise csv format with each row containing:
userID, itemId, rating, timestamp

Use TrainValidTestSplit.py to preprocess the data
If you want to split the data by rating, also use TrainValidTestSplit.py to create a split between test sets
If you want to split the data by user, you can use eval_mode='ablation'

train.py contains all of the parameters for running the model. 
model.py contains the model definitions.
data_reader.py contains code for reading data and generating batches to feed to the model.

Requires:
- Tensorflow 1.3.0
- Keras 2.0.4
- Pandas 0.20.1
- h5py
- Python 2.7
- Numpy 1.13.3
- Scipy 0.19.0

Other versions may work fine, but are untested

