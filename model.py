
"""An omnidirectional fully connected neural network for collaborative filterining recommendation

Features:
	Randomized reciprocal input-output dropout
	Configurable number of fully connected layers
	Support for various kinds of recommendation problems
	Data handling via Pandas
	Training via automatic differentiation in Keras/Tensorflow
	Conditioning on auxilliary information
	Auxilliary variable to tell the network whether or not the variable is present

"""
from __future__ import print_function
from keras.layers import Input, Dense, multiply, Lambda, concatenate, Dropout
from keras.models import Model
from keras.regularizers import l1, l2
import numpy.ma as ma
import tensorflow as tf
from keras.legacy import interfaces
from keras.engine import Layer
from keras.engine import InputSpec
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
import scipy
import numpy as np


#Model
class omni_model(object):
	def __init__(self, numlayers, num_hidden_units, input_shape, batch_size, dense_activation = 'tanh', use_causal_info=True, use_timestamps=False, use_both_masks=False, 
		l2_weight_regulatization=None, sparse_representation = False, dropout_probability = None, use_sparse_masking_layer=False):

		#The timestamps info should not be masked, becasue the timestamps for the targets are required...
		self.numlayers = numlayers
		self.num_hidden_units = num_hidden_units
		self.input_shape = input_shape
		self.batch_size = batch_size

		dataVars = Input(shape=(self.input_shape,), sparse=sparse_representation) #A Tensor containing the observed data variables
	
		output_mask = Input(shape=(self.input_shape,))

		if use_causal_info:
			#A vector representing which variables are actually present in the dataset
			observed_vars = Input(shape=(self.input_shape,), sparse=sparse_representation) #Contains ones and zeros
			x = concatenate([dataVars, observed_vars]) #Make use of the dummy variable to let the model know which variables were really observed
		else:
			x = dataVars

		if use_both_masks:
			second_mask = Input(shape=(self.input_shape,), sparse=sparse_representation)
			x = concatenate([x, second_mask])

		if use_timestamps:
			timestamps  = Input(shape=(self.input_shape,), sparse=sparse_representation)
			x = concatenate([x, timestamps])

		self.dense_layers = []

		for layer in range(self.numlayers):
			if l2_weight_regulatization is not None:
				dense_lay = Dense(self.num_hidden_units, activation=dense_activation, W_regularizer=l2(l2_weight_regulatization))
				self.dense_layers.append(dense_lay)
			else:
				dense_lay = Dense(self.num_hidden_units, activation=dense_activation)
				self.dense_layers.append(dense_lay)
			x = dense_lay(x)
			if dropout_probability is not None:
				x = Dropout(dropout_probability, noise_shape=[self.batch_size, self.num_hidden_units])(x)

		if use_sparse_masking_layer:
			if l2_weight_regulatization is not None:
				predictions = Dynamic_Masking_Layer(self.input_shape, activation='linear', W_regularizer=l2(l2_weight_regulatization))([x, output_mask])
			else:
				predictions = Dynamic_Masking_Layer(self.input_shape, activation='linear')([x, output_mask])
		else:
			if l2_weight_regulatization is not None:
				full_predictions = Dense(self.input_shape, activation='linear', W_regularizer=l2(l2_weight_regulatization))(x)
			else:
				full_predictions = Dense(self.input_shape, activation='linear')(x)

			predictions = multiply([output_mask, full_predictions]) #Multiply the output of the last layer of dense with the output mask

		#Also output the output mask for use in masking the targets.
		if use_causal_info:
			input_list = [dataVars, observed_vars, output_mask]
		else:
			input_list = [dataVars, output_mask]
		#Add optional additional input variables
		if use_timestamps:
			input_list.append(timestamps)
		if use_both_masks:
			input_list.append(second_mask)

		self.model = Model(inputs=input_list, outputs=[predictions])


	def save_weights(self, filename):
		#weights = self.model.get_weights()
		self.model.save_weights(filename)

	def load_weights(self, weights):
		self.model.set_weights(weights)

	def replace_dense_layer_weights(self, donor_model, layers_to_replace, make_layers_trainable = False):
		#Extracts the weights from the dense layers of the donor model and sets the dense weights of the recipient model (self)
		#Only works if the donor and recipient had the same number of dense layers with the same number of hidden units.
		
		donor_weights = []
		for layer in donor_model.layers:

			if (layer.input_shape[1] == self.num_hidden_units or layer.output_shape[1] == self.num_hidden_units) and 'dense' in layer.get_config()['name']:
				donor_weights.append(layer.get_weights())
		if layers_to_replace == "all":
			layers_to_replace = [True for x in range(len(donor_weights))]
		hidden_layer_number = 0
		for layer in self.model.layers:
			if (layer.input_shape[1] == self.num_hidden_units or layer.output_shape[1] == self.num_hidden_units) and 'dense' in layer.get_config()['name']:
				if layers_to_replace[hidden_layer_number]:
					layer.set_weights(donor_weights[hidden_layer_number])
					layer.trainable=make_layers_trainable
					print("Loaded weights for dense layer ", hidden_layer_number)
				hidden_layer_number+=1

	def manually_load_all_weights(self, donor_model):
		donor_weights = []
		#print("Donor layers")
		for i, old_layer in enumerate(donor_model.layers):
			new_model_layer = self.model.layers[i]
			new_model_layer.set_weights(old_layer.get_weights())

	def make_trainable(self):
		#Need to recompile the model after doing this...
		for layer in self.model.layers:
			if layer.output_shape[1] == self.num_hidden_units and 'dense' in layer.get_config()['name']:
				layer.trainable=True

	def load_and_fix_for_denoising_autoencoders(self, donor_model):
		#Extracts the weights from the dense layers of the donor model and sets the dense weights of the recipient model (self)
		#Only works if the donor and recipient had the same number of dense layers with the same number of hidden units.
		
		donor_weights = []
		for layer in donor_model.layers:
			if (layer.input_shape[1] == self.num_hidden_units or layer.output_shape[1] == self.num_hidden_units) and 'dense' in layer.get_config()['name']:
				donor_weights.append(layer.get_weights())

		print("Number of weight layers to donate", len(donor_weights))
		num_layers_to_change_on_each_side = int(len(donor_weights)/2)
		num_new_hidden_layers = 0
		for layer in self.model.layers:
			if (layer.input_shape[1] == self.num_hidden_units or layer.output_shape[1] == self.num_hidden_units) and 'dense' in layer.get_config()['name']:
				num_new_hidden_layers += 1
		hidden_layer_number = 0
		for layer in self.model.layers:
			if (layer.input_shape[1] == self.num_hidden_units or layer.output_shape[1] == self.num_hidden_units) and 'dense' in layer.get_config()['name']:
				if hidden_layer_number < num_layers_to_change_on_each_side:
					layer.set_weights(donor_weights[hidden_layer_number])
					layer.trainable=False
					print("Loaded and fixed weights for dense layer ", hidden_layer_number, " from donor dense layer ", hidden_layer_number)
				elif hidden_layer_number >= num_new_hidden_layers-num_layers_to_change_on_each_side:
					layer_number_to_grab_from = len(donor_weights)-(num_new_hidden_layers-hidden_layer_number)
					layer.set_weights(donor_weights[layer_number_to_grab_from])
					layer.trainable=False
					print("Loaded and fixed weights for dense layer ", hidden_layer_number, " from donor dense layer ", layer_number_to_grab_from)

				hidden_layer_number+=1


class Dynamic_Masking_Layer(Dense):
	@interfaces.legacy_dense_support
	def __init__(self, units,
				 activation=None,
				 use_bias=True,
				 kernel_initializer='glorot_uniform',
				 bias_initializer='zeros',
				 kernel_regularizer=None,
				 bias_regularizer=None,
				 activity_regularizer=None,
				 kernel_constraint=None,
				 bias_constraint=None,
				 **kwargs):
		super(Dense, self).__init____init__(self, units,
				 activation=None,
				 use_bias=True,
				 kernel_initializer='glorot_uniform',
				 bias_initializer='zeros',
				 kernel_regularizer=None,
				 bias_regularizer=None,
				 activity_regularizer=None,
				 kernel_constraint=None,
				 bias_constraint=None,
				 **kwargs)
		super(Dynamic_Masking_Layer, self).__init__(**kwargs)
		self.input_spec = [InputSpec(min_ndim=2), InputSpec(min_ndim=2)]

	def build(self, input_shape):
		input_shape_mask = input_shape[1]
		input_shape = input_shape[0]

		assert len(input_shape) >= 2
		print(input_shape)
		input_dim = input_shape[-1]

		self.kernel = self.add_weight(shape=(input_dim, self.units),
									  initializer=self.kernel_initializer,
									  name='kernel',
									  regularizer=self.kernel_regularizer,
									  constraint=self.kernel_constraint)
		if self.use_bias:
			self.bias = self.add_weight(shape=(self.units,),
										initializer=self.bias_initializer,
										name='bias',
										regularizer=self.bias_regularizer,
										constraint=self.bias_constraint)
		else:
			self.bias = None
		self.input_spec = [InputSpec(min_ndim=2, axes={-1: input_dim}), InputSpec(min_ndim=2, axes={-1: input_shape_mask[0]})]
		self.built = True


	def call(self, inputs):
		ip = inputs[0]
		mask = inputs[1]
		sparse_weights = self.get_sparse_weights(mask)
		output = self.multiply_dense_inputs_with_sparse_weights(ip, sparse_weights)
		if self.use_bias:
			output = K.bias_add(output, self.bias)
		if self.activation is not None:
			output = self.activation(output)
		return output

	def get_sparse_weights(self, mask):
		zero = tf.constant(0.0, dtype=tf.float32)
		locs = tf.not_equal(mask, zero)
		indices = tf.where(locs)
		sparse_weights_data = tf.gather_nd(self.kernel, indices)
		return tf.SparseTensor(indices=indices, values=sparse_weights_data, dense_shape=self.kernel.shape)

	def multiply_dense_inputs_with_sparse_weights(self, inputs, weights):
		arg1 = tf.sparse_transpose(weights)
		arg2 = tf.transpose(inputs)
		result = tf.sparse_tensor_dense_matmul(arg1, arg2)
		out = tf.transpose(result)
		return out
