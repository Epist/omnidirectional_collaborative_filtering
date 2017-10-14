
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
from keras.layers import Input, Dense, multiply, Lambda, concatenate, Dropout
from keras.models import Model
from keras.regularizers import l1, l2


#Model
class omni_model(object):
	def __init__(self, numlayers, num_hidden_units, input_shape, batch_size, dense_activation = 'tanh', use_causal_info=True, use_timestamps=False, use_both_masks=False, l2_weight_regulatization=None, sparse_representation = False, dropout_probability = None):
		#The timestamps info should not be masked, becasue the timestamps for the targets are required...
		self.numlayers = numlayers
		self.num_hidden_units = num_hidden_units
		self.input_shape = input_shape
		self.batch_size = batch_size
		#self.aux_var_value = aux_var_value

		dataVars = Input(shape=(self.input_shape,), sparse=sparse_representation) #A Tensor containing the observed data variables

		#A vector representing which variables are actually present in the dataset
		observed_vars = Input(shape=(self.input_shape,), sparse=sparse_representation) #Contains ones and zeros


		#observed_vars = Lambda(lambda x: x*aux_var_value)(observed_vars) #Implements the aux_var_value

		output_mask = Input(shape=(self.input_shape,))

		#input_mask = Input_Omnidrop(observed_vars)
		#masked_inputs = multiply([input_mask, dataVars]) #Mask the inputs
		if use_causal_info:
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
				x = dense_lay(x)
			else:
				dense_lay = Dense(self.num_hidden_units, activation=dense_activation)
				self.dense_layers.append(dense_lay)
				x = dense_lay(x)
			if dropout_probability is not None:
				x = Dropout(dropout_probability, noise_shape=[self.batch_size, self.num_hidden_units])(x)

		#output_mask = Lambda(lambda x: (x-1)*-1)(input_mask) #Invert the input mask
		full_predictions = Dense(self.input_shape, activation='linear')(x) #Input shape is the same as the output shape

		masked_outputs = multiply([output_mask, full_predictions]) #Multiply the output of the last layer of dense with the output mask

		predictions = masked_outputs

		#Also output the output mask for use in masking the targets.
		input_list = [dataVars, observed_vars, output_mask]
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

	def replace_dense_layer_weights(self, donor_model):
		#Extracts the weights from the dense layers of the donor model and sets the dense weights of the recipient model (self)
		#Only works if the donor and recipient had the same number of dense layers with the same number of hidden units.
		
		donor_weights = []

		for layer in donor_model.layers:
			if layer.input_shape[1] == self.num_hidden_units and layer.output_shape[1] == self.num_hidden_units:
				donor_weights.append(layer.get_weights())

		#for i, layer in enumerate(self.dense_layers):
		#	layer.set_weights(donor_weights[i])
		for i, layer in enumerate(self.model.layers):
			if layer.input_shape[1] == self.num_hidden_units and layer.output_shape[1] == self.num_hidden_units:
				layer.set_weights(donor_weights[i])

"""
Notes:
The input masking can be accomplished using the basic dropout function in Keras.

The output masking can be accomplished (at least for now) by setting both the dropped outputs and their respective targets to zero thus nullifying the gradient.
The vector of outputs to drop needs to be computed fro mthe input dropout vector, which I need to obtain somehow. This might mean that I cannot use the basic dropout function for input dropping.


Could add additional functionality allowing for some outputs not to be predicted even thoguh they are not observed.

User embeddings not yet implemented... This could be a problem, but only if the dataset is in the (user, movie, rating) form
not if it is in the (user, [ratings]) form
"""
