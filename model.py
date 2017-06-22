
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
from keras.layers import Input, Dense, multiply, Lambda, concatenate
from keras.models import Model


#Model
class omni_model(object):
	def __init__(self, numlayers, num_hidden_units, input_shape, use_causal_info=True):
		self.numlayers = numlayers
		self.num_hidden_units = num_hidden_units
		self.input_shape = input_shape
		#self.aux_var_value = aux_var_value

		dataVars = Input(shape=(self.input_shape,)) #A Tensor containing the observed data variables

		#A vector representing which variables are actually present in the dataset
		observed_vars = Input(shape=(self.input_shape,)) #Contains ones and zeros

		#observed_vars = Lambda(lambda x: x*aux_var_value)(observed_vars) #Implements the aux_var_value

		output_mask = Input(shape=(self.input_shape,))

		#input_mask = Input_Omnidrop(observed_vars)
		#masked_inputs = multiply([input_mask, dataVars]) #Mask the inputs
		if use_causal_info:
			x = concatenate([dataVars, observed_vars]) #Make use of the dummy variable to let the model know which variables were really observed
		else:
			x = dataVars

		for layer in range(self.numlayers):
			x = Dense(self.num_hidden_units, activation='tanh')(x)

		#output_mask = Lambda(lambda x: (x-1)*-1)(input_mask) #Invert the input mask
		full_predictions = Dense(self.input_shape, activation='linear')(x)

		masked_outputs = multiply([output_mask,full_predictions]) #Multiply the output of the last layer of dense with the output mask

		predictions = masked_outputs

		#Also output the output mask for use in masking the targets.
		self.model = Model(inputs=[dataVars, observed_vars, output_mask], 
			outputs=[predictions])

	def save_model():
		pass


"""
Notes:
The input masking can be accomplished using the basic dropout function in Keras.

The output masking can be accomplished (at least for now) by setting both the dropped outputs and their respective targets to zero thus nullifying the gradient.
The vector of outputs to drop needs to be computed fro mthe input dropout vector, which I need to obtain somehow. This might mean that I cannot use the basic dropout function for input dropping.


Could add additional functionality allowing for some outputs not to be predicted even thoguh they are not observed.

User embeddings not yet implemented... This could be a problem, but only if the dataset is in the (user, movie, rating) form
not if it is in the (user, [ratings]) form
"""