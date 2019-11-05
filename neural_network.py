import numpy as np
import sklearn as skeet
import math


'''
Helper function to evaluate the total loss on the dataset
Model is the current version of the model {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2} (dictionary)
Parameters:
	X -> np.array of the training data
	y -> np.array of the training data labels
Returns:
	Integer loss
'''
def calculate_loss(model, X, y):



    return 


'''
Helper function to predict an ouput (0 or 1)
Model is the current version of the model {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2} (dictionary)
Parameters:
	x -> np.array sample without a label
'''
def predict(model, x):

	a = x * model["W1"] + model["b1"]
	h = np.tanh(a)
	z = h * model["W2"] + model["b2"]
	prediction = np.exp(z) / np.sum(np.exp(z))

	return sum(prediction)

'''
This function learns parameters for the neural netwrok and returns the model.
Parameters:
	X -> np.array of the training data
	y -> np.array of the training data labels
	nn_hdim -> integer number of nodes in the hidden layer
	num_passes -> integer number of passes through the training data for gradient deescent
	print_loss -> Boolean value, if true, prints the loss every 1000 iterations
'''
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):


