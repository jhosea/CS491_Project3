import numpy as np
import sklearn
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

    # Number of samples
    N = np.size(X, axis=0)

    # Number of classes (features)
    C = np.size(X, axis=1)

    sum = 0

    a = np.matmul(x, model["W1"]) + model["b1"]
    h = np.tanh(a)
    z = np.matmul(h, (model["W2"])) + model["b2"]
    probability = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    corect_logprobs = -np.log(probability[range(N), y])
    data_loss = np.sum(corect_logprobs)

    return 1/N * data_loss

# for i in range(N):
#     for j in range(C):
#         prediction = predict(model, X[i])
#         sum += prediction * np.log(y[i])

#     corect_logprobs = -np.log(probs[range(num_examples), y])
# data_loss = np.sum(corect_logprobs)

# return (-1/N) * sum


'''
Helper function to predict an ouput (0 or 1)
Model is the current version of the model {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2} (dictionary)
Parameters:
	x -> np.array is one sample without a label
'''


def predict(model, x):

    a = np.matmul(x, model["W1"]) + model["b1"]
    h = np.tanh(a)
    z = np.matmul(h, model["W2"]) + model["b2"]
    probability = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    # Either 0 or 1 class. We return the one with the largest probability
    return np.argmax(probability, axis=1)


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

    sampleSize = np.size(X, axis=0)
    learningRate = 0.01

    # Model is a dictionary
    model = {}

    # Initialize with random weights and bias can be zero
    np.random.seed(0)
    model['W1'] = np.random.randn(2, nn_hdim) / np.sqrt(2)
    model['W2'] = np.random.randn(nn_hdim, 2) / np.sqrt(nn_hdim)
    model['b1'] = np.zeros((1, nn_hdim))
    model['b2'] = np.zeros((1, 2))

    for i in range(num_passes):

        # Go through the current model
        a = np.matmul(X, model["W1"]) + model["b1"]
        h = np.tanh(a)
        z = np.matmul(h, model["W2"]) + model["b2"]
        probability = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

        # Perform Backpropagation
        delta = probability
        delta[range(sampleSize), y] -= 1

        # dyhat = probability
        dyhat = delta
        da = (1 - h**2) * np.matmul(dyhat, np.transpose(model['W2']))

        dw2 = np.matmul(np.transpose(h), dyhat)
        db2 = dyhat

        dw1 = np.matmul(np.transpose(X), (da))
        db1 = da

        # Update Model
        model['W1'] -= learningRate * dw1
        model['W2'] -= learningRate * dw2
        model['b1'] -= learningRate * np.sum(db1, axis=0)
        model['b2'] -= learningRate * np.sum(db2, axis=0)

        # Optional Print and only print about 20 times
        if print_loss and i % 1000 == 0:
            print("Loss after iteration {}: {}".format(
                i, calculate_loss(model, X, y)))

    return model
