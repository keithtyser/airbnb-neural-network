from typing import Tuple
import numpy as np


def linear_forward(A, W, b):

    Z = np.dot(W, A)+b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):

    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)

        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)

        A, activation_cache = relu(Z)
    elif activation == "linear":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = Z, Z

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(1, L):
        A_prev = A
        A, cache = A, cache = linear_activation_forward(
            A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation="relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A, parameters['W'+str(L)], parameters['b'+str(L)], activation="linear")
    caches.append(cache)
    assert(AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):

    m = Y.shape[1]

    # Compute loss from aL and y.
    err = (np.abs(AL - Y)).mean(axis=1)
    cost = np.squeeze(err)
    assert(cost.shape == ())

    return cost


def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m)*(np.dot(dZ, A_prev.T))
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def relu_backward(dA, cache):

    Z = cache
    # just converting dz to a correct object.
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_backward(dA, cache):

    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "linear":
        dZ = dA.copy()
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    # Initializing the backpropagation

    dAL = -2 * (Y-AL)

    current_cache = caches[L-1]
    dA, dW, db = linear_activation_backward(
        dAL, current_cache, activation='linear')
    grads["dA" + str(L)] = dA
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    for l in reversed(range(L-1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l+2)], current_cache, activation='relu')
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2  # number of layers in the neural network
    # Update rule for each parameter

    for l in range(L):

        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - \
            (learning_rate)*(grads["dW"+str(l+1)])
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - \
            (learning_rate)*(grads["db"+str(l+1)])

    return parameters


def initialize_parameters_deep(layer_dims):

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],
                                                   layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape ==
               (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


class Model:
    # Modify your model, default is a linear regression model with random weights
    ID_DICT = {"NAME": "Keith Tyser",
               "BU_ID": "U63550344", "BU_EMAIL": "ktyser@bu.edu"}

    def __init__(self, layers_dims=[764, 512, 128, 64, 1], learning_rate=0.0055, num_iterations=20000, batchSize=500, print_cost=True):

        self.layers_dims = layers_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.batchSize = batchSize
        self.parameters = initialize_parameters_deep(self.layers_dims)

    def preprocess(self, X: np.array, y: np.array) -> Tuple[np.array, np.array]:
        return X.T, y.T

    def train(self, X_train: np.array, y_train: np.array):

        costs = []
        # Loop (gradient descent)
        for i in range(0, self.num_iterations):

            indexes = np.random.choice(
                X_train.shape[-1], self.batchSize, replace=False)
            X_trainBatch = X_train[:, indexes]
            Y_trainBatch = y_train[:, indexes]

            AL, caches = L_model_forward(X_trainBatch, self.parameters)

            cost = compute_cost(AL, Y_trainBatch)

            grads = L_model_backward(AL, Y_trainBatch, caches)

            parameters = update_parameters(
                self.parameters, grads, self.learning_rate)
        ### END CODE HERE ###

        # Print the cost every 100 training example
            if self.print_cost and i % 100 == 0:

                print("Cost after iteration %i: MAE = %f" % (i, cost))
                if self.print_cost and i % 100 == 0:
                    costs.append(cost)

    def predict(self, X_val: np.array) -> np.array:
        """
        Predict with model and given feature
        """
        ###############################################
        ####      add model prediction code here   ####
        ###############################################
        AL, caches = L_model_forward(X_val, self.parameters)
        return AL
