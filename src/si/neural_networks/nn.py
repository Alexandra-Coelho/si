# -*- coding: utf-8 -*-

# modules
import sys
from typing import Callable

import numpy as np

sys.path.insert(0, 'src/si')
from data.dataset import Dataset
from metrics.accuracy import accuracy
from metrics.mse import mse, mse_derivate


class NN:
    """
    The NN object implements a generic multi layers neural network model that allows to define complex architectures (topologies).
    """

    def __init__(self, layers: list, loss_function: Callable = mse, epochs: int = 1000, learning_rate: float = 0.01, loss_derivation: Callable = mse_derivate, verbose: bool = False):
        """
        Initialize the NN object.

        Parameters
        ----------
        layers: list
            List of layers that are part of the neural network
        """
        self.layers = layers 
        self.epochs = epochs
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.loss_derivation = loss_derivation
        self.verbose = verbose  # when verbose=True it prints what is on training

        self.history = {}  # keeps the difference between y_pred and y_true -> erro 


    def fit(self, dataset: Dataset) -> 'NN':
        """
        Fits the model to the dataset using the forward propagation algorithm. It returns self.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """

        for epoch in range(1, self.epochs + 1): 
            y_pred = np.array(dataset.X)
            y_true = np.reshape(dataset.y, (-1,1)) # reshape -> column vector

            for i in self.layers: 
                y_pred = i.forward(y_pred)  # goal: calculate the new y_pred, pass the result to the next layer 

            # backward propagation

            # chain rule -> to calculate the error, update the weights and to do the backward propagation
            # loss derivation for each example returns the difference and pass it to the backward which update the weights according to that difference
            error = self.loss_derivation(y_true, y_pred)  # with loss function -> float
        
            # estimates the weights in order to have y_pred most similar to y_true (cost=0, find which y that minimize the cost by changing w values)
            for layer in self.layers[::-1]:  # starts with the last layer since we invert the list of layers to propagate the backward error
                error = layer.backward(error, self.learning_rate)  

            cost = self.loss_function(y_true, y_pred) # saves history cost to understand if i'm reaching the global minimum (real values)
            self.history[epoch] = cost 

            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} -- cost: {cost}")
        
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of
        """ 
        y_pred = dataset.X.copy()  
        for i in self.layers:    # forward propagation
            y_pred = i.forward(y_pred)   
        return y_pred

    def cost(self, dataset: Dataset) -> float:
        """
        Computes the cost of the model

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of
        """ 
        return self.loss_function(dataset.y, self.predict(dataset))  # mse 

    def score(self, dataset: Dataset, scoring_func: Callable = accuracy) -> float:
        """
        Computes the accuracy of the model

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of
        """
        y_pred = self.predict(dataset)
        return scoring_func(dataset.y, y_pred)


if __name__ == "__main__":
    print("--------Example 1--------")
    from neural_networks.layers import (Dense, Sigmoid_Activation)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, 0, 0, 1])
    dataset = Dataset(X, y, features=['x1', 'x2'], label='x1 XNOR x2')

    weight_matrix_1 = np.array([[20, -20], [20, -20]])   # rows = input nodes
    bias1 = np.array([[-30, 10]])

    l1 = Dense(input_size=2, output_size=2) 
    l1.weights = weight_matrix_1
    l1.bias = bias1

    weight_matrix_2 = np.array([[20], [20]]) 
    bias2 = np.array([[-10]])

    l2 = Dense(input_size=2, output_size=1)
    l2.weights = weight_matrix_2
    l2.bias = bias2


    l1_seq = Sigmoid_Activation() 
    l2_seq = Sigmoid_Activation() 

    layers = [l1, l1_seq, l2, l2_seq]
    nn_model = NN(layers) 

    nn_model.fit(dataset)
    print(nn_model.predict(dataset))  