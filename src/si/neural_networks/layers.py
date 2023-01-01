# -*- coding: utf-8 -*-

# modules
import sys

import numpy as np

sys.path.insert(0, 'src/si')
from statistic.sigmoid_function import sigmoid_function


class Dense:
    """
    The Dense object implements a fully connected neural network - dense layer.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the Dense object.

        Parameters
        ----------
        input_size: int
            The number of input nodes
        output_size: int
            The number of output nodes

        Attributes
        ----------
        weights: np.ndarray
            The weights matrix
        bias: np.ndarray
            A bias vector 
        """
        if input_size < 1:
            raise ValueError("input_size must be an integer greater than 1")

        if output_size < 1:
            raise ValueError("output_size must be an integer greater than 1")

        self.input_size = input_size 
        self.output_size = output_size 

        # weight matrix initialization 
        self.weights = np.random.randn(input_size, output_size) * 0.01   # multiplied by 0.01 -> avoid large values

        # bias initialization 
        self.bias = np.zeros((1,output_size)) # add a bias for each output
        self.X = None

    def forward(self, input_data: np.ndarray) -> np.ndarray: 
        """
        Calculates the output for the next layer, considering the weights stored in the neural network

        Parameters
        ----------
        input_data: np.ndarray
            The input data of the layer
        """
        self.X = input_data 
        return np.dot(input_data, self.weights) + self.bias  # input data must have number_examples = number_features

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Computes the backward pass of the layer. Returns the error of the previous layer.

        Parameters
        ----------
        error: np.ndarray
            The error propagated to the layer
        alpha: float
            The learning rate of the model
        """
        error_to_propagate = np.dot(error, self.weights.T)  

        # updates the weights and bias
        self.weights = self.weights - learning_rate * np.dot(self.X.T, error)  # updating the weights according to the error
        
        self.bias = self.bias - learning_rate * np.sum(error, axis=0)  # update bias with the gradient descent 
        #sum -> bias gives the nodes dimension

        return error_to_propagate


class Sigmoid_Activation: 
    """
    The Sigmoid Activation object implements an activation neural network based on the sigmoid function - sigmoid activation layer
    """

    def __init__(self): 
        self.X = None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray: 
        """
        Calculates the output for the next layer, applying the sigmoid function to input data
        Sigmoid algorithm: 1 / (1 + e** -input_data)

        Parameters
        ----------
        input_data: np.ndarray
            The input values to be activated
        """
        # input data = calculated before in the dense layer
        self.X = input_data   
        return sigmoid_function(input_data) 

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Computes the backward pass of the derivative of the sigmoid activation function. Returns the error of the previous layer.

        Parameters
        ----------
        error: np.ndarray
            The error propagated to the layer
        alpha: float
            The learning rate of the model
        """
        sigmoid_derivative = sigmoid_function(self.X) * (1 - sigmoid_function(self.X)) 
        error_to_propagate = error * sigmoid_derivative  # multiply the error returned in the forward to propagate the error

        return error_to_propagate 


class SoftMaxActivation: 
    """
    The SoftMax Activation object implements an activation neural network based on the probability of occurrence of each class - softmax activation layer
    This layer is applied to multiclass problems.
    """

    def __init__(self): 
        self.X = None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:  
        """
        Calculates the probability of occurrence of each class. 
        Softmax algorithm: e**(input_data - max(input_data)) / np.sum(e**(input_data - max(input_data)))

        Parameters
        ----------
        input_data: np.ndarray
            The input values to be activated
        """
        self.X = input_data
        exp = np.exp(input_data - np.max(input_data))
        return  exp / np.sum(exp, axis = 1, keepdims = True)

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Computes the backward pass of the derivative of the softmax activation function. Returns the error of the previous layer.
        The calculation of the softmax derivative involves the jacobian. 
        If i = j:
            y_pred(i) * (1 - y_pred(i))
        Else: 
            - y_pred(i) * y_pred(j)

        Parameters
        ----------
        error: np.ndarray
            The error propagated to the layer
        alpha: float
            The learning rate of the model
        """ 
        S = self.forward(self.X)  # softmax 
        
        # calculate the jacobian
        S_vector = S.reshape(S.shape[0], 1)  # first matrix by repeating S in rows 
        S_matrix = np.tile(S_vector,S.shape[0])  # second matrix by repeating S in columns (transposing the first matrix)

        # calculate the jacobian derivative
        softmax_derivative = np.diag(S) - (S_matrix * np.transpose(S_matrix))  # multiplying them together element-wise    
    
        error_to_propagate = error * softmax_derivative 

        return error_to_propagate 


class ReLUActivation: 
    """
    The ReLUActivation object implements an activation neural network based on the rectified linear relationship - relu activation layer
    This layer considers only positive values.
    """

    def __init__(self):
        self.X = None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray: 
        """
        Calculates the rectified linear unit, it must consider only the positive part of the input_data. 
        Its a linear function of input_data, but it zeroes out negative values
        ReLU algorithm: np.maximum(mínimo, input_data)

        Parameters
        ----------
        input_data: np.ndarray
            The input values to be activated
        """
        # compute the maximum of 0 and the input value
        self.X = input_data
        return  np.maximum(0, input_data)   # returns input value if its positive, else return 0 if its 0 or negative

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Computes the backwards pass of the rectified linear relationship. Returns the error of the previous layer.

        Parameters
        ----------
        error: np.ndarray
            The error propagated to the layer
        alpha: float
            The learning rate of the model
        """
        relu_derivative = np.where(self.X > 0, 1, 0)
        error_to_propagate = error * relu_derivative

        return error_to_propagate


class LinearActivation: 
    """
    The LinearActivation object implements an activation neural network based on linear relationship - linear activation layer
    """

    def __init__(self):
        self.X = None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray: 
        """
        Calculates the linear activation algorithm, that its output will not be confined between any range.
        Linear activation algorithm: f(input_data) = input_data

        Parameters
        ----------
        input_data: np.ndarray
            The input values to be activated
        """
        self.X = input_data
        return input_data   

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Computes the backwards pass of the identity relationship. Returns the error of the previous layer.

        Parameters
        ----------
        error: np.ndarray
            The error propagated to the layer
        alpha: float
            The learning rate of the model
        """
        identity_derivative = np.ones_like(self.X)
        error_to_propagate = error * identity_derivative

        return error_to_propagate
    

