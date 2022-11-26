# -*- coding: utf-8 -*-

# modules
import numpy as np
import sys
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

    def forward(self, input_data: np.ndarray) -> np.ndarray: 
        """
        Calculates the output for the next layer, considering the weights stored in the neural network
        """
        return np.dot(input_data, self.weights) + self.bias  # input data must have number_examples = number_features


class Sigmoid_Activation: 
    """
    The Sigmoid Activation object implements an activation neural network based on the sigmoid function - sigmoid activation layer
    """

    def __init__(self): 
        pass
        
    def forward(self, input_data: np.ndarray) -> np.ndarray:  # static method
        """
        Calculates the output for the next layer, applying the sigmoid function to input data
        Sigmoid algorithm: 1 / (1 + e** -input_data)

        Parameters
        ----------
        input_data: np.ndarray
            The input values to be activated
        """
        # input data = calculated before in the dense layer
        return sigmoid_function(input_data)  


class SoftMaxActivation: 
    """
    The SoftMax Activation object implements an activation neural network based on the probability of occurrence of each class - softmax activation layer
    This layer is applied to multiclass problems.
    """

    def __init__(self): 
        pass
        
    def forward(self, input_data: np.ndarray) -> np.ndarray:  # static method
        """
        Calculates the probability of occurrence of each class. 
        Softmax algorithm: e**(input_data - max(input_data)) / np.sum(e**(input_data - max(input_data)))

        Parameters
        ----------
        input_data: np.ndarray
            The input values to be activated
        """
        exp = np.exp(input_data - np.max(input_data))
        return  exp / np.sum(exp, axis = 1, keepdims = True)


class ReLUActivation: 
    """
    The ReLUActivation object implements an activation neural network based on the rectified linear relationship - relu activation layer
    This layer considers only positive values.
    """

    def __init__(self):
        pass
        
    def forward(self, input_data: np.ndarray) -> np.ndarray: # static method
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
        return  np.maximum(0, input_data)   # returns input value if its positive, else return 0 if its 0 or negative


class LinearActivation: 
    """
    The LinearActivation object implements an activation neural network based on linear relationship - linear activation layer
    """

    def __init__(self):
        pass
        
    def forward(self, input_data: np.ndarray) -> np.ndarray: # static method
        """
        Calculates the linear activation algorithm, that its output will not be confined between any range.
        Linear activation algorithm: f(input_data) = input_data

        Parameters
        ----------
        input_data: np.ndarray
            The input values to be activated
        """
        return input_data
    


