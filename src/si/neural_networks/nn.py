# -*- coding: utf-8 -*-

# modules
import numpy as np
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset

class NN:
    """
    The NN object implements a generic multi layers neural network model that allows to define complex architectures (topologies).
    """

    def __init__(self, layers: list):
        """
        Initialize the NN object.

        Parameters
        ----------
        layers: list
            List of layers that are part of the neural network
        """
        self.layers = layers 


    def fit(self, dataset: Dataset) -> 'NN':
        """
        Fits the model to the dataset using the forward propagation algorithm. It returns self.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """
        X = dataset.X.copy()  
        for i in self.layers: 
            X = i.forward(X)  # goal: calculate the new X, pass the result to the next layer 
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of
        """ 
        pass


if __name__ == "__main__":
    print("--------Example 1--------")
    from neural_networks.layers import Dense, Sigmoid_Activation

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, 0, 0, 1])
    dataset = Dataset(X, y, features=['x1', 'x2'], label=['AND','OR'])

    l1 = Dense(input_size=2, output_size=2) 
    l2 = Dense(input_size=2, output_size=1)

    l1_seq = Sigmoid_Activation()
    l2_seq = Sigmoid_Activation()

    layers = [l1, l1_seq, l2, l2_seq]
    nn_model = NN(layers) 

    nn_model.fit(dataset)