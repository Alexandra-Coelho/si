# -*- coding: utf-8 -*-

# modules
import numpy as np
from typing import Callable, Union
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset
from metrics.accuracy import accuracy
from statistic.euclidean_distance import euclidean_distance


class KNNClassifier:
    """
    The k-Nearst Neighbors classifier is a machine learning model that classifies new samples based on
    a similarity measure (e.g., distance functions). This algorithm predicts the classes of new samples by
    looking at the classes of the k-nearest samples in the training data.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initialize the KNN classifier object.

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use

        Attributes
        ----------
        dataset: np.ndarray
            The training data
        """
        self.k = k
        self.distance = distance  # calcula distancia de uma amostra a varias amostras do dataset train
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNClassifier':
        """
        It fits the model to the given dataset and stores the training dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """
        self.dataset = dataset
        return self

    def get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the closest label of the given sample

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of
        """
        # compute the distance between the sample of the test dataset and the training dataset
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]  # returns indexes of the k smallest distances

        # get the labels of the k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]  # labels associated with the k indexes with the smallest distance

        # get the most common label
        labels, counts = np.unique(k_nearest_neighbors_labels, return_counts=True) # get the unique classes and its counts
        return labels[np.argmax(counts)] 

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of
        """
        return np.apply_along_axis(self.get_closest_label, axis=1, arr=dataset.X)  # apply over all samples

    def score(self, dataset: Dataset) -> float:
        """
        It returns the accuracy of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on
        """
        predictions = self.predict(dataset)  # get as input the test dataset = y_true
        return round(accuracy(dataset.y, predictions), 3)


if __name__ == "__main__":
    print("--------Example 1--------")
    sys.path.insert(0, 'src')
    from si.io.csv_file import read_csv
    path = 'C:/Users/ASUS/Desktop/Bioinfo/2ano/Sistemas Inteligentes/si/datasets/iris/iris.csv'
    dataset = read_csv(path, sep = ",", features = True, label = True)
    
    from model_selection.split import train_test_split
    train, test = train_test_split(dataset, test_size=0.3, random_state=38)
    
    knn = KNNClassifier(k=3)
    knn.fit(train)
    print(knn.predict(test))
    print(knn.score(test))