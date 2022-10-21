# -*- coding: utf-8 -*-

# modules
import numpy as np
from typing import Callable, Union
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset
from metrics.rmse import rmse
from statistic.euclidean_distance import euclidean_distance


class KNNRegressor:
    """
    The k-Nearst Neighbors Regressor is a machine learning model that estimates the mean value of the k most similar examples.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initialize the KNN Regressor object.

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
        self.distance = distance 
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset and stores the training dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """
        self.dataset = dataset
        return self

    def get_mean_closest_labels(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the mean value of the closest labels to the given sample
        
        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of
        """
        # compute the distance between the sample of the test dataset and the training dataset
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k] 

        # get the labels of the k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors] 

        # calculates the mean of the labels of the k nearest neighbors
        return np.mean(k_nearest_neighbors_labels)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of
        """
        return np.apply_along_axis(self.get_mean_closest_labels, axis=1, arr=dataset.X) 

    def score(self, dataset: Dataset) -> float:
        """
        It calculates and returns the error, using RMSE algorithm, between the predicted and real/true classes

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on
        """
        predictions = self.predict(dataset) 
        return round(rmse(dataset.y, predictions), 2) 


if __name__ == "__main__":
    print("--------Example 1--------")
    sys.path.insert(0, 'src')
    from si.io.csv_file import read_csv
    path = 'C:/Users/ASUS/Desktop/Bioinfo/2ano/Sistemas Inteligentes/si/datasets/cpu.csv'
    dataset = read_csv(path, sep = ",", features = True, label = True)
    
    from model_selection.split import train_test_split
    train, test = train_test_split(dataset, test_size=0.4, random_state=24)
    
    knn = KNNRegressor(k=3)
    knn.fit(train)
    print(knn.predict(test))
    print(knn.score(test))