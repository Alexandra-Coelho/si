# -*- coding: utf-8 -*-

# modules
from typing import Callable
import numpy as np
import sys
sys.path.insert(0, 'src/si')
from statistic.euclidean_distance import euclidean_distance
from data.dataset import Dataset

class KMeans:
    """
    It performs k-means clustering on the dataset.
    It groups samples into k clusters by trying to minimize the distance between samples and their closest centroid.
    It returns the centroids and the indexes of the closest centroid for each point.
    """

    def __init__(self, k: int, max_iter: int, distance: Callable = euclidean_distance):
        """
        It initializes the K-means clustering algorithm.

        Parameters
        ----------
        k: int
            Number of clusters.
        max_iter: int
            Maximum number of iterations.
        distance: Callable
            Distance function.

        Attributes
        ----------
        centroids: np.array
            Centroids of the clusters.
        labels: np.array
            Labels of the clusters.
        """
        self.k = k
        self.max_iter = max_iter
        self.distance = distance
        self.centroids = None
        self.labels = None

    def _init_centroids(self, dataset: Dataset):
        """
        It generates initial k centroids 

        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        seeds = np.random.permutation(dataset.shape()[0])[:self.k]  # randomly selects the first k samples from the dataset
        self.centroids = dataset.X[seeds]  # initializes the centroids, and at the beginning each centroid has only one sample
    
    def get_closest_centroid(self, sample: np.ndarray) -> np.ndarray:
        """
        Returns the index of the closest centroid to each data point sample

        Parameters
        ----------
        sample : np.ndarray
            A sample.
        """
        centroids_distance = self.distance(sample, self.centroids)  # calculates the distance between the sample and each centroid (returns an array of distances)
        closest_centroid_index = np.argmin(centroids_distance, axis=0)  # returns the index of the centroid with the smallest distance, that is, the closest to the sample
        return closest_centroid_index

    def fit(self, dataset: Dataset) -> 'KMeans':
        """
        It fits k-means clustering on the dataset.
        The k-means algorithm initializes the centroids and then iteratively updates them until convergence or max_iter.
        Convergence is reached when the centroids do not change anymore.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        self._init_centroids(dataset)  # generate initial centroids
        convergence = False  
        i = 0  # check if it reached the max_iter 
        labels = np.zeros(dataset.shape()[0])   

        #while loop calculate distances, infer the closest centroids for each sample, check if labels have changed
        while not convergence and i < self.max_iter:  
            # get closest centroid
            new_labels = np.apply_along_axis(self.get_closest_centroid, axis=1, arr=dataset.X)  # apply function along each sample (axis=1)

            # compute the new centroids
            centroids = [] 
            for j in range(self.k): 
                centroid = np.mean(dataset.X[new_labels == j], axis=0) # group the samples according to the centroid to which they belong and calculates the mean over all samples that are in a centroid
                centroids.append(centroid)

            self.centroids = np.array(centroids)  
            # check if converged (convergence=True when there is no difference)
            convergence = np.any(new_labels != labels)  # check if the centroids have changed
            labels = new_labels  # replace labels
            i += 1 

        self.labels = labels
        return self

    def get_distance (self, sample: np.ndarray) -> np.ndarray:
        """
        Returns the distances between each sample and the centroids.
        
        Parameters
        ----------
        sample : np.ndarray, shape=(n_features,)
            A sample.
        """
        return self.distance(sample, self.centroids)
    
    def transform (self, dataset: Dataset) -> np.ndarray:
        """
        It transforms the dataset and computes the distances between each sample and all centroids.
        
        Parameters
        ----------
        dataset: Dataset
            Dataset object.
            """
        centroids_distance = np.apply_along_axis(self.get_distance, axis=1, arr=dataset.X)
        return centroids_distance

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        It fits and transforms the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        self.fit(dataset)
        return self.transform(dataset)
    
    def predict(self, dataset: Dataset) -> np.ndarray: 
        """
        It predicts the labels to all samples of the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        return np.apply_along_axis(self.get_closest_centroid, axis=1, arr=dataset.X)

    def fit_predict(self, dataset: Dataset) -> np.ndarray:
        """
        It fits and predicts the labels to all samples of the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        self.fit(dataset)
        return self.predict(dataset)


if __name__ == "__main__":
    print("--------Example 1--------")
    sys.path.insert(0, 'src')
    from si.io.csv_file import read_csv
    path = 'C:/Users/ASUS/Desktop/Bioinfo/2ano/Sistemas Inteligentes/si/datasets/iris/iris.csv'
    dataset = read_csv(path, sep = ",", features = True, label = True)
    kmeans = KMeans(k = 3, max_iter = 8)
    print(kmeans.fit_transform(dataset))
    print(kmeans.fit_predict(dataset))
