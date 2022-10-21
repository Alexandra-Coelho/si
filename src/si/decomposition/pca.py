# -*- coding: utf-8 -*-

# modules
import numpy as np
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset

class PCA:
    """"
    Linear algebra technique to reduce the dimensions of the dataset. 
    The PCA to be implemented uses the linear algebra technique SVD (Singular Value Decomposition)
    """

    def __init__(self, n_components: int):
        """
        It initializes the PCA algorithm.

        Parameters
        ----------
        n_components: int
            Number of components.

        Attributes
        ----------
        mean: np.array
            Mean of each sample of the dataset.
        components: np.array
            Principal components matches the first n_components of Vt.
        explained_variance: np.array
            Explained_variance matches first n_components of EV.
        """
        if n_components < 1:
            raise ValueError("n_components must be greater than 0")

        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset: Dataset):
        """
        It estimates the mean, the principal components and the explained variance. Returns self.
        
        Parameters
		----------
		dataset: Dataset
			Dataset object
        """
        # calculates the mean of each sample and centers data 
        self.mean = dataset.get_mean()   # ou np.mean(dataset.X, axis = 0) 
        self.center_data = dataset.X - self.mean  # X - mean

        # calculate SVD
        U, S, Vt = np.linalg.svd(self.center_data, full_matrices = False)

        # get principal components that matches first n_components of Vt
        self.components = Vt[:self.n_components]

        # get explained_variance that matches first n_components of EV
        EV = S**2 / (len(dataset.X) - 1)
        self.explained_variance = EV [:self.n_components]

        return self

    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        Calculates the reduced dataset using the principal components.
        
        Parameters
		----------
		dataset: Dataset
			Dataset object
        """
        V = np.transpose(self.components) #transposta
        X_reduced = np.dot (self.center_data, V)
        return X_reduced

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


if __name__ == "__main__":
    print("--------Example 1--------")
    sys.path.insert(0, 'src')
    from si.io.csv_file import read_csv
    path = 'C:/Users/ASUS/Desktop/Bioinfo/2ano/Sistemas Inteligentes/si/datasets/iris/iris.csv'
    dataset = read_csv(path, sep = ",", features = True, label = True)
    pca = PCA(10)
    print(pca.fit_transform(dataset))
    