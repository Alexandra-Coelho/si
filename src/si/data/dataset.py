import numpy as np
import pandas as pd

class Dataset:
    """
    Creates a tabular dataset for machine learning
    """

    def __init__(self, X: np.ndarray, y: np.ndarray=None, features: list=None, label: str=None):
        """
        Initializes the dataset
        
        Paramaters
        ----------
        X: np.ndarray
            A matrix with the dataset features
        y: np.ndarray
            Label vector
        features: list of strings
            Names of the features
        label: str
            Name of the label
        """

        if X is None:
            raise Exception("Trying to instanciate a Dataset without any data")
        if features is None:
            features = [str(i+1) for i in range(X.shape[1])] 
        self.X = X  
        self.y = y  
        self.features = features
        self.label = label  

    def shape(self) -> tuple:
        """
        Returns a tuple with the dataset dimensions
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Checks if the dataset contains a label
        """
        if self.y is not None:  #se o y existe entao é supervisionado
            return True
        return False

    def get_classes(self) -> np.ndarray:
        """
        Returns a np.ndarray with the dataset's unique classes 
        """
        if self.y is None: 
            raise ValueError("Not possible to retrieve the unique classes due to an unsupervised dataset (y is None)")
        return np.unique(self.y) 

    def get_mean(self) -> np.ndarray:
        """
        Returns a np.ndarray with the mean for each feature of the dataset
        """
        return np.mean(self.X, axis = 0) 

    def get_variance(self) -> np.ndarray:
        """
        Returns a np.ndarray with the variance for each feature of the dataset
        """
        return np.var(self.X, axis = 0)

    def get_median(self) -> np.ndarray:
        """
        Returns a np.ndarray with the median for each feature of the dataset
        """
        return np.median(self.X, axis = 0)

    def get_min(self) -> np.ndarray:
        """
        Returns a np.ndarray with the minimum value for each feature of the dataset
        """
        return np.min(self.X, axis = 0)

    def get_max(self) -> np.ndarray:
        """
        Returns a np.ndarray with the maximum value for each feature of the dataset
        """
        return np.max(self.X, axis = 0)

    def summary(self) -> pd.DataFrame:
        """
		Returns a pd.DataFrame containing descriptive metrics of each feature
		"""
        return pd.DataFrame(
            {'mean': self.get_mean(), 'median': self.get_median(),'variance': self.get_variance(),'min': self.get_min(),'max': self.get_max()})

    def dropna(self):
        """
        Remove all samples that contain at least one null value (NaN)
        """
        if self.shape()[0] != len(self.y):
            raise ValueError("Number of examples must be equal to the length of y")

        if self.has_label(): # se tiver y
            self.y = self.y[~np.isnan(self.X).any(axis=1)]

        self.X = self.X[~np.isnan(self.X).any(axis=1)]  # ~ faz o oposto, retorna todas as linhas que nao têm NaN

    def fillna(self, value: int):
        """
        Replaces all null values ​​for a given value

        Paramaters
        ----------
        value: int
            Given value to replace the NaN values with
        """
        self.X[np.isnan(self.X)] = value  # no indice onde é True (é NaN) substitui isso por um valor
        # ou np.nan_to_num(self.X, nan = value)


if __name__ == '__main__':
    print("--------Example 1--------")
    x = np.array([[1,2,3], [1,2,3], [1,2,6]])  # matriz
    y = np.array([1,2,2])  # vetor
    features = ['A', 'B', 'C']
    label = 'y'  # nome do vetor
    dataset = Dataset(X=x, y=y, features=features, label=label)
    print(dataset.has_label())   # verifica se o dataset é supervisionado ou nao
    print(dataset.get_classes())
    print(dataset.summary())

    print("--------Example 2--------")
    x1 = np.array([[1,2,3], [1,np.nan,3], [1,2,np.nan],[8,6,9]]) 
    y1 = np.array([1,2,2]) 
    dataset2 = Dataset(X=x1, y=y1)
    print(dataset2.shape())  # before
    dataset2.dropna()
    print(dataset2.shape())  # after

    print("--------Example 3--------")
    x1 = np.array([[1,2,3], [1,np.nan,3], [1,2,np.nan]]) 
    y1 = np.array([1,2,2]) 
    dataset3 = Dataset(X=x1, y=y1)
    print(dataset3.shape())  # before
    dataset3.dropna()
    print(dataset3.shape())  # after

    print("--------Example 4--------")
    x1 = np.array([[1,2,3], [1,np.nan,3], [1,2,np.nan],[8,6,9]]) 
    y1 = np.array([1,2,2]) 
    dataset4 = Dataset(X=x1, y=y1)
    print(dataset4.shape())  # before
    dataset4.fillna(4)
    print(dataset4.shape())  # after