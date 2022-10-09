
import numpy as np
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset


class VarianceThreshold:
    """
    Variance Threshold feature selection.
    Features with a training-set variance lower than this threshold will be removed from the dataset.
    """

    def __init__(self, threshold: float = 0.0):
        """
        Variance Threshold feature selection.
        Features with a training-set variance lower than this threshold will be removed from the dataset.

        Parameters
        ----------
        threshold: float
            The threshold value to use for feature selection. Features with a
            training-set variance lower than this threshold will be removed.

        Attributes
        ----------
        variance: array-like, shape (n_features,)
            The variance of each feature.
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        self.threshold = threshold 
        self.variance = None  

    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Fit the VarianceThreshold model according to the given training data. Returns self.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit.
        """
        self.variance = np.var(dataset.X, axis=0)  # self.variance = Dataset.get_var()
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It removes all features whose variance does not meet the threshold. Returns a new Dataset object.
        Parameters
        ----------
        dataset: Dataset 
        """
        X = dataset.X
        features_mask = self.variance > self.threshold 
        X = X[:, features_mask]  #seleciona colunas que têm var superior ao threshold
        features = np.array(dataset.features)[features_mask]
        return Dataset(X=X, y=dataset.y, features=list(features), label=dataset.label) # novo dataset nao tem colunas com pouca var

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit to data, then transform it. Returns a new Dataset object.
        Parameters
        ----------
        dataset: Dataset 
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == "__main__":
    dataset = Dataset(X=np.array([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]), y=np.array([0, 1, 0]), features=["f1", "f2", "f3", "f4"], label="y")
    selector = VarianceThreshold()
    #dataset = Dataset.from_random(100,10,2)
    #selector = VarianceThreshold(threshold=0.1)
    selector = selector.fit(dataset) #aprende a var
    dataset = selector.transform(dataset) 
    print(dataset.features)
