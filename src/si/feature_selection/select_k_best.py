# modules 
from typing import Callable
import numpy as np
import warnings
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset
from statistic.f_classification import f_classification


class SelectKBest:
    """
    Select features according to the k highest scores.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
    """

    def __init__(self, k: int, score_func: Callable = f_classification):
        """
        Select features according to the k highest scores.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        k: int, default=10
            Number of top features to select.

        Attributes
        ----------
        F: array, shape (n_features,)
            F scores of features.
        p: array, shape (n_features,)
            p-values of F-scores.
        """
        if k > 0:
            self.k = k
        else:
            raise  ValueError("Invalid feature number, k must be greater than 0")
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        It fits SelectKBest to compute the F scores and p-values. Returns self.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        self.F, self.p = self.score_func(dataset) #calcula o F e p_value consoante a funçao
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the k highest scoring features. Return a new labeled Dataset with the k highest scoring features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        if any(np.isnan(elem) for elem in self.F):
            warnings.warn('Fit the selected k best before the transform method')
        idxs = np.argsort(self.F)[-self.k:]  #indexes do F por ordem crescente, depois vou buscar as k melhores features (as ultimas k)
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectKBest and transforms the dataset by selecting the k highest scoring features.
        Returns a new Dataset object with the k highest scoring features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == "__main__":
    print("--------Example 1--------")
    selector = SelectKBest(k=2, score_func = f_classification)
    dataset = Dataset(X=np.array([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]), y=np.array([1, 1, 0]), features=["f1", "f2", "f3", "f4"], label="y")
    #print(f_classification(dataset))
    new_dataset = selector.fit_transform(dataset)
    print(new_dataset.features)

    print("--------Example 2--------")
    selector = SelectKBest(k=2, score_func = f_classification)
    dataset = Dataset(X=np.array([[9, 2, 7, 5],[2, 8, 4, 3],[5, 3, 6, 9]]), y=np.array([1, 1, 0]), features=["f1", "f2", "f3", "f4"], label="y")
    new_dataset = selector.fit_transform(dataset)
    print(new_dataset.features)
    

