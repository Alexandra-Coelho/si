# modules
from typing import Callable
import numpy as np
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset
from statistic.f_classification import f_classification


class SelectPercentile:
    """
    Select features with the highest scores according to the given percentile.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.
    """

    def __init__(self, percentile: float, score_func: Callable = f_classification):
        """
        Select the best features according to the given percentile.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        percentile: float
            The percentile of features to select.

        Attributes
        ----------
        F: array, shape (n_features,)
            F scores of features.
        p: array, shape (n_features,)
            p-values of F-scores.
        """
        if not 0 < percentile < 1:
            raise  ValueError ("The percentile must be a value between 0 and 1.")
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        It fits SelectPercentile to compute the F scores and p-values for each feature of the dataset. Returns self.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        self.F, self.p = self.score_func(dataset) #calcula o F e p_value consoante a funçao
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the features with the highest F value up to the given percentile. Returns a new dataset object with the highest score selected features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        number_features = int(len(dataset.features) * self.percentile)
        idxs = np.argsort(self.F)[-number_features:]  
        features = np.array(dataset.features)[idxs]
        return Dataset(dataset.X[:, idxs], dataset.y, list(features), dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectPercentile and transforms the dataset by selecting the percentile highest scoring features.
        Returns a new dataset object with the selected features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == "__main__":
    perc = SelectPercentile(0.25, score_func = f_classification)
    dataset = Dataset(X=np.array([[9, 2, 7, 5],[2, 8, 4, 3],[5, 3, 6, 9]]), y=np.array([1, 1, 0]), features=["f1", "f2", "f3", "f4"], label="y")
    #print(f_classification(dataset))
    new_dataset = perc.fit_transform(dataset)
    print(new_dataset.features)