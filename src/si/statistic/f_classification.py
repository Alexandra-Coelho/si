# modules
from typing import Tuple, Union
import numpy as np
from scipy import stats
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset


def f_classification(dataset: Dataset) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
    """
    Scoring function for classification problems. It computes one-way ANOVA F-value for the
    provided dataset. The F-value scores allows analyzing if the mean between two or more groups (factors)
    are significantly different. Samples are grouped by the labels of the dataset.
    Returns a tuple with np.arrays (F, p), which contains the F scores and the corresponding p-values of the features.

    Parameters
    ----------
    dataset: Dataset
        A labeled dataset

    Returns
    -------
    F: np.array, shape (n_features,)
        F scores
    p: np.array, shape (n_features,)
        p-values
    """
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]  # agrupa samples por classes
    F, p = stats.f_oneway(*groups) # * se tenho numa lista, vai extrair 
    return F, p