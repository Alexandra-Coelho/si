# -*- coding: utf-8 -*-

# modules
import numpy as np
import sys
sys.path.insert(0, 'src/si')

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the accuracy of the model on the given dataset. The accuracy algorithm: (TN+TP) / (TN+TP+FP+FN)

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    return np.sum(y_true == y_pred) / len(y_true)