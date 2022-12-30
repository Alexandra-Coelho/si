# -*- coding: utf-8 -*-

# modules
import numpy as np

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the cross-entropy loss function of the model on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    return - np.sum((y_true * np.log(y_pred))/ len(y_true))


def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the derivative of the cross-entropy loss function on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset
    """
    return ((-y_true / y_pred) + ((1 - y_true) / (1 - y_pred))) / len(y_true)