# -*- coding: utf-8 -*-

# modules
import numpy as np
import sys
sys.path.insert(0, 'src/si')

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the RMSE of the model on the given dataset. The RMSE algorithm: sqrt(sum( (y_true_i - y_pred_i)**2) ) / N)

    Parameters
    ----------
    y_true: np.ndarray
        The true values
    y_pred: np.ndarray
        The predicted values
    """
    return np.sqrt(np.sum((y_true - y_pred)**2) / len(y_true))