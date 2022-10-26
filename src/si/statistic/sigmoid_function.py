# -*- coding: utf-8 -*-

# modules
import numpy as np
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset

def sigmoid_function(X: np.ndarray) -> np.ndarray:
    """
    Calculates the function sigmoid to X. It returns the probability of the values ​being equal to 1. Sigmoid algorithm: 1 / (1 + e**-X)

    Parameters
    ----------
    X: np.ndarray
        The input values
    """
    return 1 / (1 + np.exp(-X))