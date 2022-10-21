# -*- coding: utf-8 -*-

# modules
import numpy as np
import sys
sys.path.insert(0, 'src/si')

def euclidean_distance(x: np.ndarray, y:np.ndarray) -> np.ndarray:    # pex: x=[0,1,2,3,...]   y=[[1,2,3], [4,5,6]]
    """
    It computes the euclidean distance of a point (x) to a set of points y.
    Returns the euclidean distance for each point in y.

    Parameters
    ----------
    x: np.ndarray of 1 dimension
        Point.
    y: np.ndarray of 2 dimensions (contains several samples)
        Set of points.
    """
    return np.sqrt(((x - y) **2).sum(axis = 1)) 