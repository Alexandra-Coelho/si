# -*- coding: utf-8 -*-

# modules
import numpy as np
from typing import Tuple
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset

def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets. Returns a tuple with the training and testing dataset.

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator
    """
    np.random.seed(random_state)   # set random state to always generates the same samples in the training and test dataset

    n_samples = dataset.shape()[0]       
    n_test = int(n_samples * test_size)  # get number of samples in the test dataset
    permutations = np.random.permutation(n_samples)  # get the dataset permutations
    
    test_idxs = permutations[:n_test]   # get samples in the test dataset
    train_idxs = permutations[n_test:]  # get samples in the training dataset

    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)  # features and label remain the same bc we only operate at the samples level
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test
