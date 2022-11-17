# -*- coding: utf-8 -*-

# modules
import itertools
import sys

sys.path.insert(0, 'src/si')
from typing import *

from data.dataset import Dataset
from model_selection.cross_validate import cross_validate


def grid_search_cv(model, dataset: Dataset, parameter_grid: Dict[str, Tuple], scoring: Callable = None, cv: int = 3, test_size: float = 0.2) -> List[Dict[str, Any]]:
    """
    Performs a grid search cross validation on a model.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    parameter_grid: Dict[str, Tuple]
        The parameter grid to use.
    scoring: Callable
        The scoring function to use.
    cv: int
        The cross validation folds.
    test_size: float
        The test size.
    """
    # validate the parameter grid by checking if the given parameters in parameter_grid exists in the model
    for parameter in parameter_grid: 
        if not hasattr(model, parameter): 
            raise AttributeError(f"Model {model} does not haver parameter {parameter}")  # Alternative: Return None
            
    scores = []
    for combination in itertools.product(*parameter_grid.values()): 
        parameters = {} # parameter configuration

        for parameter, value in zip(parameter_grid.keys(), combination):  # set the parameter configuration
            setattr(model, parameter, value) 
            parameters[parameter] = value

        # cross validate the model
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size) 

        score['parameters'] = parameters  # add the parameter configuration
        scores.append(score)  # add the score

    return scores 


if __name__ == '__main__':
    print("--------Example 1--------")
    from linear_model.logistic_regression import LogisticRegression

    # load the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    knn = LogisticRegression(use_adaptive_alpha=False)  # initialize the Logistic Regression model

    # parameter grid
    parameter_grid_ = {'l2_penalty': (1, 10), 'alpha': (0.001, 0.0001), 'max_iter': (1000, 2000)}
    scores_ = grid_search_cv(knn, dataset_, parameter_grid=parameter_grid_, cv=3)  # cross validate the model
    print(f"Scores: {scores_}\n")  # print the scores


    print("--------Example 2--------")
    sys.path.insert(0, 'src')
    from si.io.csv_file import read_csv
    path = 'C:/Users/ASUS/Desktop/Bioinfo/2ano/Sistemas Inteligentes/si/datasets/breast-bin.csv'
    breast = read_csv(path, sep = ",", features = False, label = True)
    
    from sklearn.preprocessing import StandardScaler
    breast.X = StandardScaler().fit_transform(breast.X) 
    
    log_reg = LogisticRegression(use_adaptive_alpha=False)
    parameter_grid = {'l2_penalty': (1, 10), 'alpha': (0.001, 0.0001), 'max_iter': (1000, 2000)}
    scores = grid_search_cv(log_reg, breast, parameter_grid, cv=3)
    print(f"Scores: {scores}") 
