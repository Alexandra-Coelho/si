# -*- coding: utf-8 -*-

# modules
import numpy as np
import sys
sys.path.insert(0, 'src/si')
from typing import *
from data.dataset import Dataset
from model_selection.split import train_test_split


def cross_validate(model, dataset: Dataset, scoring: Callable = None, cv: int = 3, test_size: float = 0.2,) -> Dict[str, List[float]]:
    """
    It performs cross validation on the given model and dataset. It returns the scores of the model on the dataset.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    scoring: Callable
        The scoring function to use.
    cv: int
        The cross validation folds.
    test_size: float
        The test size.
    """
    scores = {'seeds': [], "train": [], "test": []}

    for i in range(cv):  # for each fold/cross_validation
        seed = np.random.randint(0, 1000) 
        scores["seeds"].append(seed)  # store seed
        
        train, test = train_test_split(dataset=dataset, test_size=test_size, random_state=seed)  # split the dataset
        model.fit(train)  # fit the model on the train set
       
        if scoring is None:  # uses the model's scoring function
            scores["train"].append(model.score(train))   
            scores["test"].append(model.score(test))   

        else:  # uses the scoring function provided by the user
            y_train = train.y
            y_test = test.y
   
            scores["train"].append(scoring(y_train, model.predict(train)))  # store the train score
            scores["test"].append(scoring(y_test, model.predict(test)))     # store the test score

    return scores  


if __name__ == '__main__':
    print("--------Example 1--------")
    from neighbors.knn_classifier import KNNClassifier

    # load the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    knn = KNNClassifier(k=3)  # initialize the KNN

    scores_ = cross_validate(knn, dataset_, cv=5)  # cross validate the model
    print(f"Scores: {scores_}\n")   # print the scores


    print("--------Example 2--------")
    sys.path.insert(0, 'src')
    from si.io.csv_file import read_csv
    path = 'C:/Users/ASUS/Desktop/Bioinfo/2ano/Sistemas Inteligentes/si/datasets/breast-bin.csv'
    breast = read_csv(path, sep = ",", features = False, label = True)
    
    from sklearn.preprocessing import StandardScaler
    breast.X = StandardScaler().fit_transform(breast.X) 
    
    from si.linear_model.logistic_regression import LogisticRegression
    log_reg = LogisticRegression(use_adaptive_alpha=False)
    scores = cross_validate(log_reg, breast, cv=5) 
    print(f"Scores: {scores}") 