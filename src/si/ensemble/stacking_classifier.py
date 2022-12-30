# -*- coding: utf-8 -*-

# modules
import sys
from typing import *

import numpy as np

sys.path.insert(0, 'src/si')
from data.dataset import Dataset
from metrics.accuracy import accuracy


class StackingClassifier:
    """
    The StackingClassifier is an ensemble model that uses a set of models to generate predictions.
    These predictions are then used to train the final model. The final model can then be used to predict the output variable (Y).
    """

    def __init__(self, models: list, final_model: Callable):
        """
        Initializes the StackingClassifier object.

        Parameters
        ----------
        models: list
            List of initialized models of classifiers
        final_model: Callable
            Final model classifier
        """
        self.models = models 
        self.final_model = final_model

    def fit(self, dataset: Dataset) -> "StackingClassifier":
        """
        It fits the ensemble models.
        Before fitting the final model, it trains intermediate models and use their predictions as additional features to enrich the X and test datasets.
        These new features which were created for the X dataset can then be used for training the final_model
        
        Parameters
        ----------
        dataset : Dataset
            A Dataset object 
        """
        for model in self.models:  # fit the ensemble models 
            model.fit(dataset)

        predictions = np.array([model.predict(dataset) for model in self.models])  # creates a list with the predictions of each model using the training dataset
        self.final_model.fit(Dataset(np.transpose(predictions), dataset.y))  # the predictions from the previous ensemble models will be use as new_features to fit the final_model
        return self
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the output variable (class labels for samples). 
        The final predictions are obtained by using the final model and the predictions of the ensemble models.

        Parameters
        ----------
        dataset : Dataset
            A Dataset object 
        """
        predictions = np.array([model.predict(dataset) for model in self.models])  # predictions for each model based on the test dataset
        final_predict = self.final_model.predict(Dataset(np.transpose(predictions), dataset.y))
        return final_predict

    def score(self, dataset: Dataset) -> float: 
        """
        Returns the accuracy score of the final model.
        
        Parameters
        ----------
        dataset : Dataset
            A Dataset object
        """
        return round(accuracy(dataset.y, self.predict(dataset)), 4)


if __name__ == "__main__":
    print("--------Example 1--------")
    sys.path.insert(0, 'src')
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split
    path = 'C:/Users/ASUS/Desktop/Bioinfo/2ano/Sistemas Inteligentes/si/datasets/breast-bin.csv'
    breast = read_csv(path, sep = ",", features = False, label = True)
    
    from sklearn.preprocessing import StandardScaler
    breast.X = StandardScaler().fit_transform(breast.X) # para normalizar as features
    train, test = train_test_split(breast, test_size=0.3, random_state=2)
    
    from si.linear_model.logistic_regression import LogisticRegression
    from si.neighbors.knn_classifier import KNNClassifier
    knn = KNNClassifier(k=3)
    lg = LogisticRegression(use_adaptive_alpha=False)
    final_model = KNNClassifier(k=2)

    stacking = StackingClassifier([knn, lg], final_model)
    stacking.fit(train)
    print(f"Predictions: {stacking.predict(test)}") 
    print(f"Score: {stacking.score(test)}") 


