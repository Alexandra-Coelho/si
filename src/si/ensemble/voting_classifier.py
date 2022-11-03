# -*- coding: utf-8 -*-

# modules
import numpy as np
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset
from metrics.accuracy import accuracy


class VotingClassifier:
    """
    The VotingClassifier is an ensemble model that uses voting as a combination function.
    The ensemble sets of models are based on the combination of predictions from several models.
    """

    def __init__(self, models: list):
        """
        Initializes the VotingClassifier object.

        Parameters
        ----------
        models: list
            List of initialized models of classifiers
        """
        self.models = models 

    def fit(self, dataset: Dataset) -> "VotingClassifier":
        """
        Fits the models using the training dataset. Returns self. 
        
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the models to
        """
        for model in self.models:
            model.fit(dataset)
        return self
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the output variable (class labels for samples) using the trained models and combines the predictions of each model using the voting technique. 
        
        Parameters
        ----------
        dataset : Dataset
            A Dataset object (test dataset)
        """
        predictions = np.array([model.predict(dataset) for model in self.models]) 
        
        def get_majority_vote(pred: np.ndarray) -> int: 
            """
            Helper function, which returns the majority vote of the given predictions (by choosing the most represented label)
            
            Parameters
            ----------
            pred: np.ndarray
                The predictions for a certain sample to get the majority vote of
            """
            labels, counts = np.unique(pred, return_counts=True)  # get the unique classes and its counts
            return labels[np.argmax(counts)]
        
        # ou posso fazer a transposta das predictions e colocar axis=1
        return np.apply_along_axis(get_majority_vote, axis=0, arr=predictions)  # apply a function that returns the most frequent label for each example

    def score(self, dataset: Dataset) -> float:
        """
        Returns the accuracy score on the given test data and labels.
        
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
    
    from si.neighbors.knn_classifier import KNNClassifier
    from si.linear_model.logistic_regression import LogisticRegression
    knn = KNNClassifier(k=3)
    lg = LogisticRegression(use_adaptive_alpha=False)
    
    voting = VotingClassifier([knn, lg])
    voting.fit(train)
    print(f"Predictions: {voting.predict(test)}") 
    print(f"Score: {voting.score(test)}") # VotingClassifier behaves very well in this dataset, as the score is close to 1.



