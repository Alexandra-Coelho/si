# -*- coding: utf-8 -*-

# modules
import numpy as np
import sys

sys.path.insert(0, 'src/si')
from data.dataset import Dataset
from metrics.accuracy import accuracy
from statistic.sigmoid_function import sigmoid_function


class LogisticRegression:
    """
    The LogisticRegression is a logistic model using the L2 regularization.
    This model solves the logistic regression problem using an adapted Gradient Descent technique
    """

    def __init__(self, use_adaptive_alpha: bool, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000):
        """
        Initializes the LogisticRegression object.

        Parameters
        ----------
        use_adaptive_alpha: bool
            If the adaptive alpha is used or not in the gradient descent, which implies the use of different fit methods
        l2_penalty: float
            The L2 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations

        Attributes
        ----------
        theta: np.array
            The model parameters, namely the coefficients of the linear model.
            For example, x0 * theta[0] + x1 * theta[1] + ...
        theta_zero: float
            The model parameter, namely the intercept of the linear model.
            For example, theta_zero * 1
        cost_history: dict
            The key is the descent gradient iteration number and the value is the cost in that iteration (using the cost function).
        """
        # parameters
        self.use_adaptive_alpha = use_adaptive_alpha
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # attributes
        self.theta = None
        self.theta_zero = None
        self.cost_history = {}

    def gradient_descent(self, dataset: Dataset) -> None:  
        """
        Implements the gradient descent algorithm 

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """
        m = dataset.shape()[0] 
        y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)  # Estimates the y values using the sigmoid function (y_predicted)

        # computing and updating the gradient with the learning rate
        gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X) 

        # computing the penalty
        penalization_term = self.alpha * (self.l2_penalty / m) * self.theta  

        # updating the model parameters
        self.theta = self.theta - gradient - penalization_term  
        self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)  

    def _regular_fit(self, dataset: Dataset) -> "LogisticRegression":
        """
        Executes the gradient descent algorithm, that must stop when the value of the cost function doesn't change.
        When the difference between the cost of the previous and the current iteration is less than 0.0001, the Gradient Descent must stop.
        Returns self.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)  # tamanho =  nº de features
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter):
            self.gradient_descent(dataset)
            self.cost_history[i] = self.cost(dataset) # iteration number: iteration cost

            if i !=0 and self.cost_history[i-1] - self.cost_history[i] < 0.0001: # check the difference between the cost of the previous and the current iteration
                break
        return self

    def _adaptive_fit(self, dataset: Dataset) -> "LogisticRegression":
        """
        Executes the gradient descent algorithm, that must decrease the alpha value when the value of the cost function doesn't change.
        Whenever the difference between the cost of the previous and the current iteration is less than 0.0001, the alpha value is decreased by half.
        Returns self.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n) 
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter):
            self.gradient_descent(dataset)
            self.cost_history[i] = self.cost(dataset) 
            
            if i !=0 and self.cost_history[i-1] - self.cost_history[i] < 0.0001: 
                self.alpha = self.alpha / 2  # updating the learning rate
        return self

    def fit(self, dataset: Dataset) -> "LogisticRegression":
        """
        Fits the model to the dataset depending on the boolean value of the use_adaptative_alpha. 
        If this attribute is True, performs the fit method by updating the learning rate (using the _adaptive_fit). 
        If it's False, performs the _regular_fit method by breaking the gradient descent algorithm when the difference between the cost of the previous and the current iteration < 0.0001.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """
        if self.use_adaptive_alpha:
            self._adaptive_fit(dataset)
        else:
            self._regular_fit(dataset)
    
    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset using the sigmoid function

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of
        """ 
        y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero) 
        binarizar = [1 if elem >= 0.5 else 0 for elem in y_pred]  # se y_pred > 0.5 retorna 1, caso contrário retorna 0
        return binarizar  # classificaçao

    def score(self, dataset: Dataset) -> float:
        """
        Computes and returns the accuracy of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
        """
        y_pred = self.predict(dataset) 
        return round(accuracy(dataset.y, y_pred), 4)

    def cost(self, dataset: Dataset) -> float:
        """
        Computes and returns the cost function (J function) of the model on the dataset using L2 regularization

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on
        """
        # nao podemos usar o predict (dá o binário), queremos os valores estimados na regressao
        m, n = dataset.shape()
        y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        first = (-1 / m) * np.sum(dataset.y * np.log(y_pred) + (1 - dataset.y) * np.log(1 - y_pred))
        score = first + (self.l2_penalty / (2 * m)) * np.sum(self.theta **2)
        return round(score, 4)


if __name__ == "__main__":
    print("--------Example 1--------")
    sys.path.insert(0, 'src')
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split
    path = 'C:/Users/ASUS/Desktop/Bioinfo/2ano/Sistemas Inteligentes/si/datasets/breast-bin.csv'
    breast = read_csv(path, sep = ",", features = True, label = True)
    
    from sklearn.preprocessing import StandardScaler
    breast.X = StandardScaler().fit_transform(breast.X) # para normalizar dados
    train, test = train_test_split(breast, test_size=0.3, random_state=2)

    breast_log = LogisticRegression(use_adaptive_alpha=True, max_iter=2000)
    breast_log.fit(train) 
    #print(breast_logistic.cost_history)
    #print(len(breast_logistic.cost_history))
    print(f"Predictions: {breast_log.predict(test)}")
    print(f"Score: {breast_log.score(test)}")
    print(f"Cost: {breast_log.cost(test)}")
    