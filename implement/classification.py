"""
Classification implementation
Based on:
- https://towardsdatascience.com/softmax-regression-in-python-multi-class-classification-3cb560d90cb2
"""

import numpy as np
from typing import Any
from nptyping import NDArray
from sklearn.metrics import accuracy_score


class LogisticRegression:

    def __init__(self):
        self._norm = None
        self._weights = None

    def fit(
            self,
            data: NDArray[(Any, Any), float],
            response: NDArray[Any, int],
            iterations: int = 100,
            learning_rate: float = 0.05
    ) -> None:
        self._norm = np.linalg.norm(data)
        rows = data.shape[1]
        categories = max(response)+1
        scaled_data = data / self._norm
        self._weights = np.random.rand(rows, categories).reshape(rows, categories)
        for i in range(iterations):
            loss, grad = self._loss(scaled_data, response)
            if not i % 1000:
                print(f"iteration {i+1} loss: {loss}")
            self._weights -= learning_rate * grad

    def predict(self, data: NDArray[(Any, Any), float]) -> NDArray[int]:
        scaled_data = data / self._norm
        probabilities = self._softmax(np.dot(scaled_data, self._weights))
        return np.argmax(probabilities, axis=1).squeeze()

    def _loss(
            self,
            data: NDArray[(Any, Any), float],
            response: NDArray[int],
    ):
        response_matrix = self._ohe(response)
        probabilities = self._softmax(np.dot(data, self._weights))
        loss = self._cross_entropy(response, probabilities)
        grad = self._grad(data, response_matrix, probabilities)
        return loss, grad

    @staticmethod
    def _softmax(scores):
        exp_scores = np.exp(scores)
        return exp_scores/np.sum(exp_scores, axis=1).reshape(-1, 1)

    @staticmethod
    def _cross_entropy(response: NDArray[int], probabilities: NDArray[(Any, Any), float]) -> NDArray[float]:
        return -np.mean(np.log(probabilities[range(len(probabilities)), response]))

    @staticmethod
    def _grad(
            data,
            response_matrix: NDArray[(Any, Any), float],
            probabilities: NDArray[(Any, Any), float]
    ) -> NDArray[(Any, Any), float]:
        # y_hat - y = softmax derivative
        # dot product of data and gradient calculates partial derivative for each column (Jacobian)
        # the denominator scales the results giving us an average value
        return (1/probabilities.shape[0]) * np.dot(data.T, probabilities - response_matrix)

    @staticmethod
    def _ohe(vector: NDArray[int]) -> NDArray[(Any, Any), int]:
        return np.eye(max(vector) + 1)[vector]


if __name__ == "__main__":
    iris_data = np.genfromtxt('../data/iris.csv', delimiter=',', skip_header=1, usecols=(0, 1, 2, 3))
    iris_labels = np.genfromtxt('../data/iris.csv', delimiter=',', dtype="|U5", skip_header=1, usecols=(4,))
    n_rows, n_cols = iris_data.shape
    _, label_encoding = np.unique(iris_labels, return_inverse=True)
    le = LogisticRegression()
    le.fit(iris_data.reshape(n_rows, n_cols), label_encoding, iterations=100000, learning_rate=0.3)
    predictions = le.predict(iris_data.reshape(n_rows, n_cols))
    accuracy = accuracy_score(label_encoding, predictions)
    print(f"accuracy: {accuracy}")  # ~90% accuracy
