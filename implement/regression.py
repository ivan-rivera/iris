"""Basic regression implementation"""

import numpy as np


class Regression:
    """
    Given a matrix where column 0 is the response and columns 1...N are predictors
    fit a linear model that predicts our response using the predictors.
    Note that coefficients can be represented by: B = (X^tX)^-1X^ty (least squares)
    Another alternative is to calculate the coefficients using:
    for i in range(n):
        numer += (X[i] - mean_x) * (Y[i] - mean_y)
        denom += (X[i] - mean_x) ** 2
    m = numer / denom
    c = mean_y - (m * mean_x)
    more info here: https://www.edureka.co/blog/least-square-regression/
    """

    def __init__(self):
        self._norm = None
        self._coefficients = None

    def fit(self, data: np.array):
        rows, columns = data.shape
        if columns < 2:
            raise ValueError("data must contain at least 2 columns")
        response, predictors = data[:, 0], data[:, 1:]
        self._norm = np.linalg.norm(predictors)
        scaled_predictors = self._add_bias(predictors / self._norm)
        self._coefficients = np.linalg.lstsq(scaled_predictors, response, rcond=0)[0]

    def predict(self, data: np.array) -> np.array:
        if data.shape[1] != self._coefficients.shape[0]-1:
            raise ValueError("incompatible dimensions")
        scaled_predictors = self._add_bias(data / self._norm)
        return np.sum(scaled_predictors * self._coefficients, axis=1)

    @staticmethod
    def _add_bias(data):
        return np.concatenate([np.ones((data.shape[0], 1)), data], axis=1)


def score_by_coefficients(data: np.array, coefficients: np.array) -> np.array:
    """Given a set of coefficients and a data matrix, generate scores for each row"""
    rows, columns = data.shape
    if columns+1 != coefficients.shape[0]:
        raise ValueError("dimensions do not match")
    bias_multiplier = np.ones(rows).reshape(rows, 1)
    data_w_bias = np.concatenate((bias_multiplier, data), axis=1)
    coefficient_matrix = np.tile(coefficients, (rows, 1))
    return np.sum(data_w_bias * coefficient_matrix, axis=1)


if __name__ == "__main__":
    custom_coefficients = np.array([0.3, 1.2, 3.5, 2.1])
    raw_data = np.array([
        [2.2, 3.3, 4.4],
        [1.1, 2.9, 3.4],
        [0.5, 0.1, 0.9],
        [1.5, 2.3, 5.5]
    ])
    regression = Regression()
    regression.fit(raw_data)
    fitted_scores = regression.predict(raw_data[:, 1:])
    custom_coefficient_scores = score_by_coefficients(raw_data, custom_coefficients)
    print(f"{custom_coefficient_scores=}")
    print(f"{fitted_scores=}")
