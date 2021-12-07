"""
A basic regression model
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


NUMERIC_COLS = [
    "sepal.length",
    "sepal.width",
    "petal.length",
    "petal.width"
]

CATEGORICAL_COLS = [
    "variety"
]


def main(target: str = "sepal.length") -> None:

    numeric_without_label = [c for c in NUMERIC_COLS if c != target]
    iris = pd.read_csv("../data/iris.csv")
    iris_label = iris[target]
    iris_predictors = iris[numeric_without_label + CATEGORICAL_COLS]
    train_predictors, test_predictors, train_label, test_label = train_test_split(
        iris_predictors, iris_label, test_size=0.3
    )
    column_transformer = make_column_transformer(
        (StandardScaler(), numeric_without_label),
        (OneHotEncoder(), CATEGORICAL_COLS),
    )
    pipe = Pipeline([
        ("transform", column_transformer),
        ("predict", LinearRegression())
    ])
    pipe.fit(train_predictors, train_label)
    scores = pipe.predict(test_predictors)
    mse = mean_squared_error(test_label, scores)
    print(f"The mse is {mse}")


if __name__ == '__main__':
    main()
