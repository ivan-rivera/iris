"""
A basic classification model
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


NUMERIC_COLS = [
    "sepal.length",
    "sepal.width",
    "petal.length",
    "petal.width"
]


def main(target: str = "variety") -> None:

    iris = pd.read_csv("../data/iris.csv")
    iris_label = iris[target]
    iris_predictors = iris[NUMERIC_COLS]
    train_predictors, test_predictors, train_label, test_label = train_test_split(
        iris_predictors, iris_label, test_size=0.3
    )
    column_transformer = make_column_transformer(
        (StandardScaler(), NUMERIC_COLS),
    )
    pipe = Pipeline([
        ("transform", column_transformer),
        ("predict", GaussianNB())
    ])
    pipe.fit(train_predictors, train_label)
    scores = pipe.predict(test_predictors)
    accuracy = accuracy_score(test_label, scores)
    print(f"The accuracy is {accuracy:0.3f}")


if __name__ == '__main__':
    main()
