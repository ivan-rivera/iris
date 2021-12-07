"""
A basic regression model
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC_COLS = [
    "sepal.length",
    "sepal.width",
    "petal.length",
    "petal.width"
]


def main(max_clusters: int = 20) -> None:
    iris = pd.read_csv("../data/iris.csv")
    iris_predictors = iris[NUMERIC_COLS]
    train_predictors, test_predictors = train_test_split(iris_predictors, test_size=0.3)
    column_transformer = make_column_transformer(
        (StandardScaler(), NUMERIC_COLS),
    )
    base_pipe = Pipeline([
        ("transform", column_transformer),
    ])
    base_pipe.fit(train_predictors)
    transformed_data = base_pipe.transform(train_predictors)
    transformed_test_data = base_pipe.transform(test_predictors)
    scores = {}
    for k in range(2, max_clusters):
        clusterer = KMeans(n_clusters=k)
        clusters = clusterer.fit_predict(transformed_data)
        scores[k] = silhouette_score(train_predictors, clusters)
        print(f"SS for clusters of size {k} is {scores[k]}")

    best_cluster_size = max(scores, key=lambda z: scores.get(z))
    clusterer = KMeans(n_clusters=best_cluster_size)
    clusterer.fit(transformed_data)
    predicted_clusters = clusterer.predict(transformed_test_data)
    print(f"the first predicted cluster is {predicted_clusters[0]}")


if __name__ == '__main__':
    main()
