"""
Basic K-means implementation
"""

import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


class KMeans:
    """
    K-means algorithm
    First we randomly assign k clusters, then assign points
    to those clusters based on proximity and then we shift
    cluster centroids based on the means of the assigned
    points. We do this for N iterations or until the
    assignments stop changing
    # Complexity: O(iterations * rows * columns * k)
    """

    def __init__(self, k: int, iterations: int = 100, tolerance: float = 0.001) -> None:
        self.k = k
        self.iterations = iterations
        self.tolerance = tolerance
        self._norm = None
        self._centroids = None

    def fit(self, data) -> None:
        rows, _ = data.shape
        if rows < self.k:
            raise ValueError("the number of rows must be equal to or greater than k")
        self._norm = np.linalg.norm(data)
        scaled_data = data / self._norm
        random_index = np.random.randint(low=0, high=rows-1, size=self.k)
        self._centroids = scaled_data[random_index]
        for i in range(self.iterations):
            closest_clusters = self._find_nearest_centroid(scaled_data)
            new_centroids = self._update_centroids(scaled_data, closest_clusters)
            if not self._centroids_have_changed(self._centroids, new_centroids):
                break
            self._centroids = new_centroids

    def predict(self, data: np.array) -> np.array:
        return self._find_nearest_centroid(data / self._norm)

    def fit_predict(self, data: np.array) -> np.array:
        self.fit(data)
        return self.predict(data)

    def _find_nearest_centroid(self, data: np.array) -> np.array:
        return np.array([
            closest for p in data if(
                ss := np.sum((p-self._centroids)**2, axis=1),
                closest := np.argmin(ss)
            )
        ])

    def _update_centroids(self, data: np.array, cluster_assignments: np.array) -> np.array:
        return np.array([
            new_centroid
            for k in range(self.k) if (
                cluster_points := np.where(cluster_assignments == k),
                current_centroid := self._centroids[k],
                points_allocated := len(cluster_points[0]) > 0,
                new_centroid := np.mean(data[cluster_points], axis=0) if points_allocated else current_centroid
            )
        ])

    def _centroids_have_changed(self, old_centroids: np.array, new_centroids: np.array) -> bool:
        sq_diffs = np.sqrt((old_centroids - new_centroids)**2)
        has_changed = np.mean(sq_diffs, axis=1) > self.tolerance
        return any(has_changed)


if __name__ == '__main__':
    iris_data = np.genfromtxt(
        '../data/iris.csv',
        delimiter=',',
        skip_header=1,
        usecols=(0, 1, 2, 3)
    )
    cluster = KMeans(k=5)
    assignments = cluster.fit_predict(iris_data)
    print(f"assignments: {assignments}")
    pca = PCA(n_components=2, svd_solver='full')
    components = pca.fit_transform(iris_data / np.linalg.norm(iris_data))
    plt.scatter(x=components[:, 0], y=components[:, 1], c=assignments)
    plt.show()
