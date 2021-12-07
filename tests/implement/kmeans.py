"""K-means tests"""

import pytest
import numpy as np
from implement.kmeans import KMeans


@pytest.fixture(scope="module")
def data() -> np.array:
    return np.array([[1], [2], [3], [97], [98], [99]])


def test_fewer_rows_than_k(data: np.array) -> None:
    k = 10
    expected_exception = "the number of rows must be equal to or greater than k"
    cluster = KMeans(k=k)
    with pytest.raises(ValueError) as exec_info:
        cluster.fit(data)
    assert exec_info.value.args[0] == expected_exception, "row-k comparison: exception not caught"


def test_assignments(data: np.array) -> None:
    k = 2
    cluster = KMeans(k=k)
    assignments = cluster.fit_predict(data)
    new_point = np.array([[95]])
    new_assignment = cluster.predict(new_point)
    g1, g2 = set(assignments[0:3]), set(assignments[3:])
    assert len(set(assignments)) == k, "the number of clusters is not equal to k"
    assert len(g1) == 1 and len(g2) == 1, "clusters do not divide fitted data well"
    assert new_assignment == list(g2)[0], "new prediction is not allocated correctly"
