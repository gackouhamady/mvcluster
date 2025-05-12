import numpy as np
from mvcluster.core import simple_kmeans


def test_simple_kmeans():
    X = np.array([[0, 0], [1, 1], [9, 9]])
    labels = simple_kmeans(X, n_clusters=2)
    assert len(labels) == 3
