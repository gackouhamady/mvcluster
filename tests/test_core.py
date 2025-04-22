import numpy as np
from mvcluster.core import simple_kmeans

def test_simple_kmeans():
    X = np.random.rand(100, 2)  # 100 points 2D
    labels = simple_kmeans(X, n_clusters=3)
    
    assert len(labels) == 100  # Vérifier le nombre de labels
    assert len(set(labels)) == 3  # Vérifier que les clusters sont divisés en 3
