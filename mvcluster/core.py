from sklearn.cluster import KMeans
import numpy as np

def simple_kmeans(X, n_clusters=3):
    """
    Effectue un clustering K-means sur les données X.
    
    Parameters:
    - X : np.array, données d'entrée.
    - n_clusters : int, nombre de clusters.
    
    Returns:
    - labels : np.array, indices des clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans.labels_
