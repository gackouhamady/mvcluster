from sklearn.cluster import KMeans


def simple_kmeans(X, n_clusters=3):
    """
    Effectue un clustering K-means sur les données X.

    Parameters:
    X : np.array
        Données d'entrée.
    n_clusters : int
        Nombre de clusters.

    Returns:
    np.array
        Indices des clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans.labels_
