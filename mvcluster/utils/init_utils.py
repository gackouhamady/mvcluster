import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

def init_G_F(XW, k):
    km = KMeans(k).fit(XW)
    return km.labels_, km.cluster_centers_

def init_W(X, f):
    svd = TruncatedSVD(f).fit(X)
    return svd.components_.T
