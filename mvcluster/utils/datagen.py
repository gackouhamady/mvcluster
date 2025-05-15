import os
import numpy as np
from scipy import io
from sklearn.neighbors import kneighbors_graph



def _load_mat_file(filename):
    data_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'data_lmgec', filename))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ File not found: {data_path}")
    print(f"✅ Loading {filename} from: {data_path}")
    return io.loadmat(data_path)


def acm():
    data = _load_mat_file('ACM.mat')

    X = data['features']
    A = data['PAP']
    B = data['PLP']

    Xs = [X.toarray()]
    As = [A.toarray(), B.toarray()]
    labels = data['label'].reshape(-1)

    return As, Xs, labels


def dblp():
    data = _load_mat_file('DBLP.mat')

    X = data['features']
    A = data['net_APTPA']
    B = data['net_APCPA']
    C = data['net_APA']

    Xs = [X.toarray()]
    As = [A.toarray(), B.toarray(), C.toarray()]
    labels = data['label'].reshape(-1)

    return As, Xs, labels


def imdb():
    data = _load_mat_file('IMDB.mat')

    X = data['features']
    A = data['MAM']
    B = data['MDM']

    Xs = [X.toarray()]
    As = [A.toarray(), B.toarray()]
    labels = data['label'].reshape(-1)

    return As, Xs, labels




def photos():
    data = _load_mat_file('Amazon_photos.mat')

    X = data['features']
    A = data['adj']
    labels = data['label'].reshape(-1)

 
    X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    A = A.toarray() if hasattr(A, 'toarray') else np.asarray(A)

    try:
        X2 = X @ X.T
    except Exception as e:
        print(f"⚠️ Error computing X @ X.T: {e}")
        X2 = X  

    Xs = [X, X2]
    As = [A]

    return As, Xs, labels


def wiki():
    data = _load_mat_file('wiki.mat')


    X = data['fea']
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = X.astype(float)

 
    A = data.get('W')
    if A is None:
        raise ValueError("❌ 'W' not found in wiki.mat")
    if hasattr(A, 'toarray'):
        A = A.toarray()

   
    labels = data['gnd'].reshape(-1)

   
    knn_graph = kneighbors_graph(X, 5, metric='cosine')
    if hasattr(knn_graph, 'toarray'):
        knn_graph = knn_graph.toarray()

 
    Xs = [X, np.log2(1 + X)]
    As = [A, knn_graph]

    return As, Xs, labels




def datagen(dataset):
    if dataset == 'acm':
        return acm()
    elif dataset == 'dblp':
        return dblp()
    elif dataset == 'imdb':
        return imdb()
    elif dataset == 'photos':
        return photos()
    elif dataset == 'wiki':
        return wiki()
    else:
        raise ValueError(f"❌ Unknown dataset: {dataset}")
