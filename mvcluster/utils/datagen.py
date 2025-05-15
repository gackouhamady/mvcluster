from scipy import io
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.neighbors import kneighbors_graph

def acm():
    data = io.loadmat('../datasets/data_lmgec/ACM.mat')
    X = data['features']
    A, B = data['PAP'], data['PLP']
    return [A.toarray(), B.toarray()], [X.toarray()], data['label'].reshape(-1)

def imdb():
    data = io.loadmat('../datasets/data_lmgec/IMDB.mat')
    X = data['features']
    A, B = data['PAP'], data['PLP']
    return [A.toarray(), B.toarray()], [X.toarray()], data['label'].reshape(-1)

def photos():
    data = io.loadmat('../datasets/data_lmgec/Amazon_Photos.mat')
    X = data['features']
    A, B = data['PAP'], data['PLP']
    return [A.toarray(), B.toarray()], [X.toarray()], data['label'].reshape(-1)

def wiki():
    data = io.loadmat('../datasets/data_lmgec/Wiki.mat')
    X = data['features']
    A, B = data['PAP'], data['PLP']
    return [A.toarray(), B.toarray()], [X.toarray()], data['label'].reshape(-1)



 

def dblp():
    data_path = os.path.normpath(os.path.join(os.path.dirname(__file__),  '..', 'datasets', 'data_lmgec', 'DBLP.mat'))

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ File not found at: {data_path}")

    print(f"✅ Loading DBLP.mat from: {data_path}")
    data = io.loadmat(data_path)

    X = data['features']
    A = data['net_APTPA']
    B = data['net_APCPA']
    C = data['net_APA']

    return [A.toarray(), B.toarray(), C.toarray()], [X.toarray()], data['label'].reshape(-1)



# idem pour imdb(), dblp(), wiki(), photos()

def datagen(dataset):
    if dataset == 'imdb': return imdb() # type: ignore
    elif dataset == 'dblp': return dblp() # type: ignore
    elif dataset == 'acm': return acm()
    elif dataset == 'photos': return photos() # type: ignore
    elif dataset == 'wiki': return wiki() # type: ignore
    else: raise ValueError(f"Unknown dataset {dataset}")
