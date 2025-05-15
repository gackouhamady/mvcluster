import numpy as np
import tensorflow as tf  # type: ignore
from time import time
from itertools import product
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.preprocessing import StandardScaler

from ..utils.datagen import datagen
from ..utils.preprocess import preprocess_dataset
from ..utils.metrics import clustering_accuracy, clustering_f1_score
from ..cluster.lmgec import LMGEC

def run_lmgec_experiment(dataset, temperature=1, beta=1, max_iter=10, tolerance=1e-7, runs=1):
    print(f'Running on dataset: {dataset}')
    As, Xs, labels = datagen(dataset)
    k = len(np.unique(labels))
    views = list(product(As, Xs))

    # Prétraitement des vues
    for v in range(len(views)):
        A, X = views[v]
        tf_idf = dataset in ['acm', 'dblp', 'imdb', 'photos']
        norm_adj, features = preprocess_dataset(A, X, tf_idf=tf_idf, beta=beta)
        views[v] = (np.asarray(norm_adj), features.toarray() if hasattr(features, 'toarray') else features)

    metrics = {key: [] for key in ['acc', 'nmi', 'ari', 'f1', 'loss', 'time']}

    for _ in range(runs):
        t0 = time()

        # Propagation : H_v = A * X
        Hs = [StandardScaler(with_std=False).fit_transform(S @ X) for S, X in views]

        # Entraînement du modèle LMGEC
        model = LMGEC(
            n_clusters=k,
            embedding_dim=k + 1,
            temperature=temperature,
            max_iter=max_iter,
            tolerance=tolerance
        )
        model.fit(Hs)

        # Collecte des résultats
        metrics['time'].append(time() - t0)
        metrics['acc'].append(clustering_accuracy(labels, model.labels_))
        metrics['nmi'].append(nmi(labels, model.labels_))
        metrics['ari'].append(ari(labels, model.labels_))
        metrics['f1'].append(clustering_f1_score(labels, model.labels_, average='macro'))
        metrics['loss'].append(model.loss_history_[-1])

    # Résumé
    results = {
        'mean': {k: round(np.mean(v), 4) for k, v in metrics.items()},
        'std': {k: round(np.std(v), 4) for k, v in metrics.items()}
    }

    print('ACC & F1 & NMI & ARI:')
    print(results['mean']['acc'], results['mean']['f1'], results['mean']['nmi'], results['mean']['ari'], sep=' & ')
    return results


# Interface CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="acm")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    run_lmgec_experiment(
        dataset=args.dataset,
        temperature=args.temperature,
        beta=args.beta,
        max_iter=args.max_iter,
        tolerance=args.tol,
        runs=args.runs,
    )
