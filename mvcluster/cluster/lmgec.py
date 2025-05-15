import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from ..utils.init_utils import init_G_F, init_W  # type: ignore
from ..models.lmgec_core import train_loop       # type: ignore

class LMGEC(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, embedding_dim=10, temperature=0.5, max_iter=30, tolerance=1e-6):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(self, X_views, y=None):
        n_views = len(X_views)
        alphas = np.zeros(n_views)
        XW_consensus = 0

        # Initialisation par vue
        for v, Xv in enumerate(X_views):
            Wv = init_W(Xv, self.embedding_dim)
            XWv = Xv @ Wv
            Gv, Fv = init_G_F(XWv, self.n_clusters)
            inertia = np.linalg.norm(XWv - Fv[Gv])
            alphas[v] = np.exp(-inertia / self.temperature)
            XW_consensus += alphas[v] * XWv

        # Moyenne pondérée
        XW_consensus /= alphas.sum()

        # Réinitialisation du clustering sur consensus
        G, F = init_G_F(XW_consensus, self.n_clusters)

        # Entraînement
        G, F, XW_final, losses = train_loop(
            X_views, F, G, alphas, self.n_clusters, self.max_iter, self.tolerance
        )

        # Stockage des résultats
        self.labels_ = G.numpy() if hasattr(G, 'numpy') else G
        self.F_ = F
        self.XW_ = XW_final
        self.loss_history_ = losses

        return self

    def predict(self, X_views):
        return self.labels_

    def transform(self, X_views):
        return self.XW_
