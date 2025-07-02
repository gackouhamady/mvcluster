"""
LMGEC clustering model implementation.

This module provides the LMGEC class, which implements the Localized
Multi-Graph Embedding Clustering (LMGEC) algorithm. It extends
scikit-learn's BaseEstimator and ClusterMixin interfaces, offering
fit, predict, and transform methods for clustering across multiple views.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from ..utils.init_utils import init_G_F, init_W  # type: ignore
from ..models.lmgec_core import train_loop  # type: ignore


class LMGEC(BaseEstimator, ClusterMixin):
    """
    Localized Multi-Graph Embedding Clustering (LMGEC) model.

    :param int n_clusters: Number of clusters to form.
    :param int embedding_dim: Dimension of embedding space.
    :param float temperature: Temperature for view weighting.
    :param int max_iter: Max training iterations.
    :param float tolerance: Convergence threshold.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        embedding_dim: int = 10,
        temperature: float = 0.5,
        max_iter: int = 30,
        tolerance: float = 1e-6,
    ):  # noqa: D107
        """
        Initialize the LMGEC (Localized Multi-Graph Embedding Clustering)
        model.

        Sets the main hyperparameters of the model including number of
        clusters, dimensionality of learned embeddings, and optimization
        settings.

        :param int n_clusters: Number of clusters to find in the data.
        :param int embedding_dim: Dimensionality of the latent embedding.
        :param float temperature: View weighting temperature parameter.
        :param int max_iter: Max number of optimization iterations.
        :param float tolerance: Convergence threshold for training loop.
        :returns: An instance of the LMGEC clustering model.
        :rtype: self
        """
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(self, X_views, y=None):  # noqa: D102
        """
        Fit the LMGEC model to multiple data views.

        This method initializes embeddings and performs consensus clustering
        across all views. It uses a weighted embedding fusion guided by
        inertia and iteratively refines the cluster assignments.

        :param list X_views: List of 2D arrays (one per view), shape
            (n_samples, n_features) for each.
        :param y: Ignored, for API compatibility.
        :returns: The fitted estimator.
        :rtype: self
        """
        for i, X in enumerate(X_views):
            print(f"View {i} shape: {X.shape}, sum: {np.sum(X)}, any NaN: {np.isnan(X).any()}")  # noqa: E501
        n_views = len(X_views)
        alphas = np.zeros(n_views)
        XW_consensus = 0
        Ws = []

        for v, Xv in enumerate(X_views):
            Wv = init_W(Xv, self.embedding_dim)
            Ws.append(Wv)
            XWv = Xv @ Wv
            print(f"View {v}: XWv norm = {np.linalg.norm(XWv)}")
            Gv, Fv = init_G_F(XWv, self.n_clusters)
            inertia = np.linalg.norm(XWv - Fv[Gv])
            alphas[v] = np.exp(-inertia / self.temperature)
            XW_consensus += alphas[v] * XWv

        alpha_sum = alphas.sum()
        print("DEBUG: Alphas before normalization =", alphas)
        print("DEBUG: Alpha sum =", alpha_sum)
        if alpha_sum == 0 or np.isnan(alpha_sum):
            print("DEBUG: alphas =", alphas)
            print("DEBUG: XWv norms =", [np.linalg.norm(X @ W) for X, W in zip(X_views, Ws)])  # noqa: E501
            raise ValueError("Alpha weights sum to zero or NaN; cannot normalize XW_consensus.")  # noqa: E501
        XW_consensus /= alpha_sum

        G, F = init_G_F(XW_consensus, self.n_clusters)

        G, F, XW_final, losses = train_loop(
            X_views,
            F,  # type: ignore
            G,  # type: ignore
            alphas,  # type: ignore
            self.n_clusters,
            self.max_iter,
            self.tolerance,
        )

        self.labels_ = G.numpy() if hasattr(G, "numpy") else G
        self.F_ = F
        self.XW_ = XW_final
        self.loss_history_ = losses

        return self

    def predict(self, X_views):  # noqa: D102
        """
        Predict cluster labels for input views after fitting.

        :param list X_views: List of feature matrices (ignored).
        :returns: Cluster labels from fit.
        :rtype: array-like
        """
        return self.labels_

    def transform(self, X_views):  # noqa: D102
        """
        Transform input views into the final embedding space.

        :param list X_views: List of feature matrices (ignored).
        :returns: Consensus embedding from fit.
        :rtype: array-like
        """
        return self.XW_
