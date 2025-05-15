import tensorflow as tf  # type: ignore

def update_rule_F(XW, G, k):
    return tf.math.unsorted_segment_mean(XW, G, k)

def update_rule_W(X, F, G):
    _, U, V = tf.linalg.svd(tf.transpose(X) @ tf.gather(F, G), full_matrices=False)
    return U @ tf.transpose(V)

def update_rule_G(XW, F):
    centroids_expanded = F[:, None, ...]  # shape: [k, 1, d]
    distances = tf.reduce_mean(tf.math.squared_difference(XW, centroids_expanded), axis=2)  # [k, n]
    return tf.math.argmin(distances, axis=0, output_type=tf.dtypes.int32)



def train_loop(Xs, F, G, alphas, k, max_iter, tolerance):
    """
    Alternating optimization for LMGEC.
    No @tf.function: works with dynamic shapes (e.g. Xs as lists).
    """
    n_views = len(Xs)
    n_samples = Xs[0].shape[0]
    embedding_dim = F.shape[1]

    losses = []
    prev_loss = float('inf')

    for i in range(max_iter):
        loss = 0.0
        XW_consensus = tf.zeros((n_samples, embedding_dim), dtype=tf.float64)

        for v in range(n_views):
            Xv = tf.cast(Xs[v], tf.float64)
            Wv = update_rule_W(Xv, F, G)
            XWv = tf.matmul(Xv, Wv)
            XW_consensus += alphas[v] * XWv

            Fg = tf.gather(F, G)
            recon = tf.matmul(Fg, tf.transpose(Wv))
            loss_v = tf.norm(Xv - recon)
            loss += alphas[v] * loss_v

        G = update_rule_G(XW_consensus, F)
        F = update_rule_F(XW_consensus, G, k)

        losses.append(loss.numpy())

        if abs(prev_loss - loss) < tolerance:
            break

        prev_loss = loss

    return G, F, XW_consensus, tf.convert_to_tensor(losses, dtype=tf.float64)
