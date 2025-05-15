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
    n_views = len(Xs)
    losses = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
    prev_loss = tf.constant(float('inf'), dtype=tf.float64)

    for i in tf.range(max_iter):
        loss = tf.constant(0.0, dtype=tf.float64)
        XW_consensus = tf.zeros((tf.shape(Xs[0])[0], tf.shape(F)[1]), dtype=tf.float64)


        for v in range(n_views):
            Wv = update_rule_W(Xs[v], F, G)
            XWv = tf.matmul(Xs[v], Wv)
            XW_consensus = tf.zeros((tf.shape(Xs[0])[0], tf.shape(F)[1]), dtype=tf.float64)


            F_gathered = tf.gather(F, G)
            reconstruction = tf.matmul(F_gathered, tf.transpose(Wv))
            loss_v = tf.norm(Xs[v] - reconstruction)
            loss += alphas[v] * loss_v

        G = update_rule_G(XW_consensus, F)
        F = update_rule_F(XW_consensus, G, k)

        losses = losses.write(i, loss)

        # Convergence check
        if tf.abs(prev_loss - loss) < tolerance:
            break

        prev_loss = loss

    return G, F, XW_consensus, losses.stack()
