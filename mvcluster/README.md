## 📦 Graph Clustering Architectures: GCC and LMGEC
This README provides a detailed architectural overview for two unsupervised graph clustering methods implemented in this repository:

- GCC – Graph Convolutional Clustering (single-view)

- LMGEC – Linear Multi-view Graph Embedding and Clustering (multi-view)

The pipeline stages for each model are presented as text-based flow diagrams to enhance clarity, reproducibility, and implementation alignment.

## 🧠 GCC — Graph Convolutional Clustering

``` text 
                     ┌─────────────────────────────┐
                     │    INPUT GRAPH DATA         │
                     │  A ∈ ℝⁿˣⁿ: adjacency matrix │
                     │  X ∈ ℝⁿˣᵈ: node features     │
                     └────────────┬────────────────┘
                                  │
                                  ▼
                  ┌─────────────────────────────────────┐
                  │ GRAPH PROPAGATION (SGC-like)        │
                  │ T = D_T⁻¹ (I + S̃), TᵖX ← T^p X       │
                  └────────────┬────────────────────────┘
                               │
                               ▼
                ┌──────────────────────────────────────┐
                │ LINEAR EMBEDDING                     │
                │ Z = TᵖX W,  W ∈ ℝᵈˣᶠ                  │
                │ Orthogonal constraint: WᵗW = I       │
                └────────────┬─────────────────────────┘
                             │
                             ▼
         ┌──────────────────────────────────────────────┐
         │ CLUSTERING                                   │
         │ G ∈ {0,1}ⁿˣᵏ: assignment matrix              │
         │ F ∈ ℝᵏˣᶠ: cluster centroids in embedding     │
         └────────────┬─────────────────────────────────┘
                      │
                      ▼
       ┌──────────────────────────────────────────────────────┐
       │ JOINT OBJECTIVE FUNCTION                             │
       │ L = ‖TᵖX - TᵖXWWᵗ‖² + ‖TᵖXW - GF‖²                    │
       │  → Equivalent to: ‖TᵖX - GFWᵗ‖²                      │
       └────────────┬─────────────────────────────────────────┘
                    │
                    ▼
      ┌────────────────────────────────────────────────────────┐
      │ OPTIMIZATION (ALTERNATING MINIMIZATION)                │
      │ 1. Update F ← least squares (centroid step)            │
      │ 2. Update W ← Procrustes via SVD                       │
      │ 3. Update G ← k-means assignment in embedded space     │
      └────────────────────────────────────────────────────────┘

                            🔚 Final Output:
                            - Clustered node labels G
                            - Embedding space Z = TᵖXW

```
##  LMGEC — Linear Multi-view Graph Embedding and Clustering

```text
                     ┌──────────────────────────────────────┐
                     │    INPUT MULTI-VIEW GRAPH DATA       │
                     │  V views: {(A_v, X_v)}               │
                     │  - A_v ∈ ℝⁿˣⁿ: adjacency matrix       │
                     │  - X_v ∈ ℝⁿˣᵈᵛ: feature matrix        │
                     └────────────┬─────────────────────────┘
                                  │
                                  ▼
          ┌──────────────────────────────────────────────────────┐
          │ PER-VIEW PROPAGATION                                 │
          │ H_v = S_v X_v,  S_v = (D_v + βI)⁻¹ (A_v + βI)         │
          └────────────┬─────────────────────────────────────────┘
                       │
                       ▼
       ┌───────────────────────────────────────────────────────┐
       │ PER-VIEW EMBEDDING                                    │
       │ W_v ∈ ℝᵈᵛˣᶠ,   Z_v = H_v W_v                           │
       │ Constraint: W_v W_vᵗ = I                               │
       └────────────┬──────────────────────────────────────────┘
                    │
                    ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ GLOBAL CLUSTERING                                           │
   │ - Shared cluster assignments G ∈ {0,1}ⁿˣᵏ                   │
   │ - Shared centroids F ∈ ℝᵏˣᶠ                                 │
   └────────────┬───────────────────────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ ATTENTION WEIGHTS OVER VIEWS                                 │
   │ α_v = softmax(-I_v / τ),  I_v = ‖H_v - G_v F_v‖              │
   └────────────┬────────────────────────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ JOINT OBJECTIVE FUNCTION                                     │
   │ L = Σ_v α_v ‖H_v - G F W_vᵗ‖²                                │
   └────────────┬────────────────────────────────────────────────┘
                │
                ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ OPTIMIZATION (BLOCK COORDINATE DESCENT)                       │
  │ 1. Update W_v ← SVD(H_vᵗ G F) → W_v = VUᵗ                     │
  │ 2. Update F ← Least squares on α-weighted sum of views       │
  │ 3. Update G ← Assign via argmin of distance to α-weighted F  │
  │ 4. Re-estimate α_v using updated reconstruction errors       │
  └───────────────────────────────────────────────────────────────┘

                            🔚 Final Output:
                            - Shared cluster labels G
                            - Per-view projections {W_v}
                            - Consensus centroids F

```
## 🔁 LMGEC Execution Pipeline (Code-Level Overview)
``` text
                      🔽 ENTRY POINT : Evaluation script
   mvcluster/benchmark/lmgec_benchmark.py
   └── run_lmgec_experiment(dataset, ...)
        │
        ▼
───────────────────────────────────────────────────────────────
📦 Step 1: Load Dataset

  datagen(dataset)                [utils/datagen.py]
  ├── acm(), dblp(), imdb(), etc.
  └── Returns: As, Xs, labels

───────────────────────────────────────────────────────────────
📦 Step 2: Preprocess Views

  preprocess_dataset(A, X, tf_idf, beta)  [utils/preprocess.py]
  ├── Normalize adjacency with β
  ├── Normalize features (L2 or TF-IDF)
  └── Returns: adj_norm, features_norm

───────────────────────────────────────────────────────────────
📦 Step 3: Graph Propagation

  H_v = S @ X                                 [lmgec_benchmark.py]
  H_v = StandardScaler().fit_transform(H_v)

───────────────────────────────────────────────────────────────
📦 Step 4: Train the Model

  model = LMGEC(...)                        [cluster/lmgec.py]
  model.fit(Hs)
     ├── init_W(X, f)                      [utils/init_utils.py]
     ├── init_G_F(XW, k)                   [utils/init_utils.py]
     └── train_loop(...)                   [models/lmgec_core.py]
         ├── update_rule_W / G / F
         └── α_v computed via inertia

───────────────────────────────────────────────────────────────
📦 Step 5: Predict + Embedding

  model.predict(X_views) → cluster labels
  model.transform(X_views) → projected embeddings

───────────────────────────────────────────────────────────────
📦 Step 6: Evaluate

  clustering_accuracy(...)        [utils/metrics.py]
  clustering_f1_score(...)        [utils/metrics.py]
  normalized_mutual_info_score   [sklearn]
  adjusted_rand_score            [sklearn]

───────────────────────────────────────────────────────────────
📦 Final Results

  Print: ACC, F1, NMI, ARI scores (averaged across runs)
```



# LMGEC Model: Mathematical Formulation and Weight Update Pseudocode

## 1. Mathematical Formulation

### Notations
- \( V \): Number of views, indexed by \( i = 1, \ldots, V \)
- For each view \( i \):
  - \( \mathbf{S}_i \in \mathbb{R}^{n \times n} \): normalized adjacency matrix
  - \( \mathbf{X}_i \in \mathbb{R}^{n \times d_i} \): feature matrix
- \( n \): number of samples
- Parameters:
  - \( T > 0 \): temperature controlling view weighting sharpness
  - \( \beta > 0 \): graph regularization parameter
  - \( k \): number of clusters (embedding dimension is \( k+1 \))

### View Preprocessing
\[
(\mathbf{S}_i, \mathbf{X}_i) \leftarrow \text{preprocess\_dataset}(\mathbf{A}_i, \mathbf{X}_i, \beta)
\]

\[
\mathbf{H}_i = \text{StandardScaler}(\mathbf{S}_i \mathbf{X}_i)
\]


### Score Computation per View
\[
s_i = \|\mathbf{X}_i \mathbf{W} \mathbf{v}\|_F
\]
where \( \mathbf{W}, \mathbf{v} \) are model parameters and \(\|\cdot\|_F\) is Frobenius norm.

### Weight Computation (Softmax with Temperature)
\[
\alpha_i = \frac{\exp(s_i / T)}{\sum_{j=1}^V \exp(s_j / T)}
\]

- \( T \to 0 \): weights become one-hot (selecting one view)
- \( T \to \infty \): weights become uniform

### Fusion of Views
\[
\mathbf{Z} = \sum_{i=1}^V \alpha_i \mathbf{H}_i
\]

### Objective Function
\[
\mathcal{L} = \text{ClusteringLoss}(\mathbf{Z}) + \beta \cdot \text{GraphRegularization}(\{\mathbf{S}_i\})
\]

---

## 2. Weight Update Pseudocode

```python
initialize model parameters W, v, ...

for iteration in range(max_iter):
    scores = []
    for i in range(V):
        Z_i = X_i @ W @ v           # Project view i
        s_i = norm(Z_i, 'fro')      # Compute score for view i
        scores.append(s_i)
    
    # Compute weights alpha with softmax temperature T
    exp_scores = [exp(s / T) for s in scores]
    sum_exp = sum(exp_scores)
    alphas = [e / sum_exp for e in exp_scores]
    
    # Fuse views with weights
    Z = sum(alphas[i] * H_i for i in range(V))
    
    # Update model parameters (e.g., gradient step)
    update_parameters()
    
    if convergence_criteria_met:
        break
```