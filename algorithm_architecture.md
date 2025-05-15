# 📦 Graph Clustering Architectures: GCC and LMGEC

This README describes the full pipeline architecture for two unsupervised graph clustering methods:

* **GCC**: Graph Convolutional Clustering (single-view)
* **LMGEC**: Linear Multi-view Graph Embedding and Clustering (multi-view)

Both architectures are presented as textual flow diagrams to facilitate understanding and implementation.

---

## 🧠 GCC — Graph Convolutional Clustering

```text
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

---

## 🧩 LMGEC — Linear Multi-view Graph Embedding and Clustering

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
