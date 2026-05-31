# Phase 6: Graph Construction + GraphSAGE/GAT Router — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build leakage-safe KNN graph infrastructure and GNN-based routing layer that combines with MMoE gates to form the full Graph-Gated MoE (GG-MoE) architecture.

**Architecture:** KNN graphs built from fused multimodal embeddings. GraphSAGE/GAT routers aggregate neighborhood information. Router output is combined with MMoE task gates via `softmax(gate_logit + graph_logit)` to produce expert routing weights.

**Tech Stack:** PyTorch Geometric (or pure PyG), scikit-learn for KNN, UMAP for visualization, Matplotlib for plots.

---

## Context from Previous Phases

### What's Already Built

- **Data loaders**: `src/data/daic_loader.py`, `src/data/mosei_loader.py`, `src/data/fi_loader.py`
- **Multimodal dataset**: `src/data/multimodal_dataset.py` — unified dataset with modality masks, task masks
- **Feature manifest**: `data/features/manifest.json` — 22,777 MOSEI + DAIC + FI samples with text/audio/video feature paths
- **Phase 5 MMoEEx**: `scripts/phase05_mmoe_ex.py` — trained checkpoint at `artifacts/tables/mmoe_ex_best.pt`
- **Fusion**: `scripts/phase04_fusion.py` — LMF/gated late fusion baselines
- **Graph builder stub**: `src/data/graph_builder.py` — all functions raise `NotImplementedError`

### Key Numbers

| Aspect | Value |
|--------|-------|
| DAIC train/val/test | 107 / 35 / 47 participants |
| MOSEI train/val/test | ~16k / ~1.8k / ~4.6k utterances |
| FI train/val/test | 6k / 2k / 2k clips |
| MMoEEx experts | 8 total: [0,1] DAIC-isolated, [2,3] MOSEI, [4,5] FI-isolated, [6,7] shared |
| Phase 5 DAIC AUROC | 0.5471 (below text-only baseline 0.6991 — regression issue) |
| Phase 5 MOSEI Sentiment CCC | 0.4762 |
| Phase 5 MOSEI Emotion AUC | 0.6906 |
| Phase 5 FI Avg CCC | 0.5688 |

### Dataset Paths

| Dataset | Path |
|---------|------|
| DAIC-WOZ | `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/daic/raw` |
| CMU-MOSEI | `/home/anilson/projects/posei-dataset/data/CMU-MOSEI` |
| ChaLearn FI | `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/fi/raw` |
| Feature cache | `/home/anilson/thesis/thesis-experiment-5-unified-model/data/features/` |
| Manifest | `/home/anilson/thesis/thesis-experiment-5-unified-model/data/features/manifest.json` |

### Critical Leakage Rules

1. **Split-local graph** (primary): Train/val/test graphs built separately — NO cross-split edges
2. **Inductive graph** (final eval): Test nodes connect only to train nodes — test nodes never connect to each other
3. **Transductive graph** (ablation only): All nodes can connect — clearly marked, NOT primary result

---

## Graph Variants (G0–G4)

| ID | Node features | File/Method | Purpose |
|----|--------------|-------------|---------|
| G0 | LMF fused embedding | Default | Baseline graph |
| G1 | Concatenated projected modality embeddings | Ablation | Modality contribution |
| G2 | ImageBind-style shared embedding | Ablation (if available) | Foundation multimodal space |
| G3 | LLM-enriched fused embedding | Ablation (Phase 8) | LLM-enhanced topology |
| G4 | Task-specific embedding | Ablation | Per-task graph topology |

**For Phase 6, implement G0 and G1 only.** G2–G4 are future ablative variants.

---

## File Structure

```
src/
  data/
    graph_builder.py    # MODIFY: implement KNN graph building
  models/
    gnn_router.py       # CREATE: GraphSAGE + GAT routers
    unified_moe.py      # MODIFY: add GG-MoE integration
scripts/
  phase06_graph.py      # MODIFY: full training + ablation script
artifacts/
  figures/
    phase_06_graph/     # visualizations
  tables/
    ggmoe_results.csv   # ablation results
    ggmoe_best.pt       # best checkpoint
```

---

## Task 1: KNN Graph Builder

**Files:**
- Modify: `src/data/graph_builder.py` (currently stub)

### Steps

- [ ] **Step 1: Implement `build_knn_graph`** — scikit-learnNearestNeighbors with cosine similarity, returns `(edge_index, edge_weight)`

```python
def build_knn_graph(
    embeddings: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """Build KNN graph over sample embeddings.
    Returns edge_index (2, num_edges), edge_weight (num_edges,).
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    # Exclude self-edges (first column is always self)
    neighbor_indices = indices[:, 1:]  # (N, k)
    self_distances = distances[:, 1:]
    # Build edge_index: [src_nodes, dst_nodes]
    src_nodes = np.repeat(np.arange(len(embeddings)), k)
    dst_nodes = neighbor_indices.flatten()
    edge_weight = 1.0 / (1.0 + self_distances.flatten())  # convert distance to similarity
    edge_index = np.stack([src_nodes, dst_nodes])
    return edge_index, edge_weight
```

- [ ] **Step 2: Implement `build_split_local_graph`** — train/val/test separately, verify no cross-split edges

```python
def build_split_local_graph(
    embeddings: np.ndarray,
    split_ids: np.ndarray,  # 0=train, 1=val, 2=test
    k: int = 10,
    cross_dataset_edges: bool = True,
) -> tuple[dict[str, tuple], dict[str, bool]]:
    """Build separate KNN graphs per split.
    Returns {split: (edge_index, edge_weight)}, leakage_check dict.
    """
    results = {}
    splits = {0: "train", 1: "val", 2: "test"}
    for split_id, split_name in splits.items():
        mask = split_ids == split_id
        split_embs = embeddings[mask]
        edge_index, edge_weight = build_knn_graph(split_embs, k=k)
        results[split_name] = (edge_index, edge_weight)
    # Leakage check
    leakage = validate_graph_leakage(*[r[0] for r in results.values()])
    return results, leakage
```

- [ ] **Step 3: Implement `build_inductive_graph`** — test nodes connect only to train nodes

```python
def build_inductive_graph(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build inductive graph: test nodes only connect to train nodes.
    Returns train_edge_index, train_edge_weight, test_edge_index, test_edge_weight.
    """
    # Test → Train edges
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(train_embeddings)
    distances, indices = nn.kneighbors(test_embeddings)
    src = np.arange(len(test_embeddings)).repeat(k)
    dst = indices.flatten()
    test_edge_index = np.stack([src, dst])
    test_edge_weight = 1.0 / (1.0 + distances.flatten())
    # Train → Train edges (train graph built normally)
    train_edge_index, train_edge_weight = build_knn_graph(train_embeddings, k=k)
    return train_edge_index, train_edge_weight, test_edge_index, test_edge_weight
```

- [ ] **Step 4: Implement `validate_graph_leakage`**

```python
def validate_graph_leakage(
    train_edge_index: np.ndarray,
    val_edge_index: np.ndarray,
    test_edge_index: np.ndarray,
    train_size: int,
    val_size: int,
) -> dict[str, bool]:
    """Check no cross-split edges exist."""
    val_offset = train_size
    test_offset = train_size + val_size
    # Check val edges only touch [0, train_size + val_size)
    val_ok = np.all(val_edge_index < train_size + val_size)
    # Check test edges only touch train (no val or other test)
    test_ok = np.all(test_edge_index < train_size)
    return {"val_leakage_free": bool(val_ok), "test_leakage_free": bool(test_ok)}
```

- [ ] **Step 5: Implement `build_multimodal_graph`** — cross-dataset KNN with dataset ID tracking

```python
def build_multimodal_graph(
    fused_embeddings: np.ndarray,
    dataset_ids: list[str],
    k: int = 10,
    cross_dataset_edges: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build cross-dataset KNN graph.
    Returns edge_index, edge_weight, edge_dataset_flags.
    edge_dataset_flags: 0=same-dataset, 1=cross-dataset.
    """
```

- [ ] **Step 6: Write tests**

```bash
uv run pytest tests/data/test_graph_builder.py -v
```

- [ ] **Step 7: Commit**

---

## Task 2: GraphSAGE Router

**Files:**
- Create: `src/models/gnn_router.py`

### Steps

- [ ] **Step 1: Write the failing test**

```python
def test_graphsage_router():
    from src.models.gnn_router import GraphSAGERouter
    router = GraphSAGERouter(in_dim=256, hidden_dim=128, out_dim=8, num_layers=2)
    # node_features: (batch, in_dim), edge_index: (2, num_edges)
    node_features = torch.randn(32, 256)
    edge_index = torch.randint(0, 32, (2, 100))
    out = router(node_features, edge_index)
    assert out.shape == (32, 8)
    assert torch.allclose(out.sum(dim=-1), torch.ones(32), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_gnn_router.py::test_graphsage_router -v`
Expected: FAIL with "GraphSAGERouter not found"

- [ ] **Step 3: Write minimal implementation**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGERouter(nn.Module):
    """GraphSAGE router for neighborhood-aggregated routing."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        if num_layers > 2:
            self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        else:
            self.layers = nn.ModuleList()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (num_nodes, in_dim), edge_index: (2, num_edges)
        # Simple mean aggregation
        for layer_idx in range(self.num_layers):
            if layer_idx == 0:
                h = self.fc1(x)
            elif layer_idx == self.num_layers - 1:
                h = self.fc2(h)
            else:
                h = self.layers[layer_idx - 1](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

            # Neighborhood aggregation: mean of neighbors
            row, col = edge_index[0], edge_index[1]
            aggr = torch.zeros_like(h)
            aggr.index_add_(0, row, h[col])  # sum neighbors
            count = torch.bincount(row, minlength=h.size(0)).float().clamp(min=1)
            aggr = aggr / count.unsqueeze(-1)
            h = aggr + h  # residual

        out = F.softmax(h, dim=-1)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_gnn_router.py::test_graphsage_router -v`
Expected: PASS

- [ ] **Step 5: Commit**

---

## Task 3: GAT Router

**Files:**
- Modify: `src/models/gnn_router.py`

### Steps

- [ ] **Step 1: Write the failing test**

```python
def test_gat_router():
    from src.models.gnn_router import GATRouter
    router = GATRouter(in_dim=256, hidden_dim=128, out_dim=8, num_heads=3)
    node_features = torch.randn(32, 256)
    edge_index = torch.randint(0, 32, (2, 100))
    out = router(node_features, edge_index)
    assert out.shape == (32, 8)
    assert torch.allclose(out.sum(dim=-1), torch.ones(32), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_gnn_router.py::test_gat_router -v`
Expected: FAIL with "GATRouter not found"

- [ ] **Step 3: Write minimal implementation**

```python
class GATRouter(nn.Module):
    """Graph Attention Network router for learned neighborhood weighting."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_heads: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.fc = nn.Linear(in_dim, hidden_dim)
        self.att = nn.Parameter(torch.randn(num_heads, 2 * self.head_dim))
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.att)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        h = self.fc(x).view(N, self.num_heads, self.head_dim)  # (N, heads, head_dim)
        row, col = edge_index[0], edge_index[1]

        # Compute attention scores
        h_row = h[row]  # (num_edges, heads, head_dim)
        h_col = h[col]  # (num_edges, heads, head_dim)
        h_concat = torch.cat([h_row, h_col], dim=-1)  # (num_edges, heads, 2*head_dim)
        scores = (h_concat * self.att.unsqueeze(0)).sum(dim=-1)  # (num_edges, heads)
        scores = F.leaky_relu(scores, 0.2)
        att = F.softmax(scores, dim=0)

        # Aggregate
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.index_add_(0, row, (att.unsqueeze(-1) * h_col))
        out = out.view(N, -1)
        out = self.fc_out(out)
        out = F.softmax(out, dim=-1)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_gnn_router.py::test_gat_router -v`
Expected: PASS

- [ ] **Step 5: Commit**

---

## Task 4: GG-MoE Integration

**Files:**
- Modify: `src/models/unified_moe.py`

### Context

The existing `src/models/unified_moe.py` has `MMoEEx` class with:
- 8 experts (experts[0,1]=DAIC, [2,3]=MOSEI, [4,5]=FI, [6,7]=shared)
- `compute_gate(input_emb)` → task-specific gate logits (no graph)
- `forward(x, task_ids)` → weighted expert mixture

We need to add Graph-Gated MoE (GG-MoE) where:

```python
def forward_ggmoe(self, x, task_ids, edge_index, graph_router_type="graphsage"):
    # MMoE gate logits
    gate_logits = self.compute_gate(x)  # (batch, num_experts)
    gate_probs = F.softmax(gate_logits, dim=-1)

    # Graph router (GraphSAGE or GAT)
    if graph_router_type == "graphsage":
        graph_probs = self.graphsage_router(x, edge_index)  # (batch, num_experts)
    else:
        graph_probs = self.gat_router(x, edge_index)  # (batch, num_experts)

    # Combine: weighted geometric mean of gate and graph
    combined_log_probs = torch.log(gate_probs + 1e-8) + 0.5 * torch.log(graph_probs + 1e-8)
    routing_weights = F.softmax(combined_log_probs, dim=-1)

    # Weighted expert sum
    expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (batch, num_experts, out_dim)
    out = (routing_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
    return out, routing_weights
```

### Steps

- [ ] **Step 1: Read existing `src/models/unified_moe.py`**
- [ ] **Step 2: Add `GraphSAGERouter` and `GATRouter` imports to unified_moe.py**
- [ ] **Step 3: Modify `MMoEEx.__init__` to accept `graph_router_type: str = None`**

```python
if graph_router_type == "graphsage":
    self.graphsage_router = GraphSAGERouter(
        in_dim=fusion_dim, hidden_dim=128, out_dim=len(self.experts)
    )
elif graph_router_type == "gat":
    self.gat_router = GATRouter(
        in_dim=fusion_dim, hidden_dim=128, out_dim=len(self.experts), num_heads=3
    )
else:
    self.graphsage_router = None
    self.gat_router = None
```

- [ ] **Step 4: Add `forward_ggmoe` method to `MMoEEx` class**

```python
def forward_ggmoe(
    self,
    x: torch.Tensor,
    task_ids: torch.Tensor,
    edge_index: torch.Tensor = None,
    graph_router_type: str = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Graph-Gated MoE forward pass.

    Args:
        x: fused multimodal embedding (batch, fusion_dim)
        task_ids: task identifier per sample (batch,)
        edge_index: graph connectivity (2, num_edges) or None
        graph_router_type: "graphsage" | "gat" | None

    Returns:
        out: task-specific output (batch, out_dim)
        routing_weights: expert routing distribution (batch, num_experts)
    """
    # MMoE gate logits
    gate_logits = self.compute_gate(x, task_ids)
    gate_probs = F.softmax(gate_logits, dim=-1)

    # Graph routing
    if graph_router_type and edge_index is not None:
        if graph_router_type == "graphsage":
            graph_probs = self.graphsage_router(x, edge_index)
        elif graph_router_type == "gat":
            graph_probs = self.gat_router(x, edge_index)
        else:
            graph_probs = None

        if graph_probs is not None:
            # Log-space combination for smoother routing
            combined_log_probs = torch.log(gate_probs + 1e-8) + 0.5 * torch.log(graph_probs + 1e-8)
            routing_weights = F.softmax(combined_log_probs, dim=-1)
        else:
            routing_weights = gate_probs
    else:
        routing_weights = gate_probs

    # Apply task-specific output projection
    out = self.compute_expert_mixture(x, routing_weights, task_ids)
    return out, routing_weights
```

- [ ] **Step 5: Write test for GG-MoE integration**

```python
def test_ggmoe_integration():
    from src.models.unified_moe import MMoEEx
    model = MMoEEx(
        fusion_dim=256,
        expert_dim=256,
        num_experts=8,
        num_tasks=4,
        graph_router_type="graphsage",
    )
    x = torch.randn(16, 256)
    task_ids = torch.randint(0, 4, (16,))
    edge_index = torch.randint(0, 16, (2, 50))
    out, weights = model.forward_ggmoe(x, task_ids, edge_index, "graphsage")
    assert out.shape == (16, 1)
    assert weights.shape == (16, 8)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(16), atol=1e-5)
```

- [ ] **Step 6: Run test**

Run: `uv run pytest tests/models/test_gnn_router.py::test_ggmoe_integration -v`

- [ ] **Step 7: Commit**

---

## Task 5: Phase 6 Training Script

**Files:**
- Modify: `scripts/phase06_graph.py`

### Context

The Phase 5 script `scripts/phase05_mmoe_ex.py` already does:
- Unified multimodal dataset loading
- Expert isolation with task_to_experts mapping
- Per-task heads (DAIC binary, MOSEI sentiment, MOSEI emotion, FI regression)
- NLL loss + homoscedastic uncertainty weighting
- Mixed precision, checkpointing, early stopping
- Temperature-balanced sampling (T=3.0) for MOSEI dominance prevention
- Training history logging to `artifacts/tables/mmoe_ex_results.csv`

The Phase 6 script should:
1. Reuse the same dataset, model architecture, training logic
2. Add graph construction (per split, leakage-safe)
3. Add `--graph_type {split-local|inductive|transductive}` flag
4. Add `--router {graphsage|gat|none}` flag
5. Run ablation: no-graph (MMoEEx) vs GraphSAGE vs GAT for each graph type
6. Save results to `artifacts/tables/ggmoe_results.csv` and checkpoint to `artifacts/tables/ggmoe_best.pt`

### Key Implementation Details

1. **Graph construction**: Build KNN graph from fused embeddings (LMF output)
2. **Edge index batching**: For each batch, extract local subgraph edges for nodes in batch
3. **Split-local default**: Primary results use split-local graph (no cross-split edges)
4. **Inductive mode**: Test nodes connect only to train neighbors (better for generalization)
5. **Transductive mode**: Mark clearly as ablation, NOT primary result

### Steps

- [ ] **Step 1: Read `scripts/phase05_mmoe_ex.py`** (1259 lines — understand structure)
- [ ] **Step 2: Copy and extend** with graph routing flags and logic
- [ ] **Step 3: Implement graph construction** using manifest embeddings
- [ ] **Step 4: Implement edge index batching** (extract local subgraph per batch)
- [ ] **Step 5: Add GG-MoE training loop** (use `model.forward_ggmoe()`)
- [ ] **Step 6: Add ablation runs** (no-graph vs GraphSAGE vs GAT × 3 graph types)
- [ ] **Step 7: Generate figures** (see Task 6)
- [ ] **Step 8: Run training** for each ablation variant

Training command pattern:
```bash
setsid uv run python scripts/phase06_graph.py \
    --graph_type split-local \
    --router graphsage \
    --epochs 150 \
    --k 10 \
    >> /tmp/phase06_graphsage.log 2>&1 &

setsid uv run python scripts/phase06_graph.py \
    --graph_type split-local \
    --router gat \
    --epochs 150 \
    --k 10 \
    >> /tmp/phase06_gat.log 2>&1 &
```

- [ ] **Step 9: Commit**

---

## Task 6: Graph Visualizations

**Files:**
- Create: `src/evaluation/visualizations.py` (extend if exists)
- Output: `artifacts/figures/phase_06_graph/`

### Required Visualizations

All figures should use Matplotlib, save to `artifacts/figures/phase_06_graph/`.

1. **Degree distribution by dataset** (`01_degree_distribution.png`)
   - Histogram of node degrees per dataset (DAIC/MOSEI/FI)
   - Log-scale y-axis if needed
   - Colored by dataset, with mean degree annotations

2. **Cross-dataset edge heatmap** (`02_cross_dataset_edges.png`)
   - 3×3 matrix: DAIC/MOSEI/FI × DAIC/MOSEI/FI
   - Cell value = count or fraction of edges between dataset pairs
   - Annotated with counts

3. **KNN similarity distribution** (`03_knn_similarity_hist.png`)
   - Histogram of edge similarity scores (edge_weight)
   - Separate distributions for same-dataset vs cross-dataset edges

4. **UMAP projection with graph edges** (`04_umap_graph_edges.png`)
   - All nodes projected to 2D via UMAP
   - Colored by dataset (marker) and by task label (color)
   - Lines show KNN edges (alpha=0.1, no arrowheads)

5. **Sample local subgraphs** (`05_local_subgraphs.png`)
   - 3 rows (DAIC, MOSEI, FI) × 3 columns (different k values)
   - Show ego network of selected node with its k nearest neighbors
   - Label nodes by dataset and task label

6. **Router entropy over epochs** (`06_router_entropy.png`)
   - Line plot: x=epoch, y=routing entropy per task
   - Separate lines per task (DAIC, MOSEI sentiment, MOSEI emotion, FI)
   - Shaded region = ±1 std

7. **Expert routing heatmap** (`07_expert_routing_heatmap.png`)
   - x = experts (0–7), y = tasks/datasets
   - Color = mean routing weight
   - Annotated with values

8. **Graph ablation comparison** (`08_ablation_comparison.png`)
   - Bar chart: no-graph vs GraphSAGE vs GAT
   - Separate subplots per dataset and metric
   - Error bars = 95% CI

### Steps

- [ ] **Step 1: Implement each visualization function**
- [ ] **Step 2: Add to Phase 6 script or create `scripts/phase06_visualizations.py`**
- [ ] **Step 3: Run and verify all figures save without error**
- [ ] **Step 4: Commit**

---

## Acceptance Criteria

| Task | Criterion |
|------|-----------|
| Task 1 | `build_split_local_graph` produces no cross-split edges; `build_inductive_graph` produces test→train-only edges; all three modes documented |
| Task 2 | GraphSAGE router output sums to 1.0 (softmax); gradient flows through neighborhood aggregation |
| Task 3 | GAT router output sums to 1.0; multi-head attention produces distinct weight distributions per head |
| Task 4 | GG-MoE combines MMoE gate + graph router; combined routing still sums to 1.0; expert collapse check passes |
| Task 5 | All 6 ablation variants run to completion (150 epochs); no NaN; checkpoint saves best model; CSV logs all metrics |
| Task 6 | All 8 visualizations produced and saved; figures are legible, labeled, and thesis-ready |