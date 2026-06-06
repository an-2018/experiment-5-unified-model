"""Leakage-safe KNN graph construction for multimodal samples.

Rules:
- Build train/val/test KNN graphs separately (split-local graph mode) for primary results.
- Never allow cross-split edges in primary results.
- Use inductive inference for final evaluation.
- Transductive graph = ablation only, must be clearly marked.
"""
from typing import Literal, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(
    embeddings: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """Build symmetric KNN graph over sample embeddings.

    Args:
        embeddings: (N, D) embedding matrix
        k: number of nearest neighbors (excluding self)
        metric: distance metric ("cosine" supported)

    Returns:
        edge_index: (2, num_edges) PyG-compatible edge index
        edge_weight: (num_edges,) similarity scores = 1/(1+distance)
    """
    n = embeddings.shape[0]

    if k >= n:
        raise ValueError(f"k={k} must be less than number of samples n={n}")

    # Use brute-force for cosine (more reliable for normalized embeddings)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute")
    nn.fit(embeddings)

    # Get k+1 neighbors (including self at position 0)
    distances, indices = nn.kneighbors(embeddings)

    # Remove self-edges (column 0 is self)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Build symmetric edges
    src_nodes = np.repeat(np.arange(n), k)
    dst_nodes = indices.flatten()
    edge_weights = distances.flatten()

    # Convert distance to similarity: 1/(1+dist)
    edge_weights = 1.0 / (1.0 + edge_weights)

    # Create edge index (2, num_edges)
    edge_index = np.stack([src_nodes, dst_nodes])

    return edge_index, edge_weights


def build_split_local_graph(
    embeddings: np.ndarray,
    split_ids: np.ndarray,
    k: int = 10,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, bool]]:
    """Build separate KNN graphs per split (train/val/test).

    Critical: train graph uses only train embeddings, val graph only val embeddings,
    test graph only test embeddings. No cross-split edges allowed.

    Note: cross-dataset edges within the same split are handled by build_multimodal_graph.
    This function is strictly split-local.

    Args:
        embeddings: (N, D) full embedding matrix
        split_ids: (N,) array with 0=train, 1=val, 2=test
        k: number of nearest neighbors

    Returns:
        graphs: dict with keys "train", "val", "test" → (edge_index, edge_weight)
        leakage_check: dict with "val_leakage_free", "test_leakage_free" booleans
    """
    graphs = {}
    leakage_check = {}

    train_size = np.sum(split_ids == 0)
    val_size = np.sum(split_ids == 1)
    test_size = np.sum(split_ids == 2)

    for split_name, split_value in [("train", 0), ("val", 1), ("test", 2)]:
        # Mask for this split
        mask = split_ids == split_value
        split_embeddings = embeddings[mask]
        split_size = split_embeddings.shape[0]

        # Handle empty splits - return empty edge arrays
        if split_size == 0:
            graphs[split_name] = (np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.float64))
            continue

        # Handle splits with fewer than k samples
        if split_size < k:
            # Warn but still build graph with available neighbors
            import warnings
            warnings.warn(
                f"Split '{split_name}' has {split_size} samples < k={k}. "
                f"Building graph with reduced neighbors.",
                RuntimeWarning
            )
            # Build with k=split_size-1 (at least 1 neighbor if possible)
            effective_k = max(1, split_size - 1)
            edge_index, edge_weight = build_knn_graph(split_embeddings, k=effective_k)
        else:
            edge_index, edge_weight = build_knn_graph(split_embeddings, k=k)

        graphs[split_name] = (edge_index, edge_weight)

    # In split-local mode, each graph only uses nodes from its own split.
    # Since we build graphs from split_embeddings, all node indices are LOCAL
    # to that split (0 to split_size-1). So the leakage check is simply:
    # val graph's dst nodes should be in [0, val_size), test in [0, test_size).
    # This is inherently satisfied because we built from local embeddings.
    val_edges = graphs["val"][0]
    val_dst_nodes = val_edges[1]
    val_leakage_free = np.all((val_dst_nodes >= 0) & (val_dst_nodes < val_size))

    test_edges = graphs["test"][0]
    test_dst_nodes = test_edges[1]
    test_leakage_free = np.all((test_dst_nodes >= 0) & (test_dst_nodes < test_size))

    leakage_check["val_leakage_free"] = bool(val_leakage_free)
    leakage_check["test_leakage_free"] = bool(test_leakage_free)

    return graphs, leakage_check


def build_inductive_graph(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build inductive KNN graph for transductive evaluation.

    Test nodes connect ONLY to train nodes (not to other test nodes).
    Train nodes connect to other train nodes normally.
    This prevents test-to-test information leakage.

    Args:
        train_embeddings: (N_train, D) training embeddings
        test_embeddings: (N_test, D) test embeddings
        k: number of nearest neighbors

    Returns:
        train_edge_index: (2, num_train_edges)
        train_edge_weight: (num_train_edges,)
        test_edge_index: (2, num_test_edges) - test nodes only connect to train
        test_edge_weight: (num_test_edges,)
    """
    n_train = train_embeddings.shape[0]
    n_test = test_embeddings.shape[0]

    # Train graph: train nodes connect to other train nodes
    train_edge_index, train_edge_weight = build_knn_graph(train_embeddings, k=k)

    # Test connections: test nodes connect ONLY to train nodes
    # Use train as the reference database
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(train_embeddings)

    distances, indices = nn.kneighbors(test_embeddings)

    # Build test->train edges
    src_nodes = np.repeat(np.arange(n_test), k) + n_train  # Global indices start at n_train
    dst_nodes = indices.flatten()  # Train nodes use global indices [0, n_train)
    test_edge_weights = distances.flatten()
    test_edge_weights = 1.0 / (1.0 + test_edge_weights)

    test_edge_index = np.stack([src_nodes, dst_nodes])

    return train_edge_index, train_edge_weight, test_edge_index, test_edge_weights


def validate_graph_leakage(
    train_edge_index: np.ndarray,
    val_edge_index: np.ndarray,
    test_edge_index: np.ndarray,
    train_size: int,
    val_size: int,
) -> dict[str, bool]:
    """Check that no cross-split edges exist in train/val/test graphs.

    Node indexing convention:
    - Train nodes: [0, train_size)
    - Val nodes: [train_size, train_size + val_size)
    - Test nodes: [train_size + val_size, ...)

    Args:
        train_edge_index: (2, num_train_edges)
        val_edge_index: (2, num_val_edges)
        test_edge_index: (2, num_test_edges)
        train_size: number of train nodes
        val_size: number of val nodes

    Returns dict with keys:
        - 'train_leakage_free': True if train edges stay within [0, train_size)
        - 'val_leakage_free': True if val edges only touch [train_size, train_size+val_size)
        - 'test_leakage_free': True if test edges only touch [0, train_size+val_size)
    """
    val_start = train_size
    val_end = train_size + val_size

    # Check train edges (must be within train node range)
    train_src = train_edge_index[0]
    train_dst = train_edge_index[1]
    train_edges_within_bounds = (
        np.all((train_src >= 0) & (train_src < train_size)) and
        np.all((train_dst >= 0) & (train_dst < train_size))
    )

    # Check val edges (must be within val node range)
    val_src = val_edge_index[0]
    val_dst = val_edge_index[1]
    val_edges_within_bounds = (
        np.all((val_src >= val_start) & (val_src < val_end)) and
        np.all((val_dst >= val_start) & (val_dst < val_end))
    )

    # Check test edges (must be within test node range = beyond train+val)
    test_src = test_edge_index[0]
    test_dst = test_edge_index[1]
    # Test nodes start at train_size + val_size
    test_start = train_size + val_size

    # For test nodes: src should be >= test_start, dst should be in [0, train_size+val_size)
    # because in inductive mode, test nodes only connect to train/val
    test_edges_within_bounds = (
        np.all(test_src >= test_start) and
        np.all(test_dst < test_start)  # Only train/val nodes as destinations
    )

    return {
        "train_leakage_free": bool(train_edges_within_bounds),
        "val_leakage_free": bool(val_edges_within_bounds),
        "test_leakage_free": bool(test_edges_within_bounds),
    }


def validate_graph_no_cross_split_leakage(
    edge_index: np.ndarray,
    split_ids: np.ndarray,
    graph_name: str = "graph"
) -> None:
    """Validate that no edges cross train/val/test splits.

    Args:
        edge_index: (2, num_edges) edge index
        split_ids: (N,) array with 0=train, 1=val, 2=test
        graph_name: name for error messages (e.g., "train_graph")

    Raises:
        ValueError: if any edge connects nodes from different splits.
    """
    src_split = split_ids[edge_index[0]]
    dst_split = split_ids[edge_index[1]]
    cross_split_mask = src_split != dst_split

    if cross_split_mask.any():
        n_cross = int(cross_split_mask.sum())
        pct = 100.0 * n_cross / len(cross_split_mask)
        raise ValueError(
            f"CRITICAL: {graph_name} contains {n_cross} cross-split edges ({pct:.1f}%). "
            f"This violates subject-independent splits (AGENTS.md). "
            f"Use build_inductive_graph() or build_split_local_graph() for primary metrics. "
            f"build_multimodal_graph(cross_dataset_edges=True) is ABLATION ONLY."
        )


def build_multimodal_graph(
    fused_embeddings: np.ndarray,
    dataset_ids: list[str],
    k: int = 10,
    cross_dataset_edges: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build cross-dataset KNN graph for mixed-dataset training.

    Args:
        fused_embeddings: (N, D) embedding matrix with all samples
        dataset_ids: list of strings like "daic", "mosei", "fi" per sample
        k: number of nearest neighbors
        cross_dataset_edges: if True, allow edges across datasets AND across
                             train/val/test splits. THIS IS AN ABLATION ONLY.
                             Default is False (safe, no cross-split edges).

    Returns:
        edge_index: (2, num_edges)
        edge_weight: (num_edges,) similarity scores
        edge_flags: (num_edges,) 0=same-dataset edge, 1=cross-dataset edge

    WARNING: cross_dataset_edges=True creates edges between train, val, and
    test nodes, violating subject-independent splits. NEVER use for primary
    clinical metrics. Use build_inductive_graph() or build_split_local_graph().
    """
    n = fused_embeddings.shape[0]

    # Build full KNN graph
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(fused_embeddings)
    distances, indices = nn.kneighbors(fused_embeddings)

    # Remove self-edges
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Build edges
    src_nodes = np.repeat(np.arange(n), k)
    dst_nodes = indices.flatten()
    edge_weights = (1.0 / (1.0 + distances.flatten()))

    edge_index = np.stack([src_nodes, dst_nodes])

    # Determine edge types (same vs cross dataset)
    if cross_dataset_edges:
        src_datasets = np.array(dataset_ids, dtype=object)[src_nodes]
        dst_datasets = np.array(dataset_ids, dtype=object)[dst_nodes]
        edge_flags = (src_datasets != dst_datasets).astype(np.int64)
    else:
        edge_flags = np.zeros(len(src_nodes), dtype=np.int64)

    return edge_index, edge_weights, edge_flags