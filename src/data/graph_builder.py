"""Leakage-safe KNN graph construction for multimodal samples.

Rules:
- Build train/val/test KNN graphs separately (split-local graph mode) for primary results.
- Never allow cross-split edges in primary results.
- Use inductive inference for final evaluation.
- Transductive graph = ablation only, must be clearly marked.
"""
from typing import Literal, Optional
import numpy as np


def build_knn_graph(
    embeddings: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
    split: Optional[str] = None,
    mode: Literal["split-local", "inductive", "transductive"] = "split-local",
) -> tuple[np.ndarray, np.ndarray]:
    """Build KNN graph over sample embeddings.

    Returns:
        edge_index: (2, num_edges) PyG-compatible edge index
        edge_weight: (num_edges,) similarity scores
    """
    raise NotImplementedError("Phase 6: Graph MoE Architect will implement.")


def build_multimodal_graph(
    fused_embeddings: np.ndarray,
    dataset_ids: list[str],
    k: int = 10,
    cross_dataset_edges: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build cross-dataset KNN graph for mixed-dataset training."""
    raise NotImplementedError("Phase 6: Graph MoE Architect will implement.")


def validate_graph_leakage(
    train_edge_index: np.ndarray,
    val_edge_index: np.ndarray,
    test_edge_index: np.ndarray,
) -> dict[str, bool]:
    """Check that no cross-split edges exist in train/val/test graphs.

    Returns dict with keys 'val_leakage_free', 'test_leakage_free'.
    """
    raise NotImplementedError("Phase 6: Graph MoE Architect will implement.")