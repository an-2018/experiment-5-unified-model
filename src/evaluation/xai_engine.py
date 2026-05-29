"""XAI engine: SHAP for feature/modality importance, perturbation tests.

XAI explanations must be validated with perturbation/counterfactual tests, not just visual appeal.
"""
import numpy as np
import torch
from typing import Optional


class SHAPExplainer:
    """SHAP-based explainer for multimodal models."""

    def __init__(self, model: torch.nn.Module, background_data: Optional[torch.Tensor] = None):
        self.model = model
        self.background_data = background_data

    def compute_modality_shap(self, sample: dict, background_size: int = 100) -> dict[str, float]:
        """Compute SHAP values for text/audio/video modality contributions.

        Returns dict: {modality: shap_value}.
        """
        raise NotImplementedError("Phase 11: Evaluation XAI Engineer will implement.")

    def compute_feature_shap(self, input_tensor: torch.Tensor, feature_names: list[str]) -> dict[str, float]:
        """Compute per-feature SHAP values."""
        raise NotImplementedError("Phase 11: Evaluation XAI Engineer will implement.")


class GNNExplainerWrapper:
    """Wrapper around PyG's GNNExplainer for graph-subgraph explanations."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def explain_node(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explain prediction for a single node.

        Returns:
            subgraph_edge_mask: importance of each edge in subgraph
            feature_mask: importance of each node feature
        """
        raise NotImplementedError("Phase 11: Evaluation XAI Engineer will implement.")

    def explain_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explain prediction for the whole graph."""
        raise NotImplementedError("Phase 11: Evaluation XAI Engineer will implement.")


def perturbation_test(
    sample: dict,
    model: torch.nn.Module,
    modality_to_remove: str,
) -> float:
    """Perturbation test: remove one modality, measure prediction change.

    Returns delta in prediction probability.
    """
    raise NotImplementedError("Phase 11: Evaluation XAI Engineer will implement.")


def counterfactual_test(
    sample: dict,
    model: torch.nn.Module,
    target_delta: float = 0.1,
) -> dict[str, float]:
    """Counterfactual test: find minimal modality changes to flip prediction.

    Returns dict of minimal changes per modality.
    """
    raise NotImplementedError("Phase 11: Evaluation XAI Engineer will implement.")