"""GraphXAI: mapping GNN subgraphs and SHAP values into LLM-generated narrative explanations.

GraphXAIN uses LLM text encoders to generate human-readable explanations from:
- Subgraph structure (influential neighbors)
- SHAP values (modality and feature importance)
- Sample metadata (dataset, task, subject context)
"""
import torch
from typing import Optional


class GraphXAINNarrator:
    """Generate narrative explanations from graph and SHAP data."""

    def __init__(self, llm_name: str = "mistral"):
        self.llm_name = llm_name

    def generate_explanation(
        self,
        subgraph_edge_index: torch.Tensor,
        subgraph_edge_weights: torch.Tensor,
        shap_values: dict[str, float],
        sample_metadata: dict,
        top_k_neighbors: int = 5,
    ) -> str:
        """Generate a narrative explanation for a sample prediction.

        Args:
            subgraph_edge_index: influential edges in the subgraph
            subgraph_edge_weights: edge importance weights
            shap_values: {modality: shap_value} dict
            sample_metadata: {dataset, task, subject_id, prediction, confidence}
            top_k_neighbors: number of top neighbors to describe

        Returns:
            str: human-readable narrative explanation
        """
        raise NotImplementedError("Phase 11: Evaluation XAI Engineer will implement.")

    def format_subgraph_context(
        self,
        neighbor_ids: list[int],
        neighbor_distances: list[float],
        neighbor_datasets: list[str],
    ) -> str:
        """Format subgraph neighborhood as a text description."""
        parts = []
        for nid, dist, ds in zip(neighbor_ids[:top_k_neighbors], neighbor_distances[:top_k_neighbors], neighbor_datasets[:top_k_neighbors]):
            parts.append(f"  - Neighbor {nid} (dataset={ds}, similarity={dist:.3f})")
        return "\n".join(parts) if parts else "  (no strong neighbors)"


def build_graphxain_prompt(
    prediction: str,
    confidence: float,
    top_modalities: dict[str, float],
    top_neighbors: str,
    task: str,
    dataset: str,
) -> str:
    """Build a prompt for LLM narrative generation."""
    template = """You are a clinical explainability assistant. A multimodal graph-gated model made the following prediction:

Task: {task}
Dataset: {dataset}
Prediction: {prediction}
Confidence: {confidence:.2f}

Key modality contributions (SHAP values):
{modality_lines}

Influential neighbors in the routing graph:
{neighbor_lines}

Write a concise, clinically grounded explanation (2-3 sentences) of why the model made this prediction.
Focus on which modalities and similar samples drove the decision.
"""
    modality_lines = "\n".join([f"  - {m}: {v:.4f}" for m, v in sorted(top_modalities.items(), key=lambda x: -x[1])])
    return template.format(
        task=task,
        dataset=dataset,
        prediction=prediction,
        confidence=confidence,
        modality_lines=modality_lines,
        neighbor_lines=top_neighbors,
    )