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
            subgraph_edge_index: influential edges in the subgraph (2, num_edges)
            subgraph_edge_weights: edge importance weights (num_edges,)
            shap_values: {modality: shap_value} dict
            sample_metadata: {dataset, task, subject_id, prediction, confidence}
            top_k_neighbors: number of top neighbors to describe

        Returns:
            str: human-readable narrative explanation
        """
        # Extract top-k neighbors by edge weight
        n_edges = subgraph_edge_weights.shape[0]
        if n_edges == 0:
            top_neighbors_str = "  (no strong neighbors)"
        else:
            # Sort edges by weight descending
            sorted_idx = torch.argsort(subgraph_edge_weights, descending=True)
            top_k = min(top_k_neighbors, n_edges)

            neighbor_parts = []
            for i in range(top_k):
                ei = sorted_idx[i].item()
                src = int(subgraph_edge_index[0, ei])
                dst = int(subgraph_edge_index[1, ei])
                weight = float(subgraph_edge_weights[ei])
                neighbor_parts.append(f"  - Edge {src}→{dst}: importance={weight:.4f}")

            # Infer dataset context
            ds = sample_metadata.get("dataset", "unknown")
            neighbor_parts.append(f"    (inferred context: {ds})")
            top_neighbors_str = "\n".join(neighbor_parts) if neighbor_parts else "  (no strong neighbors)"

        # Build prompt
        task = sample_metadata.get("task", "unknown")
        dataset = sample_metadata.get("dataset", "unknown")
        pred_val = sample_metadata.get("prediction", "N/A")
        confidence = float(sample_metadata.get("confidence", 0.0))

        # Create prediction label string
        if isinstance(pred_val, (int, float)):
            if task in ("depression", "daic"):
                prediction_str = f"Depression {'detected' if pred_val > 0.5 else 'not detected'} (score={pred_val:.3f})"
            else:
                prediction_str = f"Prediction value: {pred_val:.3f}"
        else:
            prediction_str = str(pred_val)

        prompt = build_graphxain_prompt(
            prediction=prediction_str,
            confidence=confidence,
            top_modalities=shap_values,
            top_neighbors=top_neighbors_str,
            task=task,
            dataset=dataset,
        )

        # Try LLM generation if model is available
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_name = self.llm_name
            if model_name == "mistral":
                model_name = "mistralai/Mistral-7B-Instruct-v0.3"

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=150, temperature=0.7, do_sample=True
                )
            narrative = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the output
            if prompt in narrative:
                narrative = narrative[len(prompt):].strip()
            return narrative
        except (ImportError, OSError, RuntimeError) as e:
            # Fallback: return template-filled prompt as explanation
            return f"[LLM unavailable ({type(e).__name__}: {e})]\n\nTemplate explanation:\n{prompt}"

    def format_subgraph_context(
        self,
        neighbor_ids: list[int],
        neighbor_distances: list[float],
        neighbor_datasets: list[str],
        top_k: int = 5,
    ) -> str:
        """Format subgraph neighborhood as a text description."""
        parts = []
        for nid, dist, ds in zip(neighbor_ids[:top_k], neighbor_distances[:top_k], neighbor_datasets[:top_k]):
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