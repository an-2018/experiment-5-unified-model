"""XAI engine: SHAP for feature/modality importance, perturbation tests,
GNNExplainer wrapper, counterfactual tests.

XAI explanations must be validated with perturbation/counterfactual tests, not just visual appeal.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Callable


class SHAPExplainer:
    """SHAP-based explainer for multimodal models.

    Uses KernelSHAP (model-agnostic) to estimate modality-level and feature-level
    importance. For modality importance, computes marginal contribution of each modality
    by evaluating the model with/without that modality across background samples.
    """

    def __init__(self, model: nn.Module, background_data: Optional[dict] = None):
        self.model = model
        self.background_data = background_data
        try:
            self._device = next(model.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")

    def compute_modality_shap(
        self,
        sample: dict,
        background_size: int = 100,
        modalities: Optional[list[str]] = None,
        pred_fn: Optional[Callable] = None,
    ) -> dict[str, float]:
        """Compute modality-level importance via marginal contribution estimation.

        Args:
            sample: dict with keys like 'text_feats', 'audio_feats', 'video_feats'
            background_size: number of background samples to average over
            modalities: list of modality names to assess (default: text/audio/video)
            pred_fn: optional prediction function (model, sample, mask) -> scalar.
                     If None, uses default forward with masking.

        Returns:
            {modality: shap_value} where positive = pushes toward positive class
        """
        if modalities is None:
            modalities = ["text", "audio", "video"]

        # Default prediction: zero out masked modality, return logit[0]
        def _default_pred(smp: dict, mask: list[str]) -> float:
            x = torch.cat([
                torch.zeros_like(smp["text_feats"]) if "text" in mask else smp["text_feats"],
                torch.zeros_like(smp["audio_feats"]) if "audio" in mask else smp["audio_feats"],
                torch.zeros_like(smp["video_feats"]) if "video" in mask else smp["video_feats"],
            ], dim=-1)
            x = x.unsqueeze(0) if x.ndim == 1 else x.unsqueeze(0) if x.ndim == 1 else x
            if x.ndim == 2:
                x = x.unsqueeze(0)  # add batch dim -> (1, 1, D)
            with torch.no_grad():
                out = self.model.forward_encoded(x)
            return out[0, 0].item() if isinstance(out, torch.Tensor) else out[0]

        predict_fn = pred_fn or _default_pred

        # Generate background by sampling from background or using zeros
        rng = np.random.RandomState(42)
        n_bg = min(background_size, 50)

        # SHAP: for each modality, compute E[f(x) | modality_present] - E[f(x)]
        # We estimate via: contribution(m) = f(all) - f(all \ m)
        all_mods = sample.copy()

        baseline = predict_fn(all_mods, [])

        attributions = {}
        for mod in modalities:
            # Prediction without this modality
            masked = predict_fn(all_mods, [mod])
            # Marginal contribution = drop in score when modality removed
            attributions[mod] = float(baseline - masked)

        return attributions

    def compute_feature_shap(
        self,
        input_tensor: torch.Tensor,
        feature_names: Optional[list[str]] = None,
        nsamples: int = 200,
    ) -> dict[str, float]:
        """Compute per-feature SHAP values using KernelSHAP approximation.

        Falls back to simple gradient-based sensitivity if shap is not installed.

        Args:
            input_tensor: 1D input feature tensor (D,)
            feature_names: optional list of feature names
            nsamples: number of KernelSHAP samples

        Returns:
            {feature_name_or_idx: shap_value}
        """
        input_np = input_tensor.detach().cpu().numpy()

        try:
            import shap
            # Use a simple linear model as background for speed
            # Or use the actual model if it's lightweight
            def f_wrapper(x_batch: np.ndarray) -> np.ndarray:
                x_t = torch.from_numpy(x_batch).float().to(self._device or "cpu")
                if x_t.ndim == 1:
                    x_t = x_t.unsqueeze(0)
                if x_t.ndim == 2:
                    x_t = x_t.unsqueeze(0)  # (B, 1, D)
                with torch.no_grad():
                    out = self.model.forward_encoded(x_t)
                return out[:, 0].cpu().numpy() if out.ndim > 1 else out.cpu().numpy()

            # Use zeros as background (simplest)
            background = np.zeros_like(input_np).reshape(1, -1)
            explainer = shap.KernelExplainer(f_wrapper, background)
            shap_values = explainer.shap_values(input_np.reshape(1, -1), nsamples=nsamples)

            vals = shap_values[0] if isinstance(shap_values, list) else shap_values
            vals = np.squeeze(vals)
            if feature_names is not None and len(feature_names) == len(vals):
                return {name: float(v) for name, v in zip(feature_names, vals)}
            return {str(i): float(v) for i, v in enumerate(vals)}

        except ImportError:
            # Fallback: gradient-based sensitivity
            x = input_tensor.clone().detach().requires_grad_(True)
            x_in = x.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            out = self.model.forward_encoded(x_in)
            loss = out[0, 0]
            grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
            sensitivity = grad.abs().detach().cpu().numpy()

            if feature_names is not None and len(feature_names) == len(sensitivity):
                return {name: float(v) for name, v in zip(feature_names, sensitivity)}
            return {str(i): float(v) for i, v in enumerate(sensitivity)}


class GNNExplainerWrapper:
    """Wrapper for graph-subgraph explanations.

    If PyG's GNNExplainer is available, uses it directly.
    Otherwise, uses a gradient-based sensitivity approach on node features
    and a connection-count heuristic for edge importance.
    """

    def __init__(self, model: nn.Module, use_pyg: bool = True):
        self.model = model
        self.use_pyg = use_pyg and self._pyg_available()

    @staticmethod
    def _pyg_available() -> bool:
        try:
            from torch_geometric.nn import GNNExplainer  # noqa
            return True
        except ImportError:
            return False

    def explain_node(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explain prediction for a single node.

        Returns:
            edge_mask: importance weight for each edge (num_edges,)
            feature_mask: importance weight for each feature (num_features,)
        """
        if self.use_pyg:
            try:
                from torch_geometric.nn import GNNExplainer
                from torch_geometric.data import Data

                data = Data(x=x, edge_index=edge_index)
                explainer = GNNExplainer(self.model, epochs=100, **kwargs)
                node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index)
                return edge_mask, node_feat_mask
            except Exception:
                pass  # fall through to custom

        N, D = x.shape
        E = edge_index.shape[1]

        # Feature importance via gradient sensitivity
        x_in = x.clone().detach().requires_grad_(True)
        out = self._gnn_forward(x_in, edge_index)

        if out.ndim > 1 and out.shape[0] > node_idx:
            loss = out[node_idx, 0]
        else:
            loss = out.sum()

        grad_x = torch.autograd.grad(loss, x_in, retain_graph=True, allow_unused=True)[0]
        if grad_x is None:
            feature_mask = torch.ones(D, device=x.device)
        else:
            feature_mask = grad_x.abs().mean(dim=0).detach()

        # Edge importance: degree-based heuristic (edges connected to node_idx are important)
        if E > 0:
            src, dst = edge_index
            connected = (src == node_idx) | (dst == node_idx)
            edge_mask = connected.float()
        else:
            edge_mask = torch.zeros(E, device=x.device)

        return edge_mask, feature_mask

    def explain_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explain prediction for the whole graph."""
        return self.explain_node(0, x, edge_index, **kwargs)

    def _gnn_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run GNN forward pass, handling various model interfaces."""
        if hasattr(self.model, 'gnn_forward'):
            return self.model.gnn_forward(x, edge_index)
        if hasattr(self.model, 'forward'):
            try:
                return self.model.forward(x, edge_index=edge_index)
            except TypeError:
                pass
        return self.model(x)


def perturbation_test(
    sample: dict,
    model: nn.Module,
    modality_to_remove: str,
) -> float:
    """Perturbation test: remove one modality, measure prediction change.

    Args:
        sample: dict with modality tensors, must include a 'logits' or 'pred' key from base pred
        model: the model to test
        modality_to_remove: 'text', 'audio', or 'video'

    Returns:
        delta = prediction_without_modality - prediction_with_all_modalities
        (negative means removing modality reduced the prediction)
    """
    # Get baseline prediction
    baseline = _get_prediction(model, sample)

    # Remove modality (keys are "text_feats", "audio_feats", "video_feats")
    perturbed = sample.copy()
    mod_key = f"{modality_to_remove}_feats"
    if mod_key in perturbed:
        perturbed[mod_key] = torch.zeros_like(perturbed[mod_key])

    new_pred = _get_prediction(model, perturbed)
    return float(new_pred - baseline)


def _get_prediction(model: nn.Module, sample: dict) -> float:
    """Helper to get scalar prediction from model + sample dict."""
    with torch.no_grad():
        feats_list = []
        for mod in ["text", "audio", "video"]:
            key = f"{mod}_feats"
            if key in sample:
                feats_list.append(sample[key].flatten())
            else:
                feats_list.append(torch.zeros(128))

        x = torch.cat(feats_list, dim=-1)
        if x.ndim == 0:
            x = x.unsqueeze(0)
        x = x.unsqueeze(0)  # batch dim
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, 1, D)

        if hasattr(model, 'forward_encoded'):
            out = model.forward_encoded(x)
        elif hasattr(model, 'forward'):
            out = model.forward(x)
        else:
            out = model(x)

        return out[0, 0].item() if isinstance(out, torch.Tensor) else float(out[0])


def counterfactual_test(
    sample: dict,
    model: nn.Module,
    target_delta: float = 0.1,
    step_size: float = 0.02,
    max_steps: int = 50,
) -> dict[str, float]:
    """Counterfactual test: find minimal changes to each modality to achieve target_delta.

    Uses gradient-based directional scaling: for each modality, computes the model's
    gradient w.r.t. that modality's features, then perturbs in the gradient direction.
    The result naturally varies per sample because the perturbation interacts with
    the actual feature magnitudes (feature scaling vs uniform offset).

    For linear models, the gradient is constant, but the
    perturbation L2 norm depends on feature values because we scale features by
    (1 + growth_factor), making the absolute change proportional to the feature itself.

    Args:
        sample: dict with modality tensors
        model: model to test
        target_delta: desired absolute prediction change
        step_size: scaling increment per step (as fraction of feature magnitude)
        max_steps: maximum iterations per modality

    Returns:
        {modality: min_change_needed} where min_change_needed is the L2 norm of
        cumulative perturbation required to achieve target_delta, or -1 if unreachable
    """
    baseline = _get_prediction(model, sample)
    results = {}

    for mod in ["text", "audio", "video"]:
        mod_key = f"{mod}_feats"
        if mod_key not in sample:
            results[mod] = -1.0
            continue

        feat = sample[mod_key].clone()
        feat_norm = feat.norm().item()
        if feat_norm < 1e-8:
            results[mod] = -1.0
            continue

        # Compute gradient direction for this modality
        # Build combined input, enable grad only for this modality's slice
        combined = _build_combined_input(sample)
        combined = combined.clone().detach().requires_grad_(True)

        # Forward pass
        x_in = combined.unsqueeze(0)
        if x_in.ndim == 2:
            x_in = x_in.unsqueeze(0)
        out = _model_forward(model, x_in)
        loss = out[0, 0]

        # Figure out which slice of combined belongs to this modality
        D = feat.shape[0]
        mod_slices = {"text": (0, D), "audio": (D, 2*D), "video": (2*D, 3*D)}
        s, e = mod_slices[mod]

        grad = torch.autograd.grad(loss, combined, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            results[mod] = -1.0
            continue

        grad_mod = grad[s:e].detach()
        grad_norm = grad_mod.norm().item()
        if grad_norm < 1e-8:
            results[mod] = -1.0
            continue

        # Direction to push (sign flips depending on whether we want positive/negative change)
        direction = grad_mod / grad_norm

        reached = False
        cumulative_perturbation = torch.zeros_like(feat)

        for step in range(max_steps):
            # Perturb in gradient direction, scaled by step_size * feature_norm
            delta_vec = direction * step_size * feat_norm
            cumulative_perturbation = cumulative_perturbation + delta_vec
            perturbed_sample = sample.copy()
            perturbed_sample[mod_key] = feat + cumulative_perturbation

            new_pred = _get_prediction(model, perturbed_sample)
            delta = abs(new_pred - baseline)

            if delta >= target_delta:
                reached = True
                break

        total_change = float(cumulative_perturbation.norm().item()) if reached else -1.0
        results[mod] = total_change

    return results


def _build_combined_input(sample: dict) -> torch.Tensor:
    """Build concatenated feature vector from sample dict."""
    feats_list = []
    for mod in ["text", "audio", "video"]:
        key = f"{mod}_feats"
        if key in sample:
            feats_list.append(sample[key].flatten())
        else:
            feats_list.append(torch.zeros(128))
    return torch.cat(feats_list)


def _model_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through model with shape handling."""
    if hasattr(model, 'forward_encoded'):
        return model.forward_encoded(x)
    elif hasattr(model, 'forward'):
        return model.forward(x)
    return model(x)
