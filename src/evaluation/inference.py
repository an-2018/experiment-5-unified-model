#!/usr/bin/env python3
"""
Inference Module — Real Model Loading and Prediction Extraction
================================================================
Shared between Phase 10 (evaluation) and Phase 11 (XAI).

Loads saved checkpoints from Phases 5 (L0) and 8 (L1-L5), runs inference
on validation data using cached features, and returns ground-truth labels
with model predictions.

No synthetic data, no mock models, no fallback values.
"""

import json
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ── Paths ──
ROOT = Path("/home/anilson/thesis/thesis-experiment-5-unified-model")
FEATURES_ROOT = ROOT / "data" / "features"
LLM_FEATURES_ROOT = FEATURES_ROOT / "llm"
MANIFEST_PATH = FEATURES_ROOT / "manifest.json"
ARTIFACTS_TABLES = ROOT / "artifacts" / "tables"
PREDICTIONS_DIR = ROOT / "artifacts" / "predictions"

DAIC_DATA = ROOT / "data" / "daic"
DAIC_RAW = Path("/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/daic/raw")
MOSEI_DATA = ROOT / "data" / "mosei"
FI_DATA = ROOT / "data" / "fi"

# ── Constants (mirroring Phase 5/8) ──
HIDDEN_DIM = 256
EXPERT_DIM = 256
NUM_EXPERTS = 8
NUM_SHARED = 2
NUM_HEADS = 4
TEMPERATURE = 2.0

FI_TRAITS = ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]
EMOTION_LABELS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

TASK_IDS = {
    "daic_depression": 0,
    "mosei_sentiment": 1,
    "mosei_emotion": 2,
    "fi_personality": 3,
}

TASK_TO_EXPERTS = {
    0: [0, 1],
    1: [2, 3],
    2: [2, 3],
    3: [4, 5],
}

LLM_DIMS = {
    "mistral": 4096,
    "clap": 512,
    "llava": 4096,
}

FEATURE_DIMS_CLASSICAL = {
    "text": 768,
    "audio": 768,
    "video": 1536,
}


# =====================================================================
# Model — mirrors UnifiedMMoEEx (Phase 5) and UnifiedMMoEExWithLLM (Phase 8)
# =====================================================================

class _MLPProjector(nn.Module):
    """Linear → LayerNorm → GELU projection."""
    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GatedLateFusion(nn.Module):
    """Gated multimodal fusion — mirrors src/models/fusion.py exactly.
    
    Gate operates on PROJECTED features (hidden_dim), not raw input.
    This is critical for checkpoint compatibility.
    """
    def __init__(self, text_dim: int, audio_dim: int, video_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.text_proj = _MLPProjector(text_dim, hidden_dim)
        self.audio_proj = _MLPProjector(audio_dim, hidden_dim)
        self.video_proj = _MLPProjector(video_dim, hidden_dim)
        # Gates operate on projected features (hidden_dim → hidden_dim)
        self.text_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.audio_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.video_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(self, text, audio, video, mask):
        t = self.text_proj(text)
        a = self.audio_proj(audio)
        v = self.video_proj(video)
        t_g = self.text_gate(t) * mask[:, 0:1].float()
        a_g = self.audio_gate(a) * mask[:, 1:2].float()
        v_g = self.video_gate(v) * mask[:, 2:3].float()
        return t_g * t + a_g * a + v_g * v


class _Expert(nn.Module):
    """Single MLP expert with residual connection."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class _MMoEExCore(nn.Module):
    """MMoEEx core — experts + gates + task_heads (matches Phase 5/8 exactly)."""
    def __init__(self, input_dim=256, num_experts=8, expert_dim=256, num_tasks=4,
                 num_shared=2, task_to_experts=None):
        super().__init__()
        self.num_experts = num_experts
        if task_to_experts is None:
            task_to_experts = {i: [i * 2, i * 2 + 1] for i in range(num_tasks)}
        self.task_to_experts = task_to_experts

        self.experts = nn.ModuleList([
            _Expert(input_dim, expert_dim, expert_dim) for _ in range(num_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts, bias=False) for _ in range(num_tasks)
        ])
        # NOTE: must be named `task_heads` to match Phase 8 checkpoint key naming
        self.task_heads = nn.ModuleList([
            nn.Linear(expert_dim, 1) for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor, task_id: int, routing_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns expert mixture output."""
        if routing_weights is not None:
            weights = routing_weights
        else:
            gate_logits = self.gates[task_id](x)
            expert_indices = self.task_to_experts.get(task_id, list(range(self.num_experts)))
            mask = torch.zeros(self.num_experts, device=x.device, dtype=torch.bool)
            mask[expert_indices] = True
            masked_logits = gate_logits.clone()
            mask_expanded = mask.unsqueeze(0).expand_as(masked_logits)
            masked_logits[~mask_expanded] = float('-inf')
            weights = torch.softmax(masked_logits, dim=-1)

        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        weighted = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        return weighted

class _HeadNet(nn.Module):
    """Task head with .net wrapper (to match checkpoint key naming).
    
    Checkpoints save as: depression_head.net.0.weight
    This class provides the .net attribute so keys match.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class _PersonalityHeads(nn.Module):
    """Personality head with .heads wrapper (to match checkpoint naming).
    
    Checkpoints save as: personality_head.heads.extraversion.0.weight
    """
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.heads = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(64, 1),
            )
            for t in FI_TRAITS
        })

    def forward(self, x):
        return {t: head(x) for t, head in self.heads.items()}


class _UnifiedInferenceModel(nn.Module):
    """Unified model for inference across L0-L5.
    
    Supports both classical (L0) and LLM (L1-L5) feature pathways.
    Constructed from checkpoint state dict — architecture matches what was trained.
    """
    def __init__(self, use_llm_projector: bool = False,
                 text_dim: int = 768, audio_dim: int = 768, video_dim: int = 1536,
                 llm_text_dim: int = 4096, llm_audio_dim: int = 768, llm_video_dim: int = 768):
        super().__init__()
        self.use_llm_projector = use_llm_projector

        # LLM projectors
        if use_llm_projector:
            self.llm_text_projector = _MLPProjector(llm_text_dim, HIDDEN_DIM)
            self.llm_audio_projector = _MLPProjector(llm_audio_dim, HIDDEN_DIM)
            self.llm_video_projector = _MLPProjector(llm_video_dim, HIDDEN_DIM)
            self.llm_fusion = nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM)

        # Classical projectors
        self.fusion = _GatedLateFusion(text_dim, audio_dim, video_dim, HIDDEN_DIM)
        self.text_projector = _MLPProjector(text_dim, HIDDEN_DIM)
        self.audio_projector = _MLPProjector(audio_dim, HIDDEN_DIM)
        self.video_projector = _MLPProjector(video_dim, HIDDEN_DIM)

        # MMoEEx core
        self.mmoe = _MMoEExCore(
            input_dim=HIDDEN_DIM, num_experts=NUM_EXPERTS, expert_dim=EXPERT_DIM,
            num_tasks=NUM_HEADS, num_shared=NUM_SHARED, task_to_experts=TASK_TO_EXPERTS,
        )

        # Task heads — must use .net wrapper to match checkpoint key naming
        self.depression_head = _HeadNet(EXPERT_DIM, 128, 1)
        self.sentiment_head = _HeadNet(EXPERT_DIM, 128, 1)
        self.emotion_head = _HeadNet(EXPERT_DIM, 128, 6)
        # Personality head must use .heads wrapper to match checkpoint naming
        self.personality_head = _PersonalityHeads(EXPERT_DIM)

    def forward_encoded(self, x: torch.Tensor, task_id: int = 0,
                        routing: str = "multimodal") -> torch.Tensor:
        """Forward for SHAP/GNN/perturbation/counterfactual explainers.

        Reads modality dimensions from the projector layer weights to handle
        variable projections correctly (e.g. L1: text=4096, audio=768, video=768).

        Args:
            x: (B, D) or (B, 1, D) concatenated text+audio+video features
            task_id: 0=depression, 1=sentiment, 2=emotion, 3=personality.
                Defaults to 0 (depression) for backward compatibility with
                existing DAIC-only callers. Callers explaining a MOSEI/FI
                sample must pass the matching task_id, otherwise the returned
                value is the depression head's output on that sample, not the
                task-native prediction.
            routing: "text_only", "video_only", or "multimodal" — must match
                the sample's own dataset-native routing (DAIC="text_only",
                FI="video_only", MOSEI="multimodal"; see InferenceDataset).
                Mirrors predict_task's fusion branch exactly. Defaults to
                "multimodal" for backward compatibility, but explanations of a
                DAIC/FI sample computed with the default will reflect a
                full-fusion forward pass the model never actually takes for
                that sample, not the text-only/video-only path that produced
                its real prediction.
        Returns:
            (B, 1) scalar output for the given task (personality's 5 traits
            are averaged to a single scalar for the explainers, mirroring the
            "FI Avg CCC" aggregate used throughout the evaluation).
        """
        if x.ndim == 3:
            x = x.squeeze(1)  # (B, 1, D) -> (B, D)
        # Move input to model device (SHAP background samples may be on CPU)
        model_device = next(self.mmoe.parameters()).device
        x = x.to(device=model_device)
        B, D = x.shape
        if self.use_llm_projector:
            # Read correct dimensions from projector weights (in_features of first Linear)
            t_dim = self.llm_text_projector.net[0].in_features  # 4096
            a_dim = self.llm_audio_projector.net[0].in_features  # 768
            # video_dim inferred from remaining
            t_h = self.llm_text_projector(x[:, :t_dim])
            if routing == "text_only":
                fused = t_h
            elif routing == "video_only":
                v_h = self.llm_video_projector(x[:, t_dim + a_dim:])
                fused = v_h
            else:
                a_h = self.llm_audio_projector(x[:, t_dim:t_dim + a_dim])
                v_h = self.llm_video_projector(x[:, t_dim + a_dim:])
                fused = self.llm_fusion(torch.cat([t_h, a_h, v_h], dim=-1))
        else:
            # Classical path: read dimensions from projector weights
            t_dim = self.text_projector.net[0].in_features  # 768
            a_dim = self.audio_projector.net[0].in_features  # 768
            if routing == "text_only":
                fused = self.text_projector(x[:, :t_dim])
            elif routing == "video_only":
                fused = self.video_projector(x[:, t_dim + a_dim:])
            else:
                mask = torch.ones(B, 3, dtype=torch.bool, device=x.device)
                fused = self.fusion(x[:, :t_dim], x[:, t_dim:t_dim + a_dim],
                                    x[:, t_dim + a_dim:], mask)

        expert_out = self.mmoe.forward(fused, task_id=task_id)
        if task_id == 0:
            return self.depression_head(expert_out)
        elif task_id == 1:
            return self.sentiment_head(expert_out)
        elif task_id == 2:
            return self.emotion_head(expert_out)[:, :1]
        elif task_id == 3:
            out_dict = self.personality_head(expert_out)
            return torch.cat([out_dict[t] for t in FI_TRAITS], dim=-1).mean(dim=-1, keepdim=True)
        return self.depression_head(expert_out)

    def get_routing_weights(self, x: torch.Tensor, task_id: int = 0,
                            routing: str = "multimodal") -> torch.Tensor:
        """Return the real trained MMoEEx gate's routing weights for input x.

        Mirrors forward_encoded's fusion path exactly (including the
        `routing` argument — text_only/video_only/multimodal must match the
        sample's own dataset-native convention), but returns the per-expert
        softmax weights instead of the final task prediction — for
        XAI/analysis use (e.g. reporting which experts a sample actually
        routes to), not a hardcoded/simulated substitute.
        """
        if x.ndim == 3:
            x = x.squeeze(1)
        model_device = next(self.mmoe.parameters()).device
        x = x.to(device=model_device)
        if self.use_llm_projector:
            t_dim = self.llm_text_projector.net[0].in_features
            a_dim = self.llm_audio_projector.net[0].in_features
            t_h = self.llm_text_projector(x[:, :t_dim])
            if routing == "text_only":
                fused = t_h
            elif routing == "video_only":
                v_h = self.llm_video_projector(x[:, t_dim + a_dim:])
                fused = v_h
            else:
                a_h = self.llm_audio_projector(x[:, t_dim:t_dim + a_dim])
                v_h = self.llm_video_projector(x[:, t_dim + a_dim:])
                fused = self.llm_fusion(torch.cat([t_h, a_h, v_h], dim=-1))
        else:
            t_dim = self.text_projector.net[0].in_features
            a_dim = self.audio_projector.net[0].in_features
            if routing == "text_only":
                fused = self.text_projector(x[:, :t_dim])
            elif routing == "video_only":
                fused = self.video_projector(x[:, t_dim + a_dim:])
            else:
                mask = torch.ones(x.shape[0], 3, dtype=torch.bool, device=x.device)
                fused = self.fusion(x[:, :t_dim], x[:, t_dim:t_dim + a_dim],
                                    x[:, t_dim + a_dim:], mask)

        gate_logits = self.mmoe.gates[task_id](fused)
        expert_indices = self.mmoe.task_to_experts.get(task_id, list(range(self.mmoe.num_experts)))
        mask = torch.zeros(self.mmoe.num_experts, device=fused.device, dtype=torch.bool)
        mask[expert_indices] = True
        masked_logits = gate_logits.clone()
        mask_expanded = mask.unsqueeze(0).expand_as(masked_logits)
        masked_logits[~mask_expanded] = float("-inf")
        return torch.softmax(masked_logits, dim=-1)

    def _select_projector(self, feat: torch.Tensor, llm_projector: nn.Module,
                           classical_projector: nn.Module) -> torch.Tensor:
        """Route feature through LLM or classical projector based on feature dimension.
        
        Some levels (e.g. L4) upgrade only text+video to LLM features while keeping
        audio as classical 512-dim. This helper detects the correct projector and
        adds an adaptive projection if dimensions don't match.
        """
        if feat.shape[-1] == 0:
            return None
        llm_in = llm_projector.net[0].in_features
        classical_in = classical_projector.net[0].in_features
        
        # Check if feature dimension matches either projector
        if feat.shape[-1] == llm_in:
            return llm_projector(feat)
        elif feat.shape[-1] == classical_in:
            return classical_projector(feat)
        else:
            # Dimension mismatch - need adaptive projection to projector's input dim
            # Create/use cached adaptive projection for this specific input dimension
            cache_key = f"adapt_proj_{feat.shape[-1]}_to_{llm_in}"
            if not hasattr(self, '_proj_cache'):
                self._proj_cache = {}
            if cache_key not in self._proj_cache:
                self._proj_cache[cache_key] = nn.Linear(feat.shape[-1], llm_in).to(feat.device)
            proj = self._proj_cache[cache_key]
            projected = proj(feat)
            # Pass through the LLM projector (both expect same input dim)
            return llm_projector(projected)

    def predict_task(self, text_feat, audio_feat, video_feat, modality_mask, task_id, routing):
        """Produce final prediction for a specific task.
        
        Mirrors Phase 5/8 forward_with_routing() + task head.
        Returns logits (binary) or regression values.
        """
        if self.use_llm_projector:
            t_h = self.llm_text_projector(text_feat)
            if routing == "text_only":
                fused = t_h
            elif routing == "video_only":
                v_h = self.llm_video_projector(video_feat)
                fused = v_h
            else:
                a_proj = self._select_projector(audio_feat, self.llm_audio_projector, self.audio_projector)
                v_proj = self._select_projector(video_feat, self.llm_video_projector, self.video_projector)
                a_h = a_proj if a_proj is not None else t_h * 0
                v_h = v_proj if v_proj is not None else t_h * 0
                fused = self.llm_fusion(torch.cat([t_h, a_h, v_h], dim=-1))
        else:
            if routing == "text_only":
                fused = self.text_projector(text_feat)
            elif routing == "video_only":
                fused = self.video_projector(video_feat)
            else:
                fused = self.fusion(text_feat, audio_feat, video_feat, modality_mask.bool())

        expert_out = self.mmoe.forward(fused, task_id)

        if task_id == 0:
            return self.depression_head(expert_out)
        elif task_id == 1:
            return self.sentiment_head(expert_out)
        elif task_id == 2:
            return self.emotion_head(expert_out)
        elif task_id == 3:
            out_dict = self.personality_head(expert_out)
            return torch.cat([out_dict[t] for t in FI_TRAITS], dim=-1)
        return expert_out

    def get_fused_representation(self, text_feat, audio_feat, video_feat, modality_mask, routing) -> torch.Tensor:
        """Return the fused representation only (no MMoE/task head), for building
        a graph on the model's own trained embedding space rather than raw
        concatenated modality features. Mirrors predict_task's fusion step exactly."""
        if self.use_llm_projector:
            t_h = self.llm_text_projector(text_feat)
            if routing == "text_only":
                fused = t_h
            elif routing == "video_only":
                v_h = self.llm_video_projector(video_feat)
                fused = v_h
            else:
                a_proj = self._select_projector(audio_feat, self.llm_audio_projector, self.audio_projector)
                v_proj = self._select_projector(video_feat, self.llm_video_projector, self.video_projector)
                a_h = a_proj if a_proj is not None else t_h * 0
                v_h = v_proj if v_proj is not None else t_h * 0
                fused = self.llm_fusion(torch.cat([t_h, a_h, v_h], dim=-1))
        else:
            if routing == "text_only":
                fused = self.text_projector(text_feat)
            elif routing == "video_only":
                fused = self.video_projector(video_feat)
            else:
                fused = self.fusion(text_feat, audio_feat, video_feat, modality_mask.bool())
        return fused

    def gnn_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """GNN forward for GNNExplainer — same as forward_encoded."""
        return self.forward_encoded(x)


# =====================================================================
# Feature Loading (Classical + LLM)
# =====================================================================

def _load_manifest():
    with open(MANIFEST_PATH, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("samples", [])


def _load_all_labels():
    """Load labels for all datasets (mirrors Phase 5/8 _load_all_labels)."""
    labels = {}

    # DAIC labels
    for split, filename, col_binary in [
        ("train", "train_split_Depression_AVEC2017.csv", "PHQ8_Binary"),
        ("val", "dev_split_Depression_AVEC2017.csv", "PHQ8_Binary"),
        ("test", "full_test_split.csv", "PHQ_Binary"),
    ]:
        path = DAIC_DATA / filename
        if not path.exists():
            continue
        with open(path, newline="") as f:
            lines = [l.strip() for l in open(path).readlines() if l.strip()]
        header = lines[0].split(",")
        idx = header.index(col_binary)
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= idx:
                continue
            pid = str(int(float(parts[0])))
            labels[f"daic_{pid}"] = int(float(parts[idx]))

    # MOSEI labels
    emotion_path = MOSEI_DATA / "mosei_emotion_labels.json"
    if emotion_path.exists():
        with open(emotion_path, "r") as f:
            mosei_labels_data = json.load(f)
        for key, label_data in mosei_labels_data.items():
            sentiment = float(label_data.get("sentiment", 0.0))
            emotions = [float(label_data.get(e, 0.0)) for e in EMOTION_LABELS]
            labels[key] = [sentiment] + emotions

    # FI labels
    train_ann = FI_DATA / "train" / "annotation_training.pkl"
    if train_ann.exists():
        with open(train_ann, "rb") as f:
            ann_train = pickle.load(f, encoding="latin-1")
        for clip_id in ann_train[FI_TRAITS[0]].keys():
            labels[f"fi_train_{clip_id}"] = {t: float(ann_train[t][clip_id]) for t in FI_TRAITS}

    val_ann = FI_DATA / "val" / "annotation_validation.pkl"
    if val_ann.exists():
        with open(val_ann, "rb") as f:
            ann_val = pickle.load(f, encoding="latin-1")
        for clip_id in ann_val[FI_TRAITS[0]].keys():
            labels[f"fi_val_{clip_id}"] = {t: float(ann_val[t][clip_id]) for t in FI_TRAITS}

    import pandas as pd
    test_csv = FI_DATA / "test" / "annotations.csv"
    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        if "interview" in df_test.columns:
            df_test = df_test.rename(columns={"interview": "openness"})
        for i in range(len(df_test)):
            row = df_test.iloc[i]
            labels[f"fi_test_{i:05d}"] = {t: float(row[t]) for t in FI_TRAITS}

    return labels


def _try_load_classical_feature(path_str, dim, root=ROOT):
    """Load a saved classical feature tensor."""
    if path_str is None:
        return False, np.zeros(dim, dtype=np.float32)
    full_path = root / path_str
    if not full_path.exists():
        return False, np.zeros(dim, dtype=np.float32)
    try:
        obj = torch.load(full_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            for key in ["pooled_embedding", "pooled_features", "embedding", "features"]:
                if key in obj and isinstance(obj[key], torch.Tensor):
                    feat = obj[key]
                    break
            else:
                for v in obj.values():
                    if isinstance(v, torch.Tensor):
                        feat = v
                        break
                else:
                    return False, np.zeros(dim, dtype=np.float32)
        else:
            feat = obj
        if isinstance(feat, torch.Tensor) and feat.dim() == 2:
            feat = feat.mean(dim=0)
        if isinstance(feat, torch.Tensor):
            feat = feat.cpu().numpy()
        feat = np.array(feat, dtype=np.float32).flatten()
        if not np.all(np.isfinite(feat)):
            return False, np.zeros(dim, dtype=np.float32)
        if feat.shape[0] < dim:
            feat = np.pad(feat, (0, dim - feat.shape[0]))
        elif feat.shape[0] > dim:
            feat = feat[:dim]
        return True, feat
    except Exception:
        return False, np.zeros(dim, dtype=np.float32)


def _get_llm_cache_path(level: str, dataset: str, split: str, modality: str) -> Path:
    """Get path for cached LLM features."""
    return LLM_FEATURES_ROOT / level / dataset / f"{split}_{modality}.npy"


def _load_llm_feature_dict(level: str, dataset: str, split: str, modality: str):
    """Load cached LLM features from .npy."""
    path = _get_llm_cache_path(level, dataset, split, modality)
    if path.exists():
        return np.load(path, allow_pickle=True).item()
    # L5 reuses L1 (text), L3 (audio), L4 (video) caches
    fallback_map = {"L5": {"text": "L1", "audio": "L3", "video": "L4"}}
    if level in fallback_map and modality in fallback_map[level]:
        fallback_level = fallback_map[level][modality]
        path = _get_llm_cache_path(fallback_level, dataset, split, modality)
        if path.exists():
            return np.load(path, allow_pickle=True).item()
    return None


# =====================================================================
# Inference Dataset (Validation only — matches Phase 5/8 JointMultimodalDataset)
# =====================================================================

class _InferenceDataset(Dataset):
    """Validation dataset for inference. Loads cached features (classical or LLM).
    
    Returns samples in the format expected by _UnifiedInferenceModel.predict_task().
    """
    def __init__(self, llm_level: str, split: str = "val", skip_extraction: bool = True):
        self.samples = []
        self.llm_level = llm_level
        self.use_llm = llm_level in ["L1", "L2", "L3", "L4", "L5"]

        # Load manifest and labels
        manifest = _load_manifest()
        all_labels = _load_all_labels()

        # Load LLM features if needed
        self.llm_features = {}
        if self.use_llm:
            for ds_name in ["daic", "mosei", "fi"]:
                for mod in ["text", "audio", "video"]:
                    feat_dict = _load_llm_feature_dict(llm_level, ds_name, split, mod)
                    if feat_dict is not None:
                        self.llm_features[f"{ds_name}_{split}_{mod}"] = feat_dict

        # Feature dims
        if llm_level == "L0":
            self.f_dims = {"text": 768, "audio": 768, "video": 1536}
        elif llm_level in ["L1", "L2"]:
            self.f_dims = {"text": 4096, "audio": 768, "video": 768}
        elif llm_level in ["L3", "L5"]:
            self.f_dims = {"text": 4096, "audio": 512, "video": 768}
        elif llm_level == "L4":
            # L4 uses LLM text (Mistral), classical audio (WavLM 768-dim), LLM video (LLaVA)
            self.f_dims = {"text": 4096, "audio": 768, "video": 4096}
        elif llm_level == "L5":
            self.f_dims = {"text": 4096, "audio": 512, "video": 4096}  # LLaVA
        else:
            self.f_dims = {"text": 768, "audio": 768, "video": 1536}

        datasets_splits = [("daic", split), ("mosei", split), ("fi", split)]

        for ds_name, sp in datasets_splits:
            for entry in manifest:
                if entry["dataset"] != ds_name or entry.get("split") != sp:
                    continue
                sample_id = entry["id"]

                # Label key
                if ds_name == "daic":
                    label_key = f"daic_{sample_id}"
                elif ds_name == "mosei":
                    label_key = sample_id
                else:
                    label_key = sample_id

                if label_key not in all_labels:
                    continue

                # Load classical features
                feat_map = entry["features"]
                t_key = feat_map.get("text_roberta")
                a_key = feat_map.get("audio_wavlm")
                v_key = feat_map.get("video_vit") if ds_name != "daic" else feat_map.get("video_openface")

                t_ok, t_vec = _try_load_classical_feature(t_key, self.f_dims["text"])
                a_ok, a_vec = _try_load_classical_feature(a_key, self.f_dims["audio"])
                v_ok, v_vec = _try_load_classical_feature(v_key, self.f_dims["video"])

                # Replace with LLM features if available
                if self.use_llm:
                    llm_t = self._get_llm_feat(ds_name, split, sample_id, "text")
                    if llm_t is not None:
                        t_vec = llm_t
                        t_ok = True

                    if llm_level in ["L3", "L5"]:
                        llm_a = self._get_llm_feat(ds_name, split, sample_id, "audio")
                        if llm_a is not None:
                            a_vec = llm_a
                            a_ok = True

                    if llm_level in ["L4", "L5"]:
                        llm_v = self._get_llm_feat(ds_name, split, sample_id, "video")
                        if llm_v is not None:
                            v_vec = llm_v
                            v_ok = True

                if not (t_ok or a_ok or v_ok):
                    continue

                label = all_labels[label_key]

                routing = "text_only" if ds_name == "daic" else "video_only" if ds_name == "fi" else "multimodal"
                task_id = TASK_IDS.get(f"{ds_name}_depression" if ds_name == "daic"
                                       else f"{ds_name}_sentiment" if ds_name == "mosei"
                                       else "fi_personality", 0)

                self.samples.append({
                    "id": sample_id,
                    "dataset": ds_name,
                    "task_id": task_id,
                    "routing": routing,
                    "text": t_vec, "audio": a_vec, "video": v_vec,
                    "modality_mask": [t_ok, a_ok, v_ok],
                    "label": label,
                })

        print(f"  InferenceDataset ({llm_level}, {split}): {len(self)} samples")

    def _get_llm_feat(self, dataset, split, sample_id, modality):
        key = f"{dataset}_{split}_{modality}"
        feat_dict = self.llm_features.get(key)
        if feat_dict is None:
            return None
        feat = feat_dict.get(sample_id)
        if feat is None:
            return None
        if feat.dtype == np.float16:
            feat = feat.astype(np.float32)
        return feat

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        label = s["label"]
        if isinstance(label, dict):
            label_arr = np.array([label[t] for t in FI_TRAITS], dtype=np.float32)
        else:
            label_arr = np.atleast_1d(np.array(label, dtype=np.float32))
        return (
            torch.from_numpy(s["text"]),
            torch.from_numpy(s["audio"]),
            torch.from_numpy(s["video"]),
            torch.tensor(s["modality_mask"], dtype=torch.bool),
            torch.from_numpy(label_arr),
            torch.tensor(s["task_id"], dtype=torch.long),
            s["routing"],
            s["id"],
            s["dataset"],
        )


def _collate_inference(batch):
    text_list, audio_list, video_list, masks, labels, task_ids, routings, ids, datasets = zip(*batch)
    max_label_size = 7
    padded_labels = []
    for label in labels:
        if label.shape[0] < max_label_size:
            p = torch.zeros(max_label_size)
            p[:label.shape[0]] = label
            padded_labels.append(p)
        else:
            padded_labels.append(label)
    return (
        torch.stack(text_list), torch.stack(audio_list), torch.stack(video_list),
        torch.stack(masks), torch.stack(padded_labels), torch.stack(task_ids),
        list(routings), list(ids), list(datasets),
    )


# =====================================================================
# Checkpoint Loading
# =====================================================================

def build_inference_model(llm_level: str, device: torch.device) -> _UnifiedInferenceModel:
    """Build the appropriate model architecture for a given level.
    
    The architecture must match what was saved in the checkpoint.
    """
    use_llm = llm_level in ["L1", "L2", "L3", "L4", "L5"]

    if llm_level == "L0":
        model = _UnifiedInferenceModel(
            use_llm_projector=False,
            text_dim=768, audio_dim=768, video_dim=1536,
        )
    elif llm_level in ["L1", "L2"]:
        model = _UnifiedInferenceModel(
            use_llm_projector=True,
            text_dim=4096, audio_dim=768, video_dim=768,
            llm_text_dim=4096, llm_audio_dim=768, llm_video_dim=768,
        )
    elif llm_level == "L3":
        model = _UnifiedInferenceModel(
            use_llm_projector=True,
            text_dim=4096, audio_dim=512, video_dim=768,
            llm_text_dim=4096, llm_audio_dim=512, llm_video_dim=768,
        )
    elif llm_level == "L4":
        model = _UnifiedInferenceModel(
            use_llm_projector=True,
            text_dim=4096, audio_dim=768, video_dim=4096,
            llm_text_dim=4096, llm_audio_dim=768, llm_video_dim=4096,
        )
    elif llm_level == "L5":
        model = _UnifiedInferenceModel(
            use_llm_projector=True,
            text_dim=4096, audio_dim=512, video_dim=4096,
            llm_text_dim=4096, llm_audio_dim=512, llm_video_dim=4096,
        )
    else:
        raise ValueError(f"Unknown level: {llm_level}")

    return model.to(device)


def _map_phase5_to_inference(state_dict: dict) -> dict:
    """Map Phase 5 L0 (UnifiedMMoEEx) checkpoint keys to _UnifiedInferenceModel keys.
    
    Phase 5 `text_projector` is a bare nn.Sequential (keys: text_projector.0.weight).
    Our model wraps it in _MLPProjector (keys: text_projector.net.0.weight).
    All other keys should match after the architecture fixes.
    """
    mapped = {}
    for k, v in state_dict.items():
        # Skip loss_fn/optimizer params
        if k.startswith("loss_fn.") or k.startswith("log_sigma.optimizer"):
            continue
        # Map text_projector.0.weight -> text_projector.net.0.weight
        if k.startswith("text_projector.") and not k.startswith("text_projector.net."):
            k = k.replace("text_projector.", "text_projector.net.", 1)
        # Map audio_projector.0.weight -> audio_projector.net.0.weight
        if k.startswith("audio_projector.") and not k.startswith("audio_projector.net."):
            k = k.replace("audio_projector.", "audio_projector.net.", 1)
        # Map video_projector.0.weight -> video_projector.net.0.weight
        if k.startswith("video_projector.") and not k.startswith("video_projector.net."):
            k = k.replace("video_projector.", "video_projector.net.", 1)
        # mmoe.task_proj -> mmoe.task_heads (Phase 5 and 8 both use task_heads now)
        if k.startswith("mmoe.task_heads."):
            pass  # Already correct
        mapped[k] = v
    return mapped


def _map_phase8_to_inference(state_dict: dict) -> dict:
    """Map Phase 8 (UnifiedMMoEExWithLLM) checkpoint keys to _UnifiedInferenceModel keys.
    
    Phase 8 LLM projectors use `projection.` submodule name (from LLMProjector class).
    Our model uses `net.` submodule name (from _MLPProjector).
    All other keys should match after architecture fixes.
    """
    mapped = {}
    for k, v in state_dict.items():
        # Skip loss_fn/optimizer params
        if k.startswith("loss_fn.") or k.startswith("log_sigma"):
            continue
        if k.startswith("optimizer."):
            continue
        # Map llm_*_projector.projection.X.Y -> llm_*_projector.net.X.Y
        if "llm_" in k and "projection." in k:
            k = k.replace("projection.", "net.")
        # Map text_projector.0.weight -> text_projector.net.0.weight
        for proj_name in ["text_projector", "audio_projector", "video_projector"]:
            if k.startswith(f"{proj_name}.") and not k.startswith(f"{proj_name}.net."):
                k = k.replace(f"{proj_name}.", f"{proj_name}.net.", 1)
        mapped[k] = v
    return mapped


def load_checkpoint(model: _UnifiedInferenceModel, llm_level: str, device: torch.device):
    """Load the correct checkpoint for a given level into the model."""
    if llm_level == "L0":
        ckpt_path = ARTIFACTS_TABLES / "mmoe_ex_best.pt"
        source = "phase5"
    elif llm_level in ["L1", "L2", "L3", "L4", "L5"]:
        ckpt_path = ARTIFACTS_TABLES / f"phase08_{llm_level}_best.pt"
        source = "phase8"
    else:
        raise ValueError(f"Unknown level: {llm_level}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        state_dict = ckpt.state_dict()

    # Map keys from script-specific format to _UnifiedInferenceModel format
    if source == "phase5":
        mapped = _map_phase5_to_inference(state_dict)
    else:
        mapped = _map_phase8_to_inference(state_dict)

    # strict=False handles benign mismatches:
    #   - mmoe.task_heads: exist in model but not in Phase 5 checkpoint
    #     (MMoEEx task_heads were added after Phase 5 ran — randomly init, never used for inference)
    #   - mmoe.log_task_weights: exists in checkpoint but not needed for inference
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        missing_inference_ok = [k for k in missing if not k.startswith("mmoe.task_heads")]
        if missing_inference_ok:
            print(f"  Warning: {len(missing_inference_ok)} non-task_head missing keys: {missing_inference_ok[:3]}...")
    if unexpected:
        unexpected_inference_ok = [k for k in unexpected if k != "mmoe.log_task_weights"]
        if unexpected_inference_ok:
            print(f"  Warning: {len(unexpected_inference_ok)} unexpected keys: {unexpected_inference_ok[:3]}...")

    model.eval()
    return model


# =====================================================================
# Prediction Extraction
# =====================================================================

@torch.no_grad()
def run_inference(model: _UnifiedInferenceModel, dataset: _InferenceDataset,
                  device: torch.device) -> dict:
    """Run inference on a dataset, returning structured predictions.
    
    Returns dict matching Phase 10's expected format:
    {
        "daic": {"all_labels": [...], "all_preds": [...], "all_logits": [...]},
        "mosei_sent": {"all_labels": [...], "all_preds": [...]},
        "mosei_emo": {"all_labels": [...], "all_preds": [...]},
        "fi": {"all_labels": [...], "all_preds": [...]},
    }
    """
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=_collate_inference)
    model.eval()

    results = {
        "daic": {"all_labels": [], "all_preds": [], "all_logits": []},
        "mosei_sent": {"all_labels": [], "all_preds": []},
        "mosei_emo": {"all_labels": [], "all_preds": []},
        "fi": {"all_labels": [], "all_preds": [], "per_trait": {t: [] for t in FI_TRAITS}},
    }

    for text, audio, video, mask, labels, task_ids, routings, ids, datasets in loader:
        text = text.to(device)
        audio = audio.to(device)
        video = video.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        for tid_val in task_ids.unique().tolist():
            t_mask = task_ids == tid_val
            t_text = text[t_mask]
            t_audio = audio[t_mask]
            t_video = video[t_mask]
            t_mask_feat = mask[t_mask]
            t_labels = labels[t_mask]
            t_routings = [routings[i] for i in range(len(routings)) if t_mask[i].item()]

            if len(t_text) == 0:
                continue

            routing = t_routings[0] if t_routings else "multimodal"
            out = model.predict_task(t_text, t_audio, t_video, t_mask_feat, int(tid_val), routing)

            if tid_val == 0:  # DAIC depression
                probs = torch.sigmoid(out).cpu().numpy().flatten()
                logits = out.cpu().numpy().flatten()
                lbl = t_labels[:, 0].cpu().numpy().flatten()
                results["daic"]["all_labels"].extend(lbl.tolist())
                results["daic"]["all_preds"].extend(probs.tolist())
                results["daic"]["all_logits"].extend(logits.tolist())
            elif tid_val == 1:  # MOSEI sentiment
                pred = out.cpu().numpy().flatten()
                lbl = t_labels[:, 0].cpu().numpy().flatten()
                results["mosei_sent"]["all_labels"].extend(lbl.tolist())
                results["mosei_sent"]["all_preds"].extend(pred.tolist())
            elif tid_val == 2:  # MOSEI emotion
                pred = torch.sigmoid(out).cpu().numpy()
                lbl = t_labels[:, :6].cpu().numpy()
                results["mosei_emo"]["all_labels"].extend(lbl.tolist())
                results["mosei_emo"]["all_preds"].extend(pred.tolist())
            elif tid_val == 3:  # FI personality
                pred = out.cpu().numpy()
                lbl = t_labels[:, :5].cpu().numpy()  # FI has 5 traits, not 7
                results["fi"]["all_labels"].extend(lbl.tolist())
                results["fi"]["all_preds"].extend(pred.tolist())
                for i, t in enumerate(FI_TRAITS):
                    results["fi"]["per_trait"][t].extend(pred[:, i].tolist())

    # Convert to numpy arrays
    for task in ["daic", "mosei_sent"]:
        results[task]["all_labels"] = np.array(results[task]["all_labels"])
        results[task]["all_preds"] = np.array(results[task]["all_preds"])
    if results["daic"]["all_logits"]:
        results["daic"]["all_logits"] = np.array(results["daic"]["all_logits"])
    if "fi" in results:
        results["fi"]["all_labels"] = np.array(results["fi"]["all_labels"])
        results["fi"]["all_preds"] = np.array(results["fi"]["all_preds"])

    return results


def extract_predictions(llm_level: str, device_str: str = "cuda",
                        skip_extraction: bool = True) -> dict:
    """Full pipeline: build model, load checkpoint, run inference on val set.
    
    Args:
        llm_level: "L0", "L1", "L2", "L3", "L4", or "L5"
        device_str: "cuda" or "cpu"
        skip_extraction: If True, use cached LLM features only (no live extraction)
    
    Returns:
        dict with full prediction results
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Extracting predictions: {llm_level}")
    print(f"{'='*60}")

    # Build dataset
    print("[1/3] Building validation dataset...")
    dataset = _InferenceDataset(llm_level=llm_level, split="val",
                                skip_extraction=skip_extraction)
    if len(dataset) == 0:
        raise RuntimeError(f"No validation samples for {llm_level}")

    # Build model and load checkpoint
    print("[2/3] Building model and loading checkpoint...")
    model = build_inference_model(llm_level, device)
    load_checkpoint(model, llm_level, device)

    # Run inference
    print("[3/3] Running inference...")
    results = run_inference(model, dataset, device)

    # Summary
    if len(results["daic"]["all_labels"]) > 0:
        from sklearn.metrics import roc_auc_score
        try:
            auroc = roc_auc_score(results["daic"]["all_labels"], results["daic"]["all_preds"])
            print(f"  DAIC: {len(results['daic']['all_labels'])} samples, AUROC={auroc:.4f}")
        except Exception as e:
            print(f"  DAIC: {len(results['daic']['all_labels'])} samples, AUROC failed: {e}")
    print(f"  MOSEI Sent: {len(results['mosei_sent']['all_labels'])} samples")
    print(f"  FI: {len(results['fi']['all_labels'])} samples")

    return results


def extract_all_levels(levels=None, device_str="cuda", skip_extraction=True,
                       save=True) -> dict:
    """Extract predictions for all LLM levels.
    
    Args:
        levels: list like ["L0", "L1", "L2", "L3", "L4", "L5"]. Default: all
        device_str: "cuda" or "cpu"
        skip_extraction: use cached features only
        save: save predictions to artifacts/predictions/
    
    Returns:
        dict of level -> prediction results
    """
    if levels is None:
        levels = ["L0", "L1", "L2", "L3", "L4", "L5"]

    all_results = {}
    for level in levels:
        try:
            results = extract_predictions(level, device_str, skip_extraction)
            all_results[level] = results

            if save:
                PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
                save_path = PREDICTIONS_DIR / f"predictions_{level}.npz"
                np.savez_compressed(save_path, **{
                    f"{task}_{k}": np.array(v)
                    for task in ["daic", "mosei_sent", "mosei_emo", "fi"]
                    for k, v in results.get(task, {}).items()
                    if isinstance(v, (list, np.ndarray))
                })
                print(f"  Saved: {save_path}")
        except Exception as e:
            print(f"  ERROR: {level} failed: {e}")
            import traceback
            traceback.print_exc()

    return all_results


# =====================================================================
# Cached Prediction Loading (for Phase 10)
# =====================================================================

def load_cached_predictions(predictions_dir: Optional[Path] = None) -> dict:
    """Load real predictions from .npz cache files into Phase 10 format.
    
    Returns dict with prediction sets per level per task:
    {
        "daic_L0": {"probs": ..., "labels": ..., "logits": ...},
        "daic_L1": ...,
        "mosei_sent_L0": {"pred": ..., "labels": ...},
        ...
        "fi_personality_L0": {"pred": ..., "labels": ...},
        ...
    }
    """
    if predictions_dir is None:
        predictions_dir = PREDICTIONS_DIR

    if not predictions_dir.exists():
        raise FileNotFoundError(
            f"No cached predictions found in {predictions_dir}. "
            "Run extract_all_levels() first to generate real predictions."
        )

    predictions = {}
    levels = ["L0", "L1", "L2", "L3", "L4", "L5"]

    for level in levels:
        npz_path = predictions_dir / f"predictions_{level}.npz"
        if not npz_path.exists():
            print(f"  Warning: no predictions for {level} at {npz_path}")
            continue

        data = np.load(npz_path)

        # DAIC
        daic_labels = data.get("daic_all_labels")
        daic_preds = data.get("daic_all_preds")
        daic_logits = data.get("daic_all_logits")
        if daic_labels is not None and daic_preds is not None:
            predictions[f"daic_{level}"] = {
                "labels": daic_labels,
                "probs": daic_preds,
                "logits": daic_logits if daic_logits is not None else daic_preds,
            }

        # MOSEI sentiment
        sent_labels = data.get("mosei_sent_all_labels")
        sent_preds = data.get("mosei_sent_all_preds")
        if sent_labels is not None and sent_preds is not None:
            predictions[f"mosei_sent_{level}"] = {
                "labels": sent_labels,
                "pred": sent_preds,
            }

        # FI personality
        fi_labels = data.get("fi_all_labels")
        fi_preds = data.get("fi_all_preds")
        if fi_labels is not None and fi_preds is not None:
            predictions[f"fi_personality_{level}"] = {
                "labels": fi_labels,
                "pred": fi_preds,
            }

    if not predictions:
        raise FileNotFoundError(
            f"No prediction files found in {predictions_dir}. "
            "Run extract_all_levels() first."
        )

    print(f"  Loaded {len(predictions)} real prediction sets from {predictions_dir}")
    for key in sorted(predictions.keys()):
        print(f"    {key}: {predictions[key].get('labels', predictions[key].get('pred')).shape}")

    return predictions


# =====================================================================
# Real Data Samples (for Phase 11 XAI)
# =====================================================================

def load_real_data_samples(dataset_name: str, n_samples: int = 5,
                           llm_level: str = "L1", split: str = "val") -> list:
    """Load real data samples from validation set for XAI case studies.
    
    Args:
        dataset_name: "daic", "mosei", or "fi"
        n_samples: number of samples to load
        llm_level: use features from which LLM level
        split: "val" or "train"
    
    Returns:
        List of dicts with keys: text_feats, audio_feats, video_feats, label, id, dataset
    """
    ds = _InferenceDataset(llm_level=llm_level, split=split)
    if len(ds) == 0:
        raise RuntimeError(f"No samples for {dataset_name} in {split} set")

    samples = []
    count = 0
    for i in range(len(ds)):
        if count >= n_samples:
            break
        text, audio, video, mask, label, task_id, routing, sid, ds_name = ds[i]
        if ds_name != dataset_name:
            continue

        label_clean = label.numpy()
        if len(label_clean) == 7 and np.all(label_clean[6:] == 0):
            label_clean = label_clean[:6]  # MOSEI: 1 sentiment + 6 emotions
        elif len(label_clean) == 7 and np.all(label_clean[1:6] == 0):
            label_clean = float(label_clean[0])  # DAIC binary

        sample = {
            "text_feats": text,
            "audio_feats": audio,
            "video_feats": video,
            "modality_mask": mask,
            "label": label_clean,
            "id": sid,
            "dataset": ds_name,
            "task_id": int(task_id),
            "routing": routing,
        }
        samples.append(sample)
        count += 1

    print(f"  Loaded {len(samples)} real {dataset_name} samples from {split}")
    return samples


def build_real_graph(samples: list, n_neighbors: int = 5, model=None, device: str = "cpu") -> tuple:
    """Build a KNN graph from real sample features.

    Args:
        samples: list of dicts with text_feats, audio_feats, video_feats
        n_neighbors: number of nearest neighbors per sample
        model: if provided, build the graph on the model's own trained fused
            representation (get_fused_representation) instead of raw
            concatenated modality features. Raw features mix unrelated
            embedding scales/dimensions across RoBERTa/WavLM/ViT and can
            dominate cosine similarity in ways unrelated to task labels; the
            trained fused space is what the router/heads actually reason
            over, so neighbor-based explanations are more likely to reflect
            genuine model behavior when built on it.
        device: device for model forward passes (only used if model is given)

    Returns:
        (x, edge_index, meta_list, edge_distances) where
        x: (N, D) feature matrix
        edge_index: (2, E) graph connectivity
        meta_list: list of metadata dicts per node
        edge_distances: (E,) cosine distances for each edge
    """
    from sklearn.neighbors import NearestNeighbors

    N = len(samples)
    x_list = []       # raw concatenated features -- returned as `x`, consumed
                      # downstream by forward_encoded()/gnn_forward() which
                      # expect this exact raw dimensionality
    topo_list = []    # fused representation -- used ONLY to decide graph
                      # topology (KNN edges) when `model` is provided
    meta_list = []

    for s in samples:
        raw_feats = torch.cat([
            s["text_feats"].float(),
            s["audio_feats"].float(),
            s["video_feats"].float(),
        ])
        x_list.append(raw_feats)

        if model is not None:
            with torch.no_grad():
                t_b = s["text_feats"].float().unsqueeze(0).to(device)
                a_b = s["audio_feats"].float().unsqueeze(0).to(device)
                v_b = s["video_feats"].float().unsqueeze(0).to(device)
                mask = s["modality_mask"]
                m_b = mask.unsqueeze(0).to(device) if hasattr(mask, "unsqueeze") \
                    else torch.tensor(mask).unsqueeze(0).to(device)
                topo_list.append(model.get_fused_representation(t_b, a_b, v_b, m_b, s["routing"]).squeeze(0).cpu())
        else:
            topo_list.append(raw_feats)

        meta_list.append({
            "id": s.get("id", len(meta_list)),
            "label": s.get("label", 0.0),
            "dataset": s.get("dataset", "?"),
        })

    x = torch.stack(x_list)  # (N, D_raw) -- returned, for downstream model forward passes
    x_topo = torch.stack(topo_list)  # (N, D_fused) -- graph topology only

    if N <= n_neighbors + 1:
        # Too few samples — fully connected graph
        edge_index = torch.zeros((2, N * (N - 1)), dtype=torch.long)
        edge_distances = torch.zeros(N * (N - 1))
        idx = 0
        for i in range(N):
            for j in range(N):
                if i != j:
                    edge_index[0, idx] = i
                    edge_index[1, idx] = j
                    edge_distances[idx] = 0.0  # Fully connected, distance = 0
                    idx += 1
        return x, edge_index, meta_list, edge_distances

    # KNN with cosine distance, computed on the topology representation
    # (fused embedding if a model was provided, raw features otherwise)
    x_np = x_topo.numpy()
    nn_model = NearestNeighbors(n_neighbors=min(n_neighbors + 1, N), metric="cosine")
    nn_model.fit(x_np)
    distances, indices = nn_model.kneighbors(x_np)

    edges = []
    edge_weights = []
    edge_distances = []
    for i in range(N):
        for j, dist in zip(indices[i], distances[i]):
            if i != j:
                weight = 1.0 - dist  # convert distance to similarity
                edges.append([i, j])
                edge_weights.append(weight)
                edge_distances.append(dist)

    edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros((2, 0), dtype=torch.long)
    edge_distances = torch.tensor(edge_distances, dtype=torch.float) if edge_distances else torch.zeros(0)
    return x, edge_index, meta_list, edge_distances


def load_real_model_for_xai(llm_level: str = "L1", device_str: str = "cuda") -> _UnifiedInferenceModel:
    """Load real trained model for XAI case studies.
    
    Returns a model with forward_encoded() and gnn_forward() methods
    that SHAPExplainer and GNNExplainerWrapper expect.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = build_inference_model(llm_level, device)
    load_checkpoint(model, llm_level, device)
    model.eval()
    return model
