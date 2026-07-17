#!/usr/bin/env python3
"""
Phase 7: Joint Multitask Training with Graph-Gated MoE
======================================================
Full unified model with graph-gated MMoEEx, temperature-balanced sampling,
progressive projector unfreezing, and negative transfer monitoring.

Key design decisions:
    - Frozen projectors for first 20 epochs (prevent DAIC overfitting)
    - Progressive unfreezing: after epoch 20, unfreeze top 2 layers
    - GraphSAGE/GAT router for neighborhood-aware expert routing
    - Primary metric: DAIC AUROC (clinically most important)
    - Negative transfer monitoring: alert when task drops below 95% of isolated baseline

Architecture:
    GatedLateFusion → GG-MoE (MMoEEx + GraphSAGE/GAT) → Task Heads
         ↓                    ↓                    ↓
      3 projectors        8 experts           4 task outputs
      (frozen→unfreeze)                      (dep/sent/emo/pers)

Usage:
    uv run python scripts/phase07_joint_training.py --epochs 150 --batch_size 32 --temperature 3.0
    uv run python scripts/phase07_joint_training.py --epochs 150 --batch_size 16 --router graphsage --graph_type split-local
    # Quick test (3 epochs):
    uv run python scripts/phase07_joint_training.py --epochs 3 --quick_test --router graphsage --graph_type split-local
"""
import argparse
import json
import math
import os
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FEATURES_ROOT = ROOT / "data" / "features"
MANIFEST_PATH = FEATURES_ROOT / "manifest.json"
ARTIFACTS_FIGURES = ROOT / "artifacts" / "figures" / "phase_07_joint_training"
ARTIFACTS_TABLES = ROOT / "artifacts" / "tables"

DAIC_DATA = ROOT / "data" / "daic"
MOSEI_DATA = ROOT / "data" / "mosei"
FI_DATA = ROOT / "data" / "fi"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HIDDEN_DIM = 256
EXPERT_DIM = 256
NUM_EXPERTS = 8
NUM_SHARED = 2
NUM_HEADS = 4
BATCH_SIZE = 32
EPOCHS_DEFAULT = 150
LR_DEFAULT = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 20
TEMPERATURE = 3.0
EXPERT_ISOLATION = True
FREEZE_EPOCHS = 20  # Keep projectors frozen for first N epochs
GRAPH_WEIGHT = 0.5  # Weight for combining MMoE gates with graph router

# Expert isolation mapping
TASK_TO_EXPERTS = {
    0: [0, 1],     # DAIC depression - ISOLATED
    1: [2, 3],     # MOSEI sentiment - shared with emotion
    2: [2, 3],     # MOSEI emotion - shared with sentiment
    3: [4, 5],     # FI personality - separate from MOSEI
}

# Feature dimensions
FEATURE_DIMS = {
    "text": 768,
    "audio": 768,
    "video": 1536,
}

FI_TRAITS = ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]
EMOTION_LABELS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

TASK_IDS = {
    "daic_depression": 0,
    "mosei_sentiment": 1,
    "mosei_emotion": 2,
    "fi_personality": 3,
}

# ---------------------------------------------------------------------------
# Imports from Phase 5
# ---------------------------------------------------------------------------
from phase05_mmoe_ex import (
    JointMultimodalDataset,
    collate_joint,
    compute_ccc,
    compute_ccc_per_trait,
    TaskLosses,
    load_manifest,
    load_all_labels,
    FEATURE_DIMS as PH5_FEATURE_DIMS,
    FI_TRAITS as PH5_FI_TRAITS,
    TASK_IDS as PH5_TASK_IDS,
)

# ---------------------------------------------------------------------------
# Imports from Phase 6 models
# ---------------------------------------------------------------------------
from models.fusion import GatedLateFusion
from models.unified_moe import MMoEEx
from models.gnn_router import GraphSAGERouter, GATRouter
from models.task_heads import DepressionHead, SentimentHead, EmotionMultiLabelHead, PersonalityHead
from data.graph_builder import (
    build_knn_graph, build_split_local_graph, build_inductive_graph,
    build_multimodal_graph, validate_graph_leakage, validate_graph_no_cross_split_leakage
)
from utils.seed import set_seed

set_seed(42)

# ---------------------------------------------------------------------------
# Joint Training Pipeline with frozen projectors and progressive unfreezing
# ---------------------------------------------------------------------------

class JointTrainingPipeline(nn.Module):
    """Full joint training pipeline with graph-gated MMoEEx.

    Key features:
    - Modality projectors that can be frozen/unfrozen
    - GatedLateFusion for multimodal samples
    - GraphSAGE/GAT router for neighborhood-aware routing
    - Per-task routing: DAIC text-only, FI video-only, MOSEI multimodal
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        video_dim: int = 1536,
        hidden_dim: int = 256,
        num_experts: int = 8,
        expert_dim: int = 256,
        num_tasks: int = 4,
        router: str = "graphsage",
        graph_weight: float = 0.5,
        graph_weight_mode: str = "fixed",
        freeze_epochs: int = 20,
        device: str = "cuda",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.freeze_epochs = freeze_epochs
        self.unfrozen = False
        self.router = router
        self.graph_weight = graph_weight
        self.graph_weight_mode = graph_weight_mode
        self.device = torch.device(device)

        if graph_weight_mode == "learned":
            # Per-task learnable graph-routing weight, replacing the fixed
            # scalar lambda=0.5. Initialized at logit=0 -> sigmoid=0.5, the
            # same starting point as the fixed default, so training can move
            # it up/down per task rather than trusting the graph router
            # uniformly everywhere (see context/graph_routing_improvement_review.md).
            self.graph_weight_logit = nn.Parameter(torch.zeros(num_tasks))

        # GatedLateFusion for multimodal samples
        self.fusion = GatedLateFusion(text_dim, audio_dim, video_dim, hidden_dim)

        # Modality projectors (initially frozen)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # GG-MoE (MMoEEx with graph router)
        gr_type = router if router != "none" else None
        self.mmoe = MMoEEx(
            input_dim=hidden_dim,
            num_experts=num_experts,
            expert_dim=expert_dim,
            num_tasks=num_tasks,
            num_shared=2,
            expert_isolation=EXPERT_ISOLATION,
            task_to_experts=TASK_TO_EXPERTS,
            graph_router_type=gr_type,
        )
        self.mmoe.graph_weight = graph_weight

        # Task heads
        self.depression_head = DepressionHead(expert_dim)
        self.sentiment_head = SentimentHead(expert_dim)
        self.emotion_head = EmotionMultiLabelHead(expert_dim)
        self.personality_head = PersonalityHead(expert_dim)

        # Initialize weights
        self._init_weights()

        # Freeze projectors initially
        self.freeze_projectors()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_projectors(self):
        """Freeze all projector parameters."""
        for proj in [self.text_proj, self.audio_proj, self.video_proj]:
            for param in proj.parameters():
                param.requires_grad = False

    def unfreeze_top_layers(self, num_layers: int = 2):
        """Unfreeze the top N layers of each projector.

        For a Sequential with [Linear, LayerNorm, GELU], unfreezing the last
        num_layers modules.
        """
        for proj in [self.text_proj, self.audio_proj, self.video_proj]:
            modules = list(proj.modules())
            # Unfreeze the last num_layers layers
            for layer in modules[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        self.unfrozen = True
        print(f"  ⚡ Unfreezing top {num_layers} layers of projectors")

    def check_frozen(self) -> bool:
        """Verify projectors are frozen (no gradients)."""
        for proj in [self.text_proj, self.audio_proj, self.video_proj]:
            for param in proj.parameters():
                if param.requires_grad:
                    return False
        return True

    def should_unfreeze(self, epoch: int) -> bool:
        """Return True if projectors should be unfrozen at given epoch."""
        return epoch >= self.freeze_epochs and not self.unfrozen

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        mask: torch.Tensor,
        task_id: int,
        routing: str,
        edge_index: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass with routing strategy.

        Args:
            text_feat: [batch, text_dim]
            audio_feat: [batch, audio_dim]
            video_feat: [batch, audio_dim]
            mask: [batch, 3] bool tensor (text, audio, video available)
            task_id: int (0-3)
            routing: "text_only" | "video_only" | "multimodal"
            edge_index: optional graph edges for GG-MoE routing

        Returns:
            task-specific output
        """
        # Route to appropriate embedding
        if routing == "text_only":
            h = self.text_proj(text_feat)
        elif routing == "video_only":
            h = self.video_proj(video_feat)
        else:  # multimodal
            h = self.fusion(text_feat, audio_feat, video_feat, mask.bool())

        # GG-MoE forward
        # Always return expert mixture (batched, expert_dim) WITHOUT task head.
        # Caller applies task head. This ensures consistent interface.
        if self.router != "none" and edge_index is not None:
            # GG-MoE routing: get expert mixture and routing weights
            gr_type = self.router

            # MMoE gate probabilities
            gate_logits = self.mmoe.gates[task_id](h)
            gate_probs = torch.softmax(gate_logits, dim=-1)

            # Graph router combines with MMoE gates
            if self.graph_weight_mode == "learned":
                lam = torch.sigmoid(self.graph_weight_logit[task_id])
            else:
                lam = self.graph_weight
            routing_weights = gate_probs
            if gr_type == "graphsage" and self.mmoe.graphsage_router is not None:
                graph_probs = self.mmoe.graphsage_router(h, edge_index)
                combined_log_probs = torch.log(gate_probs + 1e-8) + lam * torch.log(graph_probs + 1e-8)
                routing_weights = F.softmax(combined_log_probs, dim=-1)
            elif gr_type == "gat" and self.mmoe.gat_router is not None:
                graph_probs = self.mmoe.gat_router(h, edge_index)
                combined_log_probs = torch.log(gate_probs + 1e-8) + lam * torch.log(graph_probs + 1e-8)
                routing_weights = F.softmax(combined_log_probs, dim=-1)

            # Weighted expert mixture (NO task head - caller applies)
            expert_outputs = torch.stack([expert(h) for expert in self.mmoe.experts], dim=1)
            expert_out = (routing_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        else:
            # Standard MMoEEx routing - compute expert mixture WITHOUT task head
            gate_logits = self.mmoe.gates[task_id](h)
            gate_probs = torch.softmax(gate_logits, dim=-1)
            expert_outputs = torch.stack([expert(h) for expert in self.mmoe.experts], dim=1)
            expert_out = (gate_probs.unsqueeze(-1) * expert_outputs).sum(dim=1)
            routing_weights = None

        # Return expert mixture - caller applies task head
        return expert_out, routing_weights

    def get_routing_weights(self, h: torch.Tensor) -> torch.Tensor:
        """Get expert routing weights for analysis."""
        return self.mmoe.get_routing_weights(h)


# ---------------------------------------------------------------------------
# Negative Transfer Monitor
# ---------------------------------------------------------------------------

class NegativeTransferMonitor:
    """Alert when joint training causes regression vs isolated baselines.

    Isolated baselines are from Phase 3/4 experiments where each task was
    trained separately or with simpler fusion.
    """

    # Isolated baselines (from Phase 3/4)
    ISOLATED_BASELINES = {
        "daic_auroc": 0.6991,       # DAIC text-only baseline
        "mosei_sentiment_ccc": 0.5123,  # MOSEI sentiment unimodal
        "mosei_emotion_auc": 0.6906,   # MOSEI emotion unimodal
        "fi_avg_ccc": 0.5688,         # FI video-only baseline
    }

    def __init__(self, tolerance: float = 0.95):
        self.tolerance = tolerance
        self.regressions = []

    def check(self, task_name: str, metric_value: float, epoch: int) -> bool:
        """Check if a metric has regressed below the tolerance threshold.

        Returns True if regression detected.
        """
        if task_name in self.ISOLATED_BASELINES:
            threshold = self.ISOLATED_BASELINES[task_name] * self.tolerance
            if metric_value is not None and metric_value < threshold:
                msg = (f"⚠ EPOCH {epoch}: {task_name} regression! "
                       f"{metric_value:.4f} < threshold {threshold:.4f} "
                       f"(baseline: {self.ISOLATED_BASELINES[task_name]:.4f})")
                print(msg)
                self.regressions.append({
                    "epoch": epoch,
                    "task": task_name,
                    "value": metric_value,
                    "threshold": threshold,
                    "baseline": self.ISOLATED_BASELINES[task_name],
                })
                return True
        return False

    def summary(self) -> dict:
        """Return summary of all detected regressions."""
        return {
            "total_regressions": len(self.regressions),
            "regressions": self.regressions,
        }


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def load_all_dataset_embeddings(device: torch.device, hidden_dim: int = 256) -> tuple:
    """Load all datasets and create fused embeddings via GatedLateFusion.

    Returns:
        all_embeddings: dict {dataset: {split: np.ndarray (N, D)}}
        all_metadata: dict {dataset: {split: {'ids': [], 'masks': [], 'task_ids': [], 'labels': {}}}}
    """
    from phase05_mmoe_ex import load_manifest

    manifest = load_manifest()
    all_embeddings = {}
    all_metadata = {}

    # Create dataset-specific fusion modules
    fusion_modules = {}
    for dataset in ['daic', 'mosei', 'fi']:
        # Detect dimensions
        text_dim = FEATURE_DIMS["text"]
        audio_dim = FEATURE_DIMS["audio"]
        video_dim = FEATURE_DIMS["video"]

        fusion_modules[dataset] = GatedLateFusion(text_dim, audio_dim, video_dim, hidden_dim)
        fusion_modules[dataset].to(device)
        fusion_modules[dataset].eval()

    for dataset in ['daic', 'mosei', 'fi']:
        all_embeddings[dataset] = {}
        all_metadata[dataset] = {}

        for split in ['train', 'val', 'test']:
            samples = [s for s in manifest if s['dataset'] == dataset and s.get('split') == split]

            if len(samples) == 0:
                continue

            embeddings_list = []
            sample_ids = []
            masks_list = []
            task_ids_list = []
            labels = {'depression': [], 'sentiment': [], 'emotion': [], 'personality': []}

            # Modality mask mapping
            modality_mask_map = {
                'daic': (True, True, True),
                'mosei': (True, True, True),
                'fi': (False, True, True),
            }
            base_mask = list(modality_mask_map[dataset])

            for sample in samples:
                sample_id = sample['id']
                feats = sample.get('features', {})

                # Load features
                text_feat = load_feature_tensor(feats.get('text_roberta'), FEATURE_DIMS["text"])
                audio_feat = load_feature_tensor(feats.get('audio_wavlm'), FEATURE_DIMS["audio"])
                video_feat = load_feature_tensor(feats.get('video_vit'), FEATURE_DIMS["video"])

                embeddings_list.append((text_feat, audio_feat, video_feat))
                sample_ids.append(sample_id)

                # Build modality mask
                mask = [
                    'text_roberta' in feats,
                    'audio_wavlm' in feats,
                    'video_vit' in feats,
                ]
                for i in range(3):
                    if not base_mask[i]:
                        mask[i] = False
                masks_list.append(mask)

                # Task ID
                if dataset == 'daic':
                    task_ids_list.append(0)
                elif dataset == 'mosei':
                    task_ids_list.append(1)
                else:
                    task_ids_list.append(3)

            # Batch process
            fusion = fusion_modules[dataset]
            batch_size = 256
            all_embs = []

            for i in range(0, len(embeddings_list), batch_size):
                batch_end = min(i + batch_size, len(embeddings_list))
                batch = embeddings_list[i:batch_end]

                text_batch = torch.stack([b[0] for b in batch]).to(device)
                audio_batch = torch.stack([b[1] for b in batch]).to(device)
                video_batch = torch.stack([b[2] for b in batch]).to(device)

                batch_mask = torch.tensor([[m[0], m[1], m[2]] for m in masks_list[i:batch_end]],
                                          dtype=torch.bool, device=device)

                with torch.no_grad():
                    fused = fusion(text_batch, audio_batch, video_batch, batch_mask)
                all_embs.append(fused.cpu())

            embeddings = torch.cat(all_embs, dim=0).numpy()
            all_embeddings[dataset][split] = embeddings
            all_metadata[dataset][split] = {
                'ids': sample_ids,
                'masks': masks_list,
                'task_ids': task_ids_list,
                'labels': labels,
            }
            print(f"  Loaded {dataset}/{split}: {len(sample_ids)} samples, shape {embeddings.shape}")

    return all_embeddings, all_metadata


def load_feature_tensor(path_str, dim):
    """Load a feature tensor from path."""
    if path_str is None:
        return torch.zeros(dim)
    full_path = ROOT / path_str
    if not full_path.exists():
        return torch.zeros(dim)
    try:
        obj = torch.load(full_path, map_location='cpu', weights_only=False)
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
                    return torch.zeros(dim)
        else:
            feat = obj

        if feat.dim() == 2:
            feat = feat.mean(dim=0)

        if isinstance(feat, torch.Tensor):
            feat = feat.cpu()

        feat = feat.numpy().flatten()
        if feat.shape[0] < dim:
            feat = np.pad(feat, (0, dim - feat.shape[0]))
        elif feat.shape[0] > dim:
            feat = feat[:dim]

        if not np.all(np.isfinite(feat)):
            return torch.zeros(dim)
        return torch.from_numpy(feat).float()
    except Exception:
        return torch.zeros(dim)


def concatenate_all_splits(all_embeddings: dict, all_metadata: dict,
                           hidden_dim: int = 256) -> tuple:
    """Concatenate all splits into single arrays for graph construction.

    Returns:
        global_embeddings: (N, D) concatenated embeddings
        dataset_ids: list of actual sample IDs (NOT dataset names)
        global_split_ids: (N,) array of split labels (0=train, 1=val, 2=test)
        global_task_ids: (N,) array of task IDs
        index_map: {(dataset, split): (global_start, global_end)} — global positions
                   into the concatenated arrays
    """
    all_embs = []
    dataset_ids = []
    split_ids = []
    task_ids = []
    index_map = {}

    split_map = {'train': 0, 'val': 1, 'test': 2}
    global_cumulative = 0  # Track global position across ALL concatenated arrays

    for dataset in ['daic', 'mosei', 'fi']:
        for split in ['train', 'val', 'test']:
            if dataset not in all_embeddings or split not in all_embeddings[dataset]:
                continue
            embs = all_embeddings[dataset][split]
            meta = all_metadata[dataset][split]

            split_id = split_map[split]
            global_start = global_cumulative
            global_end = global_start + len(embs)
            global_cumulative = global_end

            all_embs.append(embs)
            dataset_ids.extend(meta['ids'])
            split_ids.extend([split_id] * len(embs))
            task_ids.extend(meta['task_ids'])

            # Store GLOBAL positions (not local-within-split)
            index_map[(dataset, split)] = (global_start, global_end)

    global_embeddings = np.vstack(all_embs) if all_embs else np.empty((0, hidden_dim))
    global_split_ids = np.array(split_ids, dtype=np.int64)
    global_task_ids = np.array(task_ids, dtype=np.int64)

    return global_embeddings, dataset_ids, global_split_ids, global_task_ids, index_map


def construct_graphs(global_embeddings: np.ndarray, dataset_ids: list,
                     split_ids: np.ndarray, k: int, graph_type: str) -> tuple:
    """Construct graphs based on graph_type."""
    if graph_type == 'split-local':
        graphs, leakage_check = build_split_local_graph(global_embeddings, split_ids, k=k)
        print(f"  Split-local leakage check: {leakage_check}")

        train_idx, train_w = graphs['train']
        val_idx, val_w = graphs['val']
        test_idx, test_w = graphs['test']

        # NOTE: We skip cross-split validation for split-local graphs because
        # build_split_local_graph produces LOCAL indices [0, split_size-1] within each
        # split's subgraph, which do NOT correspond to global positions in split_ids.
        # The leakage_check dict (val/test) already verifies bounds within each split.
        # Cross-dataset edges within the same split are a feature, not leakage.
        return train_idx, train_w, val_idx, val_w, test_idx, test_w

    elif graph_type == 'inductive':
        train_mask = split_ids == 0
        val_mask = split_ids == 1
        test_mask = split_ids == 2

        train_embs = global_embeddings[train_mask]
        val_embs = global_embeddings[val_mask]
        test_embs = global_embeddings[test_mask]

        print(f"  Inductive: train={len(train_embs)}, val={len(val_embs)}, test={len(test_embs)}")

        train_edge_index, train_edge_weight = build_knn_graph(train_embs, k=k)

        val_edge_index, val_edge_weight, _, _ = build_inductive_graph(train_embs, val_embs, k=k)
        test_edge_index, test_edge_weight, _, _ = build_inductive_graph(train_embs, test_embs, k=k)

        val_start = train_mask.sum()
        val_size = val_mask.sum()
        leakage = validate_graph_leakage(train_edge_index, val_edge_index, test_edge_index,
                                         train_mask.sum(), val_size)
        print(f"  Inductive leakage check: {leakage}")

        # NOTE: We skip ALL cross-split validation for inductive graphs (train, val, test).
        # In inductive mode, val→train and test→train edges are EXPECTED and CORRECT.
        # validate_graph_leakage (above) already verifies edges stay within designated bounds.
        # validate_graph_no_cross_split_leakage is wrong for inductive since it checks
        # split_id(src)==split_id(dst), but val→train edges intentionally have different split_ids.
        return train_edge_index, train_edge_weight, val_edge_index, val_edge_weight, \
               test_edge_index, test_edge_weight

    elif graph_type == 'transductive':
        edge_index, edge_weights, edge_flags = build_multimodal_graph(
            global_embeddings, dataset_ids, k=k, cross_dataset_edges=True
        )

        dst_nodes = edge_index[1]

        # FIX (2026-07-16): split membership must be looked up per-node via
        # split_ids[dst_nodes], not inferred from cumulative index-count
        # thresholds (dst_nodes < train_mask.sum() etc). The global array is
        # ordered dataset-major (all DAIC train/val/test, then all MOSEI
        # train/val/test, then all FI train/val/test), NOT split-major (all
        # train, then all val, then all test), so the threshold form silently
        # misclassified most edges' split membership. See
        # context/graph_routing_improvement_review.md.
        dst_split = split_ids[dst_nodes]
        train_edges = dst_split == 0
        val_edges = dst_split == 1
        test_edges = dst_split == 2

        train_idx = edge_index[:, train_edges]
        train_w = edge_weights[train_edges]
        val_idx = edge_index[:, val_edges]
        val_w = edge_weights[val_edges]
        test_idx = edge_index[:, test_edges]
        test_w = edge_weights[test_edges]

        print(f"  Transductive (ABLATION): train_edges={train_idx.shape[1]}, "
              f"val_edges={val_idx.shape[1]}, test_edges={test_idx.shape[1]}")

        # NOTE: We intentionally do NOT call validate_graph_no_cross_split_leakage
        # for transductive mode — cross-split edges are expected and documented.
        # The ABLATION warning above is the documented acknowledgment.

        return train_idx, train_w, val_idx, val_w, test_idx, test_w


# ---------------------------------------------------------------------------
# PyG Data for graph-enhanced training
# ---------------------------------------------------------------------------

class GraphEnhancedDataset(Dataset):
    """Dataset that provides graph-enhanced training with global KNN edges.

    Each sample has:
    - Multimodal features
    - Task ID
    - Split ID
    - Global node index (for looking up graph edges)
    """

    def __init__(self, embeddings: np.ndarray, task_ids: np.ndarray,
                 split_ids: np.ndarray, index_map: dict, dataset_ids: list,
                 all_labels: dict, manifest_data: list,
                 feature_dims: dict, temperature: float = 3.0,
                 target_split: str = None):
        """Graph-enhanced dataset.

        Args:
            target_split: If set, only include samples from this split
                          (e.g., 'train', 'val', 'test'). If None, includes all.
        """
        self.samples = []
        self.embeddings = embeddings  # Global (N, D) embeddings
        self.task_ids = task_ids
        self.split_ids = split_ids
        self.index_map = index_map
        self.dataset_ids = dataset_ids
        self.all_labels = all_labels
        self.manifest_data = manifest_data
        self.feature_dims = feature_dims
        self.target_split = target_split

        # Build dataset-specific splits
        # Only iterate over the target split to avoid train/val data leakage
        splits_to_iterate = [target_split] if target_split else ['train', 'val', 'test']
        for dataset in ['daic', 'mosei', 'fi']:
            for split in splits_to_iterate:
                if (dataset, split) not in index_map:
                    continue

                start_idx, end_idx = index_map[(dataset, split)]

                for idx in range(start_idx, end_idx):
                    sample_id = dataset_ids[idx]
                    # Get original manifest entry
                    entries = [e for e in manifest_data
                               if e['dataset'] == dataset and e.get('split') == split
                               and e['id'] == sample_id]
                    if not entries:
                        continue

                    entry = entries[0]
                    feat_map = entry['features']

                    # Load raw features for the model
                    t_key = feat_map.get('text_roberta')
                    a_key = feat_map.get('audio_wavlm')
                    v_key = feat_map.get('video_vit') if dataset != 'daic' else feat_map.get('video_openface')

                    t_ok, t_vec = self._try_load_feature(t_key, feature_dims["text"])
                    a_ok, a_vec = self._try_load_feature(a_key, feature_dims["audio"])
                    v_ok, v_vec = self._try_load_feature(v_key, feature_dims["video"])

                    if not (t_ok or a_ok or v_ok):
                        continue

                    # Get label
                    if dataset == 'daic':
                        label_key = f"daic_{sample_id}"
                    elif dataset == 'mosei':
                        label_key = sample_id
                    else:
                        label_key = sample_id

                    if label_key not in all_labels:
                        continue

                    label = all_labels[label_key]

                    if dataset == 'daic':
                        self.samples.append({
                            'id': sample_id,
                            'dataset': dataset,
                            'split': split,
                            'global_idx': idx,
                            'task_id': 0,
                            'routing': 'text_only',
                            'text': t_vec,
                            'audio': a_vec,
                            'video': v_vec,
                            'modality_mask': [t_ok, a_ok, v_ok],
                            'label': label,
                        })
                    elif dataset == 'mosei':
                        # Two tasks: sentiment and emotion
                        self.samples.append({
                            'id': f"{sample_id}_sentiment",
                            'dataset': dataset,
                            'split': split,
                            'global_idx': idx,
                            'task_id': 1,
                            'routing': 'multimodal',
                            'text': t_vec,
                            'audio': a_vec,
                            'video': v_vec,
                            'modality_mask': [t_ok, a_ok, v_ok],
                            'label': label,
                        })
                        self.samples.append({
                            'id': f"{sample_id}_emotion",
                            'dataset': dataset,
                            'split': split,
                            'global_idx': idx,
                            'task_id': 2,
                            'routing': 'multimodal',
                            'text': t_vec,
                            'audio': a_vec,
                            'video': v_vec,
                            'modality_mask': [t_ok, a_ok, v_ok],
                            'label': label,
                        })
                    else:
                        self.samples.append({
                            'id': sample_id,
                            'dataset': dataset,
                            'split': split,
                            'global_idx': idx,
                            'task_id': 3,
                            'routing': 'video_only',
                            'text': t_vec,
                            'audio': a_vec,
                            'video': v_vec,
                            'modality_mask': [t_ok, a_ok, v_ok],
                            'label': label,
                        })

        self._compute_sampling_weights(temperature)
        print(f"  GraphEnhancedDataset: {len(self)} samples")

    def _try_load_feature(self, path_str, dim):
        if path_str is None:
            return False, np.zeros(dim, dtype=np.float32)
        full_path = ROOT / path_str
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

            if feat.dim() == 2:
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

    def _compute_sampling_weights(self, temperature):
        ds_counts = defaultdict(int)
        for s in self.samples:
            ds_counts[s["dataset"]] += 1
        total = len(self.samples)
        for s in self.samples:
            freq = ds_counts[s["dataset"]] / total
            s["sample_weight"] = freq ** (1.0 / temperature)
        total_weight = sum(s["sample_weight"] for s in self.samples)
        for s in self.samples:
            s["sample_weight"] = s["sample_weight"] / total_weight

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
            torch.tensor(s["sample_weight"], dtype=torch.float32),
            s["routing"],
            s["global_idx"],
        )


def collate_graph_enhanced(batch):
    """Collate function that also provides global indices for edge lookup."""
    text_tensors = []
    audio_tensors = []
    video_tensors = []
    masks = []
    labels = []
    task_ids = []
    weights = []
    routings = []
    global_indices = []

    for t, a, v, m, y, tid, w, r, gi in batch:
        text_tensors.append(t)
        audio_tensors.append(a)
        video_tensors.append(v)
        masks.append(m)
        labels.append(y)
        task_ids.append(tid)
        weights.append(w)
        routings.append(r)
        global_indices.append(gi)

    max_label_size = 7
    padded_labels = []
    for label in labels:
        if label.shape[0] < max_label_size:
            padded = torch.zeros(max_label_size)
            padded[:label.shape[0]] = label
            padded_labels.append(padded)
        else:
            padded_labels.append(label)
    padded_labels = torch.stack(padded_labels)

    return (
        torch.stack(text_tensors),
        torch.stack(audio_tensors),
        torch.stack(video_tensors),
        torch.stack(masks),
        padded_labels,
        torch.stack(task_ids),
        torch.stack(weights),
        routings,
        torch.tensor(global_indices, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, loss_fn, scheduler, device, scaler,
                epoch, edge_index_dict, split_row_counts, monitor, router, global_embeddings):
    """Train one epoch with graph-enhanced routing."""
    model.train()
    total_loss = 0.0
    task_losses = defaultdict(list)
    n_batches = 0
    routing_entropies = []

    for batch_idx, batch in enumerate(dataloader):
        # Handle both 8-item (collate_joint) and 9-item (collate_graph_enhanced) batches
        if len(batch) == 9:
            text, audio, video, mask, labels, task_ids, weights, routings, global_indices = batch
            global_indices = global_indices.to(device) if global_indices is not None else None
        else:  # 8 items from Phase 5 collate_joint
            text, audio, video, mask, labels, task_ids, weights, routings = batch
            global_indices = None

        text = text.to(device)
        audio = audio.to(device)
        video = video.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        task_ids = task_ids.to(device)

        optimizer.zero_grad()

        with autocast():
            losses = {}
            unique_tasks = task_ids.unique().tolist()

            for tid in unique_tasks:
                t_mask = task_ids == tid
                t_text = text[t_mask]
                t_audio = audio[t_mask]
                t_video = video[t_mask]
                t_mask_feat = mask[t_mask]
                t_labels = labels[t_mask]
                t_routings = [routings[i] for i in range(len(routings)) if t_mask[i]]
                # t_global_idx not currently used (full train graph used instead)

                if len(t_text) == 0:
                    continue

                routing = t_routings[0] if t_routings else "multimodal"
                task_val = tid.item() if isinstance(tid, torch.Tensor) else tid

                # Get appropriate edge index for this batch
                # Remap global edge indices to local batch indices to avoid CUDA index errors
                if router != "none" and edge_index_dict is not None and 'train' in edge_index_dict:
                    train_edge = edge_index_dict['train'].to(device)
                    src, dst = train_edge[0], train_edge[1]

                    # SAFETY: also filter by valid data range to handle graph/dataset size mismatch
                    # The graph has 32966 nodes but dataset may have ~32473 samples.
                    # Edges with src/dst >= len(global_embeddings) are always invalid.
                    n_total = global_embeddings.shape[0]
                    valid_range = (src < n_total) & (dst < n_total)

                    # Batch membership check
                    src_in = torch.isin(src, global_indices)
                    dst_in = torch.isin(dst, global_indices)

                    edge_mask = valid_range & src_in & dst_in

                    if edge_mask.sum() == 0:
                        batch_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                    else:
                        filtered_src = src[edge_mask]
                        filtered_dst = dst[edge_mask]
                        # Build dict-based global->local mapper
                        g2l = {g.item(): i for i, g in enumerate(global_indices)}
                        src_local = torch.tensor(
                            [g2l[n.item()] for n in filtered_src],
                            dtype=torch.long, device=device
                        )
                        dst_local = torch.tensor(
                            [g2l[n.item()] for n in filtered_dst],
                            dtype=torch.long, device=device
                        )
                        batch_edge_index = torch.stack([src_local, dst_local], dim=0)
                else:
                    batch_edge_index = None

                # SAFETY: if edge indices exceed h.size(0), skip graph routing
                if batch_edge_index is not None and batch_edge_index.numel() > 0:
                    max_idx = batch_edge_index.max().item()
                    if max_idx >= t_text.size(0):
                        # Out-of-bounds edges — skip graph routing for this batch
                        batch_edge_index = None

                # Forward pass - returns expert mixture (not task output)
                expert_out, routing_weights = model(
                    t_text, t_audio, t_video, t_mask_feat,
                    task_val, routing, batch_edge_index
                )

                # Apply task head
                if task_val == 0:
                    out = model.depression_head(expert_out)
                elif task_val == 1:
                    out = model.sentiment_head(expert_out)
                elif task_val == 2:
                    out = model.emotion_head(expert_out)
                else:
                    out = model.personality_head(expert_out)

                # Compute loss
                if task_val == 0:  # DAIC depression
                    l = loss_fn.depression_loss(out, t_labels)
                elif task_val == 1:  # MOSEI sentiment
                    l = loss_fn.sentiment_loss(out, t_labels)
                elif task_val == 2:  # MOSEI emotion
                    l = loss_fn.emotion_loss(out, t_labels)
                else:  # FI personality
                    l = loss_fn.personality_loss(out, t_labels)

                losses[tid] = l

                # Track routing entropy
                if routing_weights is not None:
                    entropy = -(routing_weights * torch.log(routing_weights + 1e-8)).sum(dim=-1).mean()
                    routing_entropies.append(entropy.item())

            combined = sum(losses.values()) if losses else torch.tensor(0.0, device=device)

        if combined.item() > 0:
            scaler.scale(combined).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += combined.item()
            n_batches += 1
            for tid, l in losses.items():
                task_losses[tid].append(l.item())

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(n_batches, 1)
    avg_entropy = np.mean(routing_entropies) if routing_entropies else 0.0

    return avg_loss, task_losses, avg_entropy


def evaluate(model, dataloader, device, edge_index_dict, router, global_embeddings):
    """Evaluate model on all tasks."""
    model.eval()

    results = {
        "daic": {"all_labels": [], "all_preds": [], "auroc": None},
        "mosei_sentiment": {"all_labels": [], "all_preds": [], "ccc": None},
        "mosei_emotion": {"all_labels": [], "all_preds": [], "auc": None},
        "fi": {"all_labels": [], "all_preds": [], "avg_ccc": None, "per_trait": {}},
    }

    with torch.no_grad():
        for batch in dataloader:
            # Handle both 8-item (collate_joint) and 9-item (collate_graph_enhanced) batches
            if len(batch) == 9:
                text, audio, video, mask, labels, task_ids, weights, routings, global_indices = batch
                global_indices = global_indices.to(device) if global_indices is not None else None
            else:  # 8 items from Phase 5 collate_joint
                text, audio, video, mask, labels, task_ids, weights, routings = batch
                global_indices = None

            text = text.to(device)
            audio = audio.to(device)
            video = video.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            task_ids = task_ids.to(device)

            for tid in task_ids.unique().tolist():
                t_mask = task_ids == tid
                t_text = text[t_mask]
                t_audio = audio[t_mask]
                t_video = video[t_mask]
                t_mask_feat = mask[t_mask]
                t_labels = labels[t_mask]
                t_routings = [routings[i] for i in range(len(routings)) if t_mask[i]]

                if len(t_text) == 0:
                    continue

                routing = t_routings[0] if t_routings else "multimodal"
                task_val = tid.item() if isinstance(tid, torch.Tensor) else tid

                # Use val graph for validation — filter and remap to local batch indices
                batch_edge_index = None
                if router != "none" and edge_index_dict is not None:
                    if 'val' in edge_index_dict and global_indices is not None:
                        val_edge = edge_index_dict['val'].to(device)
                        src, dst = val_edge[0], val_edge[1]
                        n_total = global_embeddings.shape[0]
                        valid_range = (src < n_total) & (dst < n_total)
                        src_in = torch.isin(src, global_indices)
                        dst_in = torch.isin(dst, global_indices)
                        edge_mask = valid_range & src_in & dst_in
                        if edge_mask.sum() > 0:
                            filtered_src = src[edge_mask]
                            filtered_dst = dst[edge_mask]
                            g2l = {g.item(): i for i, g in enumerate(global_indices)}
                            src_local = torch.tensor(
                                [g2l[n.item()] for n in filtered_src],
                                dtype=torch.long, device=device
                            )
                            dst_local = torch.tensor(
                                [g2l[n.item()] for n in filtered_dst],
                                dtype=torch.long, device=device
                            )
                            batch_edge_index = torch.stack([src_local, dst_local], dim=0)
                        else:
                            batch_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

                # SAFETY: skip graph routing if indices exceed batch size
                if batch_edge_index is not None and batch_edge_index.numel() > 0:
                    if batch_edge_index.max().item() >= t_text.size(0):
                        batch_edge_index = None

                out, _ = model(t_text, t_audio, t_video, t_mask_feat, task_val, routing, batch_edge_index)

                if tid == 0:
                    out = model.depression_head(out)
                    out_np = torch.sigmoid(out).cpu().numpy().flatten()
                    lbl = t_labels[:, 0].cpu().numpy().flatten()
                    results["daic"]["all_labels"].extend(lbl.tolist())
                    results["daic"]["all_preds"].extend(out_np.tolist())
                elif tid == 1:
                    out_np = model.sentiment_head(out).cpu().numpy().flatten()
                    lbl = t_labels[:, 0].cpu().numpy().flatten()
                    results["mosei_sentiment"]["all_labels"].extend(lbl.tolist())
                    results["mosei_sentiment"]["all_preds"].extend(out_np.tolist())
                elif tid == 2:
                    out = model.emotion_head(out)
                    out_np = torch.sigmoid(out).cpu().numpy()
                    lbl = t_labels[:, :6].cpu().numpy()
                    results["mosei_emotion"]["all_labels"].append(lbl)
                    results["mosei_emotion"]["all_preds"].append(out_np)
                else:
                    out_dict = model.personality_head(out)
                    out_vals = torch.cat([out_dict[t].cpu() for t in FI_TRAITS], dim=-1).numpy()
                    lbl = t_labels.cpu().numpy()
                    results["fi"]["all_labels"].append(lbl)
                    results["fi"]["all_preds"].append(out_vals)

    # Compute metrics
    try:
        from sklearn.metrics import roc_auc_score
        y_true = np.array(results["daic"]["all_labels"])
        y_pred = np.array(results["daic"]["all_preds"])
        if len(np.unique(y_true)) >= 2 and len(y_pred) > 0:
            results["daic"]["auroc"] = roc_auc_score(y_true, y_pred)
    except Exception:
        results["daic"]["auroc"] = None

    try:
        y_true = np.array(results["mosei_sentiment"]["all_labels"])
        y_pred = np.array(results["mosei_sentiment"]["all_preds"])
        results["mosei_sentiment"]["ccc"] = compute_ccc(y_true, y_pred)
    except Exception:
        results["mosei_sentiment"]["ccc"] = None

    try:
        y_true = np.vstack(results["mosei_emotion"]["all_labels"])
        y_pred = np.vstack(results["mosei_emotion"]["all_preds"])
        y_true_binary = (y_true >= 0.3).astype(int)
        from sklearn.metrics import roc_auc_score
        aucs = []
        for i in range(y_true_binary.shape[1]):
            if len(np.unique(y_true_binary[:, i])) >= 2:
                aucs.append(roc_auc_score(y_true_binary[:, i], y_pred[:, i]))
        results["mosei_emotion"]["auc"] = np.mean(aucs) if aucs else None
    except Exception:
        results["mosei_emotion"]["auc"] = None

    try:
        y_true = np.vstack(results["fi"]["all_labels"])
        y_pred = np.vstack(results["fi"]["all_preds"])
        per_trait = compute_ccc_per_trait(y_true, y_pred, FI_TRAITS)
        results["fi"]["per_trait"] = per_trait
        results["fi"]["avg_ccc"] = np.mean(list(per_trait.values()))
    except Exception:
        results["fi"]["avg_ccc"] = None

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_training_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = ["daic_depression", "mosei_sentiment", "mosei_emotion", "fi_personality"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (task, color) in enumerate(zip(tasks, colors)):
        if task not in history:
            continue
        losses = history[task]
        epochs = range(1, len(losses) + 1)
        axes[i].plot(epochs, losses, color=color, linewidth=2)
        axes[i].set_title(task.replace("_", " ").title(), fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Phase 7: Joint Training Curves per Task", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_task_metrics(metrics_history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_to_plot = [
        ("daic_auroc", "DAIC AUROC", "red"),
        ("mosei_sentiment_ccc", "MOSEI Sentiment CCC", "blue"),
        ("mosei_emotion_auc", "MOSEI Emotion AUC", "green"),
        ("fi_avg_ccc", "FI Avg CCC", "purple"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (key, title, color) in enumerate(metrics_to_plot):
        vals = [m.get(key) for m in metrics_history if key in m and m[key] is not None]
        epochs = range(1, len(vals) + 1)
        if vals:
            axes[idx].plot(epochs, vals, color=color, linewidth=2)
            # Draw baseline threshold
            baseline = NegativeTransferMonitor.ISOLATED_BASELINES.get(key)
            if baseline:
                axes[idx].axhline(baseline * 0.95, color=color, linestyle='--', alpha=0.5,
                                 label=f"95% baseline: {baseline:.3f}")
                axes[idx].legend(fontsize=8)
        axes[idx].set_title(title, fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Score")
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle("Phase 7: Evaluation Metrics Over Training", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "metrics_over_training.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_routing_entropy(entropy_history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(entropy_history) + 1)
    ax.plot(epochs, entropy_history, 'g-', linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Routing Entropy", fontsize=12)
    ax.set_title("Expert Routing Entropy Over Training", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, "routing_entropy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 7: Joint Multitask Training")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="Temperature for dataset-balanced sampling")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--router", choices=["graphsage", "gat", "none"], default="graphsage",
                        help="Graph router type")
    parser.add_argument("--graph_type", choices=["split-local", "inductive", "transductive"],
                        default="split-local", help="Graph construction type")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    parser.add_argument("--graph_weight", type=float, default=GRAPH_WEIGHT,
                        help="Weight for combining MMoE gates with graph router (used when --graph_weight_mode=fixed)")
    parser.add_argument("--graph_weight_mode", choices=["fixed", "learned"], default="fixed",
                        help="fixed: single scalar lambda for all tasks. learned: per-task learnable lambda (sigmoid-parameterized, init 0.5)")
    parser.add_argument("--freeze_epochs", type=int, default=FREEZE_EPOCHS,
                        help="Number of epochs to keep projectors frozen")
    parser.add_argument("--quick_test", action="store_true",
                        help="Run quick test with reduced epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str,
                        default="artifacts/figures/phase_07_joint_training")
    args = parser.parse_args()

    if args.quick_test:
        args.epochs = 3

    device = torch.device(args.device)
    out_dir = ROOT / args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ARTIFACTS_TABLES, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Phase 7: Joint Multitask Training — Graph-Gated MoE")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Router: {args.router}")
    print(f"Graph type: {args.graph_type}")
    print(f"k (KNN): {args.k}")
    print(f"Graph weight: {args.graph_weight} (mode={args.graph_weight_mode})")
    print(f"Freeze epochs: {args.freeze_epochs}")
    print(f"Temperature: {args.temperature}")

    # Load data
    print("\n[1/6] Loading manifest and labels...")
    manifest_data = load_manifest()
    all_labels = load_all_labels()

    # Build graph
    print("\n[2/6] Building KNN graphs...")
    if args.quick_test:
        print("  Skipping full graph building for quick test (using local batch graphs)")
        edge_index_dict = None
        split_row_counts = None
        global_embeddings = None
    else:
        print(f"  Loading embeddings for graph construction (hidden_dim={HIDDEN_DIM})...")
        all_embs, all_meta = load_all_dataset_embeddings(device, HIDDEN_DIM)
        global_embeddings, dataset_ids, global_split_ids, global_task_ids, index_map = \
            concatenate_all_splits(all_embs, all_meta, HIDDEN_DIM)

        print(f"  Building {args.graph_type} graph with k={args.k}...")
        train_idx, train_w, val_idx, val_w, test_idx, test_w = construct_graphs(
            global_embeddings, dataset_ids, global_split_ids, args.k, args.graph_type
        )

        edge_index_dict = {
            'train': torch.tensor(train_idx, dtype=torch.long),
            'val': torch.tensor(val_idx, dtype=torch.long),
            'test': torch.tensor(test_idx, dtype=torch.long),
        }
        split_row_counts = {
            'train': int((global_split_ids == 0).sum()),
            'val': int((global_split_ids == 1).sum()),
            'test': int((global_split_ids == 2).sum()),
        }

        # Build graph-enhanced dataset
        print("  Building graph-enhanced dataset...")
        train_ds = GraphEnhancedDataset(
            embeddings=global_embeddings,
            task_ids=global_task_ids,
            split_ids=global_split_ids,
            index_map=index_map,
            dataset_ids=dataset_ids,
            all_labels=all_labels,
            manifest_data=manifest_data,
            feature_dims=FEATURE_DIMS,
            temperature=args.temperature,
            target_split='train',
        )
        val_ds = GraphEnhancedDataset(
            embeddings=global_embeddings,
            task_ids=global_task_ids,
            split_ids=global_split_ids,
            index_map=index_map,
            dataset_ids=dataset_ids,
            all_labels=all_labels,
            manifest_data=manifest_data,
            feature_dims=FEATURE_DIMS,
            target_split='val',
            temperature=1.0,  # No sampling bias for val
        )
    if edge_index_dict is None:
        # Fallback: use Phase 5 dataset
        print("  Using Phase 5 JointMultimodalDataset (fallback)")
        train_ds = JointMultimodalDataset(
            manifest_data=manifest_data,
            all_labels=all_labels,
            datasets_splits=[("daic", "train"), ("mosei", "train"), ("fi", "train")],
            feature_dims=FEATURE_DIMS,
            temperature=args.temperature,
        )
        val_ds = JointMultimodalDataset(
            manifest_data=manifest_data,
            all_labels=all_labels,
            datasets_splits=[("daic", "val"), ("mosei", "val"), ("fi", "val")],
            feature_dims=FEATURE_DIMS,
            temperature=1.0,
        )

    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")

    if len(train_ds) == 0:
        print("ERROR: No training samples loaded. Check manifest and labels.")
        return

    # Use appropriate collate function
    if edge_index_dict is not None:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_graph_enhanced, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_graph_enhanced, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_joint, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_joint, num_workers=2, pin_memory=True)

    # Build model
    print("\n[3/6] Building model...")
    model = JointTrainingPipeline(
        text_dim=FEATURE_DIMS["text"],
        audio_dim=FEATURE_DIMS["audio"],
        video_dim=FEATURE_DIMS["video"],
        hidden_dim=HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        expert_dim=EXPERT_DIM,
        num_tasks=NUM_HEADS,
        router=args.router,
        graph_weight=args.graph_weight,
        graph_weight_mode=args.graph_weight_mode,
        freeze_epochs=args.freeze_epochs,
        device=args.device,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Total parameters: {n_params:,} trainable, {frozen_params:,} frozen")
    print(f"  Projectors frozen: {model.check_frozen()}")

    # Loss, optimizer, scheduler
    loss_fn = TaskLosses()
    model.loss_fn = loss_fn
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    # Negative transfer monitor
    monitor = NegativeTransferMonitor(tolerance=0.95)

    # Resume
    start_epoch = 0
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        # Restore unfrozen state from checkpoint
        if "unfrozen" in ckpt:
            model.unfrozen = ckpt["unfrozen"]

    # Training
    print("\n[4/6] Training...")
    history = defaultdict(list)
    metrics_history = []
    entropy_history = []
    best_daic_auroc = 0.0
    patience_counter = 0
    best_checkpoint = None

    for epoch in range(start_epoch, args.epochs):
        # Check if we should unfreeze projectors
        if model.should_unfreeze(epoch):
            print(f"\n  ⚡ Epoch {epoch+1}: Unfreezing projector layers")
            model.unfreeze_top_layers(num_layers=2)
            # Re-create optimizer to include newly unfrozen params
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch, eta_min=1e-6)

        avg_loss, task_losses, avg_entropy = train_epoch(
            model, train_loader, optimizer, loss_fn, scheduler, device, scaler,
            epoch, edge_index_dict, split_row_counts, monitor, args.router,
            global_embeddings
        )

        entropy_history.append(avg_entropy)
        history["total"].append(avg_loss)
        for tid, losses in task_losses.items():
            task_name = ["daic_depression", "mosei_sentiment", "mosei_emotion", "fi_personality"][tid]
            history[task_name].append(np.mean(losses))

        # Evaluate every 5 epochs (or every epoch for quick test)
        eval_interval = 1 if args.quick_test else 5
        if (epoch + 1) % eval_interval == 0 or epoch == args.epochs - 1:
            results = evaluate(model, val_loader, device, edge_index_dict, args.router, global_embeddings)

            daic_auroc = results["daic"]["auroc"]
            mosei_sent_ccc = results["mosei_sentiment"]["ccc"]
            mosei_emo_auc = results["mosei_emotion"]["auc"]
            fi_avg_ccc = results["fi"]["avg_ccc"]

            metrics_history.append({
                "epoch": epoch + 1,
                "daic_auroc": daic_auroc,
                "mosei_sentiment_ccc": mosei_sent_ccc,
                "mosei_emotion_auc": mosei_emo_auc,
                "fi_avg_ccc": fi_avg_ccc,
                "val_loss": avg_loss,
            })

            # Check for negative transfer
            if daic_auroc is not None:
                monitor.check("daic_auroc", daic_auroc, epoch + 1)
            if mosei_sent_ccc is not None:
                monitor.check("mosei_sentiment_ccc", mosei_sent_ccc, epoch + 1)
            if mosei_emo_auc is not None:
                monitor.check("mosei_emotion_auc", mosei_emo_auc, epoch + 1)
            if fi_avg_ccc is not None:
                monitor.check("fi_avg_ccc", fi_avg_ccc, epoch + 1)

            # Format metric strings safely (avoid .4f on None)
            daic_str = f"{daic_auroc:.4f}" if daic_auroc is not None else "N/A"
            sent_str = f"{mosei_sent_ccc:.4f}" if mosei_sent_ccc is not None else "N/A"
            fi_str = f"{fi_avg_ccc:.4f}" if fi_avg_ccc is not None else "N/A"
            print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"Entropy: {avg_entropy:.4f} | "
                  f"DAIC AUROC: {daic_str} | "
                  f"MOSEI CCC: {sent_str} | "
                  f"FI CCC: {fi_str}")

            # Save best by DAIC AUROC (primary clinical metric)
            if daic_auroc is not None and daic_auroc > best_daic_auroc:
                best_daic_auroc = daic_auroc
                patience_counter = 0
                best_checkpoint = {
                    "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "daic_auroc": daic_auroc,
                }
                ckpt_path = ARTIFACTS_TABLES / "phase07_best.pt"
                torch.save(best_checkpoint, ckpt_path)
                print(f"    → Saved best model (DAIC AUROC={daic_auroc:.4f}) to {ckpt_path}")
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
        else:
            print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Entropy: {avg_entropy:.4f}")

    # Final evaluation
    print("\n[5/6] Final evaluation...")
    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint["model"])
        ckpt_auroc = best_checkpoint.get('daic_auroc')
        ckpt_str = f"{ckpt_auroc:.4f}" if ckpt_auroc is not None else "N/A"
        print(f"  Loaded best checkpoint (epoch {best_checkpoint['epoch']+1}, DAIC AUROC={ckpt_str})")

    final_results = evaluate(model, val_loader, device, edge_index_dict, args.router, global_embeddings)

    # Safely format metrics
    daic_auroc_final = final_results['daic']['auroc']
    mosei_sent_ccc_final = final_results['mosei_sentiment']['ccc']
    fi_avg_ccc_final = final_results['fi']['avg_ccc']
    daic_str = f"{daic_auroc_final:.4f}" if daic_auroc_final is not None else "N/A"
    sent_str = f"{mosei_sent_ccc_final:.4f}" if mosei_sent_ccc_final is not None else "N/A"
    fi_str = f"{fi_avg_ccc_final:.4f}" if fi_avg_ccc_final is not None else "N/A"

    print("\n  Final Results:")
    print(f"  DAIC AUROC: {daic_str}")
    print(f"  MOSEI Sentiment CCC: {sent_str}")
    mosei_emotion_auc = final_results['mosei_emotion']['auc']
    if mosei_emotion_auc is not None:
        print(f"  MOSEI Emotion AUC: {mosei_emotion_auc:.4f}")
    else:
        print("  MOSEI Emotion AUC: N/A")
    print(f"  FI Avg CCC: {fi_str}")
    if final_results['fi']['per_trait']:
        print(f"  FI Per-Trait: {final_results['fi']['per_trait']}")

    # Negative transfer summary
    regression_summary = monitor.summary()
    if regression_summary['total_regressions'] > 0:
        print(f"\n  ⚠ Negative transfer detected: {regression_summary['total_regressions']} regressions")
        for r in regression_summary['regressions']:
            print(f"    - Epoch {r['epoch']}: {r['task']} = {r['value']:.4f} (baseline: {r['baseline']:.4f})")

    # Save results
    results_path = ARTIFACTS_TABLES / "phase07_results.csv"
    with open(results_path, "w") as f:
        f.write("metric,value\n")
        # Safely format values for CSV
        daic_csv = f"{final_results['daic']['auroc']:.4f}" if final_results['daic']['auroc'] is not None else "N/A"
        mosei_sent_csv = f"{final_results['mosei_sentiment']['ccc']:.4f}" if final_results['mosei_sentiment']['ccc'] is not None else "N/A"
        mosei_emo_val = final_results['mosei_emotion']['auc']
        mosei_emo_csv = f"{mosei_emo_val:.4f}" if mosei_emo_val is not None else "N/A"
        fi_csv = f"{final_results['fi']['avg_ccc']:.4f}" if final_results['fi']['avg_ccc'] is not None else "N/A"

        f.write(f"daic_auroc,{daic_csv}\n")
        f.write(f"mosei_sentiment_ccc,{mosei_sent_csv}\n")
        f.write(f"mosei_emotion_auc,{mosei_emo_csv}\n")
        f.write(f"fi_avg_ccc,{fi_csv}\n")
        for trait, ccc in final_results['fi']['per_trait'].items():
            ccc_csv = f"{ccc:.4f}" if ccc is not None else "N/A"
            f.write(f"fi_{trait}_ccc,{ccc_csv}\n")
        f.write(f"best_daic_auroc,{best_daic_auroc:.4f}\n")
        f.write(f"total_regressions,{regression_summary['total_regressions']}\n")
    print(f"\n  Results saved to {results_path}")

    # Generate visualizations
    print("\n[6/6] Generating visualizations...")
    plot_training_curves(dict(history), out_dir)
    plot_task_metrics(metrics_history, out_dir)
    plot_routing_entropy(entropy_history, out_dir)

    print(f"\n  Visualizations saved to {out_dir}")
    print("\n✅ Phase 7 complete!")


if __name__ == "__main__":
    main()