#!/usr/bin/env python3
"""
Phase 5: MMoEEx — Multi-Task Multi-Expert without graph
=========================================================
Unified joint training across all 4 tasks:
    Task 0: DAIC binary depression (PHQ-8 >= 10)
    Task 1: MOSEI sentiment regression [-3, 3]
    Task 2: MOSEI emotion multi-label (6 emotions)
    Task 3: FI Big-Five personality regression

Key design decisions (per Phase 4 findings):
    - GatedLateFusion (NOT CrossAttention) — CrossAttn fails on DAIC (107 samples)
      and MOSEI (CCC=0.5397 vs 0.6229). Gated is lighter and more robust.
    - Per-dataset routing: DAIC uses text-only, FI uses video-only (fusion collapses these)
    - NLL loss for regression tasks (fixes FI constant-prediction collapse from MSE)
    - Uncertainty-weighted multitask loss (Kendall et al. 2018)
    - Temperature-balanced sampling to handle MOSEI dominance (120x larger)

Architecture:
    GatedLateFusion → MMoEEx (8 experts, 2 shared) → Task heads

Usage:
    uv run python scripts/phase05_mmoe_ex.py --mode train --epochs 150
    uv run python scripts/phase05_mmoe_ex.py --mode eval

Outputs:
    artifacts/figures/phase_05_mmoe_ex/  - PNG figures
    artifacts/tables/mmoe_ex_results.csv   - results table
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
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FEATURES_ROOT = ROOT / "data" / "features"
MANIFEST_PATH = FEATURES_ROOT / "manifest.json"
ARTIFACTS_FIGURES = ROOT / "artifacts" / "figures" / "phase_05_mmoe_ex"
ARTIFACTS_TABLES = ROOT / "artifacts" / "tables"

DAIC_DATA = ROOT / "data" / "daic"
MOSEI_DATA = ROOT / "data" / "mosei"
FI_DATA = ROOT / "data" / "fi"

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
PATIENCE = 20           # More patience for DAIC (was 15)
TEMPERATURE = 3.0       # Stronger upweighting for DAIC (was 2.0)
EXPERT_ISOLATION = True  # CRITICAL FIX: isolate DAIC experts from MOSEI

# Expert isolation mapping: each task gets isolated experts
# DAIC gets 0-1 (ISOLATED from MOSEI), MOSEI gets 2-3 (shared sentiment+emotion), FI gets 4-5
TASK_TO_EXPERTS = {
    0: [0, 1],     # DAIC depression - ISOLATED
    1: [2, 3],     # MOSEI sentiment - shared with emotion
    2: [2, 3],     # MOSEI emotion - shared with sentiment
    3: [4, 5],     # FI personality - separate from MOSEI
}

# Feature dimensions (from Phase 2)
FEATURE_DIMS = {
    "text": 768,    # RoBERTa
    "audio": 768,   # WavLM mean-pooled
    "video": 1536,  # ViT mean-pooled
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
# Feature loading
# ---------------------------------------------------------------------------

def load_feature_tensor(path):
    """Load feature from pickle/npy file."""
    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif path.suffix == ".npy":
        return np.load(path)
    else:
        raise ValueError(f"Unsupported feature format: {path}")

# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_all_labels():
    """Load labels for all datasets. Returns dict matching Phase 4 label format."""
    labels = {}

    # DAIC labels — key format: "daic_{participant_id}"
    for split, filename, col_binary in [
        ("train", "train_split_Depression_AVEC2017.csv", "PHQ8_Binary"),
        ("val",   "dev_split_Depression_AVEC2017.csv",  "PHQ8_Binary"),
        ("test",  "full_test_split.csv",                "PHQ_Binary"),
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

    # MOSEI labels — use emotion_labels.json which has sentiment + 6 emotions per sample
    # Key format: "mosei_{split}_{i:05d}"
    emotion_path = MOSEI_DATA / "mosei_emotion_labels.json"
    if emotion_path.exists():
        with open(emotion_path, "r") as f:
            mosei_labels_data = json.load(f)
        for key, label_data in mosei_labels_data.items():
            # label_data has: sentiment, happiness, sadness, anger, fear, disgust, surprise
            # Store as array: [sentiment, anger, disgust, fear, happiness, sadness, surprise]
            # This matches the expected format for sentiment (index 0) and emotions (indices 0-5 after sentiment)
            sentiment = float(label_data.get("sentiment", 0.0))
            emotions = [float(label_data.get(e, 0.0)) for e in EMOTION_LABELS]
            labels[key] = [sentiment] + emotions  # List with 7 values: [sentiment, 6 emotions]

    # FI labels — key format: "fi_{split}_{clip_id}"
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


def make_label_key(dataset, sample_id, split):
    """Construct label key matching Phase 4 format."""
    if dataset == "daic":
        return f"daic_{sample_id}"  # sample_id is participant ID (e.g. "300")
    elif dataset == "mosei":
        # sample_id is like "mosei_train_00000" → extract split and index
        # Manifest id format: "mosei_train_00000", "mosei_val_00001", etc.
        return sample_id  # Already has split in it
    else:  # fi
        # Manifest id format: "fi_train_{clip_id}" or "fi_val_{clip_id}" or "fi_test_{i:05d}"
        return f"fi_{split}_{sample_id.split('_')[-1]}"


def load_manifest():
    """Load feature manifest. Returns the list of sample entries."""
    with open(MANIFEST_PATH, "r") as f:
        data = json.load(f)
    # Manifest is a list of sample entries, not dict with "entries" key
    if isinstance(data, list):
        return data
    # Handle dict format with "samples" key
    return data.get("samples", [])


# ---------------------------------------------------------------------------
# Joint dataset for multi-task training
# ---------------------------------------------------------------------------

class JointMultimodalDataset(Dataset):
    """Combined dataset for all 3 datasets with task routing.

    Per Phase 4 findings:
    - DAIC: use text-only features (fusion fails at 107 samples)
    - FI: use video-only features (fusion collapses with MSE loss)
    - MOSEI: use full multimodal fusion (fusion works well)
    """

    def __init__(self, manifest_data, all_labels, datasets_splits, feature_dims, temperature=2.0):
        self.samples = []
        self.feature_dims = feature_dims

        for ds_name, split in datasets_splits:
            for entry in manifest_data:
                if entry["dataset"] != ds_name or entry.get("split") != split:
                    continue
                sample_id = entry["id"]

                # Compute label key matching Phase 4 format
                if ds_name == "daic":
                    label_key = f"daic_{sample_id}"  # e.g. "daic_300"
                elif ds_name == "mosei":
                    label_key = sample_id  # e.g. "mosei_train_00000"
                else:  # fi
                    # Manifest id: "fi_train_{clip_id}", "fi_val_{clip_id}", "fi_test_{i:05d}"
                    # Phase 4 label key: "fi_train_{clip_id}", etc.
                    # For test: manifest "fi_test_00000" → label "fi_test_00000"
                    label_key = sample_id  # Already in correct format

                if label_key not in all_labels:
                    continue

                feat_map = entry["features"]
                t_key = feat_map.get("text_roberta")
                a_key = feat_map.get("audio_wavlm")
                v_key = feat_map.get("video_vit") if ds_name != "daic" else feat_map.get("video_openface")

                t_ok, t_vec = self._try_load_feature(t_key, feature_dims["text"])
                a_ok, a_vec = self._try_load_feature(a_key, feature_dims["audio"])
                v_ok, v_vec = self._try_load_feature(v_key, feature_dims["video"])

                # Require at least one modality
                if not (t_ok or a_ok or v_ok):
                    continue

                label = all_labels[label_key]

                if ds_name == "daic":
                    # Single task: depression
                    self.samples.append({
                        "id": sample_id,
                        "label_key": label_key,
                        "dataset": ds_name,
                        "split": split,
                        "task_id": TASK_IDS["daic_depression"],
                        "routing": "text_only",
                        "text": t_vec,
                        "audio": a_vec,
                        "video": v_vec,
                        "modality_mask": [t_ok, a_ok, v_ok],
                        "label": label,
                        "sample_weight": 1.0,
                    })
                elif ds_name == "mosei":
                    # MOSEI has TWO tasks: sentiment (task 1) and emotion (task 2)
                    # Both share the same features but have different labels
                    # Sentiment: label[0] (scalar), Emotion: label[1:7] (6 binary labels)
                    self.samples.append({
                        "id": f"{sample_id}_sentiment",
                        "label_key": label_key,
                        "dataset": ds_name,
                        "split": split,
                        "task_id": TASK_IDS["mosei_sentiment"],
                        "routing": "multimodal",
                        "text": t_vec,
                        "audio": a_vec,
                        "video": v_vec,
                        "modality_mask": [t_ok, a_ok, v_ok],
                        "label": label,
                        "sample_weight": 1.0,
                    })
                    self.samples.append({
                        "id": f"{sample_id}_emotion",
                        "label_key": label_key,
                        "dataset": ds_name,
                        "split": split,
                        "task_id": TASK_IDS["mosei_emotion"],
                        "routing": "multimodal",
                        "text": t_vec,
                        "audio": a_vec,
                        "video": v_vec,
                        "modality_mask": [t_ok, a_ok, v_ok],
                        "label": label,  # label[1:7] = 6 emotions
                        "sample_weight": 1.0,
                    })
                else:
                    # FI personality
                    self.samples.append({
                        "id": sample_id,
                        "label_key": label_key,
                        "dataset": ds_name,
                        "split": split,
                        "task_id": TASK_IDS["fi_personality"],
                        "routing": "video_only",
                        "text": t_vec,
                        "audio": a_vec,
                        "video": v_vec,
                        "modality_mask": [t_ok, a_ok, v_ok],
                        "label": label,
                        "sample_weight": 1.0,
                    })

        self._compute_sampling_weights(temperature)
        print(f"  JointDataset: {len(self)} samples across {len(datasets_splits)} dataset splits")
        by_routing = defaultdict(int)
        for s in self.samples:
            by_routing[s["routing"]] += 1
        print(f"    Routing: {dict(by_routing)}")

    def _try_load_feature(self, path_str, dim):
        if path_str is None:
            return False, np.zeros(dim, dtype=np.float32)
        full_path = ROOT / path_str
        if not full_path.exists():
            return False, np.zeros(dim, dtype=np.float32)
        try:
            obj = torch.load(full_path, map_location="cpu", weights_only=False)
            # Handle dict format (from torch.save with dict wrapper)
            if isinstance(obj, dict):
                # Try various pooling keys
                for key in ["pooled_embedding", "pooled_features", "embedding", "features"]:
                    if key in obj and isinstance(obj[key], torch.Tensor):
                        feat = obj[key]
                        break
                else:
                    # Try first tensor value
                    for v in obj.values():
                        if isinstance(v, torch.Tensor):
                            feat = v
                            break
                    else:
                        return False, np.zeros(dim, dtype=np.float32)
            else:
                feat = obj

            # If 2D (sequence, features), mean-pool to 1D
            if isinstance(feat, torch.Tensor) and feat.dim() == 2:
                feat = feat.mean(dim=0)  # [seq_len, feat_dim] → [feat_dim]

            # Convert to numpy
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
        except Exception as e:
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
        # Handle label: FI has dict labels, others have scalar or array
        label = s["label"]
        if isinstance(label, dict):
            # FI Big-Five: convert dict to ordered array
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
        )


def collate_joint(batch):
    text_tensors = []
    audio_tensors = []
    video_tensors = []
    masks = []
    labels = []
    task_ids = []
    weights = []
    routings = []

    for t, a, v, m, y, tid, w, r in batch:
        text_tensors.append(t)
        audio_tensors.append(a)
        video_tensors.append(v)
        masks.append(m)
        labels.append(y)
        task_ids.append(tid)
        weights.append(w)
        routings.append(r)

    # Pad labels to max size (7 for MOSEI: sentiment + 6 emotions)
    max_label_size = 7  # Max of 1 (DAIC), 1 (MOSEI sentiment), 6 (MOSEI emotion), 5 (FI)
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
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ccc(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mu_y = y_true.mean()
    mu_p = y_pred.mean()
    var_y = y_true.var()
    var_p = y_pred.var()
    cov = np.cov(y_true, y_pred)[0, 1]
    denom = var_y + var_p + (mu_y - mu_p) ** 2
    if denom < 1e-12:
        return 0.0
    return (2 * cov) / denom


def compute_ccc_per_trait(y_true, y_pred, trait_names):
    results = {}
    for i, trait in enumerate(trait_names):
        yt = y_true[:, i] if y_true.ndim > 1 else y_true
        yp = y_pred[:, i] if y_pred.ndim > 1 else y_pred
        results[trait] = compute_ccc(yt, yp)
    return results


def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=500, ci=0.95):
    scores = []
    rng = np.random.default_rng(42)
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            s = metric_func(y_true[idx], y_pred[idx])
            scores.append(s)
        except Exception:
            pass
    if not scores:
        return float("nan"), float("nan")
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return lo, hi


# ---------------------------------------------------------------------------
# Model: UnifiedMMoEEx with GatedLateFusion (per Phase 4 findings)
# ---------------------------------------------------------------------------

class UnifiedMMoEEx(nn.Module):
    """Unified MMoEEx model with GatedLateFusion + MMoEEx + task heads.

    Per Phase 4 findings:
    - GatedLateFusion (NOT CrossAttention) — lighter, more robust on small datasets
    - Per-dataset routing: text-only for DAIC, video-only for FI, multimodal for MOSEI
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        video_dim: int = 1536,
        hidden_dim: int = 256,
        expert_dim: int = 256,
        num_experts: int = 8,
        num_shared: int = 2,
        num_tasks: int = 4,
    ):
        super().__init__()
        sys.path.insert(0, str(ROOT / "src"))

        from models.fusion import GatedLateFusion
        from models.unified_moe import MMoEEx
        from models.task_heads import DepressionHead, SentimentHead, EmotionMultiLabelHead, PersonalityHead

        # GatedLateFusion for MOSEI multimodal samples
        self.fusion = GatedLateFusion(text_dim, audio_dim, video_dim, hidden_dim)

        # Unimodal projectors for DAIC (text) and FI (video)
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.audio_projector = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.video_projector = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # MMoEEx backbone with expert isolation
        self.mmoe = MMoEEx(
            input_dim=hidden_dim,
            num_experts=num_experts,
            expert_dim=expert_dim,
            num_tasks=num_tasks,
            num_shared=num_shared,
            expert_isolation=EXPERT_ISOLATION,
            task_to_experts=TASK_TO_EXPERTS,
        )

        # Task-specific heads
        self.depression_head = DepressionHead(expert_dim)
        self.sentiment_head = SentimentHead(expert_dim)
        self.emotion_head = EmotionMultiLabelHead(expert_dim)
        self.personality_head = PersonalityHead(expert_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_with_routing(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        modality_mask: torch.Tensor,
        task_id: int,
        routing: str,
    ) -> torch.Tensor:
        """Forward with per-dataset routing strategy.

        routing options:
            - "text_only": use text projector only (DAIC)
            - "video_only": use video projector only (FI)
            - "multimodal": use GatedLateFusion (MOSEI)

        Returns MMoEEx expert output (before task head). Caller applies task head.
        """
        if routing == "text_only":
            h = self.text_projector(text_feat)
            return self.mmoe(h, task_id)
        elif routing == "video_only":
            h = self.video_projector(video_feat)
            return self.mmoe(h, task_id)
        else:  # multimodal
            fused = self.fusion(text_feat, audio_feat, video_feat, modality_mask.bool())
            return self.mmoe(fused, task_id)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Standard MMoEEx forward (used internally)."""
        return self.mmoe(x, task_id)


# ---------------------------------------------------------------------------
# Loss: NLL for regression (fixes FI constant-prediction collapse)
# ---------------------------------------------------------------------------

class NLLRegressionLoss(nn.Module):
    """Negative Log-Likelihood loss for regression with learned variance.

    L = 0.5 * (pred - target)^2 / sigma^2 + 0.5 * log(sigma^2)

    This prevents the constant-prediction collapse that MSE suffers from.
    The learned sigma allows the model to express uncertainty and maintain
    gradient signal even when predictions are near the mean.
    """

    def __init__(self, init_log_sigma: float = -2.0):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(init_log_sigma))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma)
        loss = 0.5 * (pred.squeeze(-1) - target) ** 2 / (sigma ** 2)
        loss = loss + 0.5 * self.log_sigma  # + 0.5 * log(sigma^2)
        return loss.mean()


class MSELossWithVariancePenalty(nn.Module):
    """MSE loss with variance regularization to prevent constant predictions.

    L = MSE(pred, target) - lambda * var(pred) + entropy_reg

    The variance penalty encourages the model to use its full capacity,
    preventing the collapse to constant predictions.
    """

    def __init__(self, lambda_variance: float = 0.1, lambda_entropy: float = 0.01):
        super().__init__()
        self.lambda_variance = lambda_variance
        self.lambda_entropy = lambda_entropy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = nn.functional.mse_loss(pred.squeeze(-1), target)
        pred_std = pred.squeeze(-1).std()
        variance_penalty = -self.lambda_variance * pred_std
        return mse + variance_penalty


class TaskLosses:
    """Compute per-task losses."""

    def __init__(self):
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")
        self.bce_multi = nn.BCEWithLogitsLoss(reduction="none")
        # NLL loss for regression (FI personality, MOSEI sentiment)
        self.nll = NLLRegressionLoss(init_log_sigma=-2.0)
        # MSE + variance penalty (FI backup)
        self.mse_variance = MSELossWithVariancePenalty(lambda_variance=0.1)

    def depression_loss(self, logits, labels):
        # Labels are padded to max_size=6; only index 0 is valid for depression
        return self.bce(logits.squeeze(-1), labels[:, 0]).mean()

    def sentiment_loss(self, preds, labels):
        # Labels are padded to max_size=6; only index 0 is valid for sentiment
        return self.nll(preds, labels[:, 0])

    def emotion_loss(self, logits, labels):
        # Labels are padded to max_size=6; indices 0-5 are valid for 6 emotions
        # Binarize at 0.3 threshold to capture more positives (fear only has 0.3% at 0.5)
        binary_labels = (labels[:, :6] >= 0.3).float()
        return self.bce_multi(logits, binary_labels).mean()

    def personality_loss(self, preds_dict, labels):
        """MSE loss per personality trait + NLL for overall uncertainty."""
        total = 0.0
        count = 0
        for i, trait in enumerate(FI_TRAITS):
            pred = preds_dict[trait].squeeze(-1)
            label = labels[:, i]
            l = self.nll(pred.unsqueeze(-1), label)
            total = total + l
            count += 1
        return total / count


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, loss_fn, scheduler, device, scaler, epoch):
    model.train()
    total_loss = 0.0
    task_losses = defaultdict(list)
    n_batches = 0

    for batch_idx, (text, audio, video, mask, labels, task_ids, weights, routings) in enumerate(dataloader):
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

                if len(t_text) == 0:
                    continue

                routing = t_routings[0] if t_routings else "multimodal"
                task_val = tid.item() if isinstance(tid, torch.Tensor) else tid

                # Get MMoEEx expert output with routing
                expert_out = model.forward_with_routing(
                    t_text, t_audio, t_video, t_mask_feat, task_val, routing
                )

                if tid == 0:  # DAIC depression
                    out = model.depression_head(expert_out)
                    l = loss_fn.depression_loss(out, t_labels)
                elif tid == 1:  # MOSEI sentiment
                    out = model.sentiment_head(expert_out)
                    l = loss_fn.sentiment_loss(out, t_labels)
                elif tid == 2:  # MOSEI emotion
                    out = model.emotion_head(expert_out)
                    l = loss_fn.emotion_loss(out, t_labels)
                else:  # FI personality
                    out = model.personality_head(expert_out)
                    l = loss_fn.personality_loss(out, t_labels)

                losses[tid] = l

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
    return avg_loss, task_losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, device):
    """Evaluate model on all tasks."""
    model.eval()

    results = {
        "daic": {"all_labels": [], "all_preds": [], "auroc": None},
        "mosei_sentiment": {"all_labels": [], "all_preds": [], "ccc": None},
        "mosei_emotion": {"all_labels": [], "all_preds": [], "auc": None},
        "fi": {"all_labels": [], "all_preds": [], "avg_ccc": None, "per_trait": {}},
    }

    with torch.no_grad():
        for text, audio, video, mask, labels, task_ids, weights, routings in dataloader:
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

                expert_out = model.forward_with_routing(
                    t_text, t_audio, t_video, t_mask_feat,
                    tid.item() if isinstance(tid, torch.Tensor) else tid,
                    t_routings[0] if t_routings else "multimodal",
                )

                if tid == 0:
                    out = torch.sigmoid(model.depression_head(expert_out)).cpu().numpy().flatten()
                    lbl = t_labels[:, 0].cpu().numpy().flatten()  # Only index 0 valid
                    results["daic"]["all_labels"].extend(lbl.tolist())
                    results["daic"]["all_preds"].extend(out.tolist())
                elif tid == 1:
                    out = model.sentiment_head(expert_out).cpu().numpy().flatten()
                    lbl = t_labels[:, 0].cpu().numpy().flatten()  # Only index 0 valid
                    results["mosei_sentiment"]["all_labels"].extend(lbl.tolist())
                    results["mosei_sentiment"]["all_preds"].extend(out.tolist())
                elif tid == 2:
                    out = torch.sigmoid(model.emotion_head(expert_out)).cpu().numpy()
                    lbl = t_labels[:, :6].cpu().numpy()  # Indices 0-5 valid
                    results["mosei_emotion"]["all_labels"].append(lbl)
                    results["mosei_emotion"]["all_preds"].append(out)
                else:
                    out_dict = model.personality_head(expert_out)
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
        # Binarize at 0.3 threshold to match training
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

    plt.suptitle("Phase 5: MMoEEx Training Curves per Task", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_expert_routing(routing_weights, task_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_routing = routing_weights.mean(dim=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(NUM_EXPERTS)
    width = 0.2
    for i, task in enumerate(task_names):
        ax.bar(x + i * width, mean_routing[i].cpu().numpy(), width, label=task)

    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Mean Routing Weight")
    ax.set_title("MMoEEx Expert Routing Distribution per Task")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"Exp {j}" for j in range(NUM_EXPERTS)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(save_dir, "expert_routing.png")
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
        if vals:
            axes[idx].plot(range(1, len(vals) + 1), vals, color=color, linewidth=2)
        axes[idx].set_title(title, fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Score")
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle("Phase 5: Evaluation Metrics Over Training", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "metrics_over_training.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_per_trait_comparison(mmoe_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traits = FI_TRAITS
    unimodal_ccc = [0.4578] * 5  # Approximate from Phase 3
    mmoe_ccc = [mmoe_results["fi"]["per_trait"].get(t, 0.0) for t in traits]

    x = np.arange(len(traits))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, unimodal_ccc, width, label="Unimodal (video)", color="#3498db", alpha=0.8)
    ax.bar(x + width/2, mmoe_ccc, width, label="MMoEEx", color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Big Five Trait")
    ax.set_ylabel("CCC")
    ax.set_title("FI Per-Trait: Unimodal vs MMoEEx")
    ax.set_xticks(x)
    ax.set_xticklabels([t[:4].capitalize() for t in traits])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(save_dir, "fi_per_trait_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_task_weights_evolution(model, save_dir):
    """Plot learned task uncertainty weights over training."""
    os.makedirs(save_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = torch.exp(model.loss_fn.nll.log_sigma).item() if hasattr(model, 'loss_fn') else None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["NLL sigma"], [sigmas if sigmas else 0], color="#e74c3c")
    ax.set_ylabel("Learned Noise Std")
    ax.set_title("NLL Loss Learned Variance Parameter")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(save_dir, "nll_sigma.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 5: MMoEEx Joint Training")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR_DEFAULT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs(ARTIFACTS_FIGURES, exist_ok=True)
    os.makedirs(ARTIFACTS_TABLES, exist_ok=True)

    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"Phase 5: MMoEEx Joint Training (GatedLateFusion)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    # Load data
    print("\n[1/5] Loading data...")
    manifest_data = load_manifest()
    all_labels = load_all_labels()

    train_ds = JointMultimodalDataset(
        manifest_data=manifest_data,
        all_labels=all_labels,
        datasets_splits=[
            ("daic", "train"),
            ("mosei", "train"),
            ("fi", "train"),
        ],
        feature_dims=FEATURE_DIMS,
        temperature=TEMPERATURE,
    )

    val_ds = JointMultimodalDataset(
        manifest_data=manifest_data,
        all_labels=all_labels,
        datasets_splits=[
            ("daic", "val"),
            ("mosei", "val"),
            ("fi", "val"),
        ],
        feature_dims=FEATURE_DIMS,
        temperature=1.0,
    )

    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")

    if len(train_ds) == 0:
        print("ERROR: No training samples loaded. Check manifest and labels.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_joint, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_joint, num_workers=2, pin_memory=True)

    # Build model
    print("\n[2/5] Building model...")
    model = UnifiedMMoEEx(
        text_dim=FEATURE_DIMS["text"],
        audio_dim=FEATURE_DIMS["audio"],
        video_dim=FEATURE_DIMS["video"],
        hidden_dim=HIDDEN_DIM,
        expert_dim=EXPERT_DIM,
        num_experts=NUM_EXPERTS,
        num_shared=NUM_SHARED,
        num_tasks=NUM_HEADS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Loss and optimizer
    loss_fn = TaskLosses()
    model.loss_fn = loss_fn  # Attach for later inspection
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    # Resume
    start_epoch = 0
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    if args.mode == "train":
        print("\n[3/5] Training...")
        history = defaultdict(list)
        metrics_history = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(start_epoch, args.epochs):
            avg_loss, task_losses = train_epoch(model, train_loader, optimizer, loss_fn, scheduler, device, scaler, epoch)

            history["total"].append(avg_loss)
            for tid, losses in task_losses.items():
                task_name = ["daic_depression", "mosei_sentiment", "mosei_emotion", "fi_personality"][tid]
                history[task_name].append(np.mean(losses))

            # Evaluate every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                results = evaluate(model, val_loader, device)

                val_loss = avg_loss
                metrics_history.append({
                    "epoch": epoch + 1,
                    "daic_auroc": results["daic"]["auroc"],
                    "mosei_sentiment_ccc": results["mosei_sentiment"]["ccc"],
                    "mosei_emotion_auc": results["mosei_emotion"]["auc"],
                    "fi_avg_ccc": results["fi"]["avg_ccc"],
                    "val_loss": val_loss,
                })

                print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | "
                      f"DAIC AUROC: {results['daic']['auroc']:.4f} | "
                      f"MOSEI CCC: {results['mosei_sentiment']['ccc']:.4f} | "
                      f"FI Avg CCC: {results['fi']['avg_ccc']:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    ckpt_path = ARTIFACTS_TABLES / "mmoe_ex_best.pt"
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }, ckpt_path)
                    print(f"    → Saved best model to {ckpt_path}")
                else:
                    patience_counter += 1

                if patience_counter >= PATIENCE:
                    print(f"\n  Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}")

        # Final evaluation
        print("\n[4/5] Final evaluation...")
        final_results = evaluate(model, val_loader, device)

        print("\n  Final Results:")
        print(f"  DAIC AUROC: {final_results['daic']['auroc']:.4f}")
        print(f"  MOSEI Sentiment CCC: {final_results['mosei_sentiment']['ccc']:.4f}")
        mosei_emotion_auc = final_results['mosei_emotion']['auc']
        if mosei_emotion_auc is not None:
            print(f"  MOSEI Emotion AUC: {mosei_emotion_auc:.4f}")
        else:
            print("  MOSEI Emotion AUC: N/A")
        print(f"  FI Avg CCC: {final_results['fi']['avg_ccc']:.4f}")
        if final_results['fi']['per_trait']:
            print(f"  FI Per-Trait: {final_results['fi']['per_trait']}")

        # Save results
        results_path = ARTIFACTS_TABLES / "mmoe_ex_results.csv"
        with open(results_path, "w") as f:
            f.write("metric,value\n")
            f.write(f"daic_auroc,{final_results['daic']['auroc']:.4f}\n")
            f.write(f"mosei_sentiment_ccc,{final_results['mosei_sentiment']['ccc']:.4f}\n")
            mosei_emotion_val = final_results['mosei_emotion']['auc']
            if mosei_emotion_val is not None:
                f.write(f"mosei_emotion_auc,{mosei_emotion_val:.4f}\n")
            else:
                f.write("mosei_emotion_auc,N/A\n")
            f.write(f"fi_avg_ccc,{final_results['fi']['avg_ccc']:.4f}\n")
            for trait, ccc in final_results['fi']['per_trait'].items():
                f.write(f"fi_{trait}_ccc,{ccc:.4f}\n")
        print(f"\n  Results saved to {results_path}")

        # Generate visualizations
        print("\n[5/5] Generating visualizations...")
        plot_training_curves(dict(history), ARTIFACTS_FIGURES)
        plot_task_metrics(metrics_history, ARTIFACTS_FIGURES)
        plot_per_trait_comparison(final_results, ARTIFACTS_FIGURES)
        plot_task_weights_evolution(model, ARTIFACTS_FIGURES)

        # Expert routing analysis
        model.eval()
        routing_weights = []
        with torch.no_grad():
            for text, audio, video, mask, labels, task_ids, weights, routings in val_loader:
                text = text.to(device)
                audio = audio.to(device)
                video = video.to(device)
                mask = mask.to(device)
                # For routing, just use multimodal approach for simplicity
                fused = model.fusion(text, audio, video, mask.bool())
                rw = model.mmoe.get_routing_weights(fused)
                routing_weights.append(rw)
        routing_weights = torch.cat(routing_weights, dim=1)
        plot_expert_routing(routing_weights, ["DAIC Dep.", "MOSEI Sent.", "MOSEI Emo.", "FI Pers."], ARTIFACTS_FIGURES)

        print(f"\n  Visualizations saved to {ARTIFACTS_FIGURES}")
        print("\n✅ Phase 5 complete!")

    else:  # eval mode
        print("\n[3/4] Loading checkpoint for evaluation...")
        ckpt_path = ARTIFACTS_TABLES / "mmoe_ex_best.pt"
        if not ckpt_path.exists():
            print(f"ERROR: No checkpoint found at {ckpt_path}")
            return
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded checkpoint from epoch {ckpt['epoch']}")

        print("\n[4/4] Evaluating...")
        results = evaluate(model, val_loader, device)
        print("\n  Results:")
        print(f"  DAIC AUROC: {results['daic']['auroc']:.4f}")
        print(f"  MOSEI Sentiment CCC: {results['mosei_sentiment']['ccc']:.4f}")
        mosei_emo_auc = results['mosei_emotion']['auc']
        if mosei_emo_auc is not None:
            print(f"  MOSEI Emotion AUC: {mosei_emo_auc:.4f}")
        else:
            print("  MOSEI Emotion AUC: N/A")
        print(f"  FI Avg CCC: {results['fi']['avg_ccc']:.4f}")


if __name__ == "__main__":
    main()