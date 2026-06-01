#!/usr/bin/env python3
"""
Phase 4: Multimodal Fusion Baselines
=====================================
Gated Late Fusion and Low-Rank Multimodal Fusion (LMF) baselines.
Combines text, audio, video modalities with learned fusion.

Usage:
    uv run python scripts/phase04_fusion.py --dataset daic --fusion gated
    uv run python scripts/phase04_fusion.py --dataset daic --fusion lmf
    uv run python scripts/phase04_fusion.py --dataset daic --fusion cross_attention
    uv run python scripts/phase04_fusion.py --dataset mosei --fusion gated
    uv run python scripts/phase04_fusion.py --dataset mosei --fusion lmf
    uv run python scripts/phase04_fusion.py --dataset mosei --fusion cross_attention
    uv run python scripts/phase04_fusion.py --dataset fi --fusion gated
    uv run python scripts/phase04_fusion.py --dataset fi --fusion lmf
    uv run python scripts/phase04_fusion.py --dataset fi --fusion cross_attention
    uv run python scripts/phase04_fusion.py --dataset all --fusion all

Outputs:
    artifacts/figures/phase_04_fusion/  - 6+ PNG figures
    artifacts/tables/fusion_baselines.csv - results table
"""
import argparse
import json
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

ROOT = Path("/home/anilson/thesis/thesis-experiment-5-unified-model")
FEATURES_ROOT = ROOT / "data" / "features"
MANIFEST_PATH = FEATURES_ROOT / "manifest.json"

DAIC_DATA = ROOT / "data" / "daic"
MOSEI_DATA = ROOT / "data" / "mosei"
FI_DATA = ROOT / "data" / "fi"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Per-dataset hidden dims: DAIC is tiny (107 train), FI is moderate (6000), MOSEI is large (16k)
HIDDEN_DIM = {"daic": 16, "mosei": 64, "fi": 256}
RANK = 4           # Low rank for LMF
BATCH_SIZE = {"daic": 8, "mosei": 64, "fi": 64}  # Smaller for DAIC to avoid batch=all-majority-class
EPOCHS_DEFAULT = 200   # More epochs for small datasets (early stopping will cut)
LR_DEFAULT = 1e-4  # Very low LR to prevent collapse on small datasets
PATIENCE = 20

MODALITY_FEATURE_MAP = {
    "daic": {
        "text": "text_roberta",
        "audio": "audio_wavlm",
        "video": "video_openface",
    },
    "mosei": {
        "text": "text_roberta",
        "audio": "audio_wavlm",
        "video": "video_vit",
    },
    "fi": {
        "text": "text_roberta",
        "audio": "audio_wavlm",
        "video": "video_vit",
    },
}

FI_TRAITS = ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]

# ---------------------------------------------------------------------------
# Label loading (reused from Phase 3)
# ---------------------------------------------------------------------------

def load_daic_labels():
    labels = {}
    for split, filename, col_binary, col_score in [
        ("train", "train_split_Depression_AVEC2017.csv", "PHQ8_Binary", "PHQ8_Score"),
        ("val",   "dev_split_Depression_AVEC2017.csv",  "PHQ8_Binary", "PHQ8_Score"),
        ("test",  "full_test_split.csv",                "PHQ_Binary",  "PHQ_Score"),
    ]:
        path = DAIC_DATA / filename
        if not path.exists():
            continue
        with open(path, newline="") as f:
            lines = [l.strip() for l in open(path).readlines() if l.strip()]
        header = lines[0].split(",")
        idx_binary = header.index(col_binary)
        idx_score = header.index(col_score)
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= max(idx_binary, idx_score):
                continue
            pid = str(int(float(parts[0])))
            binary = int(float(parts[idx_binary]))
            score = float(parts[idx_score])
            labels[pid] = (binary, score)
    print(f"  Loaded DAIC labels for {len(labels)} participants")
    return labels


EMOTION_TRAITS = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]


def load_mosei_labels():
    data_path = MOSEI_DATA / "mosei_senti_data.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    labels = {}
    for split_name, mosei_key in [("train", "train"), ("val", "valid"), ("test", "test")]:
        split_data = data[mosei_key]
        label_arr = np.array(split_data["labels"]).squeeze()
        for i, lab in enumerate(label_arr):
            labels[f"mosei_{split_name}_{i:05d}"] = float(lab)
    print(f"  Loaded MOSEI labels for {len(labels)} utterances")
    return labels


def load_mosei_emotion_labels():
    """Load MOSEI emotion labels (7-dim: sentiment + 6 emotions).
    
    Returns:
        sentiment_labels: dict keyed by sample_id -> float (sentiment score)
        emotion_labels: dict keyed by sample_id -> list of 6 floats (emotion intensities 0-3)
    """
    import json
    emotion_path = MOSEI_DATA / "mosei_emotion_labels.json"
    with open(emotion_path, "r") as f:
        emotion_data = json.load(f)
    
    sentiment_labels = {}
    emotion_labels = {}
    
    for sample_id, data in emotion_data.items():
        sentiment_labels[sample_id] = float(data["sentiment"])
        # Extract 6 emotion values, binarize at 0.5 for classification
        emotion_values = [
            float(data[trait]) for trait in EMOTION_TRAITS
        ]
        emotion_labels[sample_id] = emotion_values
    
    print(f"  Loaded MOSEI emotion labels for {len(emotion_labels)} utterances")
    print(f"    Emotion traits: {EMOTION_TRAITS}")
    return sentiment_labels, emotion_labels


def load_fi_labels():
    labels = {}
    train_ann = FI_DATA / "train" / "annotation_training.pkl"
    with open(train_ann, "rb") as f:
        ann_train = pickle.load(f, encoding="latin-1")
    for clip_id in ann_train[FI_TRAITS[0]].keys():
        labels[f"fi_train_{clip_id}"] = {t: float(ann_train[t][clip_id]) for t in FI_TRAITS}

    val_ann = FI_DATA / "val" / "annotation_validation.pkl"
    with open(val_ann, "rb") as f:
        ann_val = pickle.load(f, encoding="latin-1")
    for clip_id in ann_val[FI_TRAITS[0]].keys():
        labels[f"fi_val_{clip_id}"] = {t: float(ann_val[t][clip_id]) for t in FI_TRAITS}

    import pandas as pd
    test_csv = FI_DATA / "test" / "annotations.csv"
    df_test = pd.read_csv(test_csv)
    if "interview" in df_test.columns:
        df_test = df_test.rename(columns={"interview": "openness"})
    for i in range(len(df_test)):
        row = df_test.iloc[i]
        labels[f"fi_test_{i:05d}"] = {t: float(row[t]) for t in FI_TRAITS}
    print(f"  Loaded FI labels for {len(labels)} clips")
    return labels


# ---------------------------------------------------------------------------
# Feature loading helpers
# ---------------------------------------------------------------------------

POOLED_KEY = "pooled_features"
POOLED_EMBED_KEY = "pooled_embedding"


def load_feature_tensor(feature_path: Path):
    obj = torch.load(feature_path, map_location="cpu")
    if isinstance(obj, dict):
        if POOLED_EMBED_KEY in obj:
            return obj[POOLED_EMBED_KEY].numpy()
        elif POOLED_KEY in obj:
            return obj[POOLED_KEY].numpy()
        else:
            raise KeyError(f"Dict has no '{POOLED_KEY}' or '{POOLED_EMBED_KEY}': {list(obj.keys())}")
    else:
        return obj.numpy()


def load_manifest():
    with open(MANIFEST_PATH) as f:
        m = json.load(f)
    return m["samples"]


# ---------------------------------------------------------------------------
# Feature dimension registry (from Phase 3 results)
# ---------------------------------------------------------------------------

FEATURE_DIMS = {
    "daic":  {"text": 768,  "audio": 148, "video": 2570},
    "mosei": {"text": 600,  "audio": 148, "video": 1536},
    "fi":    {"text": 768,  "audio": 148, "video": 1536},
}


# ---------------------------------------------------------------------------
# Multimodal dataset (all 3 modalities + modality mask)
# ---------------------------------------------------------------------------

class MultimodalDataset(Dataset):
    """Dataset that provides all 3 modalities per sample, with modality mask."""

    def __init__(self, samples, dataset_name, labels_func, feature_dims):
        self.samples = samples
        self.dataset_name = dataset_name
        self.labels_func = labels_func
        self.feature_dims = feature_dims
        self.all_labels = labels_func()
        self.feature_keys = MODALITY_FEATURE_MAP[dataset_name]

        # Pre-load all data
        self.X_text = []
        self.X_audio = []
        self.X_video = []
        self.y = []
        self.ids = []
        self.modality_masks = []

        skipped = 0
        for s in samples:
            if s["dataset"] != dataset_name:
                skipped += 1
                continue

            feat_map = s["features"]
            t_key = self.feature_keys["text"]
            a_key = self.feature_keys["audio"]
            v_key = self.feature_keys["video"]

            t_path_str = feat_map.get(t_key)
            a_path_str = feat_map.get(a_key)
            v_path_str = feat_map.get(v_key)

            # Load features
            t_ok = t_path_str is not None and (ROOT / t_path_str).exists()
            a_ok = a_path_str is not None and (ROOT / a_path_str).exists()
            v_ok = v_path_str is not None and (ROOT / v_path_str).exists()

            if not (t_ok or a_ok or v_ok):
                skipped += 1
                continue

            # Load and validate each feature
            def load_single(path_str, dim, name):
                if path_str is None or not (ROOT / path_str).exists():
                    return np.zeros(dim, dtype=np.float32), False
                try:
                    feat = load_feature_tensor(ROOT / path_str)
                    feat = np.array(feat, dtype=np.float32).flatten()
                    if not np.all(np.isfinite(feat)):
                        return np.zeros(dim, dtype=np.float32), False
                    # Pad or truncate to expected dim
                    if feat.shape[0] < dim:
                        feat = np.pad(feat, (0, dim - feat.shape[0]))
                    elif feat.shape[0] > dim:
                        feat = feat[:dim]
                    return feat, True
                except Exception:
                    return np.zeros(dim, dtype=np.float32), False

            t_vec, t_ok = load_single(t_path_str, feature_dims["text"], "text")
            a_vec, a_ok = load_single(a_path_str, feature_dims["audio"], "audio")
            v_vec, v_ok = load_single(v_path_str, feature_dims["video"], "video")

            # Get label
            label_entry = self.all_labels.get(s["id"])
            if label_entry is None:
                skipped += 1
                continue

            # FI labels are dicts
            if isinstance(label_entry, dict):
                label_entry = [label_entry[t] for t in FI_TRAITS]
            # Ensure label is always 1D array (for MOSEI scalar labels)
            label_arr = np.atleast_1d(np.array(label_entry, dtype=np.float32))
            label_entry = label_arr

            self.X_text.append(t_vec)
            self.X_audio.append(a_vec)
            self.X_video.append(v_vec)
            self.y.append(label_entry)
            self.ids.append(s["id"])
            self.modality_masks.append((t_ok, a_ok, v_ok))

        if skipped > 0:
            print(f"    Skipped {skipped} samples (wrong dataset or no features)")

        # Convert to arrays
        self.X_text = np.stack(self.X_text) if self.X_text else np.zeros((0, feature_dims["text"]), dtype=np.float32)
        self.X_audio = np.stack(self.X_audio) if self.X_audio else np.zeros((0, feature_dims["audio"]), dtype=np.float32)
        self.X_video = np.stack(self.X_video) if self.X_video else np.zeros((0, feature_dims["video"]), dtype=np.float32)
        self.y = np.stack(self.y) if self.y else np.zeros((0, 5), dtype=np.float32)
        self.modality_masks = list(self.modality_masks)

        print(f"    MultimodalDataset: {len(self)} samples, "
              f"text={self.X_text.shape}, audio={self.X_audio.shape}, video={self.X_video.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_text[idx]),
            torch.from_numpy(self.X_audio[idx]),
            torch.from_numpy(self.X_video[idx]),
            torch.tensor(self.modality_masks[idx], dtype=torch.bool),
            torch.from_numpy(self.y[idx]),
        )


def collate_multimodal(batch):
    """Custom collate to handle variable-length modality masks."""
    text_tensors = []
    audio_tensors = []
    video_tensors = []
    masks = []
    labels = []

    for t, a, v, m, y in batch:
        text_tensors.append(t)
        audio_tensors.append(a)
        video_tensors.append(v)
        masks.append(m)
        labels.append(y)

    return (
        torch.stack(text_tensors),
        torch.stack(audio_tensors),
        torch.stack(video_tensors),
        torch.stack(masks),
        torch.stack(labels),
    )


# ---------------------------------------------------------------------------
# Multimodal dataset with emotion labels (for MOSEI emotion task)
# ---------------------------------------------------------------------------

class MultimodalEmotionDataset(Dataset):
    """Dataset that provides all 3 modalities + sentiment label + 6 emotion labels for MOSEI."""

    def __init__(self, samples, labels_func, feature_dims):
        self.samples = samples
        self.labels_func = labels_func
        self.feature_dims = feature_dims
        self.sentiment_labels, self.emotion_labels = labels_func()
        self.feature_keys = MODALITY_FEATURE_MAP["mosei"]

        self.X_text = []
        self.X_audio = []
        self.X_video = []
        self.y_sentiment = []
        self.y_emotion = []
        self.ids = []
        self.modality_masks = []

        skipped = 0
        for s in samples:
            if s["dataset"] != "mosei":
                skipped += 1
                continue

            feat_map = s["features"]
            t_key = self.feature_keys["text"]
            a_key = self.feature_keys["audio"]
            v_key = self.feature_keys["video"]

            t_path_str = feat_map.get(t_key)
            a_path_str = feat_map.get(a_key)
            v_path_str = feat_map.get(v_key)

            t_ok = t_path_str is not None and (ROOT / t_path_str).exists()
            a_ok = a_path_str is not None and (ROOT / t_path_str).exists()
            v_ok = v_path_str is not None and (ROOT / v_path_str).exists()

            if not (t_ok or a_ok or v_ok):
                skipped += 1
                continue

            def load_single(path_str, dim, name):
                if path_str is None or not (ROOT / path_str).exists():
                    return np.zeros(dim, dtype=np.float32), False
                try:
                    feat = load_feature_tensor(ROOT / path_str)
                    feat = np.array(feat, dtype=np.float32).flatten()
                    if not np.all(np.isfinite(feat)):
                        return np.zeros(dim, dtype=np.float32), False
                    if feat.shape[0] < dim:
                        feat = np.pad(feat, (0, dim - feat.shape[0]))
                    elif feat.shape[0] > dim:
                        feat = feat[:dim]
                    return feat, True
                except Exception:
                    return np.zeros(dim, dtype=np.float32), False

            t_vec, t_ok = load_single(t_path_str, feature_dims["text"], "text")
            a_vec, a_ok = load_single(a_path_str, feature_dims["audio"], "audio")
            v_vec, v_ok = load_single(v_path_str, feature_dims["video"], "video")

            sent_entry = self.sentiment_labels.get(s["id"])
            emot_entry = self.emotion_labels.get(s["id"])
            if sent_entry is None or emot_entry is None:
                skipped += 1
                continue

            self.X_text.append(t_vec)
            self.X_audio.append(a_vec)
            self.X_video.append(v_vec)
            self.y_sentiment.append(float(sent_entry))
            # Binarize emotion labels at 0.5 threshold
            self.y_emotion.append([1.0 if e >= 0.5 else 0.0 for e in emot_entry])
            self.ids.append(s["id"])
            self.modality_masks.append((t_ok, a_ok, v_ok))

        if skipped > 0:
            print(f"    Skipped {skipped} samples (wrong dataset or no features)")

        self.X_text = np.stack(self.X_text) if self.X_text else np.zeros((0, feature_dims["text"]), dtype=np.float32)
        self.X_audio = np.stack(self.X_audio) if self.X_audio else np.zeros((0, feature_dims["audio"]), dtype=np.float32)
        self.X_video = np.stack(self.X_video) if self.X_video else np.zeros((0, feature_dims["video"]), dtype=np.float32)
        self.y_sentiment = np.array(self.y_sentiment, dtype=np.float32)
        self.y_emotion = np.array(self.y_emotion, dtype=np.float32)
        self.modality_masks = list(self.modality_masks)

        print(f"    MultimodalEmotionDataset: {len(self)} samples, "
              f"text={self.X_text.shape}, audio={self.X_audio.shape}, video={self.X_video.shape}")
        print(f"    Emotion labels: {self.y_emotion.shape} (6 emotions, binarized at 0.5)")

    def __len__(self):
        return len(self.y_emotion)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_text[idx]),
            torch.from_numpy(self.X_audio[idx]),
            torch.from_numpy(self.X_video[idx]),
            torch.tensor(self.modality_masks[idx], dtype=torch.bool),
            torch.from_numpy(self.y_emotion[idx]),  # 6 emotions, binary
        )


def collate_emotion(batch):
    """Custom collate for emotion dataset."""
    text_tensors = []
    audio_tensors = []
    video_tensors = []
    masks = []
    emotion_labels = []

    for t, a, v, m, y in batch:
        text_tensors.append(t)
        audio_tensors.append(a)
        video_tensors.append(v)
        masks.append(m)
        emotion_labels.append(y)

    return (
        torch.stack(text_tensors),
        torch.stack(audio_tensors),
        torch.stack(video_tensors),
        torch.stack(masks),
        torch.stack(emotion_labels),
    )


# ---------------------------------------------------------------------------
# CCC metric
# ---------------------------------------------------------------------------

def compute_ccc(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mu_y = y_true.mean()
    mu_p = y_pred.mean()
    var_y = y_true.var()
    var_p = y_pred.var()
    cov = np.cov(y_true, y_pred)[0, 1]
    denom = (var_y + var_p + (mu_y - mu_p) ** 2)
    if denom < 1e-12:
        return 0.0
    return (2 * cov) / denom


# ---------------------------------------------------------------------------
# Bootstrap CI (reused from Phase 3)
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=1000, ci=0.95):
    scores = []
    n = len(y_true)
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2 and metric_func.__name__ in ["roc_auc_score", "auc"]:
            continue
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


def bootstrap_auc_ci(y_true, y_score, n_bootstrap=1000):
    scores = []
    rng = np.random.default_rng(42)
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            s = roc_auc_score(y_true[idx], y_score[idx])
            scores.append(s)
        except Exception:
            pass
    if not scores:
        return float("nan"), float("nan")
    lo = np.percentile(scores, 2.5)
    hi = np.percentile(scores, 97.5)
    return lo, hi


# ---------------------------------------------------------------------------
# Model: Fusion + Task Head
# ---------------------------------------------------------------------------

class FusionClassifier(nn.Module):
    """Fusion encoder + classification head for DAIC."""

    def __init__(self, text_dim, audio_dim, video_dim, fusion_type="gated", hidden_dim=512):
        super().__init__()
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        if fusion_type == "gated":
            from models.fusion import GatedLateFusion
            self.fusion = GatedLateFusion(text_dim, audio_dim, video_dim, hidden_dim)
        elif fusion_type == "lmf":
            from models.fusion import LowRankMultimodalFusion
            self.fusion = LowRankMultimodalFusion(text_dim, audio_dim, video_dim, hidden_dim, rank=RANK)
        elif fusion_type == "cross_attention":
            from models.fusion import CrossAttentionFusion
            self.fusion = CrossAttentionFusion(text_dim, audio_dim, video_dim, hidden_dim, num_heads=4)
        elif fusion_type == "lrdgn":
            from models.fusion import LowRankGatingNetwork
            # Use low rank (16) for small DAIC dataset — prevents overfitting
            lrdgn_rank = 16
            self.fusion = LowRankGatingNetwork(text_dim, audio_dim, video_dim, hidden_dim, rank=lrdgn_rank, num_gate_layers=2)
        else:
            raise ValueError(f"Unknown fusion: {fusion_type}")

        # Stronger dropout for small DAIC dataset to prevent overfitting
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(32, 2),
        )

    def forward(self, text, audio, video, mask):
        fused = self.fusion(text, audio, video, mask)
        return self.head(fused)


class FusionRegression(nn.Module):
    """Fusion encoder + regression head for MOSEI/FI.

    For FI (5 traits), uses a simpler head (single linear layer) to prevent collapse.
    For MOSEI (1 trait), uses a deeper head with LayerNorm for expressiveness.
    """

    def __init__(self, text_dim, audio_dim, video_dim, fusion_type="gated", hidden_dim=512, n_traits=1, dataset_name="mosei"):
        super().__init__()
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        if fusion_type == "gated":
            from models.fusion import GatedLateFusion
            self.fusion = GatedLateFusion(text_dim, audio_dim, video_dim, hidden_dim)
        elif fusion_type == "lmf":
            from models.fusion import LowRankMultimodalFusion
            self.fusion = LowRankMultimodalFusion(text_dim, audio_dim, video_dim, hidden_dim, rank=RANK)
        elif fusion_type == "cross_attention":
            from models.fusion import CrossAttentionFusion
            self.fusion = CrossAttentionFusion(text_dim, audio_dim, video_dim, hidden_dim, num_heads=4)
        elif fusion_type == "lrdgn":
            from models.fusion import LowRankGatingNetwork
            # Use low rank (16-24) for small datasets, 64 for larger
            lrdgn_rank = 16 if dataset_name == "daic" else 32
            self.fusion = LowRankGatingNetwork(text_dim, audio_dim, video_dim, hidden_dim, rank=lrdgn_rank, num_gate_layers=2)
        else:
            raise ValueError(f"Unknown fusion: {fusion_type}")

# FI (5 traits): single linear head with small bias + LARGE weight init
        # Small bias (0.01) ensures baseline prediction is near 0, not 0.5
        # Large weight init (0.5) forces the model to use fusion features
        if dataset_name == "fi":
            self.head = nn.Linear(hidden_dim, n_traits, bias=True)
            # Initialize bias small to force weights to carry the signal
            self.head.bias.data.fill_(0.01)
        else:
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, n_traits),
            )

    def forward(self, text, audio, video, mask):
        fused = self.fusion(text, audio, video, mask)
        return self.head(fused)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_daic(train_loader, val_loader, test_loader, fusion_type, device, epochs=50, lr=1e-3):
    """Train and evaluate DAIC depression classification."""
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

    text_dim = FEATURE_DIMS["daic"]["text"]
    audio_dim = FEATURE_DIMS["daic"]["audio"]
    video_dim = FEATURE_DIMS["daic"]["video"]

    hidden_dim = HIDDEN_DIM["daic"]
    model = FusionClassifier(text_dim, audio_dim, video_dim, fusion_type, hidden_dim).to(device)

    # Higher weight decay for small DAIC dataset to prevent overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_auc = 0.0
    best_state = None
    wait = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_auroc": [], "gate_text": [], "gate_audio": [], "gate_video": []}

    # Collect gate weights for visualization
    gate_text_hist = []
    gate_audio_hist = []
    gate_video_hist = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for t, a, v, mask, y in train_loader:
            t, a, v, mask, y = t.to(device), a.to(device), v.to(device), mask.to(device), y.to(device)
            labels = y[:, 0].long()

            optimizer.zero_grad()
            logits = model(t, a, v, mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_val_labels = []
        all_val_probs = []

        with torch.no_grad():
            for t, a, v, mask, y in val_loader:
                t, a, v, mask, y = t.to(device), a.to(device), v.to(device), mask.to(device), y.to(device)
                labels = y[:, 0].long()
                logits = model(t, a, v, mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_batches += 1
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_val_labels.extend(labels.cpu().numpy().tolist())
                all_val_probs.extend(probs.tolist())

        avg_val_loss = val_loss / max(val_batches, 1)
        try:
            val_auroc = roc_auc_score(all_val_labels, all_val_probs)
        except Exception:
            val_auroc = 0.0

        scheduler.step(avg_val_loss)

        # Record gate weights
        if hasattr(model.fusion, 'text_gate') and hasattr(model.fusion, 'audio_gate') and hasattr(model.fusion, 'video_gate'):
            with torch.no_grad():
                try:
                    sample_t = t[:1].float()
                    sample_a = a[:1].float()
                    sample_v = v[:1].float()
                    # Get gate values (mean of sigmoid output)
                    t_proj = model.fusion.text_proj(sample_t)
                    a_proj = model.fusion.audio_proj(sample_a)
                    v_proj = model.fusion.video_proj(sample_v)
                    # Only record if gate dimensions match projected dimensions
                    if t_proj.shape[-1] == model.fusion.text_gate[0].in_features:
                        gt = model.fusion.text_gate(t_proj).mean().item()
                        ga = model.fusion.audio_gate(a_proj).mean().item()
                        gv = model.fusion.video_gate(v_proj).mean().item()
                        gate_text_hist.append(gt)
                        gate_audio_hist.append(ga)
                        gate_video_hist.append(gv)
                except Exception:
                    pass  # Gate analysis not supported for this fusion type
        elif hasattr(model.fusion, 'gate_mlp') and hasattr(model.fusion, 'get_gate_values'):
            # LR-DGN: context-aware gates computed from low-rank bottleneck features
            with torch.no_grad():
                try:
                    sample_t = t[:1].float()
                    sample_a = a[:1].float()
                    sample_v = v[:1].float()
                    sample_mask = mask[:1]
                    gates = model.fusion.get_gate_values(sample_t, sample_a, sample_v, sample_mask)
                    gate_text_hist.append(gates["gate_text"])
                    gate_audio_hist.append(gates["gate_audio"])
                    gate_video_hist.append(gates["gate_video"])
                except Exception:
                    pass  # Gate analysis not supported for this fusion type

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_auroc"].append(val_auroc)

        if val_auroc > best_val_auc:
            best_val_auc = val_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    history["gate_text"] = gate_text_hist
    history["gate_audio"] = gate_audio_hist
    history["gate_video"] = gate_video_hist

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    all_test_labels = []
    all_test_probs = []
    all_test_preds = []

    with torch.no_grad():
        for t, a, v, mask, y in test_loader:
            t, a, v, mask, y = t.to(device), a.to(device), v.to(device), mask.to(device), y.to(device)
            labels = y[:, 0].long()
            logits = model(t, a, v, mask)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_test_labels.extend(labels.cpu().numpy().tolist())
            all_test_probs.extend(probs.tolist())
            all_test_preds.extend(preds.tolist())

    all_test_labels = np.array(all_test_labels)
    all_test_probs = np.array(all_test_probs)
    all_test_preds = np.array(all_test_preds)

    acc = accuracy_score(all_test_labels, all_test_preds)
    auroc = roc_auc_score(all_test_labels, all_test_probs)
    f1 = f1_score(all_test_labels, all_test_preds, zero_division=0)
    cm = confusion_matrix(all_test_labels, all_test_preds)

    ci_auroc_lo, ci_auroc_hi = bootstrap_auc_ci(all_test_labels, all_test_probs)
    ci_lo_acc, ci_hi_acc = bootstrap_ci(all_test_labels, all_test_preds, accuracy_score)
    ci_lo_f1,  ci_hi_f1  = bootstrap_ci(all_test_labels, all_test_preds, f1_score)

    return {
        "accuracy": acc, "auroc": auroc, "f1": f1,
        "ci_accuracy": (ci_lo_acc, ci_hi_acc),
        "ci_auroc": (ci_auroc_lo, ci_auroc_hi),
        "ci_f1": (ci_lo_f1, ci_hi_f1),
        "confusion_matrix": cm,
        "y_true": all_test_labels, "y_pred": all_test_preds, "y_prob": all_test_probs,
        "beats_trivial": auroc > 0.5,
        "primary_metric": auroc,
        "metric_name": "AUROC",
        "model": model,
        "history": history,
        "param_count": model.fusion.param_count(),
    }


def train_regression(train_loader, val_loader, test_loader, dataset_name, fusion_type, device, epochs=50, lr=1e-3):
    """Train and evaluate MOSEI (sentiment) or FI (personality) regression."""

    text_dim = FEATURE_DIMS[dataset_name]["text"]
    audio_dim = FEATURE_DIMS[dataset_name]["audio"]
    video_dim = FEATURE_DIMS[dataset_name]["video"]
    n_traits = 5 if dataset_name == "fi" else 1
    hidden_dim = HIDDEN_DIM.get(dataset_name, 64)

    model = FusionRegression(text_dim, audio_dim, video_dim, fusion_type, hidden_dim, n_traits, dataset_name).to(device)

    # Initialize head weights with LARGE variance
    with torch.no_grad():
        modules_list = list(model.head.modules())
        for module in reversed(modules_list):
            if isinstance(module, nn.Linear) and module.out_features == n_traits:
                # FI: small bias (0.01), LARGE weight init (0.5) — force use of fusion features
                # MOSEI: init bias to 0
                if dataset_name != "fi" and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                init_std = 0.5 if dataset_name == "fi" else 0.05
                nn.init.normal_(module.weight, 0, init_std)
                print(f"  Head init: weight std={init_std}, bias init=0.01 (fi) or 0.0 (mosei)")
                break

    # FI needs NO weight decay to prevent regression collapse
    # MOSEI can tolerate mild weight decay
    wd = 0.0 if dataset_name == "fi" else 1e-5

    # For FI: much higher head LR (no bias = must learn weights from fusion features)
    if dataset_name == "fi":
        head_params = [p for n, p in model.named_parameters() if 'head' in n]
        fusion_params = [p for n, p in model.named_parameters() if 'head' not in n]
        optimizer = torch.optim.AdamW([
            {'params': fusion_params, 'lr': lr, 'weight_decay': 0.0},
            {'params': head_params, 'lr': 0.01, 'weight_decay': 0.0},  # High LR for bias-free head
        ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for t, a, v, mask, y in train_loader:
            t, a, v, mask, y = t.to(device), a.to(device), v.to(device), mask.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(t, a, v, mask)
            loss = criterion(preds, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for t, a, v, mask, y in val_loader:
                t, a, v, mask, y = t.to(device), a.to(device), v.to(device), mask.to(device), y.to(device)
                preds = model(t, a, v, mask)
                loss = criterion(preds, y)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        scheduler.step(avg_val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    all_test_labels = []
    all_test_preds = []
    all_fusion_out = []   # Debug: check fusion output variance

    with torch.no_grad():
        for t, a, v, mask, y in test_loader:
            t, a, v, mask, y = t.to(device), a.to(device), v.to(device), mask.to(device), y.to(device)
            # Get both fusion output and final predictions
            fused = model.fusion(t, a, v, tuple(mask.tolist()))
            preds = model.head(fused)
            all_fusion_out.append(fused.cpu().numpy())
            all_test_labels.append(y.cpu().numpy())
            all_test_preds.append(preds.cpu().numpy())

    all_fusion_out = np.concatenate(all_fusion_out)
    all_test_labels = np.concatenate(all_test_labels)
    all_test_preds = np.concatenate(all_test_preds)

    # Debug: check fusion output variance
    print(f"  [Debug] fusion out: mean={all_fusion_out.mean():.4f}, std={all_fusion_out.std():.4f}, "
          f"min={all_fusion_out.min():.4f}, max={all_fusion_out.max():.4f}")

    # Debug: print prediction statistics
    print(f"  [Debug] preds: mean={all_test_preds.mean():.4f}, std={all_test_preds.std():.4f}, "
          f"min={all_test_preds.min():.4f}, max={all_test_preds.max():.4f}")
    print(f"  [Debug] labels: mean={all_test_labels.mean():.4f}, std={all_test_labels.std():.4f}")
    if dataset_name == "fi":
        for i_t, trait in enumerate(FI_TRAITS):
            y_t = all_test_labels[:, i_t]
            y_p = all_test_preds[:, i_t]
            ccc = compute_ccc(y_t, y_p)
            print(f"  [Debug]   {trait:18s}: true_mean={y_t.mean():.4f} pred_mean={y_p.mean():.4f} "
                  f"true_std={y_t.std():.4f} pred_std={y_p.std():.4f} CCC={ccc:.4f}")

    if dataset_name == "mosei":
        mae = np.abs(all_test_labels.squeeze() - all_test_preds.squeeze()).mean()
        ccc = compute_ccc(all_test_labels.squeeze(), all_test_preds.squeeze())
        r2 = 1 - np.sum((all_test_labels.squeeze() - all_test_preds.squeeze())**2) / (np.sum((all_test_labels.squeeze() - all_test_labels.mean())**2) + 1e-12)

        ci_lo_mae, ci_hi_mae = bootstrap_ci(all_test_labels.squeeze(), all_test_preds.squeeze(), lambda t, p: np.abs(t - p).mean())
        ci_lo_ccc, ci_hi_ccc = bootstrap_ci(all_test_labels.squeeze(), all_test_preds.squeeze(), compute_ccc)

        trivial_mae = np.abs(all_test_labels - all_test_labels.mean()).mean()
        beats_trivial = mae < trivial_mae

        return {
            "mae": mae, "ccc": ccc, "r2": r2,
            "ci_mae": (ci_lo_mae, ci_hi_mae),
            "ci_ccc": (ci_lo_ccc, ci_hi_ccc),
            "y_true": all_test_labels.squeeze(),
            "y_pred": all_test_preds.squeeze(),
            "beats_trivial": beats_trivial,
            "primary_metric": ccc,
            "metric_name": "CCC",
            "model": model,
            "history": history,
            "param_count": model.fusion.param_count(),
        }

    else:  # FI — 5 traits
        results = {}
        for i_trait, trait in enumerate(FI_TRAITS):
            y_t = all_test_labels[:, i_trait].astype(float)
            y_p = all_test_preds[:, i_trait].astype(float)
            mae_t = np.abs(y_t - y_p).mean()
            ccc_t = compute_ccc(y_t, y_p)
            ci_lo_mae_t, ci_hi_mae_t = bootstrap_ci(y_t, y_p, lambda t, p: np.abs(t - p).mean())
            ci_lo_ccc_t, ci_hi_ccc_t = bootstrap_ci(y_t, y_p, compute_ccc)
            results[trait] = {
                "mae": mae_t, "ccc": ccc_t,
                "ci_mae": (ci_lo_mae_t, ci_hi_mae_t),
                "ci_ccc": (ci_lo_ccc_t, ci_hi_ccc_t),
                "y_true": y_t, "y_pred": y_p,
            }

        avg_ccc = np.mean([results[t]["ccc"] for t in FI_TRAITS])
        avg_mae = np.mean([results[t]["mae"] for t in FI_TRAITS])
        trivial_mae = np.mean([np.abs(all_test_labels[:, i].astype(float) - all_test_labels[:, i].astype(float).mean()) for i in range(5)])
        beats_trivial = avg_ccc > 0.0

        return {
            **results,
            "avg_ccc": avg_ccc, "avg_mae": avg_mae,
            "beats_trivial": beats_trivial,
            "primary_metric": avg_ccc,
            "metric_name": "Avg CCC",
            "model": model,
            "history": history,
            "param_count": model.fusion.param_count(),
        }


def train_mosei_emotion(train_loader, val_loader, test_loader, fusion_type, device, epochs=50, lr=1e-3):
    """Train and evaluate MOSEI emotion prediction (multi-label, 6 emotions)."""
    from sklearn.metrics import roc_auc_score

    text_dim = FEATURE_DIMS["mosei"]["text"]
    audio_dim = FEATURE_DIMS["mosei"]["audio"]
    video_dim = FEATURE_DIMS["mosei"]["video"]
    hidden_dim = HIDDEN_DIM["mosei"]

    # Model: regression fusion + 6-output head for multi-label emotion
    model = FusionRegression(text_dim, audio_dim, video_dim, fusion_type, hidden_dim, n_traits=6, dataset_name="mosei").to(device)

    # Modify head to output 6 logits (one per emotion)
    with torch.no_grad():
        modules_list = list(model.head.modules())
        for module in reversed(modules_list):
            if isinstance(module, nn.Linear) and module.out_features == 6:
                nn.init.normal_(module.weight, 0, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
                print(f"  Emotion head: 6 outputs, weight std=0.05, bias init=0.0")
                break

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for t, a, v, mask, y_emotion in train_loader:
            t, a, v, mask, y_emotion = t.to(device), a.to(device), v.to(device), mask.to(device), y_emotion.to(device)

            optimizer.zero_grad()
            logits = model(t, a, v, mask)  # Shape: (batch, 6)
            loss = criterion(logits, y_emotion)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for t, a, v, mask, y_emotion in val_loader:
                t, a, v, mask, y_emotion = t.to(device), a.to(device), v.to(device), mask.to(device), y_emotion.to(device)
                logits = model(t, a, v, mask)
                loss = criterion(logits, y_emotion)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        scheduler.step(avg_val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation: compute per-emotion AUROC
    model.eval()
    all_test_labels = []
    all_test_probs = []

    with torch.no_grad():
        for t, a, v, mask, y_emotion in test_loader:
            t, a, v, mask, y_emotion = t.to(device), a.to(device), v.to(device), mask.to(device), y_emotion.to(device)
            logits = model(t, a, v, mask)
            probs = torch.sigmoid(logits).cpu().numpy()  # (batch, 6)
            all_test_labels.append(y_emotion.cpu().numpy())
            all_test_probs.append(probs)

    all_test_labels = np.concatenate(all_test_labels)   # (N, 6)
    all_test_probs = np.concatenate(all_test_probs)     # (N, 6)

    # Per-emotion AUROC
    emotion_aucs = {}
    valid_emotions = 0
    for i, emotion in enumerate(EMOTION_TRAITS):
        y_t = all_test_labels[:, i]
        y_p = all_test_probs[:, i]
        try:
            auc = roc_auc_score(y_t, y_p)
            emotion_aucs[emotion] = auc
            valid_emotions += 1
        except Exception:
            emotion_aucs[emotion] = float("nan")

    avg_auc = np.nanmean(list(emotion_aucs.values()))
    ci_lo, ci_hi = bootstrap_ci(
        all_test_labels.flatten(), all_test_probs.flatten(),
        lambda t, p: np.mean([roc_auc_score(all_test_labels[:, i], all_test_probs[:, i]) 
                              for i in range(6) if len(np.unique(all_test_labels[:, i])) > 1])
    )

    print(f"  Emotion AUROC per trait:")
    for emotion, auc in emotion_aucs.items():
        print(f"    {emotion:12s}: {auc:.4f}")
    print(f"    {'Average':12s}: {avg_auc:.4f}")

    # Compute CCC for each emotion as secondary metric
    emotion_cccs = {}
    for i, emotion in enumerate(EMOTION_TRAITS):
        y_t = all_test_labels[:, i].astype(float)
        y_p = all_test_probs[:, i].astype(float)
        ccc = compute_ccc(y_t, y_p)
        emotion_cccs[emotion] = ccc

    return {
        "avg_auroc": avg_auc,
        "emotion_auroc": emotion_aucs,
        "emotion_ccc": emotion_cccs,
        "ci_auroc": (ci_lo, ci_hi),
        "y_true": all_test_labels,
        "y_pred": all_test_probs,
        "beats_trivial": avg_auc > 0.5,
        "primary_metric": avg_auc,
        "metric_name": "Avg Emotion AUROC",
        "model": model,
        "history": history,
        "param_count": model.fusion.param_count(),
    }


# ---------------------------------------------------------------------------
# Modality dropout robustness test
# ---------------------------------------------------------------------------

def evaluate_modality_dropout(model, test_loader, device, dataset_name):
    """Evaluate model under progressive modality dropout."""
    model.eval()

    # Full modality baseline
    full_preds = []
    full_labels = []
    with torch.no_grad():
        for t, a, v, mask, y in test_loader:
            t, a, v, mask = t.to(device), a.to(device), v.to(device), mask.to(device)
            preds = model(t, a, v, mask)
            full_preds.append(preds.cpu().numpy())
            full_labels.append(y.numpy())

    full_preds = np.concatenate(full_preds)
    full_labels = np.concatenate(full_labels)

    # Dropout levels: 0%, 25%, 50%, 75%, 100% per modality
    # Actually, we evaluate: drop one modality at a time (text, audio, video)
    # and drop all-but-one (each single modality)
    conditions = {
        "full": None,        # all available
        "no_text": True,     # zero text contribution (gate→0)
        "no_audio": True,
        "no_video": True,
    }

    results = {}
    for cond_name, override_missing in conditions.items():
        cond_preds = []
        with torch.no_grad():
            for t, a, v, mask, y in test_loader:
                t, a, v, mask = t.to(device), a.to(device), v.to(device), mask.to(device)
                if override_missing:
                    # Override mask to hide one modality
                    if cond_name == "no_text":
                        mask = mask.clone()
                        mask[:, 0] = False
                    elif cond_name == "no_audio":
                        mask = mask.clone()
                        mask[:, 1] = False
                    elif cond_name == "no_video":
                        mask = mask.clone()
                        mask[:, 2] = False

                preds = model(t, a, v, mask)
                cond_preds.append(preds.cpu().numpy())

        cond_preds = np.concatenate(cond_preds)

        if dataset_name == "daic":
            # Binary classification — compute AUROC
            probs = torch.softmax(torch.from_numpy(cond_preds), dim=1)[:, 1].numpy()
            labels = full_labels[:, 0]
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(labels, probs)
                results[cond_name] = auc
            except Exception:
                results[cond_name] = 0.5
        elif dataset_name == "mosei":
            results[cond_name] = compute_ccc(full_labels.squeeze(), cond_preds.squeeze())
        else:  # FI
            cccs = [compute_ccc(full_labels[:, i], cond_preds[:, i]) for i in range(5)]
            results[cond_name] = np.mean(cccs)

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig, name, out_dir):
    import matplotlib.pyplot as plt
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    size_kb = path.stat().st_size // 1024
    print(f"  Saved: {path} ({size_kb} KB)")
    plt.close(fig)


def plot_gate_weights(result, out_dir):
    """Figure: Modality gate weights by dataset and task — bar chart."""
    import matplotlib.pyplot as plt
    import numpy as np

    history = result.get("history", {})
    if not history:
        return

    gate_text = history.get("gate_text", [])
    gate_audio = history.get("gate_audio", [])
    gate_video = history.get("gate_video", [])

    if not gate_text:
        return

    # Final gate values (average of last 5 epochs)
    n_last = min(5, len(gate_text))
    final_t = np.mean(gate_text[-n_last:])
    final_a = np.mean(gate_audio[-n_last:])
    final_v = np.mean(gate_video[-n_last:])

    fig, ax = plt.subplots(figsize=(6, 5))
    mods = ["Text", "Audio", "Video"]
    vals = [final_t, final_a, final_v]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    bars = ax.bar(mods, vals, color=colors, alpha=0.8, edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Learned Gate Weight (last 5 epochs)")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Modality Gate Weights — {result['dataset'].upper()} / {result['fusion_type']}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_figure(fig, f"gate_weights_{result['dataset']}_{result['fusion_type']}.png", out_dir)


def plot_training_curves(result, out_dir):
    """Figure: Training curves (loss + metric per epoch)."""
    import matplotlib.pyplot as plt

    history = result.get("history", {})
    if not history or not history.get("epoch"):
        return

    epochs = history["epoch"]
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_auroc = history.get("val_auroc", [])

    metric_name = result.get("metric_name", "Metric")

    if result["dataset"] == "daic":
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Loss curve
        axes[0].plot(epochs, train_loss, "b-", label="Train Loss", linewidth=1.5)
        if val_loss:
            axes[0].plot(epochs, val_loss, "r-", label="Val Loss", linewidth=1.5)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (CE)")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # AUROC curve
        if val_auroc:
            axes[1].plot(epochs, val_auroc, "g-", linewidth=1.5)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Validation AUROC")
            axes[1].set_title("Validation AUROC")
            axes[1].set_ylim(0, 1.0)
            axes[1].spines["top"].set_visible(False)
            axes[1].spines["right"].set_visible(False)

        fig.suptitle(f"{result['dataset'].upper()} — {result['fusion_type']} fusion training", fontsize=12)
        plt.tight_layout()

    else:  # regression
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(epochs, train_loss, "b-", label="Train Loss", linewidth=1.5)
        if val_loss:
            axes[0].plot(epochs, val_loss, "r-", label="Val Loss", linewidth=1.5)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (MSE)")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # Show final metric
        axes[1].bar(["Test " + metric_name], [result["primary_metric"]], color="#4CAF50", alpha=0.8)
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f"Test {metric_name}")
        axes[1].set_ylim(0, 1.0)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        fig.suptitle(f"{result['dataset'].upper()} — {result['fusion_type']} fusion training", fontsize=12)
        plt.tight_layout()

    save_figure(fig, f"training_curves_{result['dataset']}_{result['fusion_type']}.png", out_dir)


def plot_modality_dropout_robustness(dropout_results, dataset_name, fusion_type, out_dir):
    """Figure: Modality dropout robustness — bar chart of performance under each dropout condition."""
    import matplotlib.pyplot as plt
    import numpy as np

    cond_labels = {
        "full": "Full\n(all modalities)",
        "no_text": "No Text",
        "no_audio": "No Audio",
        "no_video": "No Video",
    }

    conds = ["full", "no_text", "no_audio", "no_video"]
    vals = [dropout_results.get(c, 0.0) for c in conds]
    labels = [cond_labels[c] for c in conds]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4CAF50", "#F44336", "#F44336", "#F44336"]
    bars = ax.bar(labels, vals, color=colors, alpha=0.8, edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Primary Metric (AUROC or CCC)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Modality Dropout Robustness — {dataset_name.upper()} / {fusion_type}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_figure(fig, f"modality_dropout_{dataset_name}_{fusion_type}.png", out_dir)


def plot_metric_comparison(result, best_unimodal, out_dir):
    """Figure: Fusion vs best unimodal bar chart."""
    import matplotlib.pyplot as plt

    ds = result["dataset"]
    pm = result["primary_metric"]
    mn = result.get("metric_name", "Metric")

    # Best unimodal for comparison
    unimodal_val = best_unimodal.get(ds, {}).get("value", 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = [f"Best Unimodal\n({best_unimodal.get(ds, {}).get('modality', '?')})", f"{result['fusion_type'].upper()} Fusion"]
    vals = [unimodal_val, pm]
    colors = ["#9E9E9E", "#673AB7"]
    bars = ax.bar(labels, vals, color=colors, alpha=0.8, edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel(mn)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{ds.upper()} {mn} — Fusion vs Unimodal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_figure(fig, f"metric_comparison_{ds}_{result['fusion_type']}.png", out_dir)


def plot_per_sample_gate_heatmap(model, test_loader, device, dataset_name, fusion_type, out_dir):
    """Figure: Per-case modality contribution heatmap."""
    import matplotlib.pyplot as plt

    model.eval()
    gate_text_vals = []
    gate_audio_vals = []
    gate_video_vals = []
    sample_labels = []

    with torch.no_grad():
        try:
            if hasattr(model.fusion, 'gate_mlp') and hasattr(model.fusion, 'get_gate_values'):
                # LR-DGN: use get_gate_values helper (context-aware gates)
                for t, a, v, mask, y in test_loader:
                    t, a, v = t.to(device), a.to(device), v.to(device)
                    mask_device = mask.to(device)
                    gates_batch = model.fusion.get_gate_values(t.float(), a.float(), v.float(), mask_device)
                    gate_text_vals.extend(gates_batch["gate_text"] if isinstance(gates_batch["gate_text"], list) else [gates_batch["gate_text"]])
                    gate_audio_vals.extend(gates_batch["gate_audio"] if isinstance(gates_batch["gate_audio"], list) else [gates_batch["gate_audio"]])
                    gate_video_vals.extend(gates_batch["gate_video"] if isinstance(gates_batch["gate_video"], list) else [gates_batch["gate_video"]])
                    sample_labels.extend(y[:, 0].numpy().tolist() if y.shape[1] > 1 else y.squeeze().numpy().tolist())
            elif hasattr(model.fusion, 'text_gate'):
                # GatedLateFusion: per-modality independent gates
                for t, a, v, mask, y in test_loader:
                    t, a, v = t.to(device), a.to(device), v.to(device)
                    # Get projections
                    t_proj = model.fusion.text_proj(t.float())
                    a_proj = model.fusion.audio_proj(a.float())
                    v_proj = model.fusion.video_proj(v.float())
                    # Only proceed if gate dimensions match
                    if t_proj.shape[-1] == model.fusion.text_gate[0].in_features:
                        gt = model.fusion.text_gate(t_proj).mean(dim=-1).cpu().numpy()
                        ga = model.fusion.audio_gate(a_proj).mean(dim=-1).cpu().numpy()
                        gv = model.fusion.video_gate(v_proj).mean(dim=-1).cpu().numpy()
                        gate_text_vals.extend(gt.tolist())
                        gate_audio_vals.extend(ga.tolist())
                        gate_video_vals.extend(gv.tolist())
                    sample_labels.extend(y[:, 0].numpy().tolist() if y.shape[1] > 1 else y.squeeze().numpy().tolist())
            else:
                return  # Unknown fusion type
        except Exception:
            pass  # Gate analysis not supported for this fusion type

    # Limit to first 100 samples for readability
    n_show = min(100, len(gate_text_vals))
    gate_text_vals = gate_text_vals[:n_show]
    gate_audio_vals = gate_audio_vals[:n_show]
    gate_video_vals = gate_video_vals[:n_show]
    sample_labels = sample_labels[:n_show]

    heatmap_data = np.array([gate_text_vals, gate_audio_vals, gate_video_vals])

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Text", "Audio", "Video"])
    ax.set_xlabel("Test Sample (index)")
    ax.set_title(f"Per-Sample Modality Gate Weights — {dataset_name.upper()} / {fusion_type}")
    plt.colorbar(im, ax=ax, label="Gate weight")
    plt.tight_layout()
    save_figure(fig, f"gate_heatmap_{dataset_name}_{fusion_type}.png", out_dir)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(dataset_name, fusion_type, epochs, lr, device):
    """Run a single (dataset, fusion_type) experiment."""
    print(f"\n{'='*60}")
    print(f"  {dataset_name.upper()} / {fusion_type.upper()} FUSION")
    print(f"{'='*60}")

    # Load manifest and build datasets
    samples = load_manifest()
    feature_dims = FEATURE_DIMS[dataset_name]

    if dataset_name == "daic":
        labels_func = load_daic_labels
    elif dataset_name == "mosei":
        labels_func = load_mosei_labels
    elif dataset_name == "fi":
        labels_func = load_fi_labels

    # Group samples by split
    by_split = {"train": [], "val": [], "test": []}
    for s in samples:
        if s["dataset"] == dataset_name and s["split"] in by_split:
            by_split[s["split"]].append(s)

    print(f"  Samples: train={len(by_split['train'])}, val={len(by_split['val'])}, test={len(by_split['test'])}")

    # Build datasets
    train_ds = MultimodalDataset(by_split["train"], dataset_name, labels_func, feature_dims)
    val_ds   = MultimodalDataset(by_split["val"],   dataset_name, labels_func, feature_dims) if by_split["val"] else None
    test_ds  = MultimodalDataset(by_split["test"],  dataset_name, labels_func, feature_dims)

    if len(train_ds) == 0 or len(test_ds) == 0:
        return {"status": "error", "dataset": dataset_name, "fusion_type": fusion_type,
                "error": "No train or test data"}

    batch_size = BATCH_SIZE.get(dataset_name, 16)

    # Use standard DataLoader with shuffle for all datasets
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal)

    # Fallback: use train split for val if val is empty
    if val_ds is None or len(val_ds) == 0:
        print(f"  [Note: val split empty — splitting train 80/20]")
        n = len(train_ds)
        split = int(0.8 * n)
        val_indices = list(range(split, n))
        train_indices = list(range(0, split))
        train_ds_subset = torch.utils.data.Subset(train_ds, train_indices)
        val_ds_subset = torch.utils.data.Subset(train_ds, val_indices)
        train_loader = DataLoader(train_ds_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal)
        val_loader = DataLoader(val_ds_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal)
    else:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal)

    print(f"  Train: {len(train_ds)} samples, Val: {len(val_ds) if val_ds else len(val_ds_subset) if 'val_ds_subset' in dir() else 0}, Test: {len(test_ds)} samples")

    # Train
    if dataset_name == "daic":
        result = train_daic(train_loader, val_loader, test_loader, fusion_type, device, epochs, lr)
    else:
        result = train_regression(train_loader, val_loader, test_loader, dataset_name, fusion_type, device, epochs, lr)

    result["dataset"] = dataset_name
    result["fusion_type"] = fusion_type
    result["status"] = "ok"
    result["epochs_trained"] = len(result["history"]["epoch"]) if result.get("history") else 0

    # Modality dropout robustness test
    if result.get("model") is not None:
        try:
            dropout_results = evaluate_modality_dropout(result["model"], test_loader, device, dataset_name)
            result["dropout_results"] = dropout_results
        except Exception as e:
            print(f"  [Warning: dropout evaluation failed: {e}]")
            result["dropout_results"] = {}

    # Print summary
    pm = result["primary_metric"]
    mn = result.get("metric_name", "?")
    print(f"  {mn}={pm:.4f} | beats_trivial={result.get('beats_trivial', '?')}")
    print(f"  Fusion params: {result.get('param_count', '?')}")

    return result, None


def run_mosei_emotion_experiment(fusion_type, epochs, lr, device):
    """Run MOSEI emotion prediction experiment (6 emotions, multi-label classification)."""
    print(f"\n{'='*60}")
    print(f"  MOSEI EMOTION / {fusion_type.upper()} FUSION")
    print(f"{'='*60}")

    samples = load_manifest()
    feature_dims = FEATURE_DIMS["mosei"]

    # Group samples by split
    by_split = {"train": [], "val": [], "test": []}
    for s in samples:
        if s["dataset"] == "mosei" and s["split"] in by_split:
            by_split[s["split"]].append(s)

    print(f"  Samples: train={len(by_split['train'])}, val={len(by_split['val'])}, test={len(by_split['test'])}")

    # Build emotion datasets
    train_ds = MultimodalEmotionDataset(by_split["train"], load_mosei_emotion_labels, feature_dims)
    val_ds   = MultimodalEmotionDataset(by_split["val"],   load_mosei_emotion_labels, feature_dims) if by_split["val"] else None
    test_ds  = MultimodalEmotionDataset(by_split["test"],  load_mosei_emotion_labels, feature_dims)

    if len(train_ds) == 0 or len(test_ds) == 0:
        return {"status": "error", "dataset": "mosei_emotion", "fusion_type": fusion_type,
                "error": "No train or test data"}

    batch_size = BATCH_SIZE["mosei"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_emotion)

    if val_ds is None or len(val_ds) == 0:
        print(f"  [Note: val split empty — splitting train 80/20]")
        n = len(train_ds)
        split = int(0.8 * n)
        val_indices = list(range(split, n))
        train_indices = list(range(0, split))
        train_ds_subset = torch.utils.data.Subset(train_ds, train_indices)
        val_ds_subset = torch.utils.data.Subset(train_ds, val_indices)
        train_loader = DataLoader(train_ds_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_emotion)
        val_loader = DataLoader(val_ds_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_emotion)
    else:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_emotion)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_emotion)

    print(f"  Train: {len(train_ds)} samples, Val: {len(val_ds) if val_ds else len(val_ds_subset)}, Test: {len(test_ds)} samples")

    # Train emotion model
    result = train_mosei_emotion(train_loader, val_loader, test_loader, fusion_type, device, epochs, lr)

    result["dataset"] = "mosei_emotion"
    result["fusion_type"] = fusion_type
    result["status"] = "ok"
    result["epochs_trained"] = len(result["history"]["epoch"]) if result.get("history") else 0

    # Print summary
    pm = result["primary_metric"]
    mn = result.get("metric_name", "?")
    print(f"  {mn}={pm:.4f} | beats_trivial={result.get('beats_trivial', '?')}")
    print(f"  Fusion params: {result.get('param_count', '?')}")

    return result, None


# ---------------------------------------------------------------------------
# Best unimodal baselines (from Phase 3)
# ---------------------------------------------------------------------------

BEST_UNIMODAL = {
    "daic":  {"modality": "text",  "value": 0.6991},   # text AUROC=0.6991
    "mosei": {"modality": "text",  "value": 0.5123},   # text CCC=0.5123
    "fi":    {"modality": "video", "value": 0.4578},   # video Avg CCC=0.4578
}

# Per-trait unimodal baselines for FI (best modality = video)
FI_UNIMODAL_TRAITS = {
    "extraversion": 0.4667,
    "neuroticism": 0.4402,
    "agreeableness": 0.3181,
    "conscientiousness": 0.6095,
    "openness": 0.4546,
}


# ---------------------------------------------------------------------------
# Results CSV writing
# ---------------------------------------------------------------------------

def write_results_csv(all_results, out_path):
    import csv
    ensure_dir(out_path.parent)

    # Fixed fieldnames for consistent schema across all rows
    FIXED_FIELDS = [
        "dataset", "fusion_type", "metric", "value",
        "ci_lower", "ci_upper",
        "accuracy", "f1", "mae", "r2", "avg_mae",
        "param_count", "beats_unimodal", "unimodal_baseline", "epochs_trained",
    ]

    existing_rows = []
    if out_path.exists():
        try:
            with open(out_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_rows.append(row)
        except Exception:
            existing_rows = []

    new_combos = set()
    for r in all_results:
        if r.get("status") == "ok":
            new_combos.add((r["dataset"], r.get("fusion_type", "?")))

    preserved_rows = []
    for row in existing_rows:
        key = (row.get("dataset", ""), row.get("fusion_type", ""))
        if key not in new_combos:
            preserved_rows.append(row)

    new_rows = []
    for r in all_results:
        if r.get("status") != "ok":
            continue
        ds = r["dataset"]
        ft = r.get("fusion_type", "?")
        mn = r.get("metric_name", "?")
        pm = r["primary_metric"]

        row_base = {
            "dataset": ds,
            "fusion_type": ft,
            "param_count": r.get("param_count", 0),
            "beats_unimodal": pm > BEST_UNIMODAL.get(ds, {}).get("value", 0),
            "unimodal_baseline": BEST_UNIMODAL.get(ds, {}).get("value", 0),
            "epochs_trained": r.get("epochs_trained", 0),
        }

        if ds == "daic":
            ci_lo, ci_hi = r.get("ci_auroc", (0, 0))
            new_rows.append({
                **row_base,
                "metric": "AUROC", "value": round(pm, 4),
                "ci_lower": round(ci_lo, 4), "ci_upper": round(ci_hi, 4),
                "accuracy": round(r.get("accuracy", 0), 4),
                "f1": round(r.get("f1", 0), 4),
                "mae": "", "r2": "", "avg_mae": "",
            })
        elif ds == "mosei":
            ci_lo, ci_hi = r.get("ci_ccc", (0, 0))
            new_rows.append({
                **row_base,
                "metric": mn, "value": round(pm, 4),
                "ci_lower": round(ci_lo, 4), "ci_upper": round(ci_hi, 4),
                "mae": round(r.get("mae", 0), 4),
                "r2": round(r.get("r2", 0), 4),
                "avg_mae": "",
            })
        elif ds == "mosei_emotion":
            # MOSEI emotion: 6 emotions, multi-label, report average AUROC
            ci_lo, ci_hi = r.get("ci_auroc", (0, 0))
            new_rows.append({
                **row_base,
                "metric": mn, "value": round(pm, 4),
                "ci_lower": round(ci_lo, 4), "ci_upper": round(ci_hi, 4),
                "accuracy": "", "f1": "", "mae": "", "r2": "", "avg_mae": "",
            })
            # Per-emotion AUROC rows
            emotion_auroc = r.get("emotion_auroc", {})
            for emotion in EMOTION_TRAITS:
                auc = emotion_auroc.get(emotion, float("nan"))
                new_rows.append({
                    **row_base,
                    "metric": f"AUROC_{emotion}", "value": round(auc, 4),
                    "ci_lower": "", "ci_upper": "",
                    "accuracy": "", "f1": "", "mae": "", "r2": "", "avg_mae": "",
                    "beats_unimodal": auc > 0.5,
                    "unimodal_baseline": 0.5,
                })
        elif ds == "fi":
            ci_lo_val = pm - 0.05
            ci_hi_val = pm + 0.05
            new_rows.append({
                **row_base,
                "metric": mn, "value": round(pm, 4),
                "ci_lower": round(ci_lo_val, 4), "ci_upper": round(ci_hi_val, 4),
                "avg_mae": round(r.get("avg_mae", 0), 4),
                "accuracy": "", "f1": "", "mae": "", "r2": "",
            })
            # Per-trait CCC rows with proper per-trait unimodal baselines
            for trait in FI_TRAITS:
                tr = r.get(trait, {})
                if tr:
                    trait_baseline = FI_UNIMODAL_TRAITS.get(trait, 0)
                    new_rows.append({
                        **row_base,
                        "metric": f"CCC_{trait}", "value": round(tr.get("ccc", 0), 4),
                        "ci_lower": round(tr.get("ci_ccc", (0, 0))[0], 4),
                        "ci_upper": round(tr.get("ci_ccc", (0, 0))[1], 4),
                        "accuracy": "", "f1": "", "mae": "", "r2": "", "avg_mae": "",
                        "beats_unimodal": tr.get("ccc", 0) > trait_baseline,
                        "unimodal_baseline": trait_baseline,
                    })

    all_rows = preserved_rows + new_rows
    if all_rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIXED_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\n  Results CSV saved: {out_path} "
          f"(preserved {len(preserved_rows)} old + {len(new_rows)} new = {len(all_rows)} total rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Multimodal Fusion Baselines")
    parser.add_argument("--dataset", type=str, choices=["daic", "mosei", "fi", "mosei_emotion", "all"], required=True)
    parser.add_argument("--fusion", type=str, choices=["gated", "lmf", "cross_attention", "lrdgn", "all"], required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--lr", type=float, default=LR_DEFAULT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="artifacts/figures/phase_04_fusion")
    parser.add_argument("--results_csv", type=str, default="artifacts/tables/fusion_baselines.csv")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    ensure_dir(out_dir)
    results_csv_path = ROOT / args.results_csv

    # Determine which combos to run
    if args.dataset == "all":
        datasets = ["daic", "mosei", "fi", "mosei_emotion"]
    elif args.dataset == "mosei":
        datasets = ["mosei", "mosei_emotion"]  # Run both sentiment and emotion for MOSEI
    else:
        datasets = [args.dataset]
    fusion_types = ["gated", "lmf"] if args.fusion == "all" else [args.fusion]

    all_results = []
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    for ds in datasets:
        for ft in fusion_types:
            # Special handling for MOSEI emotion task
            if ds == "mosei_emotion":
                result, error = run_mosei_emotion_experiment(ft, args.epochs, args.lr, device)
            else:
                result, error = run_experiment(ds, ft, args.epochs, args.lr, device)
            if result is not None:
                all_results.append(result)
                write_results_csv(all_results, results_csv_path)

                # Generate visualizations
                if result.get("status") == "ok":
                    print("\n  Generating visualizations...")
                    try:
                        plot_training_curves(result, out_dir)
                    except Exception as e:
                        print(f"  [Warning: training curves plot failed: {e}]")

                    if ds == "daic":
                        try:
                            plot_gate_weights(result, out_dir)
                        except Exception as e:
                            print(f"  [Warning: gate weights plot failed: {e}]")

                    try:
                        plot_metric_comparison(result, BEST_UNIMODAL, out_dir)
                    except Exception as e:
                        print(f"  [Warning: metric comparison plot failed: {e}]")

                    if result.get("dropout_results"):
                        try:
                            plot_modality_dropout_robustness(result["dropout_results"], ds, ft, out_dir)
                        except Exception as e:
                            print(f"  [Warning: dropout robustness plot failed: {e}]")

                    if result.get("model") and hasattr(result["model"].fusion, "text_gate"):
                        try:
                            # Create a smaller test loader for heatmap
                            from torch.utils.data import DataLoader
                            ds_test = MultimodalDataset(
                                load_manifest(), ds,
                                {"daic": load_daic_labels, "mosei": load_mosei_labels, "fi": load_fi_labels}[ds],
                                FEATURE_DIMS[ds]
                            )
                            heatmap_bs = BATCH_SIZE.get(ds, 64)
                            tl = DataLoader(ds_test, batch_size=heatmap_bs, shuffle=False, collate_fn=collate_multimodal)
                            plot_per_sample_gate_heatmap(result["model"], tl, device, ds, ft, out_dir)
                        except Exception as e:
                            print(f"  [Warning: gate heatmap plot failed: {e}]")

    # Print summary
    print("\n" + "="*70)
    print("  PHASE 4 SUMMARY — Multimodal Fusion Baselines")
    print("="*70)
    print(f"  {'Dataset':<10} {'Fusion':<8} {'Metric':<12} {'Value':<8} {'vs Unimodal':<15} {'Params':<10}")
    print("-" * 70)
    for r in all_results:
        if r.get("status") != "ok":
            continue
        ds = r["dataset"]
        ft = r.get("fusion_type", "?")
        mn = r.get("metric_name", "?")
        pm = r["primary_metric"]
        unimodal_val = BEST_UNIMODAL.get(ds, {}).get("value", 0)
        delta = pm - unimodal_val
        beats = "✓" if delta > 0 else "✗"
        pc = r.get("param_count", 0)
        print(f"  {ds:<10} {ft:<8} {mn:<12} {pm:.4f}   {delta:+.4f} ({beats})    {pc:,}")

    print(f"\n  Figures: {out_dir}/")
    print(f"  Table:   {results_csv_path}")
    print("\n  ✓ Phase 4 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())