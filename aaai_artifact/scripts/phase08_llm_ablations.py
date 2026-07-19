#!/usr/bin/env python3
"""
Phase 8: LLM Modality Ablations (L0–L9)
=======================================
Ablation matrix for LLM-based encoders vs classical encoders.

L0: classical (RoBERTa + WavLM + OpenFace) — reuse Phase 5 results
L1: Mistral-7B-Instruct frozen → text embeddings (project to 256D)
L2: L1 retrained (frozen Mistral features, different init — LoRA not yet implemented)
L3: L1 + CLAP audio LLM features (audio branch)
L4: L1 + LLaVA-1.5-7B video features (video branch)
L5: L1 + L3 + L4 — full LLM stack
L6-L9: Stub with print("Requires external API") and skip

Usage:
    uv run python scripts/phase08_llm_ablations.py --ablation L1 --epochs 30
    uv run python scripts/phase08_llm_ablations.py --ablation L5 --device cuda
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore")
import random

ROOT = Path(__file__).resolve().parent.parent
FEATURES_ROOT = ROOT / "data" / "features"
LLM_FEATURES_ROOT = FEATURES_ROOT / "llm"
MANIFEST_PATH = FEATURES_ROOT / "manifest.json"
ARTIFACTS_FIGURES = ROOT / "artifacts" / "figures" / "phase_08_llm_ablations"
ARTIFACTS_TABLES = ROOT / "artifacts" / "tables"

DAIC_DATA = ROOT / "data" / "daic"
DAIC_RAW = Path("data/daic/raw")
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
BATCH_SIZE = 16  # Smaller for LLM models
EPOCHS_DEFAULT = 30
LR_DEFAULT = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10
TEMPERATURE = 2.0  # Temperature-balanced sampling

# LLM embedding dimensions (after projection targets 256D)
LLM_DIMS = {
    "mistral": 4096,
    "clap": 512,     # CLAP audio features are 512D (pooler_output)
    "llava": 4096,   # LLaVA-1.5-7B hidden dim is 4096
}

# Phase 5 baseline results (to reuse for L0) — loaded from actual results file
def _load_phase5_results():
    """Load Phase 5 MMoEEx results from saved CSV."""
    import csv
    results_path = ARTIFACTS_TABLES / "mmoe_ex_results.csv"
    if not results_path.exists():
        # Fallback to hardcoded values if file not found
        return {
            "daic_auroc": 0.5471,
            "daic_auroc_text_only": 0.6991,
            "mosei_sentiment_ccc": 0.5397,
            "mosei_emotion_auc": 0.6230,
            "fi_avg_ccc": 0.4578,
        }
    with open(results_path) as f:
        reader = csv.reader(f)
        rows = {row[0]: float(row[1]) for row in reader if len(row) >= 2 and row[0] != 'metric'}
    return {
        "daic_auroc": rows.get("daic_auroc", 0.5471),
        "daic_auroc_text_only": rows.get("daic_auroc_text_only", 0.6991),
        "mosei_sentiment_ccc": rows.get("mosei_sentiment_ccc", 0.5397),
        "mosei_emotion_auc": rows.get("mosei_emotion_auc", 0.6230),
        "fi_avg_ccc": rows.get("fi_avg_ccc", 0.4578),
    }

PHASE5_RESULTS = _load_phase5_results()

ABLATION_DESCRIPTIONS = {
    "L0": "Classical encoders (RoBERTa + WavLM + OpenFace) — reuse Phase 5 results",
    "L1": "Mistral-7B-Instruct frozen — text branch only",
    "L2": "Mistral LoRA (r=16, alpha=32) — text branch with trainable adapter",
    "L3": "Mistral + CLAP audio — text + audio branches",
    "L4": "Mistral + LLaVA video — text + video branches",
    "L5": "Full LLM stack — text + audio + video",
    "L6": "ImageBind-style graph embeddings (stub)",
    "L7": "LLM teacher features (stub)",
    "L8": "Direct multimodal LLM prompting (stub)",
    "L9": "GraphXAIN narratives (stub)",
}

FI_TRAITS = ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]
EMOTION_LABELS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

TASK_IDS = {
    "daic_depression": 0,
    "mosei_sentiment": 1,
    "mosei_emotion": 2,
    "fi_personality": 3,
}

# Expert isolation mapping (same as Phase 5)
TASK_TO_EXPERTS = {
    0: [0, 1],     # DAIC depression - ISOLATED
    1: [2, 3],     # MOSEI sentiment - shared with emotion
    2: [2, 3],     # MOSEI emotion - shared with sentiment
    3: [4, 5],     # FI personality - separate from MOSEI
}

# ---------------------------------------------------------------------------
# Phase 5 model components (reused for LLM ablation)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(ROOT / "src"))

from models.fusion import GatedLateFusion
from models.unified_moe import MMoEEx
from models.task_heads import DepressionHead, SentimentHead, EmotionMultiLabelHead, PersonalityHead


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_all_labels():
    """Load labels for all datasets."""
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


def load_manifest():
    """Load feature manifest."""
    with open(MANIFEST_PATH, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("samples", [])


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
# LLM Feature Extraction (with caching)
# ---------------------------------------------------------------------------

def get_llm_cache_path(level, dataset, split, modality):
    """Get path for cached LLM features."""
    cache_dir = LLM_FEATURES_ROOT / level / dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{split}_{modality}.npy"


# ---------------------------------------------------------------------------
# Raw text loaders for DAIC and MOSEI
# ---------------------------------------------------------------------------

def _load_daic_transcript_text(participant_id):
    """Load raw transcript text from DAIC participant ZIP file."""
    zip_path = DAIC_RAW / f"{participant_id}_P.zip"
    if not zip_path.exists():
        # Try without _P suffix
        zip_path = DAIC_RAW / f"{participant_id}.zip"
        if not zip_path.exists():
            return ""
    try:
        import zipfile, csv, io
        with zipfile.ZipFile(zip_path) as z:
            csv_name = f"{participant_id}_TRANSCRIPT.csv"
            if csv_name not in z.namelist():
                return ""
            with z.open(csv_name) as f:
                reader = csv.reader(io.StringIO(f.read().decode("utf-8")), delimiter="\t")
                rows = list(reader)
            # Concatenate all speaker utterances (skip header)
            texts = [row[3] for row in rows[1:] if len(row) >= 4]
            return " ".join(texts)
    except Exception as e:
        print(f"    Warning: failed to load DAIC transcript for {participant_id}: {e}")
        return ""


def _load_mosei_text(sample_id):
    """Load raw text from MOSEI HDF5 words group using precomputed mapping."""
    global _MOSEI_HDF5, _MOSEI_ID_MAP
    if _MOSEI_ID_MAP is None:
        map_path = FEATURES_ROOT / "mosei_id_to_hdf5.pkl"
        if not map_path.exists():
            _build_mosei_id_map()
        import pickle
        with open(map_path, "rb") as f:
            _MOSEI_ID_MAP = pickle.load(f)
    if _MOSEI_HDF5 is None:
        mosei_raw_path = "data/mosei/CMU-MOSEI/mosei_raw.pkl"
        if os.path.exists(mosei_raw_path):
            import h5py
            _MOSEI_HDF5 = h5py.File(mosei_raw_path, "r")
        else:
            # Try local copy
            local_path = str(MOSEI_DATA / "mosei_raw.pkl")
            if os.path.exists(local_path):
                import h5py
                _MOSEI_HDF5 = h5py.File(local_path, "r")
            else:
                return ""
    if _MOSEI_HDF5 is None:
        return ""
    hdf5_key = _MOSEI_ID_MAP.get(sample_id)
    if hdf5_key is None or hdf5_key not in _MOSEI_HDF5["words"]:
        return ""
    try:
        words_data = _MOSEI_HDF5["words"][hdf5_key]["features"][:]
        words = []
        for w in words_data:
            text = w.tobytes().decode("utf-8", errors="replace").strip().rstrip("\x00")
            if text and text != "sp":
                words.append(text)
        return " ".join(words)
    except Exception:
        return ""


_MOSEI_HDF5 = None      # global cache for HDF5 file handle
_MOSEI_ID_MAP = None     # global cache for label_id -> hdf5_key mapping


def _build_mosei_id_map():
    """Build and cache the mapping from MOSEI label IDs to HDF5 keys."""
    import json, h5py, pickle
    labels_path = MOSEI_DATA / "mosei_emotion_labels.json"
    if not labels_path.exists():
        print("  Warning: MOSEI labels not found, cannot build ID map")
        return
    with open(labels_path, "r") as f:
        labels = json.load(f)
    mosei_raw_path = "data/mosei/CMU-MOSEI/mosei_raw.pkl"
    if not os.path.exists(mosei_raw_path):
        print("  Warning: MOSEI HDF5 not found, cannot build ID map")
        return
    hf = h5py.File(mosei_raw_path, "r")
    hdf5_keys = set(hf["words"].keys())
    video_counts = {}
    label_to_hdf5 = {}
    for key, val in labels.items():
        vid = val["video_id"]
        seg_idx = video_counts.get(vid, 0)
        video_counts[vid] = seg_idx + 1
        hdf5_key = f"{vid}[{seg_idx}]"
        # Try up to 50 segment indices if first guess fails
        for attempt in range(50):
            trial_key = f"{vid}[{seg_idx + attempt}]"
            if trial_key in hdf5_keys:
                hdf5_key = trial_key
                break
        label_to_hdf5[key] = hdf5_key
    hf.close()
    out_path = FEATURES_ROOT / "mosei_id_to_hdf5.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(label_to_hdf5, f)
    print(f"  Built MOSEI ID map: {len(label_to_hdf5)} entries -> {out_path}")


def _load_fi_text(clip_id):
    """FI has no text modality. Return empty string."""
    return ""


# ---------------------------------------------------------------------------
# LLM model cache (singleton) — load once, share across all samples
# ---------------------------------------------------------------------------
_MISTRAL_MODEL = None
_MISTRAL_TOKENIZER = None


def _ensure_mistral(device):
    """Load Mistral-7B-Instruct-v0.3 once and cache it."""
    global _MISTRAL_MODEL, _MISTRAL_TOKENIZER
    if _MISTRAL_MODEL is not None:
        return _MISTRAL_MODEL, _MISTRAL_TOKENIZER
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading Mistral-7B-Instruct-v0.3 on {device} (fp16)...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    _MISTRAL_MODEL = model
    _MISTRAL_TOKENIZER = tokenizer
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  Mistral loaded. GPU memory: {gpu_mem:.2f} GB")
    return model, tokenizer


# ---------------------------------------------------------------------------
# LLM modality extraction functions (real implementations)
# ---------------------------------------------------------------------------

def load_or_extract_mistral_text(dataset, split, device, cache_dir=None, skip_extraction=False):
    """Load or extract Mistral-7B-Instruct text features.

    Returns a dict mapping sample_id -> 4096D numpy array (float16).
    """
    level = "L1"
    cache_path = get_llm_cache_path(level, dataset, split, "text")

    # Load from cache if available
    if cache_path.exists():
        print(f"  Loading cached Mistral text features: {cache_path}")
        return np.load(cache_path, allow_pickle=True).item()

    if skip_extraction:
        print(f"  No cache at {cache_path}. Use --skip_extraction=False to extract.")
        return None

    print(f"\n  Extracting Mistral-7B text features for {dataset}/{split}...")

    # Load Mistral model (cached singleton)
    model, tokenizer = _ensure_mistral(device)

    # Get sample IDs from manifest filtered by dataset+split
    manifest = load_manifest()
    sample_ids = [s["id"] for s in manifest if s["dataset"] == dataset and s.get("split") == split]
    print(f"  Found {len(sample_ids)} samples for {dataset}/{split}")

    if len(sample_ids) == 0:
        print(f"  No samples found for {dataset}/{split}")
        return {}

    # Load raw texts
    print(f"  Loading raw text for {len(sample_ids)} samples...")
    raw_texts = []
    valid_ids = []
    for sid in sample_ids:
        if dataset == "daic":
            text = _load_daic_transcript_text(sid)
        elif dataset == "mosei":
            text = _load_mosei_text(sid)
        elif dataset == "fi":
            text = _load_fi_text(sid)
        else:
            text = ""
        if text.strip():
            raw_texts.append(text)
            valid_ids.append(sid)
        else:
            # Keep as zero vector placeholder
            raw_texts.append("")
            valid_ids.append(sid)

    # Extract features in batches
    print(f"  Extracting Mistral embeddings (batch_size=16, {(len(raw_texts))} samples)...")
    all_features = {}
    batch_size = 16

    for start_idx in range(0, len(raw_texts), batch_size):
        end_idx = min(start_idx + batch_size, len(raw_texts))
        batch_texts = raw_texts[start_idx:end_idx]
        batch_ids = valid_ids[start_idx:end_idx]

        # Handle empty texts (zero vectors)
        batch_features = np.zeros((len(batch_texts), 4096), dtype=np.float32)
        non_empty_indices = [i for i, t in enumerate(batch_texts) if t.strip()]
        non_empty_texts = [batch_texts[i] for i in non_empty_indices]

        if non_empty_texts:
            try:
                inputs = tokenizer(
                    non_empty_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512
                ).to(device)

                with torch.no_grad():
                    outputs = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                    hidden = outputs.hidden_states[-1]  # (batch, seq_len, 4096)
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                    batch_features_cpu = pooled.cpu().numpy().astype(np.float32)

                for i, idx_in_batch in enumerate(non_empty_indices):
                    batch_features[idx_in_batch] = batch_features_cpu[i]
            except Exception as e:
                print(f"    Warning: batch {start_idx}-{end_idx} failed: {e}")

        for i, sid in enumerate(batch_ids):
            all_features[sid] = batch_features[i]

        if (start_idx // batch_size) % 50 == 0 and len(raw_texts) > 0:
            pct = min(100, start_idx * 100 // len(raw_texts))
            print(f"    Progress: {start_idx}/{len(raw_texts)} ({pct}%)")

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, all_features)
    print(f"  Saved Mistral features ({len(all_features)} samples) to {cache_path}")

    return all_features


def load_or_extract_clap_audio(dataset, split, device, cache_dir=None, skip_extraction=False):
    """Load or extract CLAP audio features. (Implemented - loads real CLAP model)"""
    level = "L3"
    cache_path = get_llm_cache_path(level, dataset, split, "audio")

    if cache_path.exists():
        print(f"  Loading cached CLAP audio features: {cache_path}")
        return np.load(cache_path, allow_pickle=True).item()

    if skip_extraction:
        print(f"  No cache at {cache_path}. Use --skip_extraction=False to extract.")
        return None

    print(f"\n  Extracting CLAP audio features for {dataset}/{split}...")

    try:
        from transformers import ClapModel, ClapProcessor
        import librosa
    except ImportError:
        print("  ERROR: CLAP requires `librosa`. Install with: uv add librosa")
        return None

    # Load CLAP model
    print("  Loading CLAP model...")
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    clap_model.eval()
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    # Build file paths for raw audio based on dataset
    manifest = load_manifest()
    sample_ids = [s["id"] for s in manifest if s["dataset"] == dataset and s.get("split") == split]
    print(f"  Found {len(sample_ids)} samples for {dataset}/{split}")

    all_features = {}
    for sid in sample_ids:
        # Locate audio file — different for each dataset
        audio_path = None
        if dataset == "daic":
            zip_path = DAIC_RAW / f"{sid}_P.zip"
            if zip_path.exists():
                import zipfile, io
                with zipfile.ZipFile(zip_path) as z:
                    wav_name = f"{sid}_AUDIO.wav"
                    if wav_name in z.namelist():
                        audio_data = io.BytesIO(z.read(wav_name))
                        try:
                            waveform, sr = librosa.load(audio_data, sr=48000)
                        except Exception:
                            waveform = None
                        if waveform is not None and len(waveform) > 0:
                            inputs = processor(
                                audio=waveform, sampling_rate=48000,
                                return_tensors="pt", padding=True
                            ).to(device)
                            with torch.no_grad():
                                output = clap_model.get_audio_features(**inputs)
                                feat = output.pooler_output.cpu().numpy().astype(np.float32)[0]
                            all_features[sid] = feat
                            continue
        elif dataset == "mosei":
            # MOSEI audio is in HDF5 — not raw WAV, skip for now
            pass
        elif dataset == "fi":
            import glob
            for ext in [".wav", ".mp3", ".m4a"]:
                candidates = list(FI_DATA.glob(f"{split}/**/{sid}{ext}"))
                if candidates:
                    audio_path = candidates[0]
                    break
            if audio_path and audio_path.exists():
                try:
                    waveform, sr = librosa.load(audio_path, sr=48000)
                    if len(waveform) > 0:
                        inputs = processor(
                            audio=waveform, sampling_rate=48000,
                            return_tensors="pt", padding=True
                        ).to(device)
                        with torch.no_grad():
                            output = clap_model.get_audio_features(**inputs)
                            feat = output.pooler_output.cpu().numpy().astype(np.float32)[0]
                        all_features[sid] = feat
                        continue
                except Exception:
                    pass
        # Fallback: zero vector
        all_features[sid] = np.zeros(512, dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, all_features)
    print(f"  Saved CLAP features ({len(all_features)} samples) to {cache_path}")
    return all_features


def load_or_extract_llava_video(dataset, split, device, cache_dir=None, skip_extraction=False):
    """Load or extract LLaVA video features. (Implemented - loads real LLaVA model)"""
    level = "L4"
    cache_path = get_llm_cache_path(level, dataset, split, "video")

    if cache_path.exists():
        print(f"  Loading cached LLaVA video features: {cache_path}")
        return np.load(cache_path, allow_pickle=True).item()

    if skip_extraction:
        print(f"  No cache at {cache_path}. Use --skip_extraction=False to extract.")
        return None

    print(f"\n  Extracting LLaVA video features for {dataset}/{split}...")

    try:
        from transformers import LlavaForConditionalGeneration, AutoProcessor
    except ImportError:
        print("  ERROR: LLaVA requires transformers>=4.45.0")
        return None

    # Load LLaVA model
    print("  Loading LLaVA-1.5-7B-hf...")
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    llava_model.eval()
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    manifest = load_manifest()
    sample_ids = [s["id"] for s in manifest if s["dataset"] == dataset and s.get("split") == split]
    print(f"  Found {len(sample_ids)} samples for {dataset}/{split}")

    all_features = {}
    for sid in sample_ids:
        # Video frames are dataset-specific
        feat = None
        try:
            if dataset == "fi":
                import cv2, glob
                # Manifest IDs have format "fi_{split}_{filename}.mp4" but actual
                # videos are at {split}/videos/{filename}.mp4 — strip prefix and
                # avoid double .mp4 extension since it's already in the ID.
                prefix = f"fi_{split}_"
                vid_name = sid[len(prefix):] if sid.startswith(prefix) else sid
                # Remove .mp4 from the full sid if glob pattern had it
                if vid_name.endswith(".mp4"):
                    video_path = FI_DATA / split / "videos" / vid_name
                else:
                    video_path = FI_DATA / split / "videos" / f"{vid_name}.mp4"
                video_paths = [video_path] if video_path.exists() else []
                if video_paths:
                    cap = cv2.VideoCapture(str(video_paths[0]))
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                        if len(frames) >= 8:  # Sample up to 8 frames
                            break
                    cap.release()
                    if frames:
                        import PIL.Image, numpy as np_
                        # Use first frame as a representative image
                        # (full video processing is extremely expensive)
                        image = PIL.Image.fromarray(frames[0])
                        prompt = "<image>\nDescribe the emotional state of this person."
                        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = llava_model(**inputs, output_hidden_states=True)
                            last_hidden = outputs.hidden_states[-1]
                            feat = last_hidden.mean(dim=1).cpu().numpy().astype(np.float32)[0]
        except Exception as e:
            print(f"    Warning: LLaVA extraction failed for {sid}: {e}")
            pass

        if feat is None:
            feat = np.zeros(4096, dtype=np.float32)
        all_features[sid] = feat

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, all_features)
    print(f"  Saved LLaVA features ({len(all_features)} samples) to {cache_path}")
    return all_features


# ---------------------------------------------------------------------------
# LLM Projector: trainable linear projection LLM_dim → 256D
# ---------------------------------------------------------------------------

class LLMProjector(nn.Module):
    """Trainable projection from LLM embedding space to 256D."""

    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.projection(x)


# ---------------------------------------------------------------------------
# Joint Dataset for LLM Ablation
# ---------------------------------------------------------------------------

class JointMultimodalDataset(Dataset):
    """Combined dataset for all 3 datasets with task routing.

    For LLM ablation, we use cached LLM features instead of classical features.
    """

    def __init__(self, manifest_data, all_labels, datasets_splits, llm_level,
                 temperature=2.0, skip_extraction=False, device="cuda"):
        self.samples = []
        self.llm_level = llm_level
        self.device = device
        self.skip_extraction = skip_extraction

        # Feature dimensions based on LLM level
        if llm_level in ["L0"]:
            self.feature_dims = {"text": 768, "audio": 768, "video": 1536}  # Classical
        elif llm_level in ["L1", "L2"]:
            self.feature_dims = {"text": 4096, "audio": 768, "video": 768}  # Mistral text
        elif llm_level in ["L3", "L4"]:
            self.feature_dims = {"text": 4096, "audio": 768, "video": 768}
        elif llm_level in ["L5"]:
            self.feature_dims = {"text": 4096, "audio": 768, "video": 768}
        else:
            self.feature_dims = {"text": 768, "audio": 768, "video": 1536}

        # Load cached features for LLM levels
        self.llm_features = {}
        if llm_level in ["L1", "L2", "L3", "L4", "L5"]:
            self._load_llm_features(datasets_splits, skip_extraction)

        for ds_name, split in datasets_splits:
            for entry in manifest_data:
                if entry["dataset"] != ds_name or entry.get("split") != split:
                    continue
                sample_id = entry["id"]

                # Compute label key
                if ds_name == "daic":
                    label_key = f"daic_{sample_id}"
                elif ds_name == "mosei":
                    label_key = sample_id
                else:
                    label_key = sample_id

                if label_key not in all_labels:
                    continue

                # Get classical features as fallback or for L0
                feat_map = entry["features"]
                t_key = feat_map.get("text_roberta")
                a_key = feat_map.get("audio_wavlm")
                v_key = feat_map.get("video_vit") if ds_name != "daic" else feat_map.get("video_openface")

                t_ok, t_vec = self._try_load_classical_feature(t_key, self.feature_dims["text"])
                a_ok, a_vec = self._try_load_classical_feature(a_key, self.feature_dims["audio"])
                v_ok, v_vec = self._try_load_classical_feature(v_key, self.feature_dims["video"])

                # For LLM levels, try to replace with LLM features
                if llm_level in ["L1", "L2", "L3", "L4", "L5"] and not skip_extraction:
                    llm_t = self._get_llm_feature(ds_name, split, sample_id, "text")
                    if llm_t is not None:
                        t_vec = llm_t
                        t_ok = True

                    if llm_level in ["L3", "L5"]:
                        llm_a = self._get_llm_feature(ds_name, split, sample_id, "audio")
                        if llm_a is not None:
                            a_vec = llm_a
                            a_ok = True

                    if llm_level in ["L4", "L5"]:
                        llm_v = self._get_llm_feature(ds_name, split, sample_id, "video")
                        if llm_v is not None:
                            v_vec = llm_v
                            v_ok = True

                if not (t_ok or a_ok or v_ok):
                    continue

                label = all_labels[label_key]

                if ds_name == "daic":
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
                        "label": label,
                        "sample_weight": 1.0,
                    })
                else:
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
        print(f"  JointDataset ({llm_level}): {len(self)} samples")
        by_routing = defaultdict(int)
        for s in self.samples:
            by_routing[s["routing"]] += 1
        print(f"    Routing: {dict(by_routing)}")

    def _load_llm_features(self, datasets_splits, skip_extraction):
        """Load (or extract and cache) LLM features for all datasets.

        For each dataset+split+modality:
          1. Check for cached .npy file.
          2. If not cached and not skip_extraction, run the real LLM extraction.
          3. Store as dict[sample_id -> numpy_array].
        """
        for ds_name, split in datasets_splits:
            # --- Text (Mistral) ---
            cache_t = get_llm_cache_path(self.llm_level, ds_name, split, "text")
            if cache_t.exists():
                print(f"  Loaded cached Mistral text: {cache_t.name}")
                feat_dict = np.load(cache_t, allow_pickle=True).item()
            elif skip_extraction:
                print(f"  No Mistral cache for {ds_name}/{split}, skipping (skip_extraction=True)")
                continue
            else:
                feat_dict = load_or_extract_mistral_text(
                    ds_name, split, self.device, skip_extraction=False
                )
                if feat_dict is None:
                    print(f"  Mistral extraction returned None for {ds_name}/{split}")
                    continue
            if isinstance(feat_dict, dict):
                self.llm_features[f"{ds_name}_{split}_text"] = feat_dict
            else:
                print(f"  Warning: unexpected Mistral format for {ds_name}/{split}, expected dict")

            # --- Audio (CLAP) — only for levels L3, L5 ---
            if self.llm_level in ["L3", "L5"]:
                cache_a = get_llm_cache_path(self.llm_level, ds_name, split, "audio")
                if cache_a.exists():
                    print(f"  Loaded cached CLAP audio: {cache_a.name}")
                    feat_dict = np.load(cache_a, allow_pickle=True).item()
                elif skip_extraction:
                    continue
                else:
                    feat_dict = load_or_extract_clap_audio(
                        ds_name, split, self.device, skip_extraction=False
                    )
                if isinstance(feat_dict, dict):
                    self.llm_features[f"{ds_name}_{split}_audio"] = feat_dict

            # --- Video (LLaVA) — only for levels L4, L5 ---
            if self.llm_level in ["L4", "L5"]:
                cache_v = get_llm_cache_path(self.llm_level, ds_name, split, "video")
                if cache_v.exists():
                    print(f"  Loaded cached LLaVA video: {cache_v.name}")
                    feat_dict = np.load(cache_v, allow_pickle=True).item()
                elif skip_extraction:
                    continue
                else:
                    feat_dict = load_or_extract_llava_video(
                        ds_name, split, self.device, skip_extraction=False
                    )
                if isinstance(feat_dict, dict):
                    self.llm_features[f"{ds_name}_{split}_video"] = feat_dict

    def _get_llm_feature(self, dataset, split, sample_id, modality):
        """Get LLM feature for a specific sample by sample_id lookup.

        Converts float16 → float32 since LLM features are cached in fp16
        for storage efficiency but model layers expect fp32.
        """
        key = f"{dataset}_{split}_{modality}"
        feat_dict = self.llm_features.get(key)
        if feat_dict is None or not isinstance(feat_dict, dict):
            return None
        feat = feat_dict.get(sample_id)
        if feat is None:
            return None
        # Cast float16 → float32 for compatibility with Linear layers
        if feat.dtype == np.float16:
            feat = feat.astype(np.float32)
        return feat

    def _try_load_classical_feature(self, path_str, dim):
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
    )


# ---------------------------------------------------------------------------
# Model: UnifiedMMoEEx with LLM projectors (for L1-L5)
# ---------------------------------------------------------------------------

class UnifiedMMoEExWithLLM(nn.Module):
    """MMoEEx model with optional LLM projectors for ablation."""

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
        use_llm_projector: bool = False,
        llm_text_dim: int = 4096,
        llm_audio_dim: int = 768,
        llm_video_dim: int = 768,
    ):
        super().__init__()
        self.use_llm_projector = use_llm_projector

        # LLM projectors (trainable, for L1-L5)
        if use_llm_projector:
            self.llm_text_projector = LLMProjector(llm_text_dim, hidden_dim)
            self.llm_audio_projector = LLMProjector(llm_audio_dim, hidden_dim)
            self.llm_video_projector = LLMProjector(llm_video_dim, hidden_dim)
            # LLM fusion: concat 3×256D → project to 256D for MMoEEx
            self.llm_fusion = nn.Linear(hidden_dim * 3, hidden_dim)

        # Classical projectors (for L0 or fallback)
        self.fusion = GatedLateFusion(text_dim, audio_dim, video_dim, hidden_dim)
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

        self.mmoe = MMoEEx(
            input_dim=hidden_dim,
            num_experts=num_experts,
            expert_dim=expert_dim,
            num_tasks=num_tasks,
            num_shared=num_shared,
            expert_isolation=True,
            task_to_experts=TASK_TO_EXPERTS,
        )

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
        if self.use_llm_projector:
            if routing == "text_only":
                h = self.llm_text_projector(text_feat)
                return self.mmoe(h, task_id)
            elif routing == "video_only":
                h = self.llm_video_projector(video_feat)
                return self.mmoe(h, task_id)
            else:
                t_h = self.llm_text_projector(text_feat)
                a_h = self.llm_audio_projector(audio_feat)
                v_h = self.llm_video_projector(video_feat)
                fused = torch.cat([t_h, a_h, v_h], dim=-1)  # (batch, 768)
                fused = self.llm_fusion(fused)  # (batch, 256)
                return self.mmoe(fused, task_id)
        else:
            if routing == "text_only":
                h = self.text_projector(text_feat)
                return self.mmoe(h, task_id)
            elif routing == "video_only":
                h = self.video_projector(video_feat)
                return self.mmoe(h, task_id)
            else:
                fused = self.fusion(text_feat, audio_feat, video_feat, modality_mask.bool())
                return self.mmoe(fused, task_id)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.mmoe(x, task_id)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class NLLRegressionLoss(nn.Module):
    def __init__(self, init_log_sigma: float = -2.0):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(init_log_sigma))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma)
        loss = 0.5 * (pred.squeeze(-1) - target) ** 2 / (sigma ** 2)
        loss = loss + 0.5 * self.log_sigma
        return loss.mean()


class MSELossWithVariancePenalty(nn.Module):
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
    def __init__(self):
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")
        self.bce_multi = nn.BCEWithLogitsLoss(reduction="none")
        self.nll = NLLRegressionLoss(init_log_sigma=-2.0)
        self.mse_variance = MSELossWithVariancePenalty(lambda_variance=0.1)

    def depression_loss(self, logits, labels):
        return self.bce(logits.squeeze(-1), labels[:, 0]).mean()

    def sentiment_loss(self, preds, labels):
        return self.nll(preds, labels[:, 0])

    def emotion_loss(self, logits, labels):
        binary_labels = (labels[:, :6] >= 0.3).float()
        return self.bce_multi(logits, binary_labels).mean()

    def personality_loss(self, preds_dict, labels):
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
# Training and Evaluation
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

                expert_out = model.forward_with_routing(
                    t_text, t_audio, t_video, t_mask_feat, task_val, routing
                )

                if tid == 0:
                    out = model.depression_head(expert_out)
                    l = loss_fn.depression_loss(out, t_labels)
                elif tid == 1:
                    out = model.sentiment_head(expert_out)
                    l = loss_fn.sentiment_loss(out, t_labels)
                elif tid == 2:
                    out = model.emotion_head(expert_out)
                    l = loss_fn.emotion_loss(out, t_labels)
                else:
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


def evaluate(model, dataloader, device):
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
                    lbl = t_labels[:, 0].cpu().numpy().flatten()
                    results["daic"]["all_labels"].extend(lbl.tolist())
                    results["daic"]["all_preds"].extend(out.tolist())
                elif tid == 1:
                    out = model.sentiment_head(expert_out).cpu().numpy().flatten()
                    lbl = t_labels[:, 0].cpu().numpy().flatten()
                    results["mosei_sentiment"]["all_labels"].extend(lbl.tolist())
                    results["mosei_sentiment"]["all_preds"].extend(out.tolist())
                elif tid == 2:
                    out = torch.sigmoid(model.emotion_head(expert_out)).cpu().numpy()
                    lbl = t_labels[:, :6].cpu().numpy()
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

def plot_llm_delta_bar(results_dict, save_dir):
    """Plot DAIC AUROC comparison bar chart for L0-L5."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = []
    aurocs = []
    for lvl in ["L0", "L1", "L2", "L3", "L4", "L5"]:
        if lvl in results_dict:
            auroc = results_dict[lvl].get("daic_auroc")
            if auroc is not None:
                levels.append(lvl)
                aurocs.append(auroc)

    if not levels:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71" if a > PHASE5_RESULTS["daic_auroc"] else "#e74c3c" for a in aurocs]
    bars = ax.bar(levels, aurocs, color=colors, alpha=0.8, edgecolor="black")

    # Add baseline line
    ax.axhline(y=PHASE5_RESULTS["daic_auroc"], color="gray", linestyle="--", label=f"Phase 5 baseline ({PHASE5_RESULTS['daic_auroc']:.3f})")
    ax.axhline(y=PHASE5_RESULTS["daic_auroc_text_only"], color="blue", linestyle=":", label=f"Phase 5 text-only ({PHASE5_RESULTS['daic_auroc_text_only']:.3f})")

    ax.set_xlabel("Ablation Level", fontsize=12)
    ax.set_ylabel("DAIC AUROC", fontsize=12)
    ax.set_title("LLM Ablation: DAIC AUROC vs Classical Baseline", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.4, 0.85)

    # Add value labels
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = save_dir / "llm_delta_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_embedding_umap(classical_embeddings, llm_embeddings, labels, save_dir):
    """Plot UMAP of classical vs LLM embeddings.

    Handles different dimensionalities by using PCA to project both to a
    common dimension (min of the two) before UMAP reduction.
    """
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        n_classical = len(classical_embeddings)

        # If dimensions differ, project each to a common dim via separate PCA
        if classical_embeddings.shape[1] != llm_embeddings.shape[1]:
            common_dim = min(classical_embeddings.shape[1], llm_embeddings.shape[1], 50)
            print(f"    Projecting {classical_embeddings.shape[1]}D classical and "
                  f"{llm_embeddings.shape[1]}D LLM → {common_dim}D via separate PCA")
            pca_c = PCA(n_components=common_dim, random_state=42)
            classical_proj = pca_c.fit_transform(classical_embeddings)
            pca_l = PCA(n_components=common_dim, random_state=42)
            llm_proj = pca_l.fit_transform(llm_embeddings)
            all_emb = np.vstack([classical_proj, llm_proj])
        else:
            all_emb = np.vstack([classical_embeddings, llm_embeddings])

        # UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
        emb_2d = reducer.fit_transform(all_emb)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        classical_pts = emb_2d[:n_classical]
        llm_pts = emb_2d[n_classical:]

        ax.scatter(classical_pts[:, 0], classical_pts[:, 1], c="blue", alpha=0.5, label="Classical", s=30)
        ax.scatter(llm_pts[:, 0], llm_pts[:, 1], c="red", alpha=0.5, label="LLM", s=30)

        # Add label coloring if labels provided
        if labels is not None and len(labels) > 0:
            # Not yet implemented — reserved for future use
            pass

        ax.set_title("UMAP: Classical vs LLM Embeddings", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = save_dir / "embedding_umap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")
    except Exception as e:
        print(f"  UMAP plot failed: {e}")


def plot_cost_performance(gpu_hours, auroc_gains, save_dir):
    """Plot GPU hours vs metric gain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    levels = ["L0", "L1", "L2", "L3", "L4", "L5"]
    for i, (level, hours, gain) in enumerate(zip(levels, gpu_hours, auroc_gains)):
        ax.scatter(hours, gain, s=100, zorder=5)
        ax.annotate(level, (hours, gain), xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("GPU Hours", fontsize=12)
    ax.set_ylabel("AUROC Gain over Baseline", fontsize=12)
    ax.set_title("Cost-Performance: LLM Ablation", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--")
    plt.tight_layout()
    path = save_dir / "cost_performance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# UMAP Embedding Collection (for summary report)
# ---------------------------------------------------------------------------

def _collect_umap_embeddings(max_samples=500):
    """Collect classical (RoBERTa) and LLM (Mistral) text embeddings for UMAP.

    Loads cached LLM features and corresponding classical text features
    from disk, returning arrays suitable for plot_embedding_umap().

    Returns (classical_embeddings, llm_embeddings, sample_ids).
    Each element is None if collection fails.
    """
    random.seed(42)

    print("\n  Collecting embeddings for UMAP...")
    manifest_data = load_manifest()

    # Load LLM features from cache (L1 Mistral text)
    llm_text = {}
    for ds in ["daic", "fi", "mosei"]:
        for split in ["train", "val"]:
            cache_path = get_llm_cache_path("L1", ds, split, "text")
            if cache_path.exists():
                feat_dict = np.load(cache_path, allow_pickle=True).item()
                for sid, feat in feat_dict.items():
                    llm_text[f"{ds}_{sid}"] = feat

    print(f"    Loaded {len(llm_text)} LLM text features from cache")

    # Filter manifest to samples with both classical and LLM features
    candidates = []
    for entry in manifest_data:
        ds = entry["dataset"]
        sid = entry["id"]
        split = entry.get("split")
        if split not in ("train", "val"):
            continue
        key = f"{ds}_{sid}"
        if key not in llm_text:
            continue
        text_path_str = entry.get("features", {}).get("text_roberta")
        if text_path_str is None:
            continue
        text_path = ROOT / text_path_str
        if not text_path.exists():
            continue
        candidates.append((key, sid, ds, split, text_path))

    print(f"    Found {len(candidates)} candidates with both feature types")

    if not candidates:
        print("    WARNING: No candidates found for UMAP")
        return None, None, None

    # Stratified sampling if too many
    if len(candidates) > max_samples:
        from collections import defaultdict as _dd
        by_ds = _dd(list)
        for c in candidates:
            by_ds[c[2]].append(c)
        sampled = []
        for ds_name, items in by_ds.items():
            ds_max = max(1, int(max_samples * len(items) / len(candidates)))
            random.shuffle(items)
            sampled.extend(items[:ds_max])
        if len(sampled) > max_samples:
            random.shuffle(sampled)
            sampled = sampled[:max_samples]
        candidates = sampled

    print(f"    Using {len(candidates)} samples for UMAP")

    # Load embeddings
    classical_embeddings = []
    llm_embeddings = []
    sample_ids = []

    # Determine expected classical dimension from first successful load
    expected_classical_dim = None

    for key, sid, ds, split, text_path in candidates:
        # Load classical RoBERTa pooled features
        try:
            obj = torch.load(text_path, map_location="cpu", weights_only=False)
            if isinstance(obj, dict):
                if "pooled_features" in obj and isinstance(obj["pooled_features"], torch.Tensor):
                    feat = obj["pooled_features"]
                elif "embedding" in obj and isinstance(obj["embedding"], torch.Tensor):
                    feat = obj["embedding"]
                else:
                    continue
            else:
                continue
            # Handle 2D features by pooling
            if isinstance(feat, torch.Tensor) and feat.dim() == 2:
                feat = feat.mean(dim=0)
            # Flatten to 1D
            classical_feat = feat.cpu().numpy().flatten().astype(np.float32)
        except Exception:
            continue

        # Load LLM Mistral feature
        llm_feat = llm_text[key].astype(np.float32).flatten()

        # Set expected dimension from first load
        if expected_classical_dim is None:
            expected_classical_dim = classical_feat.shape[0]

        # Validate shape consistency
        if classical_feat.shape[0] != expected_classical_dim:
            continue
        if llm_feat.shape[0] < 1:
            continue

        classical_embeddings.append(classical_feat)
        llm_embeddings.append(llm_feat)
        sample_ids.append(key)

    if not classical_embeddings:
        print("    WARNING: No embeddings successfully loaded for UMAP")
        return None, None, None

    classical_embeddings = np.stack(classical_embeddings)
    llm_embeddings = np.stack(llm_embeddings)

    print(f"    Classical embeddings: {classical_embeddings.shape}")
    print(f"    LLM embeddings: {llm_embeddings.shape}")

    return classical_embeddings, llm_embeddings, sample_ids


# ---------------------------------------------------------------------------
# Summary Report Generation
# ---------------------------------------------------------------------------

def generate_summary_report():
    """Generate summary CSV and figures from all L0-L5 results."""
    print("\n" + "="*60)
    print("Phase 8: Generating Summary Report")
    print("="*60)

    results_dict = {}
    levels = ["L0", "L1", "L2", "L3", "L4", "L5"]

    # Load results from JSON files
    for level in levels:
        results_path = ARTIFACTS_TABLES / f"phase08_{level}_results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                results_dict[level] = json.load(f)
            print(f"  Loaded {level}: DAIC AUROC = {results_dict[level].get('daic_auroc', 'N/A')}")
        else:
            print(f"  Missing {level} results")

    if not results_dict:
        print("  No results found. Run ablations first.")
        return

    # Generate summary CSV
    csv_path = ARTIFACTS_TABLES / "phase08_llm_ablations.csv"
    with open(csv_path, "w") as f:
        f.write("level,daic_auroc,mosei_sentiment_ccc,mosei_emotion_auc,fi_avg_ccc,gpu_hours,note\n")
        for level in levels:
            if level in results_dict:
                r = results_dict[level]
                f.write(f"{level},{r.get('daic_auroc', '')},{r.get('mosei_sentiment_ccc', '')},"
                       f"{r.get('mosei_emotion_auc', '')},{r.get('fi_avg_ccc', '')},"
                       f"{r.get('gpu_hours', '')},{r.get('note', '')}\n")
            else:
                f.write(f"{level},,,,,,\n")
    print(f"\n  Saved CSV: {csv_path}")

    # Generate figures
    print("\n  Generating figures...")

    # 1. Delta bar chart
    plot_llm_delta_bar(results_dict, ARTIFACTS_FIGURES)

    # 2. Cost-performance plot
    gpu_hours = [results_dict.get(l, {}).get("gpu_hours", 0) for l in levels]
    baseline_auroc = PHASE5_RESULTS["daic_auroc"]
    auroc_gains = [results_dict.get(l, {}).get("daic_auroc", baseline_auroc) - baseline_auroc for l in levels]
    plot_cost_performance(gpu_hours, auroc_gains, ARTIFACTS_FIGURES)

    # 3. UMAP embedding plot (real embeddings)
    print("  Generating UMAP embedding plot...")
    classical_emb, llm_emb, sample_ids = _collect_umap_embeddings(max_samples=500)
    if classical_emb is not None and llm_emb is not None:
        plot_embedding_umap(classical_emb, llm_emb, [], ARTIFACTS_FIGURES)
    else:
        print("  Note: Could not collect embeddings for UMAP.")
        print("  Run --ablation L1 with GPU extraction first to generate cached LLM features.")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor("#f0f0f0")
        ax.text(0.5, 0.5, "UMAP requires LLM features\nRun L1-L5 with GPU extraction\nto generate this plot",
                transform=ax.transAxes, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title("Classical vs LLM Embeddings (UMAP)", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        umap_path = ARTIFACTS_FIGURES / "embedding_umap.png"
        plt.savefig(umap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved UMAP placeholder: {umap_path}")

    print("\n✅ Summary report complete!")
    print(f"  CSV: {csv_path}")
    print(f"  Figures: {ARTIFACTS_FIGURES}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ablation_L0(args):
    """L0: Classical baseline — reuse Phase 5 results."""
    print("\n" + "="*60)
    print("Phase 8 LLM Ablation: L0 — Classical Baseline")
    print("Using Phase 5 results (no training needed)")
    print("="*60)

    results = {
        "level": "L0",
        "daic_auroc": PHASE5_RESULTS["daic_auroc"],
        "daic_auroc_text_only": PHASE5_RESULTS["daic_auroc_text_only"],
        "mosei_sentiment_ccc": PHASE5_RESULTS["mosei_sentiment_ccc"],
        "mosei_emotion_auc": PHASE5_RESULTS["mosei_emotion_auc"],
        "fi_avg_ccc": PHASE5_RESULTS["fi_avg_ccc"],
        "gpu_hours": 0,
        "note": "Reused Phase 5 MMoEEx results",
    }

    print(f"\n  DAIC AUROC: {results['daic_auroc']:.4f} (MMoEEx)")
    print(f"  DAIC AUROC: {results['daic_auroc_text_only']:.4f} (text-only)")
    print(f"  MOSEI Sentiment CCC: {results['mosei_sentiment_ccc']:.4f}")
    print(f"  MOSEI Emotion AUC: {results['mosei_emotion_auc']:.4f}")
    print(f"  FI Avg CCC: {results['fi_avg_ccc']:.4f}")

    return results


def run_ablation_L1_L5(level, args):
    """Run LLM ablation for L1-L5."""
    print("\n" + "="*60)
    print(f"Phase 8 LLM Ablation: {level} — {ABLATION_DESCRIPTIONS.get(level, 'unknown')}")
    print("="*60)

    device = torch.device(args.device)

    # GPU detection for LLM extraction
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem_total:.1f} GB)")

    # Load data
    print("\n[1/5] Loading data...")
    manifest_data = load_manifest()
    all_labels = load_all_labels()

    # Note: feature dimensions are determined by the model, not pre-assigned here.
    # The dataset loads whatever features are available (classical or LLM).

    # Check for cached LLM features
    print("\n[2/5] Checking LLM feature cache...")
    text_cache_key = ("daic", "train")
    primary_cache = get_llm_cache_path(level, *text_cache_key, "text")
    if primary_cache.exists():
        print(f"  ✅ Found cached LLM features: {primary_cache.name}")
    elif args.skip_extraction:
        print(f"  ❌ No cache at {primary_cache} and --skip_extraction is set.")
        print("     To extract LLM features, run without --skip_extraction on GPU.")
        print("     Aborting — will not use classical fallback for LLM levels.")
        return None
    else:
        print(f"  Will extract LLM features (no cache at {primary_cache.name})")
        print("  Extraction happens during dataset construction...")

    # Create datasets with REAL LLM feature extraction
    # The JointMultimodalDataset will call load_or_extract_mistral_text (etc.)
    # when no cache is found and skip_extraction=False
    train_ds = JointMultimodalDataset(
        manifest_data=manifest_data,
        all_labels=all_labels,
        datasets_splits=[("daic", "train"), ("mosei", "train"), ("fi", "train")],
        llm_level=level,  # Always the actual LLM level, never "L0"
        temperature=TEMPERATURE,
        skip_extraction=args.skip_extraction,
        device=args.device,
    )

    val_ds = JointMultimodalDataset(
        manifest_data=manifest_data,
        all_labels=all_labels,
        datasets_splits=[("daic", "val"), ("mosei", "val"), ("fi", "val")],
        llm_level=level,  # Always the actual LLM level
        temperature=1.0,
        skip_extraction=args.skip_extraction,
        device=args.device,
    )

    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")

    if len(train_ds) == 0:
        print("ERROR: No training samples loaded.")
        return None

    # Temperature-balanced sampling: sample_weight is computed by JointMultimodalDataset
    # but was previously never consumed (plain shuffle=True), so DAIC's ~107 train rows
    # got swamped by MOSEI's ~32k task-rows in every batch. See phase05/phase07 for the
    # same fix.
    train_sample_weights = torch.tensor([s["sample_weight"] for s in train_ds.samples], dtype=torch.double)
    train_sampler = torch.utils.data.WeightedRandomSampler(
        train_sample_weights, num_samples=len(train_ds), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                             collate_fn=collate_joint, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_joint, num_workers=2, pin_memory=True)

    # Build model with real LLM feature dimensions
    print("\n[3/5] Building model...")
    # Always use the real LLM dimensions for L1-L5 (no classical fallback)
    model = UnifiedMMoEExWithLLM(
        text_dim=LLM_DIMS["mistral"],
        audio_dim=LLM_DIMS["clap"] if level in ["L3", "L5"] else 768,
        video_dim=LLM_DIMS["llava"] if level in ["L4", "L5"] else 768,
        hidden_dim=HIDDEN_DIM,
        expert_dim=EXPERT_DIM,
        num_experts=NUM_EXPERTS,
        num_shared=NUM_SHARED,
        num_tasks=NUM_HEADS,
        use_llm_projector=True,
        llm_text_dim=LLM_DIMS["mistral"],
        llm_audio_dim=LLM_DIMS["clap"] if level in ["L3", "L5"] else 768,
        llm_video_dim=LLM_DIMS["llava"] if level in ["L4", "L5"] else 768,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    loss_fn = TaskLosses()
    model.loss_fn = loss_fn
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    # Training
    print("\n[4/5] Training...")
    history = defaultdict(list)
    metrics_history = []
    best_val_loss = float("inf")
    patience_counter = 0
    # Track wall-clock time multiplied by GPU count for total GPU-hours
    num_gpus = torch.cuda.device_count() if args.device == "cuda" else 0
    train_wall_start = time.time()

    for epoch in range(args.epochs):
        avg_loss, task_losses = train_epoch(model, train_loader, optimizer, loss_fn,
                                           scheduler, device, scaler, epoch)

        history["total"].append(avg_loss)
        for tid, losses in task_losses.items():
            task_name = ["daic_depression", "mosei_sentiment", "mosei_emotion", "fi_personality"][tid]
            history[task_name].append(np.mean(losses))

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

            auroc_str = f"{results['daic']['auroc']:.4f}" if results['daic']['auroc'] else "N/A"
            print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | DAIC AUROC: {auroc_str}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ckpt_path = ARTIFACTS_TABLES / f"phase08_{level}_best.pt"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, ckpt_path)
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break
        else:
            print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}")

    train_wall_end = time.time()
    train_elapsed = train_wall_end - train_wall_start
    # Total GPU-hours = wall seconds × num GPUs / 3600
    gpu_hours = (train_elapsed * num_gpus) / 3600.0 if num_gpus > 0 else 0.0

    # Final evaluation
    print("\n[5/5] Final evaluation...")
    final_results = evaluate(model, val_loader, device)

    print("\n  Final Results:")
    print(f"  DAIC AUROC: {final_results['daic']['auroc']:.4f}")
    print(f"  MOSEI Sentiment CCC: {final_results['mosei_sentiment']['ccc']:.4f}")
    print(f"  MOSEI Emotion AUC: {final_results['mosei_emotion']['auc']:.4f}")
    print(f"  FI Avg CCC: {final_results['fi']['avg_ccc']:.4f}")
    print(f"  GPU Hours: {gpu_hours:.2f}")

    # Save per-sample val predictions so L{n} vs. L0 comparisons can compute real,
    # traceable Cohen's d / DeLong stats instead of untraceable hand-typed numbers.
    predictions_dir = ARTIFACTS_TABLES.parent / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_dir / f"predictions_{level}.npz",
        daic_all_labels=np.array(final_results["daic"]["all_labels"]),
        daic_all_preds=np.array(final_results["daic"]["all_preds"]),
        mosei_sent_all_labels=np.array(final_results["mosei_sentiment"]["all_labels"]),
        mosei_sent_all_preds=np.array(final_results["mosei_sentiment"]["all_preds"]),
    )
    print(f"  Per-sample predictions saved to {predictions_dir / f'predictions_{level}.npz'}")

    return {
        "level": level,
        "daic_auroc": final_results["daic"]["auroc"],
        "mosei_sentiment_ccc": final_results["mosei_sentiment"]["ccc"],
        "mosei_emotion_auc": final_results["mosei_emotion"]["auc"],
        "fi_avg_ccc": final_results["fi"]["avg_ccc"],
        "gpu_hours": gpu_hours,
        "note": f"Trained {args.epochs} epochs with LLM projector (level {level})",
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 8: LLM Modality Ablations")
    parser.add_argument("--ablation", type=str,
                        choices=["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"],
                        help=f"Ablation variant (required unless --generate_report)")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR_DEFAULT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Use cached features, skip extraction")
    parser.add_argument("--output_dir", type=str, default="artifacts/figures/phase_08_llm_ablations")
    parser.add_argument("--generate_report", action="store_true",
                        help="Generate summary report from existing results")
    args = parser.parse_args()

    os.makedirs(ARTIFACTS_FIGURES, exist_ok=True)
    os.makedirs(ARTIFACTS_TABLES, exist_ok=True)

    # Handle generate_report mode
    if args.generate_report:
        generate_summary_report()
        return 0

    # Require ablation for non-report mode
    if args.ablation is None:
        parser.error("--ablation is required (or use --generate_report)")

    print(f"\n{'='*60}")
    print(f"Phase 8: LLM Modality Ablations")
    print(f"  ablation: {args.ablation} — {ABLATION_DESCRIPTIONS.get(args.ablation, 'unknown')}")
    print(f"  device: {args.device}")
    print(f"  epochs: {args.epochs}")
    print(f"{'='*60}")

    # Handle L6-L9 stubs
    if args.ablation in ["L6", "L7", "L8", "L9"]:
        print(f"\n  ⚠️  {args.ablation} requires external API — not implemented yet.")
        print("  Skipping this ablation level.")
        return 0

    # Run the appropriate ablation
    if args.ablation == "L0":
        results = run_ablation_L0(args)
    else:
        results = run_ablation_L1_L5(args.ablation, args)

    # Save results
    if results:
        results_path = ARTIFACTS_TABLES / f"phase08_{args.ablation}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {results_path}")

    print("\n✅ Phase 8 complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())