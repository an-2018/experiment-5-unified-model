#!/usr/bin/env python3
"""
Phase 13: KNN Voting Baseline — No Graph, No GNN
===================================================
Isolates whether GraphSAGE learned aggregation adds value beyond simple
neighborhood smoothing (distance-weighted KNN voting on fused embeddings).

Uses the same fused multimodal embeddings (256-dim) from the trained Phase 5
MMoEEx model that the GraphSAGE router uses (via GatedLateFusion).

Key assumption: If GraphSAGE routing outperforms KNN voting, then the GNN
encoder + learned routing weights provide benefits beyond nearest-neighbor
label smoothing. If KNN voting matches or exceeds GraphSAGE, the graph
structure adds no value beyond neighborhood label propagation.

Usage:
    uv run python scripts/phase13_knn_voting_baseline.py

Outputs:
    artifacts/tables/knn_voting_results.csv
    Updated paper/tables/chapter8_ablation_ladder.tex
"""
import json
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FEATURES_ROOT = ROOT / "data" / "features"
MANIFEST_PATH = FEATURES_ROOT / "manifest.json"
ARTIFACTS_TABLES = ROOT / "artifacts" / "tables"
OUTPUT_CSV = ARTIFACTS_TABLES / "knn_voting_results.csv"

DAIC_DATA = ROOT / "data" / "daic"
MOSEI_DATA = ROOT / "data" / "mosei"
FI_DATA = ROOT / "data" / "fi"

HIDDEN_DIM = 256
NUM_NEIGHBORS = 10

FI_TRAITS = ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]
EMOTION_LABELS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

FEATURE_DIMS = {"text": 768, "audio": 768, "video": 1536}

TASK_TO_EXPERTS = {0: [0, 1], 1: [2, 3], 2: [2, 3], 3: [4, 5]}

# Import actual Phase 5 model components for exact checkpoint compatibility
from models.fusion import GatedLateFusion
from models.unified_moe import MMoEEx
from models.task_heads import DepressionHead, SentimentHead, EmotionMultiLabelHead, PersonalityHead


# =============================================================================
# Model — exactly matches Phase 5 UnifiedMMoEEx checkpoint structure
# =============================================================================

class UnifiedEmbeddingModel(nn.Module):
    """Matches Phase 5 UnifiedMMoEEx exactly for checkpoint loading."""

    def __init__(self, text_dim=768, audio_dim=768, video_dim=1536):
        super().__init__()
        self.fusion = GatedLateFusion(text_dim, audio_dim, video_dim, HIDDEN_DIM)
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
        )
        self.audio_projector = nn.Sequential(
            nn.Linear(audio_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
        )
        self.video_projector = nn.Sequential(
            nn.Linear(video_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
        )
        self.mmoe = MMoEEx(
            input_dim=HIDDEN_DIM, num_experts=8, expert_dim=256,
            num_tasks=4, num_shared=2, expert_isolation=True,
            task_to_experts=TASK_TO_EXPERTS,
        )
        self.depression_head = DepressionHead(256)
        self.sentiment_head = SentimentHead(256)
        self.emotion_head = EmotionMultiLabelHead(256)
        self.personality_head = PersonalityHead(256)

    def get_fused_embedding(self, text, audio, video, mask):
        return self.fusion(text, audio, video, mask)


# =============================================================================
# Data loading helpers
# =============================================================================

def load_all_labels():
    labels = {}
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

    emotion_path = MOSEI_DATA / "mosei_emotion_labels.json"
    if emotion_path.exists():
        with open(emotion_path, "r") as f:
            mosei_labels_data = json.load(f)
        for key, label_data in mosei_labels_data.items():
            sentiment = float(label_data.get("sentiment", 0.0))
            emotions = [float(label_data.get(e, 0.0)) for e in EMOTION_LABELS]
            labels[key] = [sentiment] + emotions

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
    with open(MANIFEST_PATH, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("samples", [])


def _try_load_feature(path_str, dim):
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
    except Exception:
        return False, np.zeros(dim, dtype=np.float32)


def get_label_key(entry):
    ds_name = entry["dataset"]
    sample_id = entry["id"]
    if ds_name == "daic":
        return f"daic_{sample_id}"
    elif ds_name == "mosei":
        return sample_id
    else:
        return sample_id


# =============================================================================
# Model loading
# =============================================================================

def build_and_load_model(device):
    model = UnifiedEmbeddingModel(
        text_dim=FEATURE_DIMS["text"],
        audio_dim=FEATURE_DIMS["audio"],
        video_dim=FEATURE_DIMS["video"],
    )
    ckpt_path = ARTIFACTS_TABLES / "mmoe_ex_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    for k in list(state_dict.keys()):
        if k.startswith("loss_fn.") or k.startswith("log_sigma"):
            del state_dict[k]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (should be 0): {missing}")
    if unexpected:
        ok_unexpected = [k for k in unexpected if k not in ["mmoe.log_task_weights"]]
        if ok_unexpected:
            print(f"  Warning: {len(ok_unexpected)} unexpected keys: {ok_unexpected[:3]}...")
    model.to(device)
    model.eval()
    return model


# =============================================================================
# Embedding extraction
# =============================================================================

def extract_embeddings_for_split(manifest, all_labels, split, model, device):
    embeddings = []
    meta = []

    for ds_name in ["daic", "mosei", "fi"]:
        ds_entries = [e for e in manifest if e["dataset"] == ds_name and e.get("split") == split]
        for entry in ds_entries:
            label_key = get_label_key(entry)
            if label_key not in all_labels:
                continue
            feat_map = entry["features"]
            t_key = feat_map.get("text_roberta")
            a_key = feat_map.get("audio_wavlm")
            v_key = feat_map.get("video_vit") if ds_name != "daic" else feat_map.get("video_openface")
            t_ok, t_vec = _try_load_feature(t_key, FEATURE_DIMS["text"])
            a_ok, a_vec = _try_load_feature(a_key, FEATURE_DIMS["audio"])
            v_ok, v_vec = _try_load_feature(v_key, FEATURE_DIMS["video"])
            if not (t_ok or a_ok or v_ok):
                continue

            with torch.no_grad():
                t_t = torch.from_numpy(t_vec).unsqueeze(0).to(device)
                a_t = torch.from_numpy(a_vec).unsqueeze(0).to(device)
                v_t = torch.from_numpy(v_vec).unsqueeze(0).to(device)
                mask = torch.tensor([[t_ok, a_ok, v_ok]], dtype=torch.bool, device=device)
                fused = model.get_fused_embedding(t_t, a_t, v_t, mask)
                emb = fused.cpu().numpy().flatten()

            embeddings.append(emb)
            meta.append({"dataset": ds_name, "label": all_labels[label_key], "key": label_key})

    if not embeddings:
        return np.empty((0, HIDDEN_DIM)), []
    return np.array(embeddings, dtype=np.float32), meta


# =============================================================================
# KNN Weighted Voting
# =============================================================================

def run_knn_voting():
    from sklearn.neighbors import NearestNeighbors

    device = torch.device("cpu")
    print("[1/5] Loading model...")
    model = build_and_load_model(device)

    print("[2/5] Loading manifest and labels...")
    manifest = load_manifest()
    all_labels = load_all_labels()

    print("[3/5] Extracting train embeddings...")
    train_embeddings, train_meta = extract_embeddings_for_split(manifest, all_labels, "train", model, device)

    print("[4/5] Extracting val embeddings...")
    val_embeddings, val_meta = extract_embeddings_for_split(manifest, all_labels, "val", model, device)

    n_train = len(train_embeddings)
    n_val = len(val_embeddings)
    print(f"  Train: {n_train}, Val: {n_val}")

    if n_train == 0 or n_val == 0:
        raise RuntimeError("No embeddings extracted.")

    effective_k = min(NUM_NEIGHBORS, n_train)
    print(f"[5/5] Running KNN (k={effective_k})...")

    nn_model = NearestNeighbors(n_neighbors=effective_k, metric="cosine", algorithm="brute")
    nn_model.fit(train_embeddings)
    distances, indices = nn_model.kneighbors(val_embeddings)
    similarities = 1.0 / (1.0 + np.maximum(distances, 1e-10))

    results = {
        "daic": {"all_labels": [], "all_preds": []},
        "mosei_sent": {"all_labels": [], "all_preds": []},
        "mosei_emo": {"all_labels": [], "all_preds": []},
        "fi": {"all_labels": [], "all_preds": []},
    }

    for vi, vmeta in enumerate(val_meta):
        v_ds = vmeta["dataset"]
        sims = similarities[vi]
        nbr_indices = indices[vi]

        if v_ds == "daic":
            total_w = 0.0
            weighted_sum = 0.0
            for ni, w in zip(nbr_indices, sims):
                tmeta = train_meta[ni]
                if tmeta["dataset"] == "daic":
                    weighted_sum += w * tmeta["label"]
                    total_w += w
            pred = (weighted_sum / total_w) if total_w > 0 else 0.5
            results["daic"]["all_labels"].append(vmeta["label"])
            results["daic"]["all_preds"].append(pred)

        elif v_ds == "mosei":
            v_label = vmeta["label"]
            total_w_s = 0.0
            weighted_sum_s = 0.0
            total_w_e = np.zeros(6)
            weighted_sum_e = np.zeros(6)
            for ni, w in zip(nbr_indices, sims):
                tmeta = train_meta[ni]
                if tmeta["dataset"] == "mosei":
                    t_label = tmeta["label"]
                    weighted_sum_s += w * t_label[0]
                    total_w_s += w
                    for ei in range(6):
                        weighted_sum_e[ei] += w * t_label[ei + 1]
                    total_w_e += w
            sent_pred = (weighted_sum_s / total_w_s) if total_w_s > 0 else 0.0
            emo_preds = np.divide(weighted_sum_e, total_w_e, out=np.zeros_like(weighted_sum_e), where=total_w_e > 0)
            results["mosei_sent"]["all_labels"].append(v_label[0])
            results["mosei_sent"]["all_preds"].append(sent_pred)
            results["mosei_emo"]["all_labels"].append(np.array([v_label[i] for i in range(1, 7)]))
            results["mosei_emo"]["all_preds"].append(emo_preds)

        elif v_ds == "fi":
            total_w = 0.0
            weighted_sum = np.zeros(5)
            for ni, w in zip(nbr_indices, sims):
                tmeta = train_meta[ni]
                if tmeta["dataset"] == "fi":
                    td = tmeta["label"]
                    trait_vals = np.array([td[t] for t in FI_TRAITS])
                    weighted_sum += w * trait_vals
                    total_w += w
            pred_traits = weighted_sum / total_w if total_w > 0 else np.zeros(5)
            ref_traits = np.array([vmeta["label"][t] for t in FI_TRAITS])
            results["fi"]["all_labels"].append(ref_traits)
            results["fi"]["all_preds"].append(pred_traits)

    for key in ["daic", "mosei_sent"]:
        if results[key]["all_labels"]:
            results[key]["all_labels"] = np.array(results[key]["all_labels"])
            results[key]["all_preds"] = np.array(results[key]["all_preds"])
    if results["mosei_emo"]["all_labels"]:
        results["mosei_emo"]["all_labels"] = np.vstack(results["mosei_emo"]["all_labels"])
        results["mosei_emo"]["all_preds"] = np.vstack(results["mosei_emo"]["all_preds"])
    if results["fi"]["all_labels"]:
        results["fi"]["all_labels"] = np.vstack(results["fi"]["all_labels"])
        results["fi"]["all_preds"] = np.vstack(results["fi"]["all_preds"])

    return results


# =============================================================================
# Metrics
# =============================================================================

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


def compute_auroc(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_pred)


def evaluate_results(results):
    metrics = {}
    if len(results["daic"]["all_labels"]) > 0:
        metrics["daic_auroc"] = compute_auroc(results["daic"]["all_labels"], results["daic"]["all_preds"])
        print(f"  DAIC AUROC: {metrics['daic_auroc']:.4f} ({len(results['daic']['all_labels'])} samples)")
    if len(results["mosei_sent"]["all_labels"]) > 0:
        metrics["mosei_sentiment_ccc"] = compute_ccc(results["mosei_sent"]["all_labels"], results["mosei_sent"]["all_preds"])
        print(f"  MOSEI Sent CCC: {metrics['mosei_sentiment_ccc']:.4f}")
    if len(results["mosei_emo"]["all_labels"]) > 0:
        emo_aucs = []
        for ei in range(6):
            try:
                auc = compute_auroc(
                    (results["mosei_emo"]["all_labels"][:, ei] >= 0.3).astype(int),
                    results["mosei_emo"]["all_preds"][:, ei],
                )
                emo_aucs.append(auc)
            except Exception:
                emo_aucs.append(0.5)
        metrics["mosei_emotion_auc"] = np.mean(emo_aucs)
        print(f"  MOSEI Emotion AUC: {metrics['mosei_emotion_auc']:.4f}")
    if len(results["fi"]["all_labels"]) > 0:
        per_trait_ccc = {}
        for i, trait in enumerate(FI_TRAITS):
            per_trait_ccc[trait] = compute_ccc(results["fi"]["all_labels"][:, i], results["fi"]["all_preds"][:, i])
        metrics["fi_avg_ccc"] = np.mean(list(per_trait_ccc.values()))
        for trait, ccc in per_trait_ccc.items():
            metrics[f"fi_{trait}_ccc"] = ccc
        print(f"  FI Avg CCC: {metrics['fi_avg_ccc']:.4f}")
    return metrics


# =============================================================================
# Save and update LaTeX
# =============================================================================

def save_results_csv(metrics):
    ARTIFACTS_TABLES.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w") as f:
        f.write("metric,value\n")
        for key, val in sorted(metrics.items()):
            if isinstance(val, (int, float)):
                f.write(f"{key},{val:.4f}\n")
    print(f"  Saved: {OUTPUT_CSV}")


def update_latex_table(metrics):
    tex_path = ROOT / "paper" / "tables" / "chapter8_ablation_ladder.tex"
    if not tex_path.exists():
        print(f"  Warning: {tex_path} not found")
        return

    daic_auroc = metrics.get("daic_auroc", 0.0)
    mosei_ccc = metrics.get("mosei_sentiment_ccc", 0.0)
    mosei_emo = metrics.get("mosei_emotion_auc", 0.0)
    fi_ccc = metrics.get("fi_avg_ccc", 0.0)
    new_row = f"4 & + KNN Voting (no GNN) & {daic_auroc:.4f} & {mosei_ccc:.4f} & {mosei_emo:.4f} & {fi_ccc:.4f} \\\\"

    content = tex_path.read_text()
    if "4 & + Graph (V0" not in content or "3 & + MMoEEx" not in content:
        print(f"  Could not find expected rows in LaTeX table")
        return

    lines = content.split('\n')
    new_lines = []
    inserted = False
    for line in lines:
        if not inserted and "4 & + Graph (V0" in line:
            new_lines.append(new_row)
            inserted = True
            line = line.replace('4 & + Graph (V0', '5 & + Graph (V0', 1)
        if "5 & + Graph (V3" in line:
            line = line.replace('5 & + Graph (V3', '6 & + Graph (V3', 1)
        new_lines.append(line)

    tex_path.write_text('\n'.join(new_lines))
    print(f"  Updated: {tex_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Phase 13: KNN Voting Baseline (No Graph)")
    print("=" * 60)

    results = run_knn_voting()

    print("\nEvaluating...")
    metrics = evaluate_results(results)

    print("\nSaving results...")
    save_results_csv(metrics)
    update_latex_table(metrics)

    print("\nSummary:")
    for key, val in sorted(metrics.items()):
        if isinstance(val, (int, float)):
            print(f"  {key}: {val:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
