#!/usr/bin/env python3
"""
XAI Analysis for MPDD Depression Detection
Simplified version - generates bar chart only (summary_plot causes hangs)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import shap
import json
import zipfile
from pathlib import Path
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/home/anilson/thesis/thesis-experiment-4-daic-personality/data/raw/mpdd")
OUTPUT_DIR = Path("/home/anilson/thesis/thesis-experiment-5-unified-model/artifacts/figures/xai_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_mpdd_data():
    """Load MPDD Young track data with pre-extracted features."""
    zip_path = DATA_DIR / "MPDD-Young.zip"

    with zipfile.ZipFile(zip_path, 'r') as zf:
        subject_labels_raw = zf.read("MPDD-Young/Training/labels/personalized_train.json").decode('utf-8')
        subject_labels = json.loads(subject_labels_raw)

        segment_files_raw = zf.read("MPDD-Young/Training/labels/Training_Validation_files.json").decode('utf-8')
        segment_files = json.loads(segment_files_raw)

    n_samples = len(segment_files)
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)

    splits = {"train": [], "val": [], "test": []}

    for idx, seg in enumerate(segment_files):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "val"
        else:
            split = "test"

        filename = seg['audio_feature_path']
        subject_id_raw = filename.split('_')[0]

        subject_label = subject_labels.get(subject_id_raw, {})
        if not subject_label:
            subject_label = subject_labels.get(str(int(subject_id_raw)), {})

        depression_label = seg.get('bin_category')
        if depression_label is None:
            depression_label = int(subject_label.get('binary_depression', 0))

        splits[split].append({
            "id": filename.replace('.npy', ''),
            "subject_id": subject_id_raw,
            "audio_path": f"MPDD-Young/Training/5s/Audio/wav2vec/{seg['audio_feature_path']}",
            "video_path": f"MPDD-Young/Training/5s/Visual/openface/{seg['video_feature_path']}",
            "label": depression_label
        })

    return splits

def load_features_for_split(splits_data, zip_path, split_name):
    """Load features for a specific split."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        features = []
        labels = []

        for item in splits_data:
            audio_data = np.load(BytesIO(zf.read(item["audio_path"])))
            audio_pooled = np.mean(audio_data, axis=0)

            video_data = np.load(BytesIO(zf.read(item["video_path"])))
            video_pooled = np.mean(video_data, axis=0)

            combined = np.concatenate([audio_pooled, video_pooled])
            features.append(combined)
            labels.append(item["label"])

        return np.array(features), np.array(labels)

def main():
    print("=" * 60)
    print("XAI Analysis for MPDD Depression Detection")
    print("=" * 60)

    zip_path = DATA_DIR / "MPDD-Young.zip"

    # Load data
    print("\n1. Loading MPDD data...")
    splits = load_mpdd_data()

    # Load features
    print("Loading features...")
    X_train, y_train = load_features_for_split(splits["train"], zip_path, "train")
    X_val, y_val = load_features_for_split(splits["val"], zip_path, "val")
    X_test, y_test = load_features_for_split(splits["test"], zip_path, "test")

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.1%}")
    print(f"Test positive rate: {y_test.mean():.1%}")

    # Feature names
    audio_names = [f"audio_{i}" for i in range(512)]
    video_names = [f"video_{i}" for i in range(709)]
    feature_names = audio_names + video_names

    # Train logistic regression
    print("\n2. Training Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Try different C values
    best_auc = 0
    best_model = None
    best_c = 0.01

    for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
        model = LogisticRegression(C=c, max_iter=1000, solver='lbfgs')
        model.fit(X_train_scaled, y_train)

        val_probs = model.predict_proba(X_val_scaled)[:, 1]
        auc = roc_auc_score(y_val, val_probs)

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_c = c

    print(f"Best C={best_c}, Val AUC={best_auc:.3f}")

    # Evaluate
    test_probs = best_model.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    print(f"Test AUROC: {test_auc:.3f}")

    # Compute SHAP values
    print("\n3. Computing SHAP values (this may take a few minutes)...")
    n_shap = min(20, len(X_test_scaled))

    # Use background data
    background = shap.kmeans(X_train_scaled, 50)
    explainer = shap.KernelExplainer(best_model.predict_proba, background)

    # Compute SHAP for test samples
    shap_values = explainer.shap_values(X_test_scaled[:n_shap])

    # Handle binary classification
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    print(f"SHAP values computed: {shap_vals.shape}")

    # Mean absolute SHAP values - average over samples and class dimension
    # shap_vals shape: (n_samples, n_features, n_classes) -> mean over (0, 2)
    mean_shap = np.abs(shap_vals).mean(axis=(0, 2))  # Shape: (n_features,)
    print(f"mean_shap shape: {mean_shap.shape}")

    # Top features
    top_k = 30
    top_indices = np.argsort(mean_shap)[-top_k:][::-1]
    print(f"top_indices shape: {top_indices.shape}, dtype: {top_indices.dtype}")

    print(f"\n4. Top {top_k} Important Features:")
    print("-" * 50)
    for i, idx in enumerate(top_indices[:20]):
        idx_int = int(idx)
        modality = "audio" if idx_int < 512 else "video"
        print(f"{i+1}. {modality}_{idx_int % 512 if idx_int >= 512 else idx_int}: {mean_shap[idx_int]:.4f}")

    # Audio vs Video importance
    audio_importance = mean_shap[:512].sum()
    video_importance = mean_shap[512:].sum()
    print(f"\nAudio total importance: {audio_importance:.4f}")
    print(f"Video total importance: {video_importance:.4f}")
    print(f"Video dominance: {video_importance / (audio_importance + video_importance):.1%}")

    # Create bar chart (avoid summary_plot which hangs)
    print("\n5. Creating visualizations...")
    top_names = []
    top_values = []
    top_colors = []

    for i, idx in enumerate(top_indices):
        idx_int = int(idx)
        if idx_int < 512:
            top_names.append(f"Audio_{idx_int}")
            top_colors.append('#2ecc71')  # Green for audio
        else:
            top_names.append(f"Video_{idx_int - 512}")
            top_colors.append('#3498db')  # Blue for video

    plt.figure(figsize=(12, 10))
    y_pos = np.arange(len(top_names))
    plt.barh(y_pos, [mean_shap[i] for i in top_indices], color=top_colors)
    plt.yticks(y_pos, top_names)
    plt.xlabel('Mean |SHAP Value|')
    plt.title(f'Top {top_k} Features for Depression Detection\n(Test AUROC={test_auc:.3f})')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Audio'),
                       Patch(facecolor='#3498db', label='Video')]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance to {OUTPUT_DIR / 'feature_importance.png'}")

    # Save results
    results = {
        "test_auc": float(test_auc),
        "val_auc": float(best_auc),
        "best_c": float(best_c),
        "n_shap_samples": n_shap,
        "audio_total_importance": float(audio_importance),
        "video_total_importance": float(video_importance),
        "video_dominance_pct": float(video_importance / (audio_importance + video_importance)),
        "top_features": [
            {
                "name": feature_names[i],
                "importance": float(mean_shap[i]),
                "modality": "audio" if i < 512 else "video"
            }
            for i in top_indices[:30]
        ]
    }

    with open(OUTPUT_DIR / "xai_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("XAI Analysis Complete!")
    print(f"Results saved to {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()