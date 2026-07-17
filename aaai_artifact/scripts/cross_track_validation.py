#!/usr/bin/env python3
"""
Cross-track validation: Train on MPDD-Young, evaluate on MPDD-Elderly.
This tests generalization of the depression detection model across age groups.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import json
import zipfile
from pathlib import Path
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Paths
YOUNG_DATA_DIR = Path("data/mpdd")
ELDERLY_DATA_DIR = Path("data/mpdd")
OUTPUT_DIR = Path("artifacts/figures/cross_track_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_mpdd_track(zip_path, track_name):
    """Load a MPDD track (Young or Elderly) with 5s Wav2Vec + OpenFace features."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Load labels
        labels_raw = zf.read(f"{track_name}/Training/labels/personalized_train.json").decode('utf-8')
        labels = json.loads(labels_raw)

        files_raw = zf.read(f"{track_name}/Training/labels/Training_Validation_files.json").decode('utf-8')
        files = json.loads(files_raw)

    # Build sample list
    samples = []
    for seg in files:
        filename = seg['audio_feature_path']
        # Extract subject ID (e.g., "100_A_1" -> "100")
        subject_id = filename.split('_')[0]

        # Get subject label
        subject_label = labels.get(subject_id, {})
        if not subject_label:
            subject_label = labels.get(str(int(subject_id)), {})

        depression_label = seg.get('bin_category')
        if depression_label is None:
            depression_label = int(subject_label.get('binary_depression', 0))

        samples.append({
            "id": seg['audio_feature_path'].replace('.npy', ''),
            "subject_id": subject_id,
            "audio_path": f"{track_name}/Training/5s/Audio/wav2vec/{seg['audio_feature_path']}",
            "video_path": f"{track_name}/Training/5s/Visual/openface/{seg['video_feature_path']}",
            "label": int(depression_label)
        })

    return samples

def load_features(samples, zip_path):
    """Load and concatenate audio + video features for all samples."""
    features = []
    labels = []

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for item in samples:
            # Load audio (5s Wav2Vec)
            audio_data = np.load(BytesIO(zf.read(item["audio_path"])))
            audio_pooled = np.mean(audio_data, axis=0)

            # Load video (OpenFace)
            video_data = np.load(BytesIO(zf.read(item["video_path"])))
            video_pooled = np.mean(video_data, axis=0)

            # Concatenate
            combined = np.concatenate([audio_pooled, video_pooled])
            features.append(combined)
            labels.append(item["label"])

    return np.array(features), np.array(labels)

def main():
    print("=" * 60)
    print("Cross-Track Validation: MPDD-Young → MPDD-Elderly")
    print("=" * 60)

    # Load Young track (training data)
    print("\n1. Loading MPDD-Young (training)...")
    young_zip = YOUNG_DATA_DIR / "MPDD-Young.zip"
    young_samples = load_mpdd_track(young_zip, "MPDD-Young")
    X_young, y_young = load_features(young_samples, young_zip)
    print(f"   Young: {X_young.shape[0]} samples, {X_young.shape[1]} features")
    print(f"   Positive rate: {y_young.mean():.1%}")

    # Use 70/15/15 split for Young
    n_young = len(young_samples)
    n_train = int(n_young * 0.7)
    n_val = int(n_young * 0.15)

    X_train, y_train = X_young[:n_train], y_young[:n_train]
    X_val, y_val = X_young[n_train:n_train+n_val], y_young[n_train:n_train+n_val]

    print(f"   Train: {X_train.shape[0]} samples (pos rate: {y_train.mean():.1%})")
    print(f"   Val: {X_val.shape[0]} samples (pos rate: {y_val.mean():.1%})")

    # Load Elderly track (test data)
    print("\n2. Loading MPDD-Elderly (test)...")
    elderly_zip = ELDERLY_DATA_DIR / "MPDD-Elderly.zip"
    elderly_samples = load_mpdd_track(elderly_zip, "MPDD-Elderly")
    X_elderly, y_elderly = load_features(elderly_samples, elderly_zip)
    print(f"   Elderly: {X_elderly.shape[0]} samples, {X_elderly.shape[1]} features")
    print(f"   Positive rate: {y_elderly.mean():.1%}")

    # Check feature dimensions match
    assert X_train.shape[1] == X_elderly.shape[1], \
        f"Feature dimension mismatch: Young={X_train.shape[1]}, Elderly={X_elderly.shape[1]}"

    # Scale features
    print("\n3. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_elderly_scaled = scaler.transform(X_elderly)

    # Train Logistic Regression on Young
    print("\n4. Training Logistic Regression on Young track...")
    best_auc = 0
    best_model = None
    best_c = 0.01

    for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
        model = LogisticRegression(C=c, max_iter=1000, solver='lbfgs')
        model.fit(X_train_scaled, y_train)

        val_probs = model.predict_proba(X_val_scaled)[:, 1]
        auc = roc_auc_score(y_val, val_probs)

        print(f"   C={c}: Val AUC={auc:.3f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_c = c

    print(f"\n   Best: C={best_c}, Val AUC={best_auc:.3f}")

    # Evaluate on Elderly (zero-shot cross-track)
    print("\n5. Evaluating on Elderly track (zero-shot transfer)...")
    elderly_probs = best_model.predict_proba(X_elderly_scaled)[:, 1]
    elderly_auc = roc_auc_score(y_elderly, elderly_probs)
    elderly_preds = (elderly_probs > 0.5).astype(int)
    elderly_acc = accuracy_score(y_elderly, elderly_preds)
    elderly_f1 = f1_score(y_elderly, elderly_preds)

    print(f"   Elderly AUROC: {elderly_auc:.3f}")
    print(f"   Elderly Accuracy: {elderly_acc:.3f}")
    print(f"   Elderly F1: {elderly_f1:.3f}")

    # Compare with within-track Elderly evaluation
    print("\n6. Within-track Elderly evaluation (train on Elderly, test on Elderly)...")

    # Split Elderly into train/test
    n_elderly = len(elderly_samples)
    n_elderly_train = int(n_elderly * 0.7)
    n_elderly_val = int(n_elderly * 0.15)

    X_elderly_train = X_elderly_scaled[:n_elderly_train]
    y_elderly_train = y_elderly[:n_elderly_train]
    X_elderly_test = X_elderly_scaled[n_elderly_train:n_elderly_train+n_elderly_val]
    y_elderly_test = y_elderly[n_elderly_train:n_elderly_train+n_elderly_val]

    # Train on Elderly subset
    elderly_model = LogisticRegression(C=best_c, max_iter=1000, solver='lbfgs')
    elderly_model.fit(X_elderly_train, y_elderly_train)

    elderly_within_probs = elderly_model.predict_proba(X_elderly_test)[:, 1]
    elderly_within_auc = roc_auc_score(y_elderly_test, elderly_within_probs)

    print(f"   Within-track Elderly AUC: {elderly_within_auc:.3f}")

    # Results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Young-only Val AUROC:     {best_auc:.3f}")
    print(f"Cross-track (Young→Elderly): {elderly_auc:.3f}")
    print(f"Within-track Elderly:      {elderly_within_auc:.3f}")
    print(f"Cross-track gap:           {elderly_within_auc - elderly_auc:.3f}")

    # Save results
    results = {
        "young_val_auc": float(best_auc),
        "cross_track_auc": float(elderly_auc),
        "within_track_auc": float(elderly_within_auc),
        "cross_track_gap": float(elderly_within_auc - elderly_auc),
        "best_c": float(best_c),
        "young_train_size": int(X_train.shape[0]),
        "young_pos_rate": float(y_train.mean()),
        "elderly_size": int(X_elderly.shape[0]),
        "elderly_pos_rate": float(y_elderly.mean())
    }

    with open(OUTPUT_DIR / "cross_track_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart comparison
    ax1 = axes[0]
    methods = ['Young Val\n(within)', 'Cross-Track\n(Young→Elderly)', 'Within Elderly\n(70/15 split)']
    aucs = [best_auc, elderly_auc, elderly_within_auc]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    ax1.bar(methods, aucs, color=colors)
    ax1.set_ylabel('AUROC')
    ax1.set_title('Cross-Track Validation Results')
    ax1.set_ylim([0, 1])
    for i, v in enumerate(aucs):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

    # Distribution comparison
    ax2 = axes[1]
    ax2.hist(elderly_probs[y_elderly == 0], bins=20, alpha=0.5, label='Non-depressed', color='#3498db')
    ax2.hist(elderly_probs[y_elderly == 1], bins=20, alpha=0.5, label='Depressed', color='#e74c3c')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Elderly Predictions Distribution')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cross_track_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()