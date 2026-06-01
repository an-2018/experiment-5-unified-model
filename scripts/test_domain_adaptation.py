#!/usr/bin/env python3
"""
Test if feature normalization (per-feature z-scoring) can improve cross-track transfer.
The hypothesis is that per-feature normalization on Young will generalize better to Elderly.
"""

import numpy as np
import json
import zipfile
from pathlib import Path
from io import BytesIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/home/anilson/thesis/thesis-experiment-4-daic-personality/data/raw/mpdd")
OUTPUT_DIR = Path("/home/anilson/thesis/thesis-experiment-5-unified-model/artifacts/figures/domain_adaptation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_track_features(zip_path, track_name):
    """Load features for a track."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        labels_raw = zf.read(f"{track_name}/Training/labels/personalized_train.json").decode('utf-8')
        labels = json.loads(labels_raw)

        files_raw = zf.read(f"{track_name}/Training/labels/Training_Validation_files.json").decode('utf-8')
        files = json.loads(files_raw)

    samples = []
    for seg in files:
        filename = seg['audio_feature_path']
        subject_id = filename.split('_')[0]

        subject_label = labels.get(subject_id, {})
        if not subject_label:
            subject_label = labels.get(str(int(subject_id)), {})

        depression_label = seg.get('bin_category')
        if depression_label is None:
            depression_label = int(subject_label.get('binary_depression', 0))

        samples.append({
            "audio_path": f"{track_name}/Training/5s/Audio/wav2vec/{seg['audio_feature_path']}",
            "video_path": f"{track_name}/Training/5s/Visual/openface/{seg['video_feature_path']}",
            "label": int(depression_label)
        })

    features = []
    labels = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for item in samples:
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
    print("Domain Adaptation: Feature Normalization for Cross-Track Transfer")
    print("=" * 60)

    # Load both tracks
    print("\n1. Loading track features...")
    X_young, y_young = load_track_features(DATA_DIR / "MPDD-Young.zip", "MPDD-Young")
    X_elderly, y_elderly = load_track_features(DATA_DIR / "MPDD-Elderly.zip", "MPDD-Elderly")

    print(f"   Young: {X_young.shape}, pos rate: {y_young.mean():.1%}")
    print(f"   Elderly: {X_elderly.shape}, pos rate: {y_elderly.mean():.1%}")

    # Split Young into train/val
    n_young = len(X_young)
    n_train = int(n_young * 0.7)
    n_val = int(n_young * 0.15)

    X_train, y_train = X_young[:n_train], y_young[:n_train]
    X_val, y_val = X_young[n_train:n_train+n_val], y_young[n_train:n_train+n_val]

    print(f"   Train: {X_train.shape[0]} (pos: {y_train.mean():.1%})")
    print(f"   Val: {X_val.shape[0]} (pos: {y_val.mean():.1%})")

    # Test different normalization strategies
    strategies = {
        "none": {"method": "raw features", "scaler": None},
        "standard": {"method": "StandardScaler (fit on train)", "scaler": StandardScaler()},
        "per_modality": {"method": "Separate audio/video scaling", "scaler": "per_modality"},
    }

    results = {}

    for name, config in strategies.items():
        print(f"\n2. Testing strategy: {name} ({config['method']})")

        if name == "none":
            X_train_proc = X_train
            X_val_proc = X_val
            X_elderly_proc = X_elderly
        elif name == "standard":
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train)
            X_val_proc = scaler.transform(X_val)
            X_elderly_proc = scaler.transform(X_elderly)
        elif name == "per_modality":
            # Scale audio (0-511) and video (512-1220) separately
            scaler_audio = StandardScaler()
            scaler_video = StandardScaler()

            X_train_audio = scaler_audio.fit_transform(X_train[:, :512])
            X_train_video = scaler_video.fit_transform(X_train[:, 512:])
            X_train_proc = np.hstack([X_train_audio, X_train_video])

            X_val_audio = scaler_audio.transform(X_val[:, :512])
            X_val_video = scaler_video.transform(X_val[:, 512:])
            X_val_proc = np.hstack([X_val_audio, X_val_video])

            X_elderly_audio = scaler_audio.transform(X_elderly[:, :512])
            X_elderly_video = scaler_video.transform(X_elderly[:, 512:])
            X_elderly_proc = np.hstack([X_elderly_audio, X_elderly_video])

        # Train on Young
        best_c = 0.01
        best_auc = 0
        for c in [0.001, 0.01, 0.1, 1.0]:
            lr = LogisticRegression(C=c, max_iter=1000, solver='lbfgs')
            lr.fit(X_train_proc, y_train)

            val_probs = lr.predict_proba(X_val_proc)[:, 1]
            auc = roc_auc_score(y_val, val_probs)

            if auc > best_auc:
                best_auc = auc
                best_c = c

        # Final model
        lr = LogisticRegression(C=best_c, max_iter=1000, solver='lbfgs')
        lr.fit(X_train_proc, y_train)

        # Evaluate on Elderly
        elderly_probs = lr.predict_proba(X_elderly_proc)[:, 1]
        elderly_auc = roc_auc_score(y_elderly, elderly_probs)
        elderly_preds = (elderly_probs > 0.5).astype(int)
        elderly_acc = accuracy_score(y_elderly, elderly_preds)
        elderly_f1 = f1_score(y_elderly, elderly_preds)

        results[name] = {
            "young_val_auc": best_auc,
            "elderly_auc": elderly_auc,
            "elderly_acc": elderly_acc,
            "elderly_f1": elderly_f1,
            "best_c": best_c
        }

        print(f"   Young Val AUC: {best_auc:.3f}")
        print(f"   Elderly AUC: {elderly_auc:.3f}")

    # Also test within-track Elderly baseline
    print("\n3. Within-track Elderly baseline...")
    n_elderly = len(X_elderly)
    n_elderly_train = int(n_elderly * 0.7)
    n_elderly_val = int(n_elderly * 0.15)

    X_elderly_train = X_elderly[:n_elderly_train]
    y_elderly_train = y_elderly[:n_elderly_train]
    X_elderly_val = X_elderly[n_elderly_train:n_elderly_train+n_elderly_val]
    y_elderly_val = y_elderly[n_elderly_train:n_elderly_train+n_elderly_val]

    scaler = StandardScaler()
    X_elderly_train_scaled = scaler.fit_transform(X_elderly_train)
    X_elderly_val_scaled = scaler.transform(X_elderly_val)

    lr = LogisticRegression(C=0.01, max_iter=1000)
    lr.fit(X_elderly_train_scaled, y_elderly_train)
    within_auc = roc_auc_score(y_elderly_val, lr.predict_proba(X_elderly_val_scaled)[:, 1])
    results["within_elderly"] = {"elderly_val_auc": within_auc}

    print(f"   Within-track Elderly Val AUC: {within_auc:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Young Val':<12} {'Elderly AUC':<12}")
    print("-" * 44)
    for name, res in results.items():
        young_val = res.get("young_val_auc", res.get("elderly_val_auc", 0))
        elderly_auc = res.get("elderly_auc", res.get("elderly_val_auc", 0))
        print(f"{name:<20} {young_val:<12.3f} {elderly_auc:<12.3f}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    ax1 = axes[0]
    names = list(results.keys())
    young_vals = [results[n].get("young_val_auc", results[n].get("elderly_val_auc", 0)) for n in names]
    elderly_vals = [results[n].get("elderly_auc", results[n].get("elderly_val_auc", 0)) for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, young_vals, width, label='Young Val AUC', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, elderly_vals, width, label='Elderly AUC', color='#e74c3c')

    ax1.set_ylabel('AUROC')
    ax1.set_title('Domain Adaptation: Cross-Track Transfer')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim([0, 1])

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', fontsize=8)

    # Distribution plot
    ax2 = axes[1]
    for name in ["standard", "per_modality"]:
        elderly_probs = []
        # Re-run to get predictions
        if name == "standard":
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train)
            X_elderly_proc = scaler.transform(X_elderly)
        else:
            scaler_audio = StandardScaler()
            scaler_video = StandardScaler()
            X_train_audio = scaler_audio.fit_transform(X_train[:, :512])
            X_train_video = scaler_video.fit_transform(X_train[:, 512:])
            X_train_proc = np.hstack([X_train_audio, X_train_video])
            X_elderly_audio = scaler_audio.transform(X_elderly[:, :512])
            X_elderly_video = scaler_video.transform(X_elderly[:, 512:])
            X_elderly_proc = np.hstack([X_elderly_audio, X_elderly_video])

        lr = LogisticRegression(C=0.01, max_iter=1000)
        lr.fit(X_train_proc, y_train)
        elderly_probs = lr.predict_proba(X_elderly_proc)[:, 1]

        ax2.hist(elderly_probs, bins=20, alpha=0.5, label=name)

    ax2.axvline(0.5, color='black', linestyle='--', label='Decision boundary')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Elderly Prediction Distribution')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "domain_adaptation_results.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    with open(OUTPUT_DIR / "domain_adaptation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()