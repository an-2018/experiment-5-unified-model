#!/usr/bin/env python3
"""
Cross-Dataset Validation: MPDD-Young → DAIC-WOZ
Tests if depression detection features generalize across datasets.
Uses audio-only features with dimension matching.
"""

import numpy as np
import json
import zipfile
from pathlib import Path
from io import BytesIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Paths
MPDD_DIR = Path("/home/anilson/thesis/thesis-experiment-4-daic-personality/data/raw/mpdd")
DAIC_PARQUET = Path("/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/daic/features/daic_features.parquet")
OUTPUT_DIR = Path("/home/anilson/thesis/thesis-experiment-5-unified-model/artifacts/figures/cross_dataset_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_mpdd_young():
    """Load MPDD-Young track audio features (512-dim Wav2Vec2)."""
    zip_path = MPDD_DIR / "MPDD-Young.zip"

    with zipfile.ZipFile(zip_path, 'r') as zf:
        labels_raw = zf.read("MPDD-Young/Training/labels/personalized_train.json").decode('utf-8')
        labels = json.loads(labels_raw)

        files_raw = zf.read("MPDD-Young/Training/labels/Training_Validation_files.json").decode('utf-8')
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
            "audio_path": f"MPDD-Young/Training/5s/Audio/wav2vec/{seg['audio_feature_path']}",
            "label": int(depression_label)
        })

    features = []
    labels = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for item in samples:
            audio_data = np.load(BytesIO(zf.read(item["audio_path"])))
            audio_pooled = np.mean(audio_data, axis=0)  # (512,)
            features.append(audio_pooled)
            labels.append(item["label"])

    return np.array(features), np.array(labels)

def load_daic():
    """Load DAIC-WOZ features with official splits."""
    df = pd.read_parquet(DAIC_PARQUET)

    features = []
    labels = []
    splits = []

    for idx, row in df.iterrows():
        audio = np.array(row['audio_features'])  # (768,)
        label = int(row['label_dep_binary'])
        split = row['split']

        features.append(audio)
        labels.append(label)
        splits.append(split)

    X = np.array(features)
    y = np.array(labels)

    train_idx = [i for i, s in enumerate(splits) if s == 'train']
    val_idx = [i for i, s in enumerate(splits) if s == 'val']
    test_idx = [i for i, s in enumerate(splits) if s == 'test']

    return X, y, train_idx, val_idx, test_idx

def match_feature_dim(X1, X2, dim):
    """Match feature dimensions by truncating/padding."""
    X1_matched = X1[:, :dim] if X1.shape[1] >= dim else np.pad(X1, ((0,0), (0, dim - X1.shape[1])))
    X2_matched = X2[:, :dim] if X2.shape[1] >= dim else np.pad(X2, ((0,0), (0, dim - X2.shape[1])))
    return X1_matched, X2_matched

def main():
    print("=" * 60)
    print("Cross-Dataset Validation: MPDD-Young → DAIC-WOZ")
    print("=" * 60)

    # Load MPDD (audio-only, 512-dim)
    print("\n1. Loading MPDD-Young (audio features)...")
    X_mpdd, y_mpdd = load_mpdd_young()
    print(f"   MPDD audio: {X_mpdd.shape}, pos rate: {y_mpdd.mean():.1%}")

    # Load DAIC (audio features are 768-dim)
    print("\n2. Loading DAIC-WOZ (audio features)...")
    X_daic, y_daic, train_idx, val_idx, test_idx = load_daic()
    print(f"   DAIC audio: {X_daic.shape}, pos rate: {y_daic.mean():.1%}")
    print(f"   DAIC splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Feature dimension check
    mpdd_audio_dim = X_mpdd.shape[1]  # 512
    daic_audio_dim = X_daic.shape[1]  # 768

    print(f"\n   Feature dimensions:")
    print(f"   - MPDD audio: {mpdd_audio_dim}")
    print(f"   - DAIC audio: {daic_audio_dim}")

    # Use 512-dim common space (MPDD is already 512, DAIC truncate to 512)
    common_dim = min(mpdd_audio_dim, daic_audio_dim)  # 512
    print(f"   - Using common dim: {common_dim}")

    X_mpdd, X_daic = match_feature_dim(X_mpdd, X_daic, common_dim)

    # Split MPDD for training
    n_mpdd = len(X_mpdd)
    n_train = int(n_mpdd * 0.7)
    n_val = int(n_mpdd * 0.15)

    X_mpdd_train, y_mpdd_train = X_mpdd[:n_train], y_mpdd[:n_train]
    X_mpdd_val, y_mpdd_val = X_mpdd[n_train:n_train+n_val], y_mpdd[n_train:n_train+n_val]

    print(f"\n   MPDD Train: {X_mpdd_train.shape[0]} (pos: {y_mpdd_train.mean():.1%})")
    print(f"   MPDD Val: {X_mpdd_val.shape[0]} (pos: {y_mpdd_val.mean():.1%})")

    # Prepare DAIC splits
    X_daic_train = X_daic[train_idx]
    y_daic_train = y_daic[train_idx]
    X_daic_val = X_daic[val_idx]
    y_daic_val = y_daic[val_idx]
    X_daic_test = X_daic[test_idx]
    y_daic_test = y_daic[test_idx]

    print(f"\n   DAIC Train: {X_daic_train.shape[0]} (pos: {y_daic_train.mean():.1%})")
    print(f"   DAIC Val: {X_daic_val.shape[0]} (pos: {y_daic_val.mean():.1%})")
    print(f"   DAIC Test: {X_daic_test.shape[0]} (pos: {y_daic_test.mean():.1%})")

    # Test different scaling approaches
    print("\n3. Training on MPDD, evaluating on DAIC...")
    results = {}

    for scale_method in ["none", "standard"]:
        print(f"\n   Scaling: {scale_method}")

        if scale_method == "none":
            X_mpdd_train_p = X_mpdd_train
            X_mpdd_val_p = X_mpdd_val
            X_daic_train_p = X_daic_train
            X_daic_val_p = X_daic_val
            X_daic_test_p = X_daic_test
        else:
            scaler_mpdd = StandardScaler()
            X_mpdd_train_p = scaler_mpdd.fit_transform(X_mpdd_train)
            X_mpdd_val_p = scaler_mpdd.transform(X_mpdd_val)

            # Scale DAIC with MPDD stats (domain adaptation)
            X_daic_train_p = scaler_mpdd.transform(X_daic_train)
            X_daic_val_p = scaler_mpdd.transform(X_daic_val)
            X_daic_test_p = scaler_mpdd.transform(X_daic_test)

        # Find best C on MPDD val
        best_c = 0.01
        best_auc = 0
        for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
            lr = LogisticRegression(C=c, max_iter=1000, solver='lbfgs')
            lr.fit(X_mpdd_train_p, y_mpdd_train)

            val_probs = lr.predict_proba(X_mpdd_val_p)[:, 1]
            auc = roc_auc_score(y_mpdd_val, val_probs)

            if auc > best_auc:
                best_auc = auc
                best_c = c

        # Train final model
        lr = LogisticRegression(C=best_c, max_iter=1000, solver='lbfgs')
        lr.fit(X_mpdd_train_p, y_mpdd_train)

        # Evaluate on DAIC splits
        daic_train_probs = lr.predict_proba(X_daic_train_p)[:, 1]
        daic_val_probs = lr.predict_proba(X_daic_val_p)[:, 1]
        daic_test_probs = lr.predict_proba(X_daic_test_p)[:, 1]

        daic_train_auc = roc_auc_score(y_daic_train, daic_train_probs)
        daic_val_auc = roc_auc_score(y_daic_val, daic_val_probs) if len(np.unique(y_daic_val)) > 1 else float('nan')
        daic_test_auc = roc_auc_score(y_daic_test, daic_test_probs) if len(np.unique(y_daic_test)) > 1 else float('nan')

        results[scale_method] = {
            "mpdd_val_auc": best_auc,
            "daic_train_auc": daic_train_auc,
            "daic_val_auc": daic_val_auc,
            "daic_test_auc": daic_test_auc,
            "best_c": best_c
        }

        print(f"   MPDD Val AUC: {best_auc:.3f}")
        print(f"   DAIC Train AUC: {daic_train_auc:.3f}")
        print(f"   DAIC Val AUC: {daic_val_auc:.3f}")
        print(f"   DAIC Test AUC: {daic_test_auc:.3f}")

    # Within-DAIC baseline
    print("\n4. Within-DAIC baseline...")
    scaler_daic = StandardScaler()
    X_daic_train_scaled = scaler_daic.fit_transform(X_daic_train)
    X_daic_val_scaled = scaler_daic.transform(X_daic_val)
    X_daic_test_scaled = scaler_daic.transform(X_daic_test)

    best_c = 0.01
    best_auc = 0
    for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
        lr = LogisticRegression(C=c, max_iter=1000, solver='lbfgs')
        lr.fit(X_daic_train_scaled, y_daic_train)
        val_probs = lr.predict_proba(X_daic_val_scaled)[:, 1]
        auc = roc_auc_score(y_daic_val, val_probs)
        if auc > best_auc:
            best_auc = auc
            best_c = c

    lr = LogisticRegression(C=best_c, max_iter=1000)
    lr.fit(X_daic_train_scaled, y_daic_train)
    within_val_auc = roc_auc_score(y_daic_val, lr.predict_proba(X_daic_val_scaled)[:, 1])
    within_test_auc = roc_auc_score(y_daic_test, lr.predict_proba(X_daic_test_scaled)[:, 1])

    results["within_daic"] = {
        "daic_val_auc": within_val_auc,
        "daic_test_auc": within_test_auc,
        "best_c": best_c
    }

    print(f"   Within-DAIC Val AUC: {within_val_auc:.3f}")
    print(f"   Within-DAIC Test AUC: {within_test_auc:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'MPDD Val':<12} {'DAIC Test':<12}")
    print("-" * 50)
    for name, res in results.items():
        mpdd_val = res.get("mpdd_val_auc", res.get("daic_val_auc", "-"))
        daic_test = res.get("daic_test_auc", res.get("daic_val_auc", "-"))
        mpdd_str = f"{mpdd_val:.3f}" if isinstance(mpdd_val, float) else str(mpdd_val)
        daic_str = f"{daic_test:.3f}" if isinstance(daic_test, float) else str(daic_test)
        print(f"{name:<25} {mpdd_str:<12} {daic_str:<12}")

    # Transfer gap analysis
    if 'standard' in results and 'within_daic' in results:
        transfer_gap = results['standard']['daic_val_auc'] - results['within_daic']['daic_val_auc']
        print(f"\n   Transfer gap (cross vs within): {transfer_gap:.3f}")
        if transfer_gap < -0.1:
            print("   -> NEGATIVE transfer detected!")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    ax1 = axes[0]
    methods = list(results.keys())
    mpdd_vals = [results[m].get("mpdd_val_auc", results[m].get("daic_val_auc", 0)) for m in methods]
    daic_vals = [results[m].get("daic_test_auc", results[m].get("daic_val_auc", 0)) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax1.bar(x - width/2, mpdd_vals, width, label='Source (MPDD)', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, daic_vals, width, label='Target (DAIC)', color='#e74c3c')

    ax1.set_ylabel('AUROC')
    ax1.set_title('Cross-Dataset Transfer: MPDD → DAIC (Audio)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
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
    scaler_mpdd = StandardScaler()
    X_mpdd_train_p = scaler_mpdd.fit_transform(X_mpdd_train)
    X_daic_test_p = scaler_mpdd.transform(X_daic_test)

    lr = LogisticRegression(C=results['standard']['best_c'], max_iter=1000)
    lr.fit(X_mpdd_train_p, y_mpdd_train)
    daic_test_probs = lr.predict_proba(X_daic_test_p)[:, 1]

    ax2.hist(daic_test_probs[y_daic_test == 0], bins=20, alpha=0.5, label='Non-depressed', color='#3498db')
    ax2.hist(daic_test_probs[y_daic_test == 1], bins=20, alpha=0.5, label='Depressed', color='#e74c3c')
    ax2.axvline(0.5, color='black', linestyle='--', label='Decision boundary')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('MPDD-trained Model on DAIC Test Set')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cross_dataset_mpdd_daic.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    with open(OUTPUT_DIR / "cross_dataset_results.json", "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()