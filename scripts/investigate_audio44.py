#!/usr/bin/env python3
"""
Investigate audio_44 feature and cross-track distribution shift.
"""

import numpy as np
import json
import zipfile
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Paths
DATA_DIR = Path("/home/anilson/thesis/thesis-experiment-4-daic-personality/data/raw/mpdd")

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
    print("Investigating audio_44 and Cross-Track Distribution Shift")
    print("=" * 60)

    # Load both tracks
    print("\n1. Loading track features...")
    X_young, y_young = load_track_features(DATA_DIR / "MPDD-Young.zip", "MPDD-Young")
    X_elderly, y_elderly = load_track_features(DATA_DIR / "MPDD-Elderly.zip", "MPDD-Elderly")

    print(f"   Young: {X_young.shape}, pos rate: {y_young.mean():.1%}")
    print(f"   Elderly: {X_elderly.shape}, pos rate: {y_elderly.mean():.1%}")

    # Feature 44 is the 45th audio feature (0-indexed)
    audio_44_idx = 44

    print("\n2. Analyzing audio_44 feature...")
    print(f"   Feature index: {audio_44_idx} (audio feature)")

    # Young statistics
    young_audio44 = X_young[:, audio_44_idx]
    young_pos = X_young[y_young == 1, audio_44_idx]
    young_neg = X_young[y_young == 0, audio_44_idx]

    print(f"\n   YOUNG TRACK:")
    print(f"   - Mean (all): {young_audio44.mean():.4f} (std: {young_audio44.std():.4f})")
    print(f"   - Mean (depressed): {young_pos.mean():.4f} (std: {young_pos.std():.4f})")
    print(f"   - Mean (non-depressed): {young_neg.mean():.4f} (std: {young_neg.std():.4f})")
    print(f"   - Difference: {young_pos.mean() - young_neg.mean():.4f}")

    # Elderly statistics
    elderly_audio44 = X_elderly[:, audio_44_idx]
    elderly_pos = X_elderly[y_elderly == 1, audio_44_idx]
    elderly_neg = X_elderly[y_elderly == 0, audio_44_idx]

    print(f"\n   ELDERLY TRACK:")
    print(f"   - Mean (all): {elderly_audio44.mean():.4f} (std: {elderly_audio44.std():.4f})")
    print(f"   - Mean (depressed): {elderly_pos.mean():.4f} (std: {elderly_pos.std():.4f})")
    print(f"   - Mean (non-depressed): {elderly_neg.mean():.4f} (std: {elderly_neg.std():.4f})")
    print(f"   - Difference: {elderly_pos.mean() - elderly_neg.mean():.4f}")

    # Cross-track comparison
    print(f"\n   CROSS-TRACK COMPARISON:")
    print(f"   - Young mean: {young_audio44.mean():.4f}")
    print(f"   - Elderly mean: {elderly_audio44.mean():.4f}")
    print(f"   - Distribution shift: {abs(young_audio44.mean() - elderly_audio44.mean()):.4f}")

    # Check if audio_44 is predictive in Elderly
    print("\n3. Testing audio_44 predictive power in Elderly...")
    if elderly_pos.shape[0] > 0 and elderly_neg.shape[0] > 0:
        # Simple comparison
        elderly_combined = np.concatenate([elderly_pos, elderly_neg])
        elderly_labels = np.concatenate([np.ones(len(elderly_pos)), np.zeros(len(elderly_neg))])

        # Use ranks to compute AUC (single feature)
        from scipy.stats import rankdata
        ranks = rankdata(elderly_combined)
        auc_single = roc_auc_score(elderly_labels, ranks)
        print(f"   audio_44 AUC on Elderly: {auc_single:.3f}")

    # Check top features in Elderly
    print("\n4. Finding most predictive features in Elderly...")

    # Use within-track LR to find important features for Elderly
    n_train = int(len(y_elderly) * 0.7)
    X_elderly_train = X_elderly[:n_train]
    y_elderly_train = y_elderly[:n_train]

    scaler = StandardScaler()
    X_elderly_train_scaled = scaler.fit_transform(X_elderly_train)

    lr = LogisticRegression(C=0.01, max_iter=1000)
    lr.fit(X_elderly_train_scaled, y_elderly_train)

    # Get coefficients
    coefs = lr.coef_[0]
    top_elderly_idx = np.argsort(np.abs(coefs))[-20:][::-1]

    print("\n   Top 20 features for Elderly (by LR coefficient magnitude):")
    for i, idx in enumerate(top_elderly_idx):
        modality = "audio" if idx < 512 else "video"
        print(f"   {i+1}. {modality}_{idx % 512}: coef={coefs[idx]:.4f}")

    # Check if audio_44 is in top features for Elderly
    if audio_44_idx in top_elderly_idx:
        print(f"\n   audio_44 IS in top 20 for Elderly (rank: {list(top_elderly_idx).index(audio_44_idx)+1})")
    else:
        print(f"\n   audio_44 NOT in top 20 for Elderly (coef={coefs[audio_44_idx]:.4f})")

    # Distribution shift analysis
    print("\n5. Overall distribution shift analysis...")

    # Compute mean shift per modality
    audio_shift = np.abs(X_young[:, :512].mean(axis=0) - X_elderly[:, :512].mean(axis=0)).mean()
    video_shift = np.abs(X_young[:, 512:].mean(axis=0) - X_elderly[:, 512:].mean(axis=0)).mean()

    print(f"   Mean absolute shift (audio): {audio_shift:.4f}")
    print(f"   Mean absolute shift (video): {video_shift:.4f}")
    print(f"   Video shift relative: {video_shift/audio_shift:.2f}x audio")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: audio_44 distribution comparison
    ax1 = axes[0, 0]
    ax1.hist(young_neg, bins=30, alpha=0.5, label='Young Non-Dep', color='#3498db')
    ax1.hist(young_pos, bins=30, alpha=0.5, label='Young Dep', color='#e74c3c')
    ax1.axvline(young_neg.mean(), color='#3498db', linestyle='--', linewidth=2)
    ax1.axvline(young_pos.mean(), color='#e74c3c', linestyle='--', linewidth=2)
    ax1.set_xlabel('audio_44 value')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Young: audio_44 (Dep diff: {young_pos.mean()-young_neg.mean():.3f})')
    ax1.legend()

    # Plot 2: audio_44 in Elderly
    ax2 = axes[0, 1]
    ax2.hist(elderly_neg, bins=30, alpha=0.5, label='Elderly Non-Dep', color='#3498db')
    ax2.hist(elderly_pos, bins=30, alpha=0.5, label='Elderly Dep', color='#e74c3c')
    ax2.axvline(elderly_neg.mean(), color='#3498db', linestyle='--', linewidth=2)
    ax2.axvline(elderly_pos.mean(), color='#e74c3c', linestyle='--', linewidth=2)
    ax2.set_xlabel('audio_44 value')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Elderly: audio_44 (Dep diff: {elderly_pos.mean()-elderly_neg.mean():.3f})')
    ax2.legend()

    # Plot 3: Cross-track distribution of audio_44
    ax3 = axes[1, 0]
    ax3.hist(young_audio44, bins=30, alpha=0.5, label='Young', color='#2ecc71')
    ax3.hist(elderly_audio44, bins=30, alpha=0.5, label='Elderly', color='#9b59b6')
    ax3.set_xlabel('audio_44 value')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Cross-Track: audio_44 Distribution Shift')
    ax3.legend()

    # Plot 4: Top features comparison
    ax4 = axes[1, 1]
    # Get top features for Young (from previous SHAP analysis)
    young_top = [44, 388, 226, 433, 55, 438, 472, 182, 2, 103]  # audio features from SHAP
    elderly_top = [int(i) for i in top_elderly_idx[:10]]

    # Check overlap
    young_audio_top = set([i for i in young_top if i < 512])
    elderly_audio_top = set([i for i in elderly_top if i < 512])
    overlap = young_audio_top & elderly_audio_top

    ax4.bar(['Young\nTop Audio', 'Elderly\nTop Audio', 'Overlap'], 
            [len(young_audio_top), len(elderly_audio_top), len(overlap)],
            color=['#2ecc71', '#9b59b6', '#e74c3c'])
    ax4.set_ylabel('Count')
    ax4.set_title(f'Audio Feature Overlap: {len(overlap)} shared')

    plt.tight_layout()
    plt.savefig(Path("/home/anilson/thesis/thesis-experiment-5-unified-model/artifacts/figures/cross_track_validation/audio_44_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()