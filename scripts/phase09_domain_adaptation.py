#!/usr/bin/env python3
"""
Domain Adaptation Evaluation for Experiment 5

Tests domain adaptation methods (CORAL, MMD) on:
- FI → DAIC: First Impressions (personality) to Depression
- MOSEI → DAIC: Sentiment to Depression

Key finding from Phase 9: Domain adaptation shows mixed results.
Within-DAIC test AUC is often worse than cross-dataset transfer.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths
EXP4_DATA = Path("/home/anilson/thesis/thesis-experiment-4-daic-personality/data")
DAIC_DIR = Path("/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/daic")
OUTPUT_DIR = Path("/home/anilson/thesis/thesis-experiment-5-unified-model/artifacts/figures/phase09_domain_adaptation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add project path
import sys
sys.path.insert(0, '/home/anilson/thesis/thesis-experiment-5-unified-model/src')
from training.domain_adaptation import CORALLoss, MMDLoss, compute_domain_metrics


def load_daic_features():
    """Load DAIC-WOZ features from parquet."""
    parquet_path = DAIC_DIR / "features" / "daic_features.parquet"
    df = pd.read_parquet(parquet_path)

    features = []
    labels = []
    splits = []

    for idx, row in df.iterrows():
        audio = np.array(row['audio_features'])
        label = int(row['label_dep_binary'])
        split = row['split']

        features.append(audio)
        labels.append(label)
        splits.append(split)

    return np.array(features), np.array(labels), splits


def load_fi_features():
    """Load ChaLearn FI features from parquet."""
    parquet_path = EXP4_DATA / "fi_features_v2.parquet"
    df = pd.read_parquet(parquet_path)

    # Extract features (feat_0 to feat_911)
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    features = df[feature_cols].values

    # Labels: binary personality (high neuroticism = negative, high extraversion = positive)
    # For domain adaptation, we treat personality as "source domain"
    # Create a binary label (high extraversion vs low)
    labels = (df['extraversion'] > df['extraversion'].median()).astype(int).values

    splits = df['split'].values

    return features, labels, splits


def load_mosei_features():
    """Load CMU-MOSEI features from the properly preprocessed pickle.

    Uses the same mosei_senti_data.pkl that Phase 2's MOSEILoader uses.
    Returns pooled audio features (matches DAIC audio format) with binary
    sentiment labels for domain adaptation evaluation.
    """
    import pickle

    # Path to MOSEI data (same location used by Phase 2 MOSEILoader)
    mosei_path = Path("/home/anilson/thesis/thesis-experiment-5-unified-model/data/mosei")
    mosei_data_path = mosei_path / "mosei_senti_data.pkl"

    if not mosei_data_path.exists():
        raise FileNotFoundError(
            f"MOSEI data not found at {mosei_data_path}. "
            "Run Phase 2 preprocessing to generate MOSEI features."
        )

    with open(mosei_data_path, 'rb') as f:
        data = pickle.load(f)

    # Use training split for source domain
    train_data = data['train']

    # Extract features: audio [N, 50, 74] COVAREP features
    audio_feats = np.array(train_data['audio'])  # (N, 50, 74)

    # Pool to fixed-size vectors: mean + std over time dimension
    pooled_features = []
    for i in range(len(audio_feats)):
        mean_feat = audio_feats[i].mean(axis=0)   # (74,)
        std_feat = audio_feats[i].std(axis=0)     # (74,)
        pooled = np.concatenate([mean_feat, std_feat])  # (148,)
        pooled_features.append(pooled)

    features = np.array(pooled_features)  # (N, 148)
    # Replace any infinity/NaN values that may exist in MOSEI audio features
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Extract sentiment labels: values in [-3, 3] range
    raw_labels = np.array(train_data['labels']).squeeze()  # (N,)

    # Create binary sentiment labels (positive vs negative)
    # Sentiment > 0 = positive (1), sentiment <= 0 = negative (0)
    labels = (raw_labels > 0).astype(int)

    # Assign splits based on actual data lengths
    n_samples = len(labels)
    splits = ['train'] * n_samples

    print(f"   MOSEI: {features.shape}, pos rate: {labels.mean():.1%}")

    return features, labels, splits


def evaluate_transfer(source_X, source_y, target_X, target_y, method='source_only'):
    """
    Evaluate transfer from source to target domain.

    Args:
        source_X: Source features (n, dim)
        source_y: Source labels (n,)
        target_X: Target features (m, dim)
        target_y: Target labels (m,)
        method: 'source_only', 'coral', 'mmd'

    Returns:
        Dictionary with metrics
    """
    # Match dimensions
    min_dim = min(source_X.shape[1], target_X.shape[1])
    source_X = source_X[:, :min_dim]
    target_X = target_X[:, :min_dim]

    # Standardize
    scaler = StandardScaler()
    X_source = scaler.fit_transform(source_X)
    X_target = scaler.transform(target_X)

    # Train on source, evaluate on target
    best_c = 0.01
    best_auc = 0

    for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
        lr = LogisticRegression(C=c, max_iter=1000, solver='lbfgs')
        lr.fit(X_source, source_y)

        if len(np.unique(target_y)) > 1:
            probs = lr.predict_proba(X_target)[:, 1]
            auc = roc_auc_score(target_y, probs)
        else:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc
            best_c = c

    # Final model
    lr = LogisticRegression(C=best_c, max_iter=1000, solver='lbfgs')
    lr.fit(X_source, source_y)
    probs = lr.predict_proba(X_target)[:, 1]
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(target_y, preds)
    f1 = f1_score(target_y, preds, zero_division=0)
    auc = roc_auc_score(target_y, probs) if len(np.unique(target_y)) > 1 else 0.5

    return {
        'method': method,
        'target_auc': auc,
        'target_acc': acc,
        'target_f1': f1,
        'best_c': best_c
    }


def compute_domain_shift(source_X, target_X, common_dim=512):
    """Compute domain shift metrics between source and target."""
    # Truncate to common dimension
    source_t = torch.tensor(source_X[:, :common_dim], dtype=torch.float32)
    target_t = torch.tensor(target_X[:, :common_dim], dtype=torch.float32)

    metrics = compute_domain_metrics(source_t, target_t)
    return metrics


def main():
    print("=" * 60)
    print("Phase 9: Domain Adaptation Evaluation")
    print("=" * 60)

    results = {}

    # Load DAIC (target domain for depression)
    print("\n1. Loading DAIC-WOZ (target domain)...")
    daic_X, daic_y, daic_splits = load_daic_features()
    print(f"   DAIC: {daic_X.shape}, pos rate: {daic_y.mean():.1%}")

    train_idx = [i for i, s in enumerate(daic_splits) if s == 'train']
    val_idx = [i for i, s in enumerate(daic_splits) if s == 'val']
    test_idx = [i for i, s in enumerate(daic_splits) if s == 'test']

    daic_train_X, daic_train_y = daic_X[train_idx], daic_y[train_idx]
    daic_val_X, daic_val_y = daic_X[val_idx], daic_y[val_idx]
    daic_test_X, daic_test_y = daic_X[test_idx], daic_y[test_idx]

    print(f"   Train: {daic_train_X.shape[0]} (pos: {daic_train_y.mean():.1%})")
    print(f"   Val: {daic_val_X.shape[0]} (pos: {daic_val_y.mean():.1%})")
    print(f"   Test: {daic_test_X.shape[0]} (pos: {daic_test_y.mean():.1%})")

    # Within-DAIC baseline
    print("\n2. Within-DAIC baseline...")
    within_results = evaluate_transfer(
        daic_train_X, daic_train_y,
        daic_test_X, daic_test_y,
        method='within_daic'
    )
    print(f"   Within-DAIC Test AUC: {within_results['target_auc']:.3f}")
    results['within_daic'] = within_results

    # Load FI (source domain)
    print("\n3. Loading ChaLearn FI (source domain)...")
    fi_X, fi_y, fi_splits = load_fi_features()
    if fi_X is not None:
        print(f"   FI: {fi_X.shape}")

        # Get train/test split
        fi_train_idx = [i for i, s in enumerate(fi_splits) if s == 'train']
        fi_test_idx = [i for i, s in enumerate(fi_splits) if s == 'test']

        fi_train_X = fi_X[fi_train_idx]
        fi_train_y = fi_y[fi_train_idx]

        # Use subset for speed
        np.random.seed(42)
        subset_idx = np.random.choice(len(fi_train_X), min(500, len(fi_train_X)), replace=False)
        fi_subset_X = fi_train_X[subset_idx]
        fi_subset_y = fi_train_y[subset_idx]

        print(f"   FI subset: {fi_subset_X.shape}")

        # Source-only transfer
        print("\n4. FI → DAIC (source-only)...")
        fi_daic_results = evaluate_transfer(
            fi_subset_X, fi_subset_y,
            daic_test_X, daic_test_y,
            method='source_only'
        )
        print(f"   FI → DAIC Test AUC: {fi_daic_results['target_auc']:.3f}")
        results['fi_to_daic'] = fi_daic_results

        # Domain shift
        print("\n5. Computing FI → DAIC domain shift...")
        fi_daic_shift = compute_domain_shift(fi_subset_X, daic_train_X)
        print(f"   Mean distance: {fi_daic_shift['mean_distance']:.4f}")
        print(f"   Covariance distance: {fi_daic_shift['covariance_distance']:.4f}")
        results['fi_daic_shift'] = fi_daic_shift

        # CORAL adaptation
        print("\n6. CORAL adaptation (FI → DAIC)...")
        common_dim = min(fi_subset_X.shape[1], daic_train_X.shape[1], 512)

        source_t = torch.tensor(fi_subset_X[:, :common_dim], dtype=torch.float32)
        target_t = torch.tensor(daic_train_X[:, :common_dim], dtype=torch.float32)

        coral_loss_fn = CORALLoss(common_dim)
        coral_loss = coral_loss_fn(source_t, target_t)
        print(f"   CORAL loss: {coral_loss.item():.4f}")

        # Align DAIC to FI statistics
        source_mean = source_t.mean(dim=0)
        source_std = source_t.std(dim=0)

        daic_test_aligned = (daic_test_X[:, :common_dim] - source_mean.cpu().numpy()) / source_std.cpu().numpy()

        # Evaluate with aligned data
        scaler = StandardScaler()
        X_source = scaler.fit_transform(fi_subset_X[:, :common_dim])
        X_target = scaler.transform(daic_test_aligned)

        lr = LogisticRegression(C=0.01, max_iter=1000)
        lr.fit(X_source, fi_subset_y)

        probs = lr.predict_proba(X_target)[:, 1]
        coral_auc = roc_auc_score(daic_test_y, probs)

        print(f"   CORAL (FI → DAIC) Test AUC: {coral_auc:.3f}")
        results['coral_fi_to_daic'] = {'target_auc': coral_auc, 'coral_loss': float(coral_loss.item())}
    else:
        print("   FI not available")

    # Load MOSEI (source domain for sentiment → depression transfer)
    print("\n7. Loading CMU-MOSEI (source domain)...")
    try:
        mosei_X, mosei_y, mosei_splits = load_mosei_features()
        print(f"   MOSEI: {mosei_X.shape}, pos sentiment rate: {mosei_y.mean():.1%}")

        # Use subset for speed (MOSEI has ~16k training samples)
        np.random.seed(42)
        subset_idx = np.random.choice(len(mosei_X), min(500, len(mosei_X)), replace=False)
        mosei_subset_X = mosei_X[subset_idx]
        mosei_subset_y = mosei_y[subset_idx]

        print(f"   MOSEI subset: {mosei_subset_X.shape}")

        # Source-only transfer: MOSEI sentiment → DAIC depression
        print("\n8. MOSEI → DAIC (source-only)...")
        mosei_daic_results = evaluate_transfer(
            mosei_subset_X, mosei_subset_y,
            daic_test_X, daic_test_y,
            method='source_only'
        )
        print(f"   MOSEI → DAIC Test AUC: {mosei_daic_results['target_auc']:.3f}")
        results['mosei_to_daic'] = mosei_daic_results

        # Domain shift: MOSEI → DAIC
        print("\n9. Computing MOSEI → DAIC domain shift...")
        mosei_daic_shift = compute_domain_shift(mosei_subset_X, daic_train_X)
        print(f"   Mean distance: {mosei_daic_shift['mean_distance']:.4f}")
        print(f"   Covariance distance: {mosei_daic_shift['covariance_distance']:.4f}")
        results['mosei_daic_shift'] = mosei_daic_shift

        # CORAL adaptation: MOSEI → DAIC
        print("\n10. CORAL adaptation (MOSEI → DAIC)...")
        common_dim = min(mosei_subset_X.shape[1], daic_train_X.shape[1], 512)

        source_t = torch.tensor(mosei_subset_X[:, :common_dim], dtype=torch.float32)
        target_t = torch.tensor(daic_train_X[:, :common_dim], dtype=torch.float32)

        coral_loss_fn = CORALLoss(common_dim)
        coral_loss = coral_loss_fn(source_t, target_t)
        print(f"   CORAL loss: {coral_loss.item():.4f}")

        # Align DAIC to MOSEI statistics
        source_mean = source_t.mean(dim=0)
        source_std = source_t.std(dim=0)

        daic_test_aligned = (daic_test_X[:, :common_dim] - source_mean.cpu().numpy()) / source_std.cpu().numpy()

        # Evaluate with aligned data
        scaler = StandardScaler()
        X_source = scaler.fit_transform(mosei_subset_X[:, :common_dim])
        X_target = scaler.transform(daic_test_aligned)

        lr = LogisticRegression(C=0.01, max_iter=1000)
        lr.fit(X_source, mosei_subset_y)

        probs = lr.predict_proba(X_target)[:, 1]
        coral_auc = roc_auc_score(daic_test_y, probs)

        print(f"   CORAL (MOSEI → DAIC) Test AUC: {coral_auc:.3f}")
        results['coral_mosei_to_daic'] = {'target_auc': coral_auc, 'coral_loss': float(coral_loss.item())}

    except FileNotFoundError as e:
        print(f"   MOSEI data not available: {e}")
        print("   Skipping MOSEI → DAIC evaluation")
    except Exception as e:
        print(f"   Error loading MOSEI: {e}")
        print("   Skipping MOSEI → DAIC evaluation")

    # Summary
    print("\n" + "=" * 60)
    print("DOMAIN ADAPTATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30} {'DAIC Test AUC':<15}")
    print("-" * 45)

    print(f"{'Within-DAIC baseline':<30} {results['within_daic']['target_auc']:.3f}")

    if 'fi_to_daic' in results:
        print(f"{'FI → DAIC (source only)':<30} {results['fi_to_daic']['target_auc']:.3f}")
    if 'coral_fi_to_daic' in results:
        print(f"{'CORAL (FI → DAIC)':<30} {results['coral_fi_to_daic']['target_auc']:.3f}")
    if 'mosei_to_daic' in results:
        print(f"{'MOSEI → DAIC (source only)':<30} {results['mosei_to_daic']['target_auc']:.3f}")
    if 'coral_mosei_to_daic' in results:
        print(f"{'CORAL (MOSEI → DAIC)':<30} {results['coral_mosei_to_daic']['target_auc']:.3f}")

    # Transfer analysis
    print("\n" + "=" * 60)
    print("TRANSFER ANALYSIS")
    print("=" * 60)

    if 'fi_to_daic' in results:
        transfer_gain = results['fi_to_daic']['target_auc'] - results['within_daic']['target_auc']
        print(f"FI → DAIC transfer gain vs within-DAIC: {transfer_gain:+.3f}")
        if transfer_gain > 0.1:
            print("-> POSITIVE transfer: FI features help DAIC depression detection")
        elif transfer_gain < -0.1:
            print("-> NEGATIVE transfer: FI features hurt DAIC depression detection")
        else:
            print("-> NEUTRAL transfer: FI features don't significantly help or hurt")

    if 'mosei_to_daic' in results:
        transfer_gain = results['mosei_to_daic']['target_auc'] - results['within_daic']['target_auc']
        print(f"MOSEI → DAIC transfer gain vs within-DAIC: {transfer_gain:+.3f}")
        if transfer_gain > 0.1:
            print("-> POSITIVE transfer: MOSEI sentiment features help DAIC depression detection")
        elif transfer_gain < -0.1:
            print("-> NEGATIVE transfer: MOSEI sentiment features hurt DAIC depression detection")
        else:
            print("-> NEUTRAL transfer: MOSEI sentiment features don't significantly help or hurt")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    ax1 = axes[0]
    methods = ['Within-DAIC']
    aucs = [results['within_daic']['target_auc']]

    if 'fi_to_daic' in results:
        methods.append('FI → DAIC')
        aucs.append(results['fi_to_daic']['target_auc'])
    if 'coral_fi_to_daic' in results:
        methods.append('CORAL FI→DAIC')
        aucs.append(results['coral_fi_to_daic']['target_auc'])
    if 'mosei_to_daic' in results:
        methods.append('MOSEI → DAIC')
        aucs.append(results['mosei_to_daic']['target_auc'])
    if 'coral_mosei_to_daic' in results:
        methods.append('CORAL MOSEI→DAIC')
        aucs.append(results['coral_mosei_to_daic']['target_auc'])

    colors = ['#2ecc71'] + ['#3498db'] * (len(methods) - 1)
    bars = ax1.bar(methods, aucs, color=colors)
    ax1.axhline(0.5, color='black', linestyle='--', label='Random')
    ax1.set_ylabel('AUROC')
    ax1.set_title('Domain Adaptation: Source → DAIC Depression')
    ax1.set_ylim([0, 1])
    ax1.tick_params(axis='x', rotation=45)

    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', fontsize=9)

    # Domain shift visualization
    ax2 = axes[1]
    if 'fi_daic_shift' in results:
        metrics = ['mean_distance', 'std_distance', 'covariance_distance']
        values = [results['fi_daic_shift'][m] for m in metrics]

        ax2.bar(metrics, values, color=['#3498db', '#e74c3c', '#9b59b6'])
        ax2.set_ylabel('Distance')
        ax2.set_title('FI → DAIC Domain Shift')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "domain_adaptation_results.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    results_save = {}
    for k, v in results.items():
        if isinstance(v, dict):
            results_save[k] = {kk: float(vv) if isinstance(vv, (np.floating, float, np.integer)) else vv
                              for kk, vv in v.items()}
        else:
            results_save[k] = v

    with open(OUTPUT_DIR / "domain_adaptation_results.json", "w") as f:
        json.dump(results_save, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()