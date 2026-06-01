#!/usr/bin/env python3
"""
Phase 10: Calibration and Statistical Validation for Experiment 5

Tests:
1. Calibration methods (Temperature, Platt, Isotonic) on DAIC depression
2. Bootstrap confidence intervals
3. DeLong test for AUROC comparison
4. Effect sizes for ablation comparisons
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, brier_score_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths
DAIC_DIR = Path("/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/daic")
OUTPUT_DIR = Path("/home/anilson/thesis/thesis-experiment-5-unified-model/artifacts/figures/phase10_calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add project paths
import sys
sys.path.insert(0, '/home/anilson/thesis/thesis-experiment-5-unified-model/src')
from training.calibration import (
    TemperatureScaling, PlattScaling, IsotonicCalibrator,
    compute_ece, compute_brier_score, calibrate_logits
)
from evaluation.statistics import bootstrap_ci, delong_auroc_test, compute_cohens_d

import torch


def load_daic():
    """Load DAIC features and labels."""
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


def train_model_and_get_probs(X_train, y_train, X_test):
    """Train LR model and return test probabilities."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(C=0.01, max_iter=1000, solver='lbfgs')
    model.fit(X_train_scaled, y_train)

    probs = model.predict_proba(X_test_scaled)[:, 1]
    logits = model.decision_function(X_test_scaled)

    return probs, logits


def compute_calibration_metrics(probs, labels):
    """Compute calibration metrics."""
    brier = brier_score_loss(labels, probs)
    ece = compute_ece(probs, labels, n_bins=10)
    return {'brier': brier, 'ece': ece}


def plot_reliability_diagram(probs, labels, method_name, save_path):
    """Create reliability diagram (calibration curve)."""
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if in_bin.sum() > 0:
            bin_accs.append(labels[in_bin].mean())
            bin_confs.append(probs[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accs.append(0)
            bin_confs.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bin_counts.append(0)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    # Plot calibration curve
    ax.plot(bin_confs, bin_accs, 'o-', color='#3498db', label=f'{method_name}')

    # Fill between for gap visualization
    ax.fill_between(bin_confs, bin_accs, bin_confs, alpha=0.2, color='#3498db')

    ax.set_xlabel('Confidence (Predicted Probability)')
    ax.set_ylabel('Accuracy (Proportion Positive)')
    ax.set_title(f'Reliability Diagram: {method_name}')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()

    # Add ECE annotation
    ece = compute_ece(probs, labels, n_bins)
    ax.annotate(f'ECE = {ece:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("Phase 10: Calibration and Statistical Validation")
    print("=" * 60)

    # Load DAIC
    print("\n1. Loading DAIC-WOZ...")
    X, y, splits = load_daic()

    train_idx = [i for i, s in enumerate(splits) if s == 'train']
    val_idx = [i for i, s in enumerate(splits) if s == 'val']
    test_idx = [i for i, s in enumerate(splits) if s == 'test']

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"   Train: {X_train.shape[0]} (pos: {y_train.mean():.1%})")
    print(f"   Val: {X_val.shape[0]} (pos: {y_val.mean():.1%})")
    print(f"   Test: {X_test.shape[0]} (pos: {y_test.mean():.1%})")

    # Train base model
    print("\n2. Training base model...")
    probs_train, logits_train = train_model_and_get_probs(X_train, y_train, X_train)
    probs_val, logits_val = train_model_and_get_probs(X_train, y_train, X_val)
    probs_test, logits_test = train_model_and_get_probs(X_train, y_train, X_test)

    print(f"   Base model - Val AUROC: {roc_auc_score(y_val, probs_val):.3f}")
    print(f"   Base model - Test AUROC: {roc_auc_score(y_test, probs_test):.3f}")

    # Compute base calibration metrics
    print("\n3. Computing base calibration metrics...")
    base_metrics = compute_calibration_metrics(probs_test, y_test)
    print(f"   Test Brier Score: {base_metrics['brier']:.4f}")
    print(f"   Test ECE: {base_metrics['ece']:.4f}")

    results = {
        'base': {
            'test_auc': float(roc_auc_score(y_test, probs_test)),
            'brier': float(base_metrics['brier']),
            'ece': float(base_metrics['ece'])
        }
    }

    # Test calibration methods
    print("\n4. Testing calibration methods...")

    calibration_methods = ['none', 'temperature', 'platt', 'isotonic']
    calibration_results = {}

    for method in calibration_methods:
        print(f"\n   {method} calibration:")

        # Use validation set for fitting calibration
        if method == 'none':
            cal_probs_test = probs_test
            cal_probs_val = probs_val
        else:
            # Calibrate using validation logits
            cal_result, info = calibrate_logits(
                logits_train, y_train,
                method=method,
                val_logits=logits_val,
                val_labels=y_val
            )
            # Apply to test
            cal_result_test, _ = calibrate_logits(
                logits_test, y_test,
                method=method,
                val_logits=logits_val,
                val_labels=y_val
            )
            cal_probs_test = cal_result_test

        # Compute metrics
        metrics = compute_calibration_metrics(cal_probs_test, y_test)
        test_auc = roc_auc_score(y_test, cal_probs_test)
        test_acc = accuracy_score(y_test, (cal_probs_test > 0.5).astype(int))

        print(f"   - Test AUROC: {test_auc:.3f}")
        print(f"   - Test Brier: {metrics['brier']:.4f}")
        print(f"   - Test ECE: {metrics['ece']:.4f}")

        calibration_results[method] = {
            'test_auc': float(test_auc),
            'brier': float(metrics['brier']),
            'ece': float(metrics['ece'])
        }

        # Create reliability diagram
        plot_reliability_diagram(
            cal_probs_test, y_test, method.capitalize(),
            OUTPUT_DIR / f"reliability_{method}.png"
        )

    results['calibration'] = calibration_results

    # Bootstrap confidence intervals
    print("\n5. Computing bootstrap confidence intervals...")

    # Bootstrap on validation set for AUROC
    boot_vals = []
    for _ in range(2000):
        idx = np.random.choice(len(y_val), len(y_val), replace=True)
        if len(np.unique(y_val[idx])) > 1:
            boot_vals.append(roc_auc_score(y_val[idx], probs_val[idx]))

    boot_mean, boot_ci_lower, boot_ci_upper = bootstrap_ci(np.array(boot_vals))
    print(f"   Val AUROC: {boot_mean:.3f} (95% CI: {boot_ci_lower:.3f} - {boot_ci_upper:.3f})")

    results['bootstrap_ci'] = {
        'val_auc_mean': float(boot_mean),
        'val_auc_ci_lower': float(boot_ci_lower),
        'val_auc_ci_upper': float(boot_ci_upper)
    }

    # DeLong test (compare base vs best calibration)
    print("\n6. DeLong test for AUROC comparison...")

    best_cal_method = min(calibration_results.keys(),
                         key=lambda k: calibration_results[k]['ece'] if k != 'none' else float('inf'))
    print(f"   Best calibration method: {best_cal_method} (ECE={calibration_results[best_cal_method]['ece']:.4f})")

    if best_cal_method != 'none':
        # Recompute best calibration for test
        best_cal_probs_test, _ = calibrate_logits(
            logits_test, y_test,
            method=best_cal_method,
            val_logits=logits_val,
            val_labels=y_val
        )

        delong_result = delong_auroc_test(y_test, probs_test, best_cal_probs_test)
        z_stat = delong_result['z_statistic']
        p_value = delong_result['p_value']
        print(f"   DeLong z-statistic: {z_stat:.3f}")
        print(f"   DeLong p-value: {p_value:.4f}")

        results['delong_test'] = {
            'z_statistic': float(z_stat),
            'p_value': float(p_value)
        }

    # Effect size (Cohen's d) between uncalibrated and calibrated
    print("\n7. Effect size analysis...")

    if best_cal_method != 'none':
        best_cal_probs_test, _ = calibrate_logits(
            logits_test, y_test,
            method=best_cal_method,
            val_logits=logits_val,
            val_labels=y_val
        )

        d = compute_cohens_d(probs_test, best_cal_probs_test)
        print(f"   Cohen's d (base vs {best_cal_method}): {d:.3f}")

        results['effect_size'] = {
            'cohens_d': float(d),
            'comparison': f'base vs {best_cal_method}'
        }

    # Summary
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'AUROC':<10} {'Brier':<10} {'ECE':<10}")
    print("-" * 50)
    for method, metrics in calibration_results.items():
        print(f"{method:<20} {metrics['test_auc']:.3f}     {metrics['brier']:.4f}   {metrics['ece']:.4f}")

    print(f"\n   Base model Val AUROC: {results['bootstrap_ci']['val_auc_mean']:.3f}")
    print(f"   95% CI: [{results['bootstrap_ci']['val_auc_ci_lower']:.3f}, {results['bootstrap_ci']['val_auc_ci_upper']:.3f}]")

    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of calibration methods
    ax1 = axes[0]
    methods = list(calibration_results.keys())
    eces = [calibration_results[m]['ece'] for m in methods]

    bars = ax1.bar(methods, eces, color=['#e74c3c', '#2ecc71', '#3498db', '#9b59b6'])
    ax1.set_ylabel('ECE (Expected Calibration Error)')
    ax1.set_title('Calibration Quality by Method')
    ax1.tick_params(axis='x', rotation=45)

    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', fontsize=9)

    # AUROC comparison
    ax2 = axes[1]
    aucs = [calibration_results[m]['test_auc'] for m in methods]

    bars = ax2.bar(methods, aucs, color=['#e74c3c', '#2ecc71', '#3498db', '#9b59b6'])
    ax2.set_ylabel('AUROC')
    ax2.set_title('Test AUROC by Calibration Method')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim([0, 1])
    ax2.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Random')

    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibration_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    with open(OUTPUT_DIR / "calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()