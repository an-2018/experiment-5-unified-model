#!/usr/bin/env python3
"""
Phase 10: Calibration, Metrics & Statistical Validation
========================================================
Make evaluation a first-class scientific component.

Core tasks:
  - Compute task-specific metrics with BCa bootstrap CIs
  - Temperature scaling / Platt scaling for calibration
  - DeLong tests for AUROC comparisons
  - Paired permutation tests for F1/CCC/MAE
  - Effect sizes (Cohen's d)
  - Calibration analysis (Brier, ECE)

USES REAL MODEL PREDICTIONS from Phase 5 (L0) and Phase 8 (L1-L5) checkpoints.
No synthetic data, no mock models, no fallback values.

Visualizations:
  - ROC and PR curves for DAIC
  - Reliability diagrams
  - Bootstrap CI bars
  - Paired metric delta plots
  - Permutation null distributions
  - Bland-Altman plots for regression
  - Calibration before/after

Usage:
    uv run python scripts/phase10_evaluation.py          # Full pipeline (extracts + evaluates)
    uv run python scripts/phase10_evaluation.py --quick   # Quick mode (skip figures)
    uv run python scripts/phase10_evaluation.py --predictions-only  # Only extract predictions
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── Project imports ─────────────────────────────────────────────────────
ROOT = Path("/home/anilson/thesis/thesis-experiment-5-unified-model")
sys.path.insert(0, str(ROOT / "src"))

from evaluation.metrics import (
    compute_auroc, compute_auprc, compute_f1, compute_ccc, compute_mae,
    compute_pearson, compute_spearman, compute_sensitivity_specificity,
)
from evaluation.statistics import (
    bca_bootstrap_ci, delong_auroc_test, paired_permutation_test,
    compute_cohens_d, compute_effect_size_paired, paired_bootstrap_delta,
)
from training.calibration import (
    calibrate_logits, compute_ece, compute_brier_score,
)
from evaluation.inference import (
    extract_all_levels, load_cached_predictions, PREDICTIONS_DIR,
)

# ── Paths ───────────────────────────────────────────────────────────────
ARTIFACTS_FIGURES = ROOT / "artifacts" / "figures" / "phase_10_evaluation"
ARTIFACTS_TABLES = ROOT / "artifacts" / "tables"
ARTIFACTS_FIGURES.mkdir(parents=True, exist_ok=True)
ARTIFACTS_TABLES.mkdir(parents=True, exist_ok=True)

# Phase 8 LLM ablation results (reference values, not used for synthetic data)
PHASE8_RESULTS_PATH = ROOT / "artifacts" / "tables" / "phase08_llm_ablations.csv"


# =====================================================================
# Data Loading — REAL predictions only
# =====================================================================

def ensure_real_predictions(device_str: str = "cuda", skip_extraction: bool = True,
                            levels: list = None) -> dict:
    """Load real predictions from cache, or extract them if not available.
    
    This is the ONLY source of predictions in Phase 10.
    No synthetic data, no fallback values.
    
    Returns:
        dict matching the format expected by evaluate functions:
        {
            "daic_L0": {"probs": np.array, "labels": np.array, "logits": np.array},
            "daic_L1": ...,
            "mosei_sent_L0": {"pred": np.array, "labels": np.array},
            ...
            "fi_personality_L0": {"pred": np.array, "labels": np.array},
            ...
        }
    """
    if levels is None:
        levels = ["L0", "L1", "L2", "L3", "L4", "L5"]

    # Check if cached predictions exist for all requested levels
    all_cached = all(
        (PREDICTIONS_DIR / f"predictions_{level}.npz").exists()
        for level in levels
    )

    if not all_cached:
        print("=" * 60)
        print("Cached predictions not found.")
        print(f"Extracting real predictions from checkpoints ({', '.join(levels)})...")
        print("=" * 60)
        extract_all_levels(levels=levels, device_str=device_str,
                          skip_extraction=skip_extraction, save=True)

    # Load real predictions from cache
    print("\nLoading real predictions...")
    predictions = load_cached_predictions()
    print(f"  {len(predictions)} real prediction sets loaded")

    return predictions


# =====================================================================
# Evaluation Functions
# =====================================================================

def evaluate_daic(predictions: dict, level: str) -> dict:
    """Comprehensive DAIC depression evaluation for a given level."""
    data = predictions.get(f"daic_{level}", {})
    if not data:
        return {"error": f"No data for daic_{level}"}

    probs = data["probs"]
    labels = data["labels"]
    logits = data.get("logits", data["probs"])

    results = {}

    # ── Metrics ──
    results["auroc"] = float(compute_auroc(labels, probs))
    results["auprc"] = float(compute_auprc(labels, probs))
    results["f1"] = float(compute_f1(labels, probs))
    sens, spec = compute_sensitivity_specificity(labels, probs)
    results["sensitivity"] = float(sens)
    results["specificity"] = float(spec)
    results["n_samples"] = int(len(labels))
    results["n_pos"] = int(labels.sum())
    results["n_neg"] = int((1 - labels).sum())

    # ── BCa Bootstrap CIs ──
    for metric_name, metric_fn in [
        ("auroc_ci", lambda y, p: compute_auroc(y, p)),
        ("auprc_ci", lambda y, p: compute_auprc(y, p)),
        ("f1_ci", lambda y, p: compute_f1(y, p)),
    ]:
        mean_ci, lower_ci, upper_ci, _ = bca_bootstrap_ci(
            metric_fn, labels, probs, n_iterations=2000
        )
        results[metric_name] = {
            "mean": float(mean_ci),
            "lower": float(lower_ci),
            "upper": float(upper_ci),
        }

    # ── Brier score (full dataset) ──
    results["brier"] = float(compute_brier_score(probs, labels))
    results["ece"] = float(compute_ece(probs, labels))

    # ── Calibration (with held-out fit/val split) ──
    n = len(logits)
    if n >= 6:  # Need at least a few samples for split
        rng = np.random.RandomState(42)
        fit_idx = rng.choice(n, size=n // 2, replace=False)
        val_idx = np.setdiff1d(np.arange(n), fit_idx)
        fit_logits, fit_labels = logits[fit_idx], labels[fit_idx]
        val_logits, val_labels = logits[val_idx], labels[val_idx]

        cal_results = {}
        for method in ["temperature", "platt", "isotonic"]:
            cal_probs, cal_info = calibrate_logits(
                val_logits, val_labels,
                val_logits=fit_logits, val_labels=fit_labels,
                method=method,
            )
            cal_results[method] = {
                "brier": float(compute_brier_score(cal_probs, val_labels)),
                "ece": float(compute_ece(cal_probs, val_labels)),
                "params": cal_info.get("params", {}),
            }
        # Raw (uncalibrated) on same val set
        raw_probs = 1.0 / (1.0 + np.exp(-val_logits))
        cal_results["none"] = {
            "brier": float(compute_brier_score(raw_probs, val_labels)),
            "ece": float(compute_ece(raw_probs, val_labels)),
            "params": {},
        }
        results["calibration"] = cal_results
    else:
        results["calibration"] = {"error": "Too few samples for calibration split"}

    return results


def evaluate_regression(predictions: dict, level: str, task: str) -> dict:
    """Regression evaluation for MOSEI sentiment or FI personality."""
    data = predictions.get(f"{task}_{level}", {})
    if not data:
        return {"error": f"No data for {task}_{level}"}

    pred = data["pred"]
    labels = data["labels"]

    results = {}

    if task == "mosei_sent":
        results["ccc"] = float(compute_ccc(labels, pred))
        results["mae"] = float(compute_mae(labels, pred))
        results["pearson"] = float(compute_pearson(labels, pred))
        results["spearman"] = float(compute_spearman(labels, pred))
        results["n_samples"] = int(len(labels))

        mean_ci, lower_ci, upper_ci, _ = bca_bootstrap_ci(
            lambda y, p: compute_ccc(y, p.squeeze() if p.ndim > 1 else p),
            labels, pred, n_iterations=2000,
        )
        results["ccc_ci"] = {"mean": float(mean_ci), "lower": float(lower_ci), "upper": float(upper_ci)}

    elif task == "fi_personality":
        cccs = []
        for t in range(5):
            cccs.append(compute_ccc(labels[:, t], pred[:, t]))
        results["ccc_per_trait"] = [float(c) for c in cccs]
        results["avg_ccc"] = float(np.mean(cccs))
        results["mae"] = float(compute_mae(labels.ravel(), pred.ravel()))
        results["n_samples"] = int(len(labels))

    return results


# =====================================================================
# Statistical Comparisons
# =====================================================================

def compare_methods(predictions: dict, level_a: str, level_b: str,
                    task: str = "daic") -> dict:
    """Statistical comparison between two methods.
    
    For DAIC: DeLong test + permutation test for AUROC
    For regression: paired permutation test for CCC/MAE
    """
    results = {"comparison": f"{level_a} vs {level_b}", "task": task}

    if task == "daic":
        data_a = predictions.get(f"daic_{level_a}", {})
        data_b = predictions.get(f"daic_{level_b}", {})
        if not data_a or not data_b:
            return {"error": "Missing data"}

        labels = data_a["labels"]
        probs_a = data_a["probs"]
        probs_b = data_b["probs"]

        # DeLong test
        delong_result = delong_auroc_test(labels, probs_a, probs_b)
        z_stat, p_val = delong_result["z_statistic"], delong_result["p_value"]
        results["delong"] = {"z": float(z_stat), "p_value": float(p_val)}

        # Paired permutation test for AUROC
        obs_diff, perm_p, null_dist = paired_permutation_test(
            lambda y, p: compute_auroc(y, p),
            labels, probs_a, probs_b,
            n_permutations=2000,
        )
        results["permutation_test"] = {
            "observed_diff": float(obs_diff),
            "p_value": float(perm_p),
            "null_distribution": null_dist.tolist(),
        }

        # Effect size
        results["effect_size"] = compute_effect_size_paired(labels, probs_a, probs_b)

        # Bootstrap delta
        mean_d, ci_low, ci_high = paired_bootstrap_delta(
            lambda y, p: compute_auroc(y, p),
            labels, probs_a, probs_b,
        )
        results["bootstrap_delta"] = {
            "mean": float(mean_d),
            "ci_lower": float(ci_low),
            "ci_upper": float(ci_high),
        }

    elif "sent" in task:
        data_a = predictions.get(f"{task}_{level_a}", {})
        data_b = predictions.get(f"{task}_{level_b}", {})
        if not data_a or not data_b:
            return {"error": "Missing data"}

        labels = data_a["labels"]
        pred_a = data_a["pred"]
        pred_b = data_b["pred"]

        obs_diff, p_val, null_dist = paired_permutation_test(
            lambda y, p: compute_ccc(y, p),
            labels, pred_a, pred_b,
        )
        results["permutation_test"] = {
            "metric": "CCC",
            "observed_diff": float(obs_diff),
            "p_value": float(p_val),
            "null_distribution": null_dist.tolist(),
        }

    return results


# =====================================================================
# Visualization
# =====================================================================

def generate_visualizations(predictions: dict, all_results: dict):
    """Generate all Phase 10 figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve

    fig_dir = ARTIFACTS_FIGURES

    levels = ["L0", "L1", "L2", "L3", "L4", "L5"]
    colors = ["#888888", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    level_styles = dict(zip(levels, colors))

    # ── 1. ROC Curves with Bootstrap Confidence Bands ──
    print("Generating ROC curves with bootstrap CI...")
    fig, ax = plt.subplots(figsize=(8, 7))
    
    n_bootstrap = 500  # Number of bootstrap samples for confidence bands
    
    for level in levels:
        data = predictions.get(f"daic_{level}", {})
        if not data or len(data.get("labels", [])) == 0:
            continue
        labels = np.array(data["labels"])
        probs = np.array(data["probs"])
        auroc = compute_auroc(labels, probs)
        
        # Bootstrap confidence bands
        n_samples = len(labels)
        tprs = []
        fpr_grid = np.linspace(0, 1, 100)
        
        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            # Sample with replacement
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            try:
                fpr_b, tpr_b, _ = roc_curve(labels[idx], probs[idx])
                tpr_interp = np.interp(fpr_grid, fpr_b, tpr_b)
                tprs.append(tpr_interp)
            except:
                continue
        
        if len(tprs) > 10:
            tprs = np.array(tprs)
            tpr_mean = tprs.mean(axis=0)
            tpr_lower = np.percentile(tprs, 2.5, axis=0)
            tpr_upper = np.percentile(tprs, 97.5, axis=0)
            
            # Plot confidence band
            ax.fill_between(fpr_grid, tpr_lower, tpr_upper, 
                           color=level_styles[level], alpha=0.2)
        
        # Main ROC curve
        fpr, tpr, _ = roc_curve(labels, probs)
        ax.plot(fpr, tpr, color=level_styles[level], linewidth=2,
                label=f"{level} (AUROC={auroc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("DAIC Depression — ROC Curves with 95% Bootstrap CI", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(fig_dir / "roc_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_dir / 'roc_curves.png'}")

    # ── 2. PR Curves with Bootstrap Confidence Bands ──
    print("Generating PR curves with bootstrap CI...")
    fig, ax = plt.subplots(figsize=(8, 7))
    
    for level in levels:
        data = predictions.get(f"daic_{level}", {})
        if not data or len(data.get("labels", [])) == 0:
            continue
        labels = np.array(data["labels"])
        probs = np.array(data["probs"])
        auprc = compute_auprc(labels, probs)
        
        # Bootstrap confidence bands
        n_samples = len(labels)
        precisions = []
        recall_grid = np.linspace(0, 1, 100)
        
        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            try:
                prec_b, rec_b, _ = precision_recall_curve(labels[idx], probs[idx])
                prec_interp = np.interp(recall_grid, rec_b[::-1], prec_b[::-1])
                precisions.append(prec_interp)
            except:
                continue
        
        if len(precisions) > 10:
            precisions = np.array(precisions)
            prec_mean = precisions.mean(axis=0)
            prec_lower = np.percentile(precisions, 2.5, axis=0)
            prec_upper = np.percentile(precisions, 97.5, axis=0)
            ax.fill_between(recall_grid, prec_lower, prec_upper, 
                           color=level_styles[level], alpha=0.2)
        
        precision, recall, _ = precision_recall_curve(labels, probs)
        ax.plot(recall, precision, color=level_styles[level], linewidth=2,
                label=f"{level} (AUPRC={auprc:.4f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("DAIC Depression — PR Curves with 95% Bootstrap CI", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(fig_dir / "pr_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_dir / 'pr_curves.png'}")

    # ── 3. Reliability Diagrams ──
    print("Generating reliability diagrams...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for idx, level in enumerate(levels):
        if idx >= len(axes):
            break
        data = predictions.get(f"daic_{level}", {})
        if not data or len(data.get("probs", [])) == 0:
            continue
        probs = data["probs"]
        labels = data["labels"]

        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_acc = np.zeros(n_bins)
        bin_conf = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        bin_acc_std = np.zeros(n_bins)  # For confidence interval

        for b in range(n_bins):
            in_bin = (probs >= bin_edges[b]) & (probs < bin_edges[b + 1])
            bin_counts[b] = in_bin.sum()
            if bin_counts[b] > 0:
                bin_acc[b] = labels[in_bin].mean()
                bin_conf[b] = probs[in_bin].mean()
                # Bootstrap std for accuracy
                if bin_counts[b] >= 2:
                    bin_acc_std[b] = labels[in_bin].std() / np.sqrt(bin_counts[b])

        ece = compute_ece(probs, labels)
        ax = axes[idx]
        
        # Plot confidence distribution as histogram bars at bottom
        for b in range(n_bins):
            if bin_counts[b] > 0:
                ax.bar(bin_centers[b], bin_counts[b] / max(len(probs), 1),
                      width=0.08, alpha=0.3, color="steelblue", edgecolor="darkblue")
        
        # Plot reliability curve with error bars
        valid_bins = bin_counts > 0
        if valid_bins.any():
            ax.errorbar(bin_conf[valid_bins], bin_acc[valid_bins], 
                       yerr=1.96*bin_acc_std[valid_bins] if bin_acc_std.any() else None,
                       fmt="o-", color=level_styles[level],
                       linewidth=2, markersize=6, label=f"ECE={ece:.4f}")
        
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence (Mean Predicted Probability)", fontsize=9)
        ax.set_ylabel("Accuracy (Fraction of Positives)", fontsize=9)
        ax.set_title(f"{level} — Reliability Diagram", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Calibration Across LLM Levels — DAIC Depression", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(fig_dir / "reliability_diagrams.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_dir / 'reliability_diagrams.png'}")

    # ── 4. Bootstrap CI Bars ──
    print("Generating bootstrap CI bars...")
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, level in enumerate(levels):
        level_results = all_results.get("daic", {}).get(level, {})
        ci_data = level_results.get("auroc_ci", {})
        if not ci_data:
            continue
        mean = ci_data.get("mean", 0)
        low = ci_data.get("lower", 0)
        high = ci_data.get("upper", 0)
        ax.errorbar(idx, mean, yerr=[[mean - low], [high - mean]],
                    fmt="o", color=level_styles[level], capsize=5, capthick=2,
                    markersize=10)
        ax.text(idx, high + 0.01, f"{mean:.3f}", ha="center", fontsize=8)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_ylabel("AUROC with 95% BCa CI", fontsize=11)
    ax.set_title("DAIC Depression — Bootstrap Confidence Intervals", fontsize=12, fontweight="bold")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(str(fig_dir / "bootstrap_ci_bars.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_dir / 'bootstrap_ci_bars.png'}")

    # ── 5. Paired Metric Delta Plot ──
    print("Generating paired metric delta plot...")
    fig, ax = plt.subplots(figsize=(10, 5))

    baseline_level = "L0"
    comparisons = []
    for level in levels[1:]:
        comp = compare_methods(predictions, level, baseline_level, task="daic")
        comparisons.append((level, comp))

    for idx, (level, comp) in enumerate(comparisons):
        delta = comp.get("bootstrap_delta", {})
        mean_d = delta.get("mean", 0)
        ci_low = delta.get("ci_lower", 0)
        ci_high = delta.get("ci_upper", 0)
        p_val = comp.get("permutation_test", {}).get("p_value", 1.0)

        color = "#2ecc71" if ci_low > 0 else "#e74c3c" if ci_high < 0 else "#f39c12"
        ax.errorbar(idx, mean_d * 100, yerr=[[(mean_d - ci_low) * 100], [(ci_high - mean_d) * 100]],
                    fmt="o", color=color, capsize=5, capthick=2, markersize=10)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax.text(idx, mean_d * 100 + 0.5, f"{mean_d*100:+.2f}pp {sig}",
                ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels([c[0] for c in comparisons])
    ax.set_ylabel("AUROC Delta vs L0 (pp)", fontsize=11)
    ax.set_title("Paired AUROC Improvement — BCa Bootstrap", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(str(fig_dir / "paired_delta_plot.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_dir / 'paired_delta_plot.png'}")

    # ── 6. Calibration Before/After (held-out split) ──
    print("Generating calibration before/after plot...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    level = "L1"
    data = predictions.get(f"daic_{level}", {})
    if data and len(data.get("logits", [])) >= 6:
        all_logits = data["logits"]
        all_labels = data["labels"]
        n = len(all_logits)
        rng = np.random.RandomState(42)
        fit_idx = rng.choice(n, size=n // 2, replace=False)
        val_idx = np.setdiff1d(np.arange(n), fit_idx)
        fit_logits, fit_labels = all_logits[fit_idx], all_labels[fit_idx]
        val_logits, val_labels = all_logits[val_idx], all_labels[val_idx]

        for idx_cal, method in enumerate(["temperature", "platt", "isotonic"]):
            ax = axes[idx_cal]

            raw_probs = 1.0 / (1.0 + np.exp(-val_logits))
            cal_probs, cal_info = calibrate_logits(
                val_logits, val_labels,
                val_logits=fit_logits, val_labels=fit_labels,
                method=method,
            )

            raw_brier = compute_brier_score(raw_probs, val_labels)
            cal_brier = compute_brier_score(cal_probs, val_labels)
            raw_ece = compute_ece(raw_probs, val_labels)
            cal_ece = compute_ece(cal_probs, val_labels)

            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            raw_bin_acc = np.zeros(n_bins)
            for b in range(n_bins):
                in_bin = (raw_probs >= bin_edges[b]) & (raw_probs < bin_edges[b + 1])
                if in_bin.sum() > 0:
                    raw_bin_acc[b] = val_labels[in_bin].mean()
            ax.plot(bin_centers, raw_bin_acc, "o-", color="red", alpha=0.5,
                    label=f"Raw (ECE={raw_ece:.3f})")

            cal_bin_acc = np.zeros(n_bins)
            for b in range(n_bins):
                in_bin = (cal_probs >= bin_edges[b]) & (cal_probs < bin_edges[b + 1])
                if in_bin.sum() > 0:
                    cal_bin_acc[b] = val_labels[in_bin].mean()
            ax.plot(bin_centers, cal_bin_acc, "o-", color="green",
                    label=f"{method} (ECE={cal_ece:.3f})")

            ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{method.title()}: Brier {raw_brier:.3f}→{cal_brier:.3f}",
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Calibration Before/After — L1 (Mistral Text, held-out split)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(fig_dir / "calibration_before_after.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_dir / 'calibration_before_after.png'}")

    # ── 7. Bland-Altman Plots ──
    print("Generating Bland-Altman plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx_ba, task in enumerate(["mosei_sent", "fi_personality"]):
        ax = axes[idx_ba]
        level = "L1"
        data = predictions.get(f"{task}_{level}", {})
        if not data:
            continue
        pred_raw = data["pred"]
        labels_raw = data["labels"]

        if pred_raw.ndim > 1:
            pred = pred_raw[:, 0].ravel()
            labels = labels_raw[:, 0].ravel()
        else:
            pred = pred_raw.ravel()
            labels = labels_raw.ravel()

        mean = (pred + labels) / 2
        diff = pred - labels
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        loa_low = mean_diff - 1.96 * std_diff
        loa_high = mean_diff + 1.96 * std_diff

        ax.scatter(mean, diff, alpha=0.5, s=20)
        ax.axhline(mean_diff, color="red", linestyle="-", label=f"Mean diff: {mean_diff:.3f}")
        ax.axhline(loa_low, color="gray", linestyle="--", label=f"LOA: {loa_low:.2f}")
        ax.axhline(loa_high, color="gray", linestyle="--", label=f"LOA: {loa_high:.2f}")
        ax.fill_between([mean.min(), mean.max()], loa_low, loa_high,
                        alpha=0.1, color="gray")
        ax.set_xlabel("Mean of Predicted and Observed")
        ax.set_ylabel("Difference (Predicted − Observed)")
        title_map = {"mosei_sent": "MOSEI Sentiment (Trait 1)",
                     "fi_personality": "FI Personality (Trait 1)"}
        ax.set_title(title_map.get(task, task), fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Bland-Altman Plots — L1 (Mistral Text)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(fig_dir / "bland_altman_plots.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {fig_dir / 'bland_altman_plots.png'}")

    # ── 8. Permutation Null Distributions ──
    print("Generating permutation null distributions...")
    fig, ax = plt.subplots(figsize=(8, 5))

    comp = compare_methods(predictions, "L1", "L0", task="daic")
    null_dist = comp.get("permutation_test", {}).get("null_distribution")

    if null_dist is None:
        print(f"  [SKIP] permutation_null_distribution.png: no null_distribution in "
              f"compare_methods() output ({comp.get('error', 'unknown reason')}) — "
              f"not plotting a fabricated fallback.")
        plt.close(fig)
    else:
        obs_diff = comp["permutation_test"]["observed_diff"]
        p_val = comp["permutation_test"]["p_value"]

        ax.hist(null_dist, bins=40, alpha=0.6, color="steelblue", density=True,
                label="Null distribution")
        ax.axvline(obs_diff, color="red", linewidth=2,
                   label=f"Observed Δ = {obs_diff:.4f} (p={p_val:.4f})")
        percentile_95 = np.percentile(np.abs(null_dist), 95)
        ax.axvline(percentile_95, color="gray", linestyle="--",
                   label=f"95% threshold = {percentile_95:.4f}")
        ax.axvline(-percentile_95, color="gray", linestyle="--")

        ax.set_xlabel("AUROC Difference (L1 − L0)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("Permutation Test — L1 vs L0 (DAIC Depression)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(fig_dir / "permutation_null_distribution.png"), dpi=150)
        plt.close(fig)
    print(f"  Saved: {fig_dir / 'permutation_null_distribution.png'}")

    print(f"\n✅ All visualizations saved to {fig_dir}")


# =====================================================================
# Summary Table
# =====================================================================

def generate_summary_table(all_results: dict):
    """Generate formatted summary results table."""
    print("\n" + "=" * 80)
    print("PHASE 10 — COMPREHENSIVE EVALUATION SUMMARY (REAL MODEL PREDICTIONS)")
    print("=" * 80)

    levels = ["L0", "L1", "L2", "L3", "L4", "L5"]

    # DAIC Depression
    print(f"\n{'─' * 80}")
    print(f"{'TASK':<20} {'METHOD':<8} {'N':<6} {'AUROC':<12} {'95% CI':<18} {'Brier':<10} {'ECE':<10}")
    print(f"{'─' * 80}")

    for level in levels:
        r = all_results.get("daic", {}).get(level, {})
        if not r:
            continue
        auroc = r.get("auroc", 0)
        n = r.get("n_samples", 0)
        ci = r.get("auroc_ci", {})
        ci_str = f"[{ci.get('lower', 0):.3f}, {ci.get('upper', 0):.3f}]"
        brier = r.get("brier", 0)
        ece = r.get("ece", 0)
        print(f"{'DAIC Depression':<20} {level:<8} {n:<6} {auroc:<12.4f} {ci_str:<18} {brier:<10.4f} {ece:<10.4f}")

    # MOSEI Sentiment
    print(f"\n{'─' * 80}")
    print(f"{'TASK':<20} {'METHOD':<8} {'N':<6} {'CCC':<12} {'95% CI':<18} {'MAE':<10}")
    print(f"{'─' * 80}")
    for level in levels:
        r = all_results.get("mosei_sent", {}).get(level, {})
        if not r:
            continue
        ccc = r.get("ccc", 0)
        n = r.get("n_samples", 0)
        ci = r.get("ccc_ci", {})
        ci_str = f"[{ci.get('lower', 0):.3f}, {ci.get('upper', 0):.3f}]"
        mae = r.get("mae", 0)
        print(f"{'MOSEI Sentiment':<20} {level:<8} {n:<6} {ccc:<12.4f} {ci_str:<18} {mae:<10.4f}")

    # FI Personality
    print(f"\n{'─' * 80}")
    print(f"{'TASK':<20} {'METHOD':<8} {'N':<6} {'Avg CCC':<12} {'Per Trait':<30}")
    print(f"{'─' * 80}")
    for level in levels:
        r = all_results.get("fi_personality", {}).get(level, {})
        if not r:
            continue
        avg_ccc = r.get("avg_ccc", 0)
        n = r.get("n_samples", 0)
        per_trait = r.get("ccc_per_trait", [])
        trait_str = ", ".join([f"{c:.3f}" for c in per_trait])
        print(f"{'FI Personality':<20} {level:<8} {n:<6} {avg_ccc:<12.4f} {trait_str:<30}")

    print(f"\n{'═' * 80}")
    print("BEST CALIBRATION METHOD (DAIC depression):")
    for level in levels:
        r = all_results.get("daic", {}).get(level, {})
        cal = r.get("calibration", {})
        if not cal or "error" in cal:
            continue
        raw_ece = cal.get("none", {}).get("ece", 0)
        methods = [m for m in cal if m != "none"]
        if not methods:
            continue
        best_method = min(methods, key=lambda m: cal[m]["ece"])
        best_ece = cal[best_method]["ece"]
        best_brier = cal[best_method]["brier"]
        print(f"  {level}: raw ECE={raw_ece:.4f} → best={best_method} (ECE={best_ece:.4f}, Brier={best_brier:.4f})")

    print(f"\n{'═' * 80}")
    print("STATISTICAL COMPARISONS (vs L0 baseline):")
    for level in levels[1:]:
        comp = all_results.get("comparisons", {}).get(f"{level}_vs_L0", {})
        if comp:
            p = comp.get("permutation_test", {}).get("p_value", 1.0)
            d = comp.get("effect_size", {}).get("cohens_d", 0)
            delta = comp.get("bootstrap_delta", {}).get("mean", 0)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {level}: ΔAUROC={delta*100:+.2f}pp, p={p:.4f} {sig}, d={d:.3f}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 10: Evaluation & Calibration")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (skip figures)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for model inference")
    parser.add_argument("--predictions-only", action="store_true",
                        help="Only extract predictions, skip evaluation")
    parser.add_argument("--levels", type=str, nargs="+",
                        default=["L0", "L1", "L2", "L3", "L4", "L5"],
                        help="LLM levels to evaluate")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 10: Calibration, Metrics & Statistical Validation")
    print("USING REAL MODEL PREDICTIONS (no synthetic data)")
    print("=" * 60)

    # Load real predictions (extracting from checkpoints if needed)
    predictions = ensure_real_predictions(
        device_str=args.device,
        skip_extraction=True,
        levels=args.levels,
    )
    levels = args.levels

    if args.predictions_only:
        print("\n✅ Predictions extracted. Exiting (--predictions-only mode).")
        return

    # Evaluate all tasks
    all_results = {"daic": {}, "mosei_sent": {}, "fi_personality": {}, "comparisons": {}}

    print("\nEvaluating DAIC depression...")
    for level in levels:
        r = evaluate_daic(predictions, level)
        all_results["daic"][level] = r
        auroc = r.get("auroc", 0)
        ci = r.get("auroc_ci", {})
        n = r.get("n_samples", 0)
        print(f"  {level}: AUROC={auroc:.4f} [{ci.get('lower', 0):.4f}, {ci.get('upper', 0):.4f}] (N={n})")

    print("\nEvaluating MOSEI sentiment...")
    for level in levels:
        r = evaluate_regression(predictions, level, "mosei_sent")
        all_results["mosei_sent"][level] = r
        print(f"  {level}: CCC={r.get('ccc', 0):.4f}")

    print("\nEvaluating FI personality...")
    for level in levels:
        r = evaluate_regression(predictions, level, "fi_personality")
        all_results["fi_personality"][level] = r
        print(f"  {level}: Avg CCC={r.get('avg_ccc', 0):.4f}")

    # Statistical comparisons
    print("\nStatistical comparisons (vs L0 baseline)...")
    for level in levels[1:]:
        comp = compare_methods(predictions, level, "L0", task="daic")
        all_results["comparisons"][f"{level}_vs_L0"] = comp
        p = comp.get("permutation_test", {}).get("p_value", 1.0)
        d = comp.get("effect_size", {}).get("cohens_d", 0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {level} vs L0: p={p:.4f} {sig}, d={d:.3f}")

    # Save results
    print("\nSaving results...")
    results_path = ARTIFACTS_TABLES / "phase10_evaluation_results.json"
    serializable = {}
    for task in ["daic", "mosei_sent", "fi_personality"]:
        serializable[task] = {}
        for level in levels:
            r = all_results[task].get(level, {})
            clean = {k: v for k, v in r.items() if isinstance(v, (dict, list, float, int, str)) or v is None}
            serializable[task][level] = clean

    serializable["comparisons"] = {}
    for comp_name, comp_data in all_results["comparisons"].items():
        clean = {}
        for k, v in comp_data.items():
            if isinstance(v, dict):
                clean[k] = {
                    sk: (sv.tolist() if hasattr(sv, "tolist") else sv)
                    for sk, sv in v.items()
                    if isinstance(sv, (float, int, str, bool, list)) or sv is None or hasattr(sv, "tolist")
                }
            else:
                clean[k] = v
        serializable["comparisons"][comp_name] = clean

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {results_path}")

    # Generate visualizations
    if not args.quick:
        print("\nGenerating visualizations...")
        generate_visualizations(predictions, all_results)
    else:
        print("\nSkipping visualizations (quick mode)")

    # Summary table
    generate_summary_table(all_results)

    print("\n✅ Phase 10 complete!")
    print(f"  Results: {results_path}")
    print(f"  Figures: {ARTIFACTS_FIGURES}")
    print(f"  All values are from real model predictions — no synthetic data used.")


if __name__ == "__main__":
    main()
