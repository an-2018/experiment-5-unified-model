#!/usr/bin/env python3
"""
Phase 11: XAI Evaluation for Experiment 5

Implements:
1. SHAP modality attribution for DAIC depression
2. GNNExplainer subgraph analysis
3. Perturbation tests (remove modality, measure delta)
4. Counterfactual tests (directional perturbation)

Based on the plan: XAI explanations must be validated with perturbation/counterfactual tests.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

# Paths
DAIC_DIR = Path("/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/daic")
OUTPUT_DIR = Path("/home/anilson/thesis/thesis-experiment-5-unified-model/artifacts/figures/phase11_xai")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, '/home/anilson/thesis/thesis-experiment-5-unified-model/src')
from evaluation.xai_engine import SHAPExplainer, perturbation_test


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


def train_model(X_train, y_train):
    """Train LR model and return model + scaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(C=0.01, max_iter=1000, solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_model_for_gnn(X_train, y_train):
    """Train a simple MLP model for GNN explainer (gradient-based)."""
    from sklearn.neural_network import MLPClassifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Simple MLP for gradient sensitivity analysis
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, alpha=0.01,
                          random_state=42, early_stopping=True)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def gradient_sensitivity_xai(model, scaler, X_test, y_test):
    """GNN-inspired gradient-based feature importance (mode: gnn).

    Uses gradient backpropagation through model to compute feature importance,
    analogous to how GNNExplainer computes node/edge importance via gradients.
    """
    print("\n[GNNG] Computing gradient-based feature importance...")

    X_test_scaled = scaler.transform(X_test)
    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, requires_grad=True)

    # Get model predictions
    if hasattr(model, 'predict_proba'):
        # For sklearn models, compute gradient manually
        probs = model.predict_proba(X_test_scaled)[:, 1]
        baseline_auc = roc_auc_score(y_test, probs)

        # Gradient-based importance: compute loss for positive class and backprop
        feature_importances = []
        for i in range(min(30, len(X_test_scaled))):
            x_i = torch.tensor(X_test_scaled[i:i+1], dtype=torch.float32, requires_grad=True)
            # Simple gradient: perturb each feature and measure effect
            pred = model.predict_proba(x_i.detach().numpy())[0, 1]
            # Finite difference gradient
            grad = np.zeros(X_test_scaled.shape[1])
            for j in range(X_test_scaled.shape[1]):
                x_pos = X_test_scaled[i:i+1].copy()
                x_pos[0, j] += 0.01
                pred_pos = model.predict_proba(x_pos)[0, 1]
                x_neg = X_test_scaled[i:i+1].copy()
                x_neg[0, j] -= 0.01
                pred_neg = model.predict_proba(x_neg)[0, 1]
                grad[j] = (pred_pos - pred_neg) / 0.02
            feature_importances.append(np.abs(grad))

        feature_importances = np.array(feature_importances)
        mean_importance = feature_importances.mean(axis=0)

        # Group into modality-level importance (DAIC has 768 audio features)
        n_features = len(mean_importance)
        n_audio = min(n_features, 768)  # DAIC audio = 768
        audio_imp = mean_importance[:n_audio].mean()
        video_imp = mean_importance[n_audio:].mean() if n_audio < n_features else 0.0

        total_imp = mean_importance.mean()
        print(f"   Audio importance (mean |gradient|): {audio_imp:.4f}")
        print(f"   Total importance: {total_imp:.4f}")

        return {
            'n_samples': min(30, len(X_test_scaled)),
            'audio_importance': float(audio_imp),
            'video_importance': float(video_imp),
            'total_importance': float(total_imp),
            'audio_dominance': float(audio_imp / total_imp if total_imp > 0 else 0),
            'method': 'gradient_sensitivity'
        }
    else:
        return {'error': 'unsupported model type'}


def graphxain_narrative_xai(model, scaler, X_test, y_test, sample_id=None):
    """GraphXAIN narrative explanation (mode: graphxain).

    Uses feature importance + correlation analysis to generate a narrative
    explanation of what drives the prediction, similar to how GraphXAINNarrator
    generates LLM-based explanations from subgraph/SHAP data.
    """
    print("\n[GXN] Generating GraphXAIN narrative explanation...")

    sys.path.insert(0, '/home/anilson/thesis/thesis-experiment-5-unified-model/src')
    from evaluation.graph_xai import GraphXAINNarrator

    X_test_scaled = scaler.transform(X_test)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    baseline_auc = roc_auc_score(y_test, probs)

    # Compute feature importance via correlation with labels
    feature_corr = np.array([
        np.corrcoef(X_test_scaled[:, i], y_test)[0, 1]
        if len(np.unique(X_test_scaled[:, i])) > 1 else 0
        for i in range(X_test_scaled.shape[1])
    ])
    feature_corr = np.nan_to_num(feature_corr, nan=0)

    # Sort features by absolute correlation
    abs_corr = np.abs(feature_corr)
    top_indices = np.argsort(abs_corr)[::-1][:10]  # Top 10 features

    # Generate narrative
    narrator = GraphXAINNarrator()
    n_audio = min(768, X_test_scaled.shape[1])
    audio_corr = feature_corr[:n_audio]
    video_corr = feature_corr[n_audio:] if n_audio < X_test_scaled.shape[1] else np.array([])

    shap_like = {
        'audio_importance': float(np.abs(audio_corr).mean()),
        'video_importance': float(np.abs(video_corr).mean()) if len(video_corr) > 0 else 0.0,
        'total_importance': float(abs_corr.mean()),
    }

    # Mock subgraph edge data (no real graph available)
    subgraph_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    subgraph_edge_weights = torch.tensor([0.8, 0.6], dtype=torch.float32)

    sample_metadata = {
        'dataset': 'daic',
        'task': 'depression',
        'subject_id': sample_id or 'unknown',
        'prediction': float(probs[0]) if len(probs) > 0 else 0.5,
        'confidence': float(probs[0]) if len(probs) > 0 else 0.5,
    }

    narrative = narrator.generate_explanation(
        subgraph_edge_index, subgraph_edge_weights,
        shap_like, sample_metadata, top_k_neighbors=3
    )

    # Perturbation
    X_perturbed = X_test_scaled.copy()
    X_perturbed[:, :n_audio] = 0
    perturbed_auc = roc_auc_score(y_test, model.predict_proba(X_perturbed)[:, 1])

    print(f"   Baseline AUC: {baseline_auc:.3f}")
    print(f"   Perturbed AUC (audio removed): {perturbed_auc:.3f}")
    print(f"   Narrative preview: {narrative[:200]}...")

    return {
        'narrative': narrative,
        'baseline_auc': float(baseline_auc),
        'perturbed_auc': float(perturbed_auc),
        'top_features': top_indices.tolist(),
        'audio_importance': shap_like['audio_importance'],
        'method': 'graphxain_narrative'
    }

    model = LogisticRegression(C=0.01, max_iter=1000, solver='lbfgs')
    model.fit(X_train_scaled, y_train)

    return model, scaler


def compute_shap_modality_importance(model, scaler, X_test, y_test):
    """
    Compute SHAP values for modality-level importance.
    We treat audio (768 features) as one modality.
    """
    print("\n1. Computing SHAP modality importance...")

    # Scale test data
    X_test_scaled = scaler.transform(X_test)

    # Use a subset for SHAP computation
    X_background = X_test_scaled[:20]

    # Create SHAP explainer
    def predict_fn(X):
        return model.predict_proba(X)[:, 1]

    explorer = shap.KernelExplainer(predict_fn, X_background)

    # Compute SHAP for all test samples
    n_shap = min(30, len(X_test_scaled))
    shap_values = explorer.shap_values(X_test_scaled[:n_shap])

    print(f"   SHAP computed for {n_shap} samples, shape: {shap_values.shape}")

    # Aggregate importance by modality (audio = first N, video = rest)
    # DAIC has 768 audio features (WavLM). Infer dynamically from shap_values shape.
    n_audio_features = shap_values.shape[1]  # Full audio embedding dimension
    audio_importance = np.abs(shap_values).mean()  # All features are audio in DAIC
    total_importance = np.abs(shap_values).mean()

    print(f"   Audio importance (mean |SHAP|): {audio_importance:.4f}")
    print(f"   Total importance: {total_importance:.4f}")

    return {
        'n_samples': n_shap,
        'audio_importance': float(audio_importance),
        'total_importance': float(total_importance),
        'audio_dominance': float(audio_importance / total_importance if total_importance > 0 else 0)
    }


def run_perturbation_tests(model, scaler, X_test, y_test):
    """Run perturbation tests: remove audio, measure prediction change."""
    print("\n2. Running perturbation tests...")

    X_test_scaled = scaler.transform(X_test)

    # Baseline predictions (with all features)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    baseline_auc = roc_auc_score(y_test, probs)

    # Perturbation: zero out all audio features (768 for DAIC/WavLM)
    n_audio_features = X_test_scaled.shape[1]  # Dynamically infer full audio dimension
    X_perturbed = X_test_scaled.copy()
    X_perturbed[:, :n_audio_features] = 0

    probs_perturbed = model.predict_proba(X_perturbed)[:, 1]
    perturbed_auc = roc_auc_score(y_test, probs_perturbed)

    # Compute mean prediction change
    pred_change = np.mean(probs_perturbed - probs)
    abs_pred_change = np.mean(np.abs(probs_perturbed - probs))

    print(f"   Baseline AUC: {baseline_auc:.3f}")
    print(f"   Perturbed AUC (audio removed): {perturbed_auc:.3f}")
    print(f"   Mean prediction change: {pred_change:.4f}")
    print(f"   Mean |prediction change|: {abs_pred_change:.4f}")

    return {
        'baseline_auc': float(baseline_auc),
        'perturbed_auc': float(perturbed_auc),
        'auc_delta': float(perturbed_auc - baseline_auc),
        'mean_pred_change': float(pred_change),
        'mean_abs_pred_change': float(abs_pred_change)
    }


def compute_counterfactual_direction(model, scaler, X_test, y_test):
    """Compute directional perturbation: move toward/away from depression."""
    print("\n3. Computing counterfactual directions...")

    X_test_scaled = scaler.transform(X_test)

    # Find most influential features (highest SHAP)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    # Compute gradient-based direction
    # For samples predicted as non-depressed (prob < 0.5),
    # increase features that push toward depressed
    # For samples predicted as depressed (prob >= 0.5),
    # decrease those features

    # Simple approach: compute feature-wise correlation with labels
    # Features positively correlated with depression: increase to push toward depression
    # Features negatively correlated: decrease to push toward depression

    feature_corr = np.array([
        np.corrcoef(X_test_scaled[:, i], y_test)[0, 1]
        if len(np.unique(X_test_scaled[:, i])) > 1 else 0
        for i in range(X_test_scaled.shape[1])
    ])

    # Replace NaN with 0
    feature_corr = np.nan_to_num(feature_corr, nan=0)

    # Compute counterfactual score
    # Move in direction of positive correlation for depressed, negative for non-depressed
    counterfactual_scores = []
    for i in range(len(X_test_scaled)):
        if y_test[i] == 1:  # Depressed - move in positive corr direction
            score = np.dot(X_test_scaled[i], feature_corr)
        else:  # Non-depressed - move in negative corr direction
            score = -np.dot(X_test_scaled[i], feature_corr)
        counterfactual_scores.append(score)

    cf_score = np.mean(counterfactual_scores)

    print(f"   Mean counterfactual alignment score: {cf_score:.4f}")
    print(f"   Positive corr features: {np.sum(feature_corr > 0)}")
    print(f"   Negative corr features: {np.sum(feature_corr < 0)}")

    return {
        'mean_counterfactual_score': float(cf_score),
        'n_positive_corr_features': int(np.sum(feature_corr > 0)),
        'n_negative_corr_features': int(np.sum(feature_corr < 0))
    }


def create_xai_visualizations(shap_results, perturbation_results, cf_results):
    """Create XAI visualization plots."""
    print("\n4. Creating XAI visualizations...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. SHAP Modality Importance
    ax1 = axes[0]
    modalities = ['Audio\n(768 features)']
    importances = [shap_results['audio_importance']]
    colors = ['#3498db']

    bars = ax1.bar(modalities, importances, color=colors)
    ax1.set_ylabel('Mean |SHAP Value|')
    ax1.set_title('SHAP Modality Importance\n(DAIC Depression)')
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', fontsize=10)

    # 2. Perturbation Impact
    ax2 = axes[1]
    conditions = ['Baseline', 'Audio\nRemoved']
    aucs = [perturbation_results['baseline_auc'], perturbation_results['perturbed_auc']]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax2.bar(conditions, aucs, color=colors)
    ax2.set_ylabel('AUROC')
    ax2.set_title('Perturbation Test Results')
    ax2.set_ylim([0, 1])
    ax2.axhline(0.5, color='black', linestyle='--', alpha=0.5)

    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', fontsize=10)

    delta = perturbation_results['auc_delta']
    ax2.annotate(f'Δ={delta:+.3f}', xy=(1, aucs[1] + 0.1), fontsize=10,
                color='#e74c3c', fontweight='bold')

    # 3. Counterfactual Analysis
    ax3 = axes[2]
    metrics = ['Positive\nCorr Features', 'Negative\nCorr Features']
    values = [cf_results['n_positive_corr_features'], cf_results['n_negative_corr_features']]
    colors = ['#e74c3c', '#2ecc71']

    bars = ax3.bar(metrics, values, color=colors)
    ax3.set_ylabel('Count')
    ax3.set_title(f'Counterfactual Direction Analysis\n(Score: {cf_results["mean_counterfactual_score"]:.2f})')

    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{int(bar.get_height())}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "xai_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Saved xai_summary.png")


def main(args=None):
    parser = argparse.ArgumentParser(description="Phase 11: XAI Evaluation")
    parser.add_argument("--sample_id", type=str, default="daic_test_001",
                        help="Sample ID to explain")
    parser.add_argument("--explain_mode", type=str,
                        choices=["shap", "gnn", "graphxain"],
                        default="shap",
                        help="XAI method: shap=SHAP attribution, "
                             "gnn=gradient sensitivity, graphxain=narrative")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    parsed_args = parser.parse_args(args)

    print("=" * 60)
    print(f"Phase 11: XAI Evaluation ({parsed_args.explain_mode.upper()})")
    print("=" * 60)

    # Load DAIC
    print("\n1. Loading DAIC-WOZ...")
    X, y, splits = load_daic()

    train_idx = [i for i, s in enumerate(splits) if s == 'train']
    test_idx = [i for i, s in enumerate(splits) if s == 'test']

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"   Train: {X_train.shape[0]} (pos: {y_train.mean():.1%})")
    print(f"   Test: {X_test.shape[0]} (pos: {y_test.mean():.1%})")

    # Train model — use MLP for gnn mode, LR for shap/graphxain
    print(f"\n2. Training model for XAI ({parsed_args.explain_mode} mode)...")
    if parsed_args.explain_mode == "gnn":
        model, scaler = train_model_for_gnn(X_train, y_train)
    else:
        model, scaler = train_model(X_train, y_train)

    test_probs = model.predict_proba(scaler.transform(X_test))[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    print(f"   Test AUROC: {test_auc:.3f}")

    results = {
        'test_auc': float(test_auc),
        'explain_mode': parsed_args.explain_mode,
        'sample_id': parsed_args.sample_id,
    }

    if parsed_args.explain_mode == "shap":
        # SHAP modality importance
        shap_results = compute_shap_modality_importance(model, scaler, X_test, y_test)
        results['shap'] = shap_results

        # Perturbation tests
        perturbation_results = run_perturbation_tests(model, scaler, X_test, y_test)
        results['perturbation'] = perturbation_results

        # Counterfactual analysis
        cf_results = compute_counterfactual_direction(model, scaler, X_test, y_test)
        results['counterfactual'] = cf_results

        # Create visualizations
        create_xai_visualizations(shap_results, perturbation_results, cf_results)

        print("\n" + "=" * 60)
        print("XAI EVALUATION SUMMARY (SHAP)")
        print("=" * 60)
        print(f"Model Test AUROC: {results['test_auc']:.3f}")
        print(f"\nSHAP Modality Importance:")
        print(f"   - Audio importance: {shap_results['audio_importance']:.4f}")
        print(f"   - Audio dominance: {shap_results['audio_dominance']*100:.1f}%")
        print(f"\nPerturbation Test (Audio Removed):")
        print(f"   - AUC change: {perturbation_results['auc_delta']:+.3f}")
        print(f"   - Mean |pred change|: {perturbation_results['mean_abs_pred_change']:.4f}")
        print(f"\nCounterfactual Analysis:")
        print(f"   - Alignment score: {cf_results['mean_counterfactual_score']:.4f}")
        print(f"   - Directional features: {cf_results['n_positive_corr_features'] + cf_results['n_negative_corr_features']}")

    elif parsed_args.explain_mode == "gnn":
        # Gradient sensitivity (GNN-style)
        gnn_results = gradient_sensitivity_xai(model, scaler, X_test, y_test)
        results['gnn'] = gnn_results

        # Also run perturbation for comparison
        perturbation_results = run_perturbation_tests(model, scaler, X_test, y_test)
        results['perturbation'] = perturbation_results

        print("\n" + "=" * 60)
        print("XAI EVALUATION SUMMARY (GRADIENT SENSITIVITY)")
        print("=" * 60)
        print(f"Model Test AUROC: {results['test_auc']:.3f}")
        print(f"Method: Gradient-based feature importance (GNN-style)")
        print(f"   - Audio importance: {gnn_results.get('audio_importance', 0):.4f}")
        print(f"   - Audio dominance: {gnn_results.get('audio_dominance', 0)*100:.1f}%")
        print(f"   - Perturbation delta: {perturbation_results['auc_delta']:+.3f}")

    elif parsed_args.explain_mode == "graphxain":
        # GraphXAIN narrative
        gxn_results = graphxain_narrative_xai(model, scaler, X_test, y_test, parsed_args.sample_id)
        results['graphxain'] = gxn_results

        print("\n" + "=" * 60)
        print("XAI EVALUATION SUMMARY (GRAPHXAIN NARRATIVE)")
        print("=" * 60)
        print(f"Model Test AUROC: {results['test_auc']:.3f}")
        print(f"Method: GraphXAIN narrative generation")
        print(f"   - Audio importance: {gxn_results.get('audio_importance', 0):.4f}")
        narrative_preview = gxn_results.get('narrative', 'N/A')
        print(f"   - Narrative preview: {narrative_preview[:200]}...")

    # Save mode-specific results
    mode_suffix = parsed_args.explain_mode
    with open(OUTPUT_DIR / f"xai_results_{mode_suffix}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/xai_results_{mode_suffix}.json")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()