"""Visualization utilities: loss curves, metric plots, UMAP projections, reliability diagrams.

Every phase outputs ≥1 figure to artifacts/figures/phase_XX_name/.
All figures must be publication-ready (300 DPI, legible fonts).
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional
import os


def setup_style():
    """Set publication-quality matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
    })


def plot_label_distribution(labels: dict[str, np.ndarray], dataset_name: str, output_path: str):
    """Plot label distribution (histogram for regression, bar for classification)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in labels.items():
        ax.hist(values, alpha=0.6, label=name, bins=30)
    ax.set_title(f"{dataset_name} — Label Distribution")
    ax.set_xlabel("Label value")
    ax.set_ylabel("Count")
    ax.legend()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_missing_modality_heatmap(modality_counts: dict[str, dict], output_path: str):
    """Plot missing modality heatmap per dataset."""
    setup_style()
    # modality_counts: {dataset: {modality: count}}
    fig, ax = plt.subplots(figsize=(8, 4))
    # Convert to matrix form for heatmap
    datasets = list(modality_counts.keys())
    modalities = list(modality_counts[datasets[0]].keys())
    matrix = np.array([[modality_counts[d][m] for m in modalities] for d in datasets])
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=modalities, yticklabels=datasets, ax=ax)
    ax.set_title("Modality Availability Heatmap")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, output_path: str = ""):
    """Reliability diagram (calibration curve) for binary classification."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    accuracies = []
    confidences = []
    for i in range(n_bins):
        in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if in_bin.sum() > 0:
            accuracies.append(y_true[in_bin].mean())
            confidences.append(y_prob[in_bin].mean())
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(confidences, accuracies, "o-", label="Model")
    ax.set_xlabel("Average predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram")
    ax.legend()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_metric_comparison(results: dict[str, dict], metric: str, output_path: str):
    """Bar plot comparing a single metric across methods/datasets."""
    setup_style()
    names = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 4))
    ax.bar(names, values, color="steelblue")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} Comparison")
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()