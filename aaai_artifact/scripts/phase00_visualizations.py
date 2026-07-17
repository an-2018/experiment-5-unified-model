"""
Phase 0 Visualization Script — Reproducibility Dashboard & Pipeline Diagram.

Outputs to: artifacts/figures/phase_00_setup/
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.utils import get_env_info, get_git_hash

OUTPUT_DIR = "artifacts/figures/phase_00_setup"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use the evaluation visualizations style
from src.evaluation.visualizations import setup_style
setup_style()


def plot_reproducibility_dashboard(output_path: str):
    """Fig 1: Reproducibility dashboard showing env versions, GPU, dataset paths, seeds."""
    env_info = get_env_info()
    git_hash = get_git_hash()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Phase 0 — Reproducibility Dashboard", fontsize=14, fontweight="bold")

    # Panel 1: Environment versions
    ax1 = axes[0, 0]
    env_items = [
        f"Python: {env_info['python'].split()[0]}",
        f"Torch: {env_info['torch']}",
        f"PyTorch Lightning: {env_info['pytorch_lightning']}",
        f"CUDA Available: {env_info['cuda_available']}",
        f"Git Hash: {git_hash}",
    ]
    ax1.text(0.05, 0.95, "\n".join(env_items), transform=ax1.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))
    ax1.axis("off")
    ax1.set_title("Environment & Versions")

    # Panel 2: Dataset paths
    ax2 = axes[0, 1]
    dataset_paths = [
        "DAIC-WOZ",
        "  data/daic/raw",
        "CMU-MOSEI",
        "  data/mosei/CMU-MOSEI",
        "ChaLearn FI",
        "  data/fi/raw",
    ]
    ax2.text(0.05, 0.95, "\n".join(dataset_paths), transform=ax2.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3))
    ax2.axis("off")
    ax2.set_title("Dataset Paths")

    # Panel 3: Seed configuration
    ax3 = axes[1, 0]
    seed_items = [
        "Global seed: 42",
        "torch.manual_seed: 42",
        "torch.cuda.manual_seed_all: 42",
        "torch.backends.cudnn.deterministic: True",
        "torch.backends.cudnn.benchmark: False",
        "random.seed: 42",
        "np.random.seed: 42",
    ]
    ax3.text(0.05, 0.95, "\n".join(seed_items), transform=ax3.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3))
    ax3.axis("off")
    ax3.set_title("Seed Configuration")

    # Panel 4: Module structure
    ax4 = axes[1, 1]
    structure_items = [
        "src/",
        "  data/         — loaders, dataset, preprocessing",
        "  models/       — encoders, fusion, MoE, GNN, heads",
        "  training/     — trainer, losses, sampler, calibration",
        "  evaluation/   — metrics, stats, XAI, graph_xai",
        "  utils/        — seed, logging, registry",
        "configs/",
        "artifacts/figures/phase_XX_name/",
    ]
    ax4.text(0.05, 0.95, "\n".join(structure_items), transform=ax4.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.3))
    ax4.axis("off")
    ax4.set_title("Project Structure")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_pipeline_diagram(output_path: str):
    """Fig 2: High-level project pipeline diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Experiment 5 — Unified Multimodal Graph-Gated MoE\nPipeline Overview (Phases 0–12)", fontsize=13, fontweight="bold")

    # Phase boxes
    phases = [
        (0.5, 7, "Phase 0\nSetup"),
        (2.5, 7, "Phase 1\nEDA"),
        (4.5, 7, "Phase 2\nPreprocess"),
        (0.5, 5, "Phase 3\nBaselines"),
        (2.5, 5, "Phase 4\nFusion"),
        (4.5, 5, "Phase 5\nMMoEEx"),
        (6.5, 5, "Phase 6\nGraph"),
        (8.5, 5, "Phase 7\nJoint Train"),
        (10.5, 5, "Phase 8\nLLM Abl"),
        (0.5, 3, "Phase 9\nDomain Adapt"),
        (2.5, 3, "Phase 10\nCalibration"),
        (4.5, 3, "Phase 11\nXAI"),
        (6.5, 3, "Phase 12\nThesis"),
    ]

    colors = ["#3498db", "#2ecc71", "#2ecc71",
              "#e74c3c", "#e74c3c", "#e74c3c", "#9b59b6", "#9b59b6", "#f39c12",
              "#f39c12", "#1abc9c", "#1abc9c", "#34495e"]

    for (x, y, label), color in zip(phases, colors):
        rect = plt.Rectangle((x - 0.4, y - 0.5), 1.8, 1.0, facecolor=color, alpha=0.7, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.5, y, label, ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    # Architecture blocks on the right
    arch_blocks = [
        (10.5, 7, "DAIC-WOZ\nDepression", "#3498db"),
        (12.0, 7, "CMU-MOSEI\nSentiment+Emotion", "#2ecc71"),
        (10.5, 6.5, "ChaLearn FI\nPersonality", "#e74c3c"),
        (10.5, 5.5, "Text Encoder\nRoBERTa/LLM", "#9b59b6"),
        (12.0, 5.5, "Audio Encoder\nWavLM/LLM", "#9b59b6"),
        (10.5, 4.5, "Video Encoder\nOpenFace/ViT", "#9b59b6"),
        (10.5, 3.5, "LMF / Gated\nFusion", "#f39c12"),
        (12.0, 3.5, "MMoEEx\nExpert Bank", "#f39c12"),
        (10.5, 2.5, "KNN Graph\nGraphSAGE/GAT", "#1abc9c"),
        (12.0, 2.5, "Task Heads\n4 tasks", "#34495e"),
        (10.5, 1.5, "XAI: SHAP\nGNNExplainer", "#34495e"),
    ]

    arch_colors = [
        "#3498db",  # DAIC
        "#2ecc71",  # MOSEI
        "#e74c3c",  # FI
        "#9b59b6",  # text encoder
        "#9b59b6",  # audio encoder
        "#9b59b6",  # video encoder
        "#f39c12",  # LMF fusion
        "#f39c12",  # MMoEEx
        "#1abc9c",  # KNN graph
        "#34495e",  # task heads
        "#34495e",  # XAI
    ]
    for (x, y, label, _), color in zip(arch_blocks, arch_colors):
        rect = plt.Rectangle((x - 0.7, y - 0.35), 1.6, 0.7, facecolor=color, alpha=0.6, edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.1, y, label, ha="center", va="center", fontsize=7, fontweight="bold", color="white")

    # Arrows (simplified)
    arrow_props = dict(arrowstyle="->", color="gray", lw=1.5)
    # Phase flow arrows
    for i in range(len(phases) - 1):
        if i in [2, 5, 7, 8, 9, 10, 11]:
            continue  # skip to next row or wrap
        x1 = phases[i][0] + 0.9
        y1 = phases[i][1]
        x2 = phases[i+1][0] - 0.9
        y2 = phases[i+1][1]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_props)

    # Flow labels
    ax.text(7.5, 6.5, "Phase 0-2: Data Pipeline", fontsize=9, style="italic", color="gray", ha="center")
    ax.text(7.5, 4.5, "Phase 3-7: Core Architecture", fontsize=9, style="italic", color="gray", ha="center")
    ax.text(7.5, 2.5, "Phase 8-12: Evaluation & Thesis", fontsize=9, style="italic", color="gray", ha="center")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_reproducibility_dashboard(os.path.join(OUTPUT_DIR, "phase_00_reproducibility_dashboard.png"))
    plot_pipeline_diagram(os.path.join(OUTPUT_DIR, "phase_00_pipeline_diagram.png"))
    print(f"\nPhase 0 visualizations saved to: {OUTPUT_DIR}")