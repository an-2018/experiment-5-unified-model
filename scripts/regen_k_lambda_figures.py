#!/usr/bin/env python3
"""Regenerate the K-sensitivity and learned-lambda figures from the post
sampler-fix real results (graph_sensitivity.csv and the V0/V0_learned
per-variant CSVs), replacing the pre-fix versions."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "artifacts" / "tables"
OUT_DIRS = [
    ROOT / "artifacts" / "figures" / "phase_06_graph",
    ROOT / "paper" / "figures",
]


def load_csv(path):
    with open(path) as f:
        return {r["metric"]: float(r["value"]) for r in csv.DictReader(f)}


def save_all(fig, names):
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")


def plot_k_sensitivity():
    df = pd.read_csv(TABLES / "graph_sensitivity.csv")
    df = df[df["variant"] == "inductive"].sort_values("k")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(df["k"], df["daic_auroc"], marker="o", label="DAIC AUROC", color="#c0392b")
    ax.plot(df["k"], df["mosei_sentiment_ccc"], marker="s", label="MOSEI Sentiment CCC", color="#2980b9")
    ax.plot(df["k"], df["fi_avg_ccc"], marker="^", label="FI Avg CCC", color="#27ae60")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (AUROC=0.5)")
    ax.set_xlabel("K (neighbours)")
    ax.set_ylabel("Metric value")
    ax.set_xticks(df["k"])
    ax.set_title("Graph K-sensitivity sweep (inductive)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_all(fig, ["11_k_sensitivity.png", "k_sensitivity.png"])
    plt.close(fig)
    print("Saved k_sensitivity figure.")


def plot_learned_lambda():
    v0 = load_csv(TABLES / "ggmoe_ablation_real" / "V0_results.csv")
    v0l = load_csv(TABLES / "ggmoe_ablation_real" / "V0_learned_results.csv")

    tasks = ["daic_auroc", "mosei_sentiment_ccc", "mosei_emotion_auc", "fi_avg_ccc"]
    labels = ["DAIC\nAUROC", "MOSEI\nSentiment CCC", "MOSEI\nEmotion AUC", "FI\nAvg CCC"]
    fixed_vals = [v0[t] for t in tasks]
    learned_vals = [v0l[t] for t in tasks]

    x = range(len(tasks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar([i - width / 2 for i in x], fixed_vals, width, label="Fixed $\\lambda=0.5$ (V0)", color="#7f8c8d")
    ax.bar([i + width / 2 for i in x], learned_vals, width, label="Learned per-task $\\lambda$ (V0)", color="#2980b9")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Metric value")
    ax.set_title("Learned vs. fixed routing weight $\\lambda$ (V0 config)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_all(fig, ["10_learned_lambda.png", "learned_lambda.png"])
    plt.close(fig)
    print("Saved learned_lambda figure.")


if __name__ == "__main__":
    plot_k_sensitivity()
    plot_learned_lambda()
