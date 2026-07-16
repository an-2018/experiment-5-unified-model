#!/usr/bin/env python3
"""Phase 13: Graph Sensitivity Sweep — K in {5, 10, 15, 20} with density metrics.

Computes real KNN-graph structural statistics (density, degree, similarity) at
each K using the actual fused DAIC/MOSEI/FI embeddings (via
scripts.phase06_graph.load_all_dataset_embeddings), rather than synthetic
random data. Real per-K task performance (DAIC AUROC, MOSEI CCC, FI CCC) is
merged in from artifacts/tables/ggmoe_ablation_real/*.csv when available —
those come from scripts/phase07_joint_training.py, the validated real-label
training pipeline (see run_ggmoe_ablation_real.sh for K=10/15 inductive runs,
and run_graph_sensitivity_extra_k.sh for the additional K=5/20 inductive runs
this sweep needs).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/anilson/thesis/thesis-experiment-5-unified-model")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from data.graph_builder import build_knn_graph, build_inductive_graph
from phase07_joint_training import load_all_dataset_embeddings, concatenate_all_splits


def load_real_embeddings(device="cpu", hidden_dim=256):
    """Load real fused embeddings across DAIC/MOSEI/FI (all splits, concatenated)."""
    all_embs, all_meta = load_all_dataset_embeddings(torch.device(device), hidden_dim)
    global_embeddings, dataset_ids, global_split_ids, global_task_ids, index_map = \
        concatenate_all_splits(all_embs, all_meta, hidden_dim)
    return global_embeddings, global_split_ids


def real_variant_results(results_dir: Path) -> dict:
    """Load real per-K task metrics from run_ggmoe_ablation_real.sh / extra-K runs."""
    metrics_by_k = {}
    variant_to_k = {"V0": 10, "V3": 15, "K5": 5, "K20": 20}
    for name, k in variant_to_k.items():
        csv_path = results_dir / f"{name}_results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        row = dict(zip(df["metric"], df["value"]))
        metrics_by_k[k] = {
            "daic_auroc": row.get("daic_auroc"),
            "mosei_sentiment_ccc": row.get("mosei_sentiment_ccc"),
            "fi_avg_ccc": row.get("fi_avg_ccc"),
        }
    return metrics_by_k


def main():
    print("Loading real fused embeddings (DAIC + MOSEI + FI, all splits)...")
    embeddings, split_ids = load_real_embeddings()
    n = embeddings.shape[0]
    print(f"  {n} real samples, dim={embeddings.shape[1]}")

    train_mask = split_ids == 0
    train_emb = embeddings[train_mask]
    other_emb = embeddings[~train_mask]
    print(f"  train={len(train_emb)}, val+test={len(other_emb)}")

    perf_by_k = real_variant_results(ROOT / "artifacts/tables/ggmoe_ablation_real")

    sweep_results = []
    for k in [5, 10, 15, 20]:
        # Full graph (all splits together) — structural reference only
        ei, ew = build_knn_graph(embeddings, k=k)
        num_edges = ei.shape[1]
        max_possible = n * (n - 1)
        sweep_results.append({
            "k": k, "variant": "full_graph",
            "num_edges": num_edges,
            "density": num_edges / max_possible if max_possible > 0 else 0,
            "avg_degree": num_edges / n,
            "avg_similarity": float(ew.mean()) if len(ew) > 0 else 0.0,
        })

        # Inductive graph (train-only + val/test connecting to train) — matches V0/V3 protocol
        train_ei, train_ew, other_ei, other_ew = build_inductive_graph(train_emb, other_emb, k=k)
        total_edges = train_ei.shape[1] + other_ei.shape[1]
        row = {
            "k": k, "variant": "inductive",
            "num_edges": total_edges,
            "density": total_edges / max_possible if max_possible > 0 else 0,
            "avg_degree": total_edges / n,
            "avg_similarity": float(np.mean([train_ew.mean(), other_ew.mean()]))
                if len(train_ew) > 0 and len(other_ew) > 0 else 0.0,
        }
        perf = perf_by_k.get(k)
        if perf:
            row.update(perf)
        else:
            print(f"  [note] no real performance run found for k={k} yet "
                  f"(expected artifacts/tables/ggmoe_ablation_real/{{V0,V3,K5,K20}}_results.csv)")
        sweep_results.append(row)

    df = pd.DataFrame(sweep_results)
    out_path = ROOT / "artifacts/tables/graph_sensitivity.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nGraph sensitivity saved to {out_path}")
    print(df.to_string())


if __name__ == "__main__":
    main()
