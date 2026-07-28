#!/usr/bin/env python3
"""Diagnose whether the real KNN routing graphs (V0-V4) carry task-relevant
signal, by measuring same-dataset edge label agreement/correlation against a
random-pairing baseline. Cheap, no retraining: reuses the exact embedding +
graph construction code path used by the real training runs.

See context/graph_routing_improvement_review.md for motivation.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/anilson/thesis/thesis-experiment-5-unified-model")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from phase07_joint_training import (
    load_all_dataset_embeddings, concatenate_all_splits, HIDDEN_DIM,
)
from phase05_mmoe_ex import load_all_labels, make_label_key
from data.graph_builder import build_knn_graph, build_inductive_graph, build_split_local_graph
from utils.seed import set_seed
import torch

FI_TRAITS = ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]


def get_dataset_for_index(idx, index_map):
    for (dataset, split), (start, end) in index_map.items():
        if start <= idx < end:
            return dataset, split
    return None, None


def get_scalar_label(dataset, sample_id, split, all_labels):
    key = make_label_key(dataset, sample_id, split)
    if key not in all_labels:
        return None
    val = all_labels[key]
    if dataset == "daic":
        return float(val)
    elif dataset == "mosei":
        return float(val[0])  # sentiment
    else:  # fi
        return float(np.mean([val[t] for t in FI_TRAITS]))


def edge_agreement(edge_index, dataset_ids, index_map, split_ids, all_labels, seed=0):
    """For same-dataset edges, compute label agreement/correlation vs a random-pairing baseline.

    DAIC: fraction of edges with matching binary label (vs random-pairing baseline).
    MOSEI/FI: Pearson correlation of connected nodes' scalar labels (vs shuffled baseline).
    """
    src, dst = edge_index[0], edge_index[1]
    results = {}

    # Precompute dataset/split/label per node lazily
    node_cache = {}

    def node_info(idx):
        if idx not in node_cache:
            dataset, split = get_dataset_for_index(idx, index_map)
            sample_id = dataset_ids[idx]
            label = get_scalar_label(dataset, sample_id, split, all_labels) if dataset else None
            node_cache[idx] = (dataset, label)
        return node_cache[idx]

    by_dataset_pairs = {"daic": [], "mosei": [], "fi": []}
    cross_dataset_edges = 0
    total_edges = len(src)

    for i in range(total_edges):
        s, d = int(src[i]), int(dst[i])
        ds_s, lbl_s = node_info(s)
        ds_d, lbl_d = node_info(d)
        if ds_s is None or ds_d is None:
            continue
        if ds_s != ds_d:
            cross_dataset_edges += 1
            continue
        if lbl_s is None or lbl_d is None:
            continue
        by_dataset_pairs[ds_s].append((lbl_s, lbl_d))

    results["total_edges"] = total_edges
    results["cross_dataset_fraction"] = cross_dataset_edges / total_edges if total_edges else 0.0

    rng = np.random.RandomState(seed)
    for dataset, pairs in by_dataset_pairs.items():
        if len(pairs) < 5:
            results[dataset] = {"n_edges": len(pairs), "metric": None, "random_baseline": None}
            continue
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])

        if dataset == "daic":
            real_metric = float(np.mean(a == b))
            b_shuffled = rng.permutation(b)
            baseline = float(np.mean(a == b_shuffled))
            metric_name = "label_agreement_fraction"
        else:
            if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                real_metric, baseline = 0.0, 0.0
            else:
                real_metric = float(np.corrcoef(a, b)[0, 1])
                b_shuffled = rng.permutation(b)
                baseline = float(np.corrcoef(a, b_shuffled)[0, 1]) if np.std(b_shuffled) > 1e-8 else 0.0
            metric_name = "pearson_correlation"

        results[dataset] = {
            "n_edges": len(pairs), "metric": metric_name,
            "real": round(real_metric, 4), "random_baseline": round(baseline, 4),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="Seeds the untrained GatedLateFusion projection used for graph "
                             "construction. Previously unseeded (a different random projection "
                             "every run, non-reproducible) -- fixed to match the seeding this "
                             "project's actual V0-V4 training runs now use.")
    args = parser.parse_args()
    set_seed(args.seed)

    print(f"Loading real fused embeddings (seed={args.seed}; this reuses the exact V0-V4 embedding pipeline)...")
    all_embs, all_meta = load_all_dataset_embeddings(torch.device("cpu"), HIDDEN_DIM)
    global_embeddings, dataset_ids, global_split_ids, global_task_ids, index_map = \
        concatenate_all_splits(all_embs, all_meta, HIDDEN_DIM)
    print(f"  {global_embeddings.shape[0]} total nodes")

    print("Loading real labels...")
    all_labels = load_all_labels()
    print(f"  {len(all_labels)} labeled samples")

    # NOTE: V2 (transductive) is excluded here. construct_graphs()'s transductive
    # branch (phase07_joint_training.py:653-655) filters train/val/test edges using
    # cumulative index-count thresholds (dst_nodes < train_mask.sum()), which is only
    # correct if the global array is ordered split-major (all train, then all val,
    # then all test). concatenate_all_splits() actually orders it dataset-major
    # (DAIC train/val/test, then MOSEI train/val/test, then FI train/val/test), so
    # V2's train/val/test edge assignment is scrambled -- a separate, real bug in the
    # training code itself, independent of this diagnostic. V2 needs a code fix and
    # rerun before its results can be trusted; not analyzed here.
    variants = [
        ("V0", "inductive", 10), ("V1", "split-local", 10),
        ("V3", "inductive", 15), ("V4", "split-local", 15),
    ]

    train_mask = global_split_ids == 0
    val_mask = global_split_ids == 1
    global_train_idx = np.where(train_mask)[0]
    global_val_idx = np.where(val_mask)[0]

    train_embs = global_embeddings[train_mask]
    val_embs = global_embeddings[val_mask]
    n_train = len(train_embs)

    all_results = {}
    for name, graph_type, k in variants:
        print(f"\n{'='*60}\n{name}: graph_type={graph_type} k={k}\n{'='*60}")

        if graph_type == "inductive":
            # build_inductive_graph returns LOCAL indices: src in [n_train, n_train+n_val)
            # for val-side nodes, dst in [0, n_train) for train-side nodes. Remap both
            # to global indices before looking up labels.
            _, _, val_edge_local, val_edge_w = build_inductive_graph(train_embs, val_embs, k=k)
            local_src = val_edge_local[0] - n_train
            local_dst = val_edge_local[1]
            global_src = global_train_idx[local_dst]  # dst=train side (nn.kneighbors reference)
            global_dst = global_val_idx[local_src]     # src=val side (query)
            # edge_agreement expects edge_index[0]=src, edge_index[1]=dst; use (val, train) pairs
            val_edge_global = np.stack([global_dst, global_src])
        else:  # split-local
            graphs, _ = build_split_local_graph(global_embeddings, global_split_ids, k=k)
            val_edge_local, _ = graphs["val"]
            global_src = global_val_idx[val_edge_local[0]]
            global_dst = global_val_idx[val_edge_local[1]]
            val_edge_global = np.stack([global_src, global_dst])

        res = edge_agreement(val_edge_global, dataset_ids, index_map, global_split_ids, all_labels)
        print(f"  val graph: {res['total_edges']} edges, "
              f"cross-dataset fraction={res['cross_dataset_fraction']:.4f}")
        for ds in ["daic", "mosei", "fi"]:
            r = res[ds]
            if r["metric"] is None:
                print(f"    {ds}: insufficient same-dataset edges ({r['n_edges']})")
            else:
                print(f"    {ds}: {r['metric']}: real={r['real']}  random_baseline={r['random_baseline']}  "
                      f"(n={r['n_edges']})")
        all_results[name] = res

    import json
    out_dir = ROOT / "artifacts" / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"homophily_untrained_embedding_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
