#!/usr/bin/env python3
"""Recompute the graph-homophily diagnostic using the TRAINED non-graph
MMoEEx checkpoint's own fused representation, instead of the untrained,
randomly-initialized GatedLateFusion that scripts/diagnose_graph_homophily.py
(and the actual V0-V4 training's graph construction, phase07_joint_training.py
line ~1424) both use via load_all_dataset_embeddings().

This determines whether the existing homophily numbers (0.568/0.555 etc.)
would look different computed on the model's own learned representation
rather than a random projection of the raw features -- i.e. whether genuine
convergent evidence exists between the E1 construct-profile finding and the
routing homophily diagnostic, or whether they are measuring different things
that happen to point the same direction by coincidence.

Reuses edge_agreement() from diagnose_graph_homophily.py unchanged so the
metric definitions are identical; only the embedding source differs.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from diagnose_graph_homophily import edge_agreement  # noqa: E402
from e1_e7_profile_gate import (  # noqa: E402
    ARTIFACTS_TABLES, LLM_LEVEL, build_inference_model, load_checkpoint_from_path,
)
from src.evaluation.inference import _InferenceDataset  # noqa: E402
from src.data.graph_builder import build_inductive_graph, build_split_local_graph  # noqa: E402

CHECKPOINTS = {
    "original": ARTIFACTS_TABLES / "mmoe_ex_best_original_seed_unknown.pt",
    "seed17": ARTIFACTS_TABLES / "mmoe_ex_best_seed17.pt",
    "seed42": ARTIFACTS_TABLES / "mmoe_ex_best_seed42.pt",
    "seed1337": ARTIFACTS_TABLES / "mmoe_ex_best_seed1337.pt",
    "seed2024": ARTIFACTS_TABLES / "mmoe_ex_best_seed2024.pt",
    "seed31415": ARTIFACTS_TABLES / "mmoe_ex_best_seed31415.pt",
}
VARIANTS = [("V0", "inductive", 10), ("V1", "split-local", 10),
            ("V3", "inductive", 15), ("V4", "split-local", 15)]


@torch.no_grad()
def extract_trained_embeddings(model, device, split: str):
    """Extract fused representations for ALL samples in a given split across
    daic/mosei/fi, using each dataset's native routing, from the trained
    checkpoint. Returns embeddings + per-sample (dataset, scalar_label)."""
    ds = _InferenceDataset(llm_level=LLM_LEVEL, split=split)
    embeddings, dataset_names, scalar_labels = [], [], []

    for s in ds.samples:
        text = torch.from_numpy(s["text"]).float().unsqueeze(0).to(device)
        audio = torch.from_numpy(s["audio"]).float().unsqueeze(0).to(device)
        video = torch.from_numpy(s["video"]).float().unsqueeze(0).to(device)
        mask = torch.tensor(s["modality_mask"], dtype=torch.bool).unsqueeze(0).to(device)
        fused = model.get_fused_representation(text, audio, video, mask, s["routing"]).squeeze(0).cpu().numpy()
        embeddings.append(fused)
        dataset_names.append(s["dataset"])

        label = s["label"]
        if s["dataset"] == "daic":
            scalar_labels.append(float(label))
        elif s["dataset"] == "mosei":
            scalar_labels.append(float(label[0]) if hasattr(label, "__len__") else float(label))
        else:  # fi -- label is a dict of {trait: value}
            if isinstance(label, dict):
                scalar_labels.append(float(np.mean(list(label.values()))))
            elif hasattr(label, "__len__"):
                scalar_labels.append(float(np.mean(label)))
            else:
                scalar_labels.append(float(label))

    return np.stack(embeddings), np.array(dataset_names), np.array(scalar_labels)


def edge_agreement_from_embeddings(edge_index, dataset_names, scalar_labels, seed=0):
    """Adapts diagnose_graph_homophily.edge_agreement's logic for a flat
    (dataset_names, scalar_labels) array instead of the original's
    index_map/dataset_ids/all_labels lookup machinery."""
    src, dst = edge_index[0], edge_index[1]
    rng = np.random.RandomState(seed)
    by_dataset_pairs = {"daic": [], "mosei": [], "fi": []}
    cross_dataset_edges = 0
    total_edges = len(src)

    for i in range(total_edges):
        s, d = int(src[i]), int(dst[i])
        ds_s, ds_d = dataset_names[s], dataset_names[d]
        if ds_s != ds_d:
            cross_dataset_edges += 1
            continue
        by_dataset_pairs[ds_s].append((scalar_labels[s], scalar_labels[d]))

    results = {"total_edges": total_edges,
               "cross_dataset_fraction": cross_dataset_edges / total_edges if total_edges else 0.0}

    for dataset, pairs in by_dataset_pairs.items():
        if len(pairs) < 5:
            results[dataset] = {"n_edges": len(pairs), "metric": None}
            continue
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        if dataset == "daic":
            real_metric = float(np.mean(a == b))
            baseline = float(np.mean(a == rng.permutation(b)))
            metric_name = "label_agreement_fraction"
        else:
            if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                real_metric, baseline = 0.0, 0.0
            else:
                real_metric = float(np.corrcoef(a, b)[0, 1])
                b_shuf = rng.permutation(b)
                baseline = float(np.corrcoef(a, b_shuf)[0, 1]) if np.std(b_shuf) > 1e-8 else 0.0
            metric_name = "pearson_correlation"
        results[dataset] = {"n_edges": len(pairs), "metric": metric_name,
                            "real": round(real_metric, 4), "random_baseline": round(baseline, 4)}
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {}

    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        if not ckpt_path.exists():
            print(f"Skipping {ckpt_name}: checkpoint not found at {ckpt_path}")
            continue
        print(f"\n{'='*70}\n{ckpt_name}\n{'='*70}")
        model = build_inference_model(LLM_LEVEL, device)
        load_checkpoint_from_path(model, ckpt_path, device)
        model.eval()

        train_emb, train_ds_names, train_labels = extract_trained_embeddings(model, device, "train")
        val_emb, val_ds_names, val_labels = extract_trained_embeddings(model, device, "val")
        print(f"  train: {len(train_emb)} samples, val: {len(val_emb)} samples")

        ckpt_results = {}
        for name, graph_type, k in VARIANTS:
            if graph_type == "inductive":
                # Val nodes connect ONLY to train nodes (mirrors build_inductive_graph's
                # test-side semantics; diagnose_graph_homophily.py analyzes this same
                # val->train edge set for the "inductive" variants).
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
                nn.fit(train_emb)
                _, indices = nn.kneighbors(val_emb)
                n_val = len(val_emb)
                src = np.repeat(np.arange(n_val), k)
                dst = indices.flatten()
                # src indexes into val arrays, dst indexes into train arrays -- edge_agreement
                # needs one combined (dataset_names, labels) array, so concatenate with a
                # dst-index offset into the train block.
                combined_ds_names = np.concatenate([val_ds_names, train_ds_names])
                combined_labels = np.concatenate([val_labels, train_labels])
                dst_offset = dst + n_val
                edge_index = np.stack([src, dst_offset])
                res = edge_agreement_from_embeddings(edge_index, combined_ds_names, combined_labels)
            else:  # split-local: val nodes connect only to other val nodes
                edge_index, _ = _build_knn_local(val_emb, k)
                res = edge_agreement_from_embeddings(edge_index, val_ds_names, val_labels)

            ckpt_results[name] = res
            print(f"  {name} (k={k}): cross-dataset-frac={res['cross_dataset_fraction']:.4f}")
            for ds in ["daic", "mosei", "fi"]:
                r = res[ds]
                if r["metric"] is None:
                    print(f"    {ds}: insufficient edges ({r['n_edges']})")
                else:
                    print(f"    {ds}: {r['metric']}: real={r['real']} random={r['random_baseline']} (n={r['n_edges']})")

        all_results[ckpt_name] = ckpt_results

    import json
    with open(REPO_ROOT / "artifacts" / "stats" / "homophily_trained_representation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: artifacts/stats/homophily_trained_representation.json")


def _build_knn_local(embeddings, k):
    from src.data.graph_builder import build_knn_graph
    return build_knn_graph(embeddings, k=k)


if __name__ == "__main__":
    main()
