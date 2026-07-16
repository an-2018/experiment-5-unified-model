#!/usr/bin/env python3
"""
Phase 11 — XAI and Graph-Based Explanation Package
===================================================
Generates multimodal and graph-based explanations for selected cases across
DAIC-WOZ (depression), CMU-MOSEI (sentiment/emotion), and ChaLearn FI (personality).

USES REAL MODEL (Phase 8 L1 — Mistral text) and REAL DATA (validation samples).
No synthetic data, no mock models, no fallback values.

Outputs:
  - artifacts/figures/phase_11_xai/*.png (visualization types)
  - artifacts/tables/phase11_xai_results.json (case studies + metrics)
"""
import os, sys, json, warnings
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.xai_engine import (
    SHAPExplainer, GNNExplainerWrapper, perturbation_test, counterfactual_test,
)
from evaluation.graph_xai import GraphXAINNarrator
from evaluation.inference import (
    load_real_model_for_xai, load_real_data_samples, build_real_graph,
)

# ── Style ──
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
})

FIG_DIR = Path("artifacts/figures/phase_11_xai")
TAB_DIR = Path("artifacts/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS (operate on any data — no synthetic assumptions)
# ═══════════════════════════════════════════════════════════════════

def plot_shap_beeswarm(shap_values_list: list[dict], output_path: str):
    """Plot horizontal beeswarm of SHAP values per modality."""
    if not shap_values_list:
        return
    modalities = list(shap_values_list[0].keys())
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, mod in enumerate(modalities):
        vals = [s.get(mod, 0) for s in shap_values_list]
        jitter = np.random.RandomState(i).randn(len(vals)) * 0.05
        ax.scatter(vals, np.full_like(vals, i, dtype=float) + jitter,
                   alpha=0.4, s=15, label=mod)

    ax.set_yticks(range(len(modalities)))
    ax.set_yticklabels(modalities)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("SHAP Value (modality contribution)")
    ax.set_title("Modality-Level SHAP Values Across Samples")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_modality_attribution(shap_values_list: list[dict], output_path: str,
                              n_samples: int = 10):
    """Plot stacked bars of modality contributions per sample."""
    if not shap_values_list:
        return
    modalities = list(shap_values_list[0].keys())
    samples_to_plot = shap_values_list[:n_samples]

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(samples_to_plot))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for i, mod in enumerate(modalities):
        vals = [s.get(mod, 0) for s in samples_to_plot]
        ax.bar(range(len(samples_to_plot)), vals, bottom=bottom,
               color=colors[i % len(colors)], label=mod, alpha=0.8)
        bottom += np.array(vals)

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Cumulative SHAP Value")
    ax.set_title("Modality Attribution per Sample")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_gnn_subgraph(edge_index: torch.Tensor, edge_weights: torch.Tensor,
                      output_path: str, query_idx: int = 0,
                      node_meta: Optional[list] = None,
                      title: str = "GNNExplainer Subgraph"):
    """Ego-centered view of the query node's neighborhood.

    The query node is fixed at the center; neighbors are placed at a radius
    proportional to their dissimilarity (1 - edge weight), so visually closer
    neighbors really are more similar in the model's own representation, not
    an artifact of a generic force-directed layout. Neighbor nodes are colored
    by their true label (binary: two colors; continuous: a diverging colormap)
    so label-homogeneity of the neighborhood -- or its absence -- is visible
    directly, rather than only reported as a separate aggregate statistic.
    """
    n_edges = edge_index.shape[1]
    if n_edges == 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "No edges in subgraph", ha="center", va="center")
        ax.set_title(title)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    weights = edge_weights.detach().cpu().numpy()
    edges_np = edge_index.cpu().numpy()

    # Keep only edges touching the query node, deduped by neighbor id (keep max weight).
    neighbor_w = {}
    for i in range(n_edges):
        src, dst = int(edges_np[0, i]), int(edges_np[1, i])
        if src == query_idx:
            nid = dst
        elif dst == query_idx:
            nid = src
        else:
            continue
        w = float(weights[i])
        if nid not in neighbor_w or w > neighbor_w[nid]:
            neighbor_w[nid] = w

    if not neighbor_w:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, f"No edges touch node {query_idx}", ha="center", va="center")
        ax.set_title(title)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    neighbor_ids = sorted(neighbor_w, key=lambda n: -neighbor_w[n])
    n_neigh = len(neighbor_ids)
    radii = [max(0.28, 1.0 - neighbor_w[nid]) for nid in neighbor_ids]
    angles = np.linspace(0, 2 * np.pi, n_neigh, endpoint=False)
    pos = {query_idx: (0.0, 0.0)}
    for nid, r, a in zip(neighbor_ids, radii, angles):
        pos[nid] = (r * np.cos(a), r * np.sin(a))

    def get_label(nid):
        if node_meta is not None and nid < len(node_meta):
            lv = node_meta[nid].get("label")
            if lv is not None:
                arr = np.asarray(lv).flatten()
                if arr.size:
                    return float(arr[0])
        return None

    query_label = get_label(query_idx)
    neighbor_labels = [get_label(nid) for nid in neighbor_ids]
    finite = [l for l in neighbor_labels + [query_label] if l is not None]
    is_binary = bool(finite) and set(round(v, 6) for v in finite).issubset({0.0, 1.0})

    fig, ax = plt.subplots(figsize=(6.6, 6.2))

    for nid in neighbor_ids:
        w = neighbor_w[nid]
        x0, y0 = pos[query_idx]
        x1, y1 = pos[nid]
        ax.plot([x0, x1], [y0, y1], color="#8a9a95", linewidth=0.6 + 3.2 * w,
                alpha=0.3 + 0.5 * w, zorder=1, solid_capstyle="round")

    legend_handles = None
    colorbar_mappable = None
    if is_binary:
        cmap = {0.0: "#4C7DA6", 1.0: "#C4622D"}
        neighbor_colors = [cmap.get(round(l, 6), "#9a9a9a") if l is not None else "#9a9a9a"
                           for l in neighbor_labels]
        legend_handles = [mpatches.Patch(color="#4C7DA6", label="Label 0"),
                          mpatches.Patch(color="#C4622D", label="Label 1")]
    else:
        vals = np.array([l if l is not None else 0.0 for l in neighbor_labels])
        lo, hi = float(min(vals.min(), 0.0)), float(max(vals.max(), 1e-6))
        norm = plt.Normalize(vmin=lo, vmax=hi)
        cmap_fn = plt.get_cmap("RdYlBu_r")
        neighbor_colors = [cmap_fn(norm(v)) for v in vals]
        colorbar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_fn)

    for nid, color in zip(neighbor_ids, neighbor_colors):
        x1, y1 = pos[nid]
        ax.scatter([x1], [y1], s=460, color=color, edgecolor="white", linewidth=1.4, zorder=3)
        ax.annotate(str(nid), (x1, y1), ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold", zorder=4)

    ax.scatter([0], [0], s=680, color="#1B2320", edgecolor="#E8C547", linewidth=2.6,
              marker="*", zorder=5)
    ax.annotate(f"query\n{query_idx}", (0, 0), xytext=(0, -0.36), ha="center", va="top",
                fontsize=8.2, color="#1B2320", fontweight="bold", zorder=6)

    if is_binary and query_label is not None:
        same = sum(1 for l in neighbor_labels if l is not None and round(l, 6) == round(query_label, 6))
        subtitle = f"{same} of {n_neigh} neighbors share the query's label"
    else:
        subtitle = f"{n_neigh} neighbors — node radius ∝ dissimilarity, color = label value"

    ax.set_title(f"{title}\n{subtitle}", fontsize=11.3)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9)
    if colorbar_mappable is not None:
        cbar = fig.colorbar(colorbar_mappable, ax=ax, fraction=0.04, pad=0.02, shrink=0.7)
        cbar.set_label("label value", fontsize=8.5)
        cbar.ax.tick_params(labelsize=7.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_top_neighbors_table(neighbors: list[dict], output_path: str,
                             title: str = "Top-K Influential Neighbors"):
    """Plot table of top-k neighbors."""
    if not neighbors:
        return

    fig, ax = plt.subplots(figsize=(8, max(3, len(neighbors) * 0.5)))
    ax.axis("off")

    col_labels = ["Neighbor ID", "Distance", "Dataset", "Influence Weight"]
    rows = []
    for n in neighbors:
        rows.append([str(n.get("id", "")), f"{n.get('distance', 0):.4f}",
                     n.get("dataset", "?"), f"{n.get('weight', 0):.4f}"])

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center", colWidths=[0.15, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    ax.set_title(title, fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_counterfactual_change(cf_results: dict, output_path: str,
                               title: str = "Counterfactual: Min Change per Modality to Flip Prediction"):
    """Plot bar chart of minimal perturbation needed per modality."""
    modalities = list(cf_results.keys())
    values = [cf_results[m] for m in modalities]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c" if v < 0 else "#3498db" for v in values]
    bars = ax.bar(modalities, values, color=colors, alpha=0.7)

    for bar, val in zip(bars, values):
        label = f"{val:.2f}" if val >= 0 else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                label, ha="center", fontsize=9)

    ax.set_ylabel("Gradient-based Perturbation Magnitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_graphxain_panel(narrative: str, shap_values: dict, neighbors: list[dict],
                         sample_meta: dict, output_path: str):
    """Combined panel with narrative + technical evidence."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    lines = [
        f"GraphXAIN Explanation Panel",
        f"{'=' * 40}",
        f"",
        f"Subject: {sample_meta.get('subject_id', 'N/A')}",
        f"Prediction: {sample_meta.get('prediction', 'N/A')}",
        f"Confidence: {sample_meta.get('confidence', 0):.2f}",
        f"Dataset: {sample_meta.get('dataset', '?')}",
        f"Task: {sample_meta.get('task', '?')}",
        f"",
        f"Modality Attributions:",
    ]
    for mod, val in sorted(shap_values.items(), key=lambda x: -abs(x[1])):
        lines.append(f"  {mod}: {val:+.4f}")

    lines.append(f"")
    lines.append(f"Top Neighbors:")
    for n in neighbors[:3]:
        lines.append(f"  ID={n.get('id','?')}, dist={n.get('distance',0):.3f}")

    lines.append(f"")
    lines.append(f"Narrative Explanation:")
    lines.append(f"{narrative[:500]}...")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontfamily="monospace", fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_multi_construct_force_panel(task_rows: list[dict], sample_meta: dict, output_path: str):
    """Multi-panel force/tornado plot, one row per task head, for a single instance.

    Adapts the standard tabular-ML "force plot" (prediction bar + per-feature
    push-left/push-right tornado + feature-value table, one row per model) to
    this multimodal setting: instead of several models explaining one instance,
    the SAME instance is explained through each of the model's four task heads
    (depression / sentiment / emotion / personality), operationalizing the
    multi-construct-characterization hypothesis as a single figure rather than
    four disconnected numbers.

    task_rows: list of dicts, one per task head, each with:
      name: str, kind: "binary" or "continuous", value: float (raw head output,
      pre-sigmoid for binary), class_labels: (neg, pos) for binary,
      value_range: (lo, hi) for continuous, shap: {modality: float},
      perturbation: {modality: float}
    """
    n = len(task_rows)
    NEG, POS = "#4C7DA6", "#C4622D"
    fig = plt.figure(figsize=(11.5, 2.5 * n + 0.9))
    gs = fig.add_gridspec(n, 3, width_ratios=[1.05, 2.0, 1.25], wspace=0.4, hspace=0.75,
                          left=0.1, right=0.97, top=0.88, bottom=0.06)

    for row, task in enumerate(task_rows):
        # Panel 1 — prediction summary
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.set_xlim(0, 1); ax0.set_ylim(0, 1); ax0.axis("off")
        ax0.set_title(task["name"], fontsize=10.5, fontweight="bold", loc="left")
        if task["kind"] == "binary":
            p = 1 / (1 + np.exp(-task["value"]))
            neg_lbl, pos_lbl = task.get("class_labels", ("Neg", "Pos"))
            ax0.barh([0.5], [1 - p], left=0, height=0.38, color=NEG)
            ax0.barh([0.5], [p], left=1 - p, height=0.38, color=POS)
            ax0.text(0.0, 0.82, neg_lbl, fontsize=8, color=NEG, fontweight="bold", transform=ax0.transAxes)
            ax0.text(0.0, 0.68, f"{1 - p:.2f}", fontsize=9.5, color=NEG, transform=ax0.transAxes)
            ax0.text(1.0, 0.82, pos_lbl, fontsize=8, color=POS, fontweight="bold", ha="right", transform=ax0.transAxes)
            ax0.text(1.0, 0.68, f"{p:.2f}", fontsize=9.5, color=POS, ha="right", transform=ax0.transAxes)
        else:
            lo, hi = task.get("value_range", (-1.0, 1.0))
            v = float(np.clip(task["value"], lo, hi))
            zero = 0.0 if lo < 0 < hi else lo
            frac = lambda x: (x - lo) / (hi - lo)
            color = POS if v >= zero else NEG
            ax0.barh([0.5], [frac(v) - frac(zero)], left=frac(zero), height=0.38, color=color)
            ax0.axvline(frac(zero), color="#666", linewidth=0.8)
            ax0.text(0.0, 0.82, f"range [{lo:g}, {hi:g}]", fontsize=7.5, color="#777",
                     transform=ax0.transAxes)
            ax0.text(0.0, 0.68, f"value = {task['value']:+.3f}", fontsize=9.5,
                     color=color, fontweight="bold", transform=ax0.transAxes)

        # Panel 2 — modality tornado
        ax1 = fig.add_subplot(gs[row, 1])
        mods = list(task["shap"].keys())
        vals = [task["shap"][m] for m in mods]
        order = np.argsort(np.abs(vals))
        mods_sorted = [mods[i] for i in order]
        vals_sorted = [vals[i] for i in order]
        colors = [POS if v >= 0 else NEG for v in vals_sorted]
        ypos = range(len(mods_sorted))
        ax1.barh(ypos, vals_sorted, color=colors, height=0.55, zorder=3)
        ax1.axvline(0, color="#444", linewidth=0.8, zorder=2)
        ax1.set_yticks(list(ypos)); ax1.set_yticklabels(mods_sorted, fontsize=9)
        span = max(1e-6, max(abs(v) for v in vals_sorted))
        for i, v in enumerate(vals_sorted):
            ax1.text(v + np.sign(v if v != 0 else 1) * span * 0.06, i, f"{v:+.3f}", fontsize=8,
                     ha="left" if v >= 0 else "right", va="center")
        ax1.set_xlim(-span * 1.4, span * 1.4)
        ax1.grid(True, axis="x", alpha=0.25, zorder=0)
        if row == n - 1:
            ax1.set_xlabel("modality contribution (SHAP, push toward each class)", fontsize=8.3)
        ax1.tick_params(axis="x", labelsize=7.5)

        # Panel 3 — value table
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.axis("off")
        pert = task.get("perturbation", {})
        rows = [[m, f"{task['shap'][m]:+.3f}", f"{pert.get(m, float('nan')):+.3f}" if m in pert else "—"]
                for m in mods]
        tbl = ax2.table(cellText=rows, colLabels=["Modality", "SHAP", "Perturb Δ"],
                        loc="center", cellLoc="center", colWidths=[0.4, 0.32, 0.32])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.35)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#d5dbd8")
            if r == 0:
                cell.set_text_props(fontweight="bold", color="#5b6b67")
                cell.set_facecolor("#eef1ef")

    fig.suptitle(
        f"Multi-construct explanation — subject {sample_meta.get('subject_id', '?')} "
        f"({sample_meta.get('dataset', '?').upper()})\n"
        "same forward pass, four task heads — not four separate models",
        fontsize=12.5, fontweight="bold", y=0.975)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Phase 11: XAI and Graph-Based Explanation Package")
    print("USING REAL MODEL (L1 — Mistral text) AND REAL DATA")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load real model ──
    print("\n[1/4] Loading real model (Phase 8 L1 — Mistral text)...")
    model = load_real_model_for_xai(llm_level="L1", device_str=str(device))
    model.eval()
    print("  Model loaded from real checkpoint — no mock model used.")

    # ── Create explainers ──
    print("\n[2/4] Initializing explainers...")
    shap_explainer = SHAPExplainer(model)
    gnn_explainer = GNNExplainerWrapper(model, use_pyg=False)
    graph_xain = GraphXAINNarrator(llm_name="mistral")

    all_results = {"case_studies": [], "perturbation_results": [], "counterfactual_results": []}

    # ── Process each dataset with real data ──
    print("\n[3/4] Loading real data and generating explanations...")
    for dataset in ["daic", "mosei", "fi"]:
        print(f"\n{'─' * 50}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'─' * 50}")

        # Load REAL data samples from validation set
        # (DAIC=~107 val, MOSEI=~2000 val, FI=~200 val — use up to 50)
        try:
            samples = load_real_data_samples(dataset, n_samples=50, llm_level="L1", split="val")
        except RuntimeError as e:
            print(f"  Skipping {dataset}: {e}")
            continue

        if len(samples) == 0:
            print(f"  No real samples found for {dataset}")
            continue

        print(f"  Loaded {len(samples)} real samples")

        # task_id for this dataset's native head (0=depression, 1=sentiment,
        # 3=personality) — SHAP/perturbation/counterfactual/logit must use
        # this so a MOSEI/FI sample's explanations reflect its own task head
        # rather than defaulting to the depression head.
        task_id_map = {"daic": 0, "mosei": 1, "fi": 3}
        task_id = task_id_map[dataset]

        # Build REAL KNN graph on the model's own trained fused representation
        # (not raw concatenated modality features -- see build_real_graph docstring;
        # this addresses the faithfulness gap noted for DAIC in the homophily
        # diagnostic, Section on graph routing, by using the same embedding
        # space the router/heads actually reason over).
        x, edge_index, meta_list, edge_distances = build_real_graph(
            samples, n_neighbors=5, model=model, device=str(device)
        )
        print(f"  Graph: {x.shape[0]} nodes, {edge_index.shape[1]} edges")

        # Run SHAP on a subset of samples
        shap_values_list = []
        for i in range(min(20, len(samples))):
            sv = shap_explainer.compute_modality_shap(samples[i], task_id=task_id,
                                                       routing=samples[i]["routing"])
            shap_values_list.append(sv)

        # Run GNNExplainer
        x_gpu = x.to(device)
        edge_mask, feat_mask = gnn_explainer.explain_node(0, x_gpu, edge_index.to(device))

        # Perturbation tests
        pert_results = []
        for i in range(min(10, len(samples))):
            for mod in ["text", "audio", "video"]:
                delta = perturbation_test(samples[i], model, mod, task_id=task_id,
                                          routing=samples[i]["routing"])
                pert_results.append({"sample": i, "modality": mod, "delta": delta})

        all_results["perturbation_results"].extend(pert_results)

        # Counterfactual tests -- same 10-sample subset as the perturbation
        # tests above, so the two aggregates are directly comparable.
        cf_result = None
        for i in range(min(10, len(samples))):
            cf_i = counterfactual_test(samples[i], model, task_id=task_id,
                                       routing=samples[i]["routing"])
            all_results["counterfactual_results"].append({"dataset": dataset, "sample": i, **cf_i})
            if i == 0:
                cf_result = cf_i  # first sample, used for the per-dataset figure below

        # Case studies
        task_map = {"daic": "depression", "mosei": "sentiment", "fi": "personality"}
        for case_idx in range(min(3, len(samples))):
            s = samples[case_idx]
            sample_routing = s["routing"]
            # Get real prediction using this dataset's own task head and the
            # sample's own routing (text_only for DAIC, video_only for FI,
            # multimodal for MOSEI) -- otherwise this reflects a full-fusion
            # forward pass the model never actually takes for that sample.
            with torch.no_grad():
                feats = torch.cat([
                    s["text_feats"].float().to(device),
                    s["audio_feats"].float().to(device),
                    s["video_feats"].float().to(device),
                ])
                inp = feats.unsqueeze(0)  # (1, D)
                logit = model.forward_encoded(inp, task_id=task_id, routing=sample_routing).item()
                prob = 1 / (1 + np.exp(-logit))

            shap_vals = shap_explainer.compute_modality_shap(s, task_id=task_id, routing=sample_routing)
            pert_deltas = {mod: perturbation_test(s, model, mod, task_id=task_id, routing=sample_routing)
                          for mod in ["text", "audio", "video"]}
            cf_vals = counterfactual_test(s, model, task_id=task_id, routing=sample_routing)

            # Multi-construct characterization profile: run the SAME sample's
            # fused representation (same routing/modality-mask the sample was
            # actually trained with) through all four task heads, not just the
            # one head matching its own dataset. This operationalizes the
            # "characterize depression alongside sentiment/emotion/personality"
            # hypothesis directly, rather than only predicting a single
            # dataset-native target per sample.
            with torch.no_grad():
                t_b = s["text_feats"].float().unsqueeze(0).to(device)
                a_b = s["audio_feats"].float().unsqueeze(0).to(device)
                v_b = s["video_feats"].float().unsqueeze(0).to(device)
                m_b = s["modality_mask"].unsqueeze(0).to(device) if hasattr(s["modality_mask"], "unsqueeze") \
                    else torch.tensor(s["modality_mask"]).unsqueeze(0).to(device)
                routing = s["routing"]

                dep_out = model.predict_task(t_b, a_b, v_b, m_b, 0, routing)
                sent_out = model.predict_task(t_b, a_b, v_b, m_b, 1, routing)
                emo_out = model.predict_task(t_b, a_b, v_b, m_b, 2, routing)
                pers_out = model.predict_task(t_b, a_b, v_b, m_b, 3, routing)

                multi_construct_profile = {
                    "depression_prob": float(torch.sigmoid(dep_out).item()),
                    "sentiment_score": float(sent_out.item()),
                    "emotion_probs": {
                        e: float(p) for e, p in zip(
                            ["happiness", "sadness", "anger", "fear", "disgust", "surprise"],
                            torch.sigmoid(emo_out).squeeze(0).cpu().tolist(),
                        )
                    },
                    "personality_scores": {
                        t: float(p) for t, p in zip(
                            ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"],
                            pers_out.squeeze(0).cpu().tolist(),
                        )
                    },
                }

            # Real neighbor info from KNN graph (use actual distances)
            n_edges = edge_index.shape[1]
            neighbor_list = []
            if n_edges > 0 and edge_distances.numel() > 0:
                # Use actual KNN distances (cosine distance)
                dists = edge_distances.detach().cpu()
                # Convert distance to similarity weight (1 - cosine_distance)
                weights = torch.clamp(1.0 - dists, min=0.0)
                sorted_idx = torch.argsort(weights, descending=True)
                for k in range(min(5, n_edges)):
                    ei = sorted_idx[k].item()
                    if ei >= n_edges:
                        continue
                    src = int(edge_index[0, ei])
                    dst = int(edge_index[1, ei])
                    weight = float(weights[ei])
                    dist = float(dists[ei])
                    neighbor_list.append({
                        "id": int(dst),
                        "distance": dist,
                        "dataset": dataset,
                        "weight": weight,
                    })

            # Real expert routing weights from the trained MMoEEx gate, using
            # this dataset's own task_id and routing (matching forward_encoded above).
            with torch.no_grad():
                real_weights = model.get_routing_weights(
                    inp, task_id=task_id, routing=sample_routing).squeeze(0).cpu()
            expert_weights = {
                f"expert_{e}": round(float(real_weights[e]), 4)
                for e in range(real_weights.numel())
                if real_weights[e] > 1e-6
            }

            # True label
            label_val = s["label"]
            if isinstance(label_val, np.ndarray):
                if label_val.ndim == 0:
                    true_label = float(label_val)
                else:
                    true_label = float(label_val[0]) if len(label_val) > 0 else 0.0
            else:
                true_label = float(label_val)

            case_study = {
                "dataset": dataset,
                "case_id": case_idx,
                "sample_id": s.get("id", f"sample_{case_idx}"),
                "task": task_map[dataset],
                "prediction": float(logit),
                "confidence": float(prob),
                "true_label": true_label,
                "shap_values": shap_vals,
                "perturbation_deltas": pert_deltas,
                "counterfactual": cf_vals,
                "top_neighbors": neighbor_list,
                "expert_routing": expert_weights,
                "multi_construct_profile": multi_construct_profile,
            }

            # Generate GraphXAIN narrative
            n_edge_index = edge_index if n_edges > 0 else torch.zeros((2, 1), dtype=torch.long)
            n_edge_weights = edge_mask if edge_mask.numel() > 0 else torch.ones(1)
            sample_meta_local = {
                "dataset": dataset,
                "task": task_map[dataset],
                "subject_id": s.get("id", f"sample_{case_idx}"),
                "prediction": prob,
                "confidence": prob,
            }
            narrative = graph_xain.generate_explanation(
                subgraph_edge_index=n_edge_index,
                subgraph_edge_weights=n_edge_weights,
                shap_values=shap_vals,
                sample_metadata=sample_meta_local,
                top_k_neighbors=5,
            )
            case_study["narrative"] = str(narrative)
            all_results["case_studies"].append(case_study)
            print(f"  Case {case_idx}: subject={case_study['sample_id']}, "
                  f"pred={prob:.4f}, true={true_label:.4f}")

        # ── Visualizations ──
        fig_base = str(FIG_DIR / f"{dataset}")

        plot_shap_beeswarm(shap_values_list, f"{fig_base}_shap_beeswarm.png")
        plot_modality_attribution(shap_values_list, f"{fig_base}_modality_attribution.png")

        plot_gnn_subgraph(edge_index, edge_mask,
                          f"{fig_base}_gnn_subgraph.png",
                          query_idx=0, node_meta=meta_list,
                          title=f"GNNExplainer Subgraph — {dataset.upper()}")

        plot_top_neighbors_table(neighbor_list, f"{fig_base}_top_neighbors.png",
                                 title=f"Top Neighbors — {dataset.upper()}")

        plot_counterfactual_change(cf_vals, f"{fig_base}_counterfactual.png")

        plot_graphxain_panel(narrative, shap_vals, neighbor_list, sample_meta_local,
                             f"{fig_base}_graphxain_panel.png")

        # Multi-construct force panel: the same instance used for the plots
        # above (last case study of this dataset), explained through all four
        # task heads in one figure -- one forward pass, four constructs, not
        # four separate models the way a tabular-ML force plot would compare.
        with torch.no_grad():
            sent_val = model.forward_encoded(inp, task_id=1, routing=sample_routing).item()
            pers_val = model.forward_encoded(inp, task_id=3, routing=sample_routing).item()
        task_specs = [
            ("Depression", 0, "binary", {"class_labels": ("Not depressed", "Depressed")}),
            ("Sentiment", 1, "continuous",
             {"value_range": (min(-0.5, sent_val * 1.4), max(0.5, sent_val * 1.4))}),
            ("Emotion — happiness (dominant)", 2, "binary", {"class_labels": ("Not happy", "Happy")}),
            ("Personality (avg. of 5 traits)", 3, "continuous",
             {"value_range": (0.0, max(1.0, pers_val * 1.3))}),
        ]
        task_rows = []
        for name, t_id, kind, extra in task_specs:
            with torch.no_grad():
                val = model.forward_encoded(inp, task_id=t_id, routing=sample_routing).item()
            t_shap = shap_explainer.compute_modality_shap(s, task_id=t_id, routing=sample_routing)
            t_pert = {mod: perturbation_test(s, model, mod, task_id=t_id, routing=sample_routing)
                     for mod in ["text", "audio", "video"]}
            row = {"name": name, "kind": kind, "value": val, "shap": t_shap, "perturbation": t_pert}
            row.update(extra)
            task_rows.append(row)

        plot_multi_construct_force_panel(
            task_rows,
            {"subject_id": case_study["sample_id"], "dataset": dataset},
            f"{fig_base}_multi_construct_force.png",
        )

    # ── Validation summary ──
    n_cases = len(all_results["case_studies"])
    n_pert = len(all_results["perturbation_results"])
    text_deltas = [r.get("perturbation_deltas", {}).get("text", 0)
                   for r in all_results["case_studies"]]
    avg_delta_text = np.mean(text_deltas) if text_deltas else 0.0

    print(f"\n{'═' * 50}")
    print("VALIDATION SUMMARY")
    print(f"{'═' * 50}")
    print(f"  Case studies: {n_cases} ({n_cases // 3} per dataset)")
    print(f"  Perturbation tests: {n_pert}")
    print(f"  Average text perturbation delta: {avg_delta_text:.4f}")
    print(f"  All results from REAL model predictions — no synthetic data used.")

    # Export results
    results_path = TAB_DIR / "phase11_xai_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # List all figures
    print(f"\nFigures in {FIG_DIR}:")
    for f in sorted(FIG_DIR.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")

    print(f"\n{'=' * 50}")
    print("Phase 11 complete!")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
