#!/usr/bin/env python3
"""
Phase 8 — Sample-Level Visualization: Feature Evolution Across LLM Levels

Shows how the same DAIC participants' feature representations change when
using classical encoders (L0) vs LLM encoders (L1/L3/L4).

Output: artifacts/figures/phase_08_llm_ablations/sample_feature_evolution.png
        artifacts/figures/phase_08_llm_ablations/sample_feature_table.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os, sys, zipfile, json, textwrap
from pathlib import Path
from sklearn.decomposition import PCA

# ── Config ──────────────────────────────────────────────────────────────
FIG_DIR = Path("artifacts/figures/phase_08_llm_ablations")
FIG_DIR.mkdir(parents=True, exist_ok=True)

DAIC_PROC = Path("data/daic/raw")
DAIC_ZIP = DAIC_PROC
FEAT_CACHE = Path("data/features/llm")

# Pick 4 diverse val participants
SAMPLE_IDS = ["382", "331", "335", "346"]
SAMPLE_LABELS = {
    "382": {"label": 0, "phq8": 0.0,  "desc": "Healthy (PHQ-8=0)"},
    "331": {"label": 0, "phq8": 8.0,  "desc": "Mild symptoms (PHQ-8=8)"},
    "335": {"label": 1, "phq8": 12.0, "desc": "Depressed (PHQ-8=12)"},
    "346": {"label": 1, "phq8": 23.0, "desc": "Severe (PHQ-8=23)"},
}

# Classical feature dimensions
CLASSICAL_DIMS = {
    "text": 384,
    "audio": 74,   # COVAREP (mean-pooled)
    "video": 31,   # 17 AU + 8 gaze + 6 headpose (mean-pooled)
}

SEED = 42
DPI = 150

# ── Data Loading ────────────────────────────────────────────────────────

def load_transcript(pid):
    """Load raw transcript from participant zip."""
    zpath = DAIC_ZIP / f"{pid}_P.zip"
    if not zpath.exists():
        return f"[No transcript for {pid}]"
    try:
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if "TRANSCRIPT" in name:
                    return z.read(name).decode("utf-8", errors="replace")
    except Exception as e:
        return f"[Error: {e}]"
    return "[No transcript found]"

def load_classical_features(pid):
    """Load classical features and return as dict of numpy arrays."""
    result = {}
    base = DAIC_PROC / "processed" / f"{pid}"
    for mod in ["text", "audio_cov", "audio_fmt", "vis_au", "vis_eg", "vis_hp"]:
        fpath = DAIC_PROC / "processed" / f"{pid}_{mod}.npy"
        if fpath.exists():
            arr = np.load(str(fpath))
            # For audio/video time-series, take mean across time
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            result[mod] = arr.astype(np.float32)
        else:
            print(f"  [WARN] Classical feature not found: {fpath}")
    return result

def load_llm_features(pid, level, modality):
    """Load LLM features for a given level and modality."""
    # modality: "text" → L1/L2, "audio" → L3, "video" → L4
    level_map = {"text": "L1", "audio": "L3", "video": "L4"}
    lvl = level_map.get(modality, level)
    path = FEAT_CACHE / lvl / "daic" / f"val_{modality}.npy"
    if not path.exists():
        path = FEAT_CACHE / lvl / "daic" / f"train_{modality}.npy"
    if not path.exists():
        print(f"  [WARN] LLM feature not found: {path}")
        return None
    feats = np.load(str(path), allow_pickle=True).item()
    return feats.get(pid, None)

def get_all_llm_text_features(split="val"):
    """Load all LLM text features for PCA background."""
    path = FEAT_CACHE / "L1" / "daic" / f"{split}_text.npy"
    if not path.exists():
        path = FEAT_CACHE / "L1" / "daic" / "train_text.npy"
    feats = np.load(str(path), allow_pickle=True).item()
    ids = list(feats.keys())
    mat = np.stack([feats[i] for i in ids])
    return ids, mat

def get_all_classical_text_features():
    """Load all available classical text features."""
    proc_dir = DAIC_PROC / "processed"
    ids, feats = [], []
    for fpath in sorted(proc_dir.glob("*_text.npy")):
        pid = fpath.name.split("_")[0]
        arr = np.load(str(fpath))
        ids.append(pid)
        feats.append(arr.astype(np.float32))
    return ids, np.stack(feats)


# ── Feature Evolution Visualization ─────────────────────────────────────

def create_sample_data_cards(ax, sample_data):
    """Draw individual data cards for each participant."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    
    colors = ["#d4edda" if s["label"] == 0 else "#f8d7da" for s in sample_data]
    border_colors = ["#155724" if s["label"] == 0 else "#721c24" for s in sample_data]
    
    for i, (pid, sdata) in enumerate(zip(SAMPLE_IDS, sample_data)):
        x = i * 2.5 + 0.1
        y = 0.1
        w, h = 2.3, 3.8
        
        box = FancyBboxPatch((x, y), w, h, 
                             boxstyle="round,pad=0.05",
                             facecolor=colors[i], 
                             edgecolor=border_colors[i], 
                             linewidth=2)
        ax.add_patch(box)
        
        # Participant ID
        ax.text(x + w/2, y + h - 0.25, f"ID: {pid}", 
                ha="center", va="top", fontsize=11, fontweight="bold",
                color=border_colors[i])
        
        # Label
        lbl = "DEPRESSED" if sdata["label"] == 1 else "CONTROL"
        ax.text(x + w/2, y + h - 0.6, lbl, 
                ha="center", va="top", fontsize=9, fontweight="bold",
                color=border_colors[i])
        
        # PHQ-8
        phq = sdata.get("phq8", "N/A")
        ax.text(x + w/2, y + h - 0.95, f"PHQ-8: {phq}", 
                ha="center", va="top", fontsize=9, color="black")
        
        # Description
        ax.text(x + w/2, y + h - 1.3, sdata.get("desc", ""), 
                ha="center", va="top", fontsize=8, color="black", style="italic")
        
        # Transcript stats
        transcript = sdata.get("transcript", "")
        lines = transcript.strip().split("\n")
        participant_lines = [l for l in lines if "Participant" in l]
        word_count = sum(len(l.split("\t")[-1].split()) for l in participant_lines if "\t" in l)
        total_turns = len([l for l in lines if "\t" in l and l.strip()])
        
        ax.text(x + w/2, y + h - 1.7, f"Participant turns: {len(participant_lines)}", 
                ha="center", va="top", fontsize=7, color="black")
        ax.text(x + w/2, y + h - 1.95, f"Words spoken: ~{word_count}", 
                ha="center", va="top", fontsize=7, color="black")
        ax.text(x + w/2, y + h - 2.2, f"Total turns: ~{total_turns}", 
                ha="center", va="top", fontsize=7, color="black")
        
        # Feature dimensions
        ax.text(x + w/2, y + 0.3, 
                f"Classical: {CLASSICAL_DIMS['text']}D text\n"
                f"            {CLASSICAL_DIMS['audio']}D audio\n"
                f"            {CLASSICAL_DIMS['video']}D video", 
                ha="center", va="bottom", fontsize=6, color="gray")
        ax.text(x + w/2, y + 0.05, "↓ LLM ↓", 
                ha="center", va="bottom", fontsize=7, color="#0066cc")
        ax.text(x + w/2, y - 0.15, 
                f"Mistral: 4096D text\n"
                f"CLAP: 512D audio\n"
                f"LLaVA: 4096D video", 
                ha="center", va="bottom", fontsize=6, color="#0066cc")


def create_text_feature_panel(ax, sample_data):
    """Compare classical text features vs LLM Mistral features via PCA."""
    # Get all classical text features for PCA background
    all_ids, classical_all = get_all_classical_text_features()
    _, llm_all = get_all_llm_text_features()
    
    # PCA on classical features
    pca_c = PCA(n_components=2, random_state=SEED)
    classical_2d = pca_c.fit_transform(classical_all)
    
    # PCA on LLM features
    pca_llm = PCA(n_components=2, random_state=SEED)
    llm_2d = pca_llm.fit_transform(llm_all.astype(np.float32))
    
    # Map IDs
    classical_idx = {pid: i for i, pid in enumerate(all_ids)}
    llm_idx = {pid: i for i, pid in enumerate(SAMPLE_IDS) if pid in all_ids}
    
    # Subplot: Classical text
    ax[0].scatter(classical_2d[:, 0], classical_2d[:, 1], 
                  c="lightgray", s=20, alpha=0.5, label="Other participants")
    colors_scatter = ["green" if s["label"] == 0 else "red" for s in sample_data]
    sizes_scatter = [120, 120, 120, 120]
    for pid, sdata, c, s in zip(SAMPLE_IDS, sample_data, colors_scatter, sizes_scatter):
        if pid in classical_idx:
            idx = classical_idx[pid]
            ax[0].scatter(classical_2d[idx, 0], classical_2d[idx, 1],
                         c=c, s=s, edgecolors="black", linewidth=1.5,
                         zorder=5, label=f"{pid} ({sdata['desc'].split('(')[0].strip()})")
    
    ax[0].set_title(f"Classical Text Features ({CLASSICAL_DIMS['text']}D → PCA)", 
                    fontsize=10, fontweight="bold")
    ax[0].set_xlabel(f"PC1 ({pca_c.explained_variance_ratio_[0]:.1%})")
    ax[0].set_ylabel(f"PC2 ({pca_c.explained_variance_ratio_[1]:.1%})")
    ax[0].legend(fontsize=6, loc="best")
    ax[0].grid(True, alpha=0.3)
    
    # Subplot: LLM text
    for pid, sdata, c, s in zip(SAMPLE_IDS, sample_data, colors_scatter, sizes_scatter):
        if pid in llm_idx:
            idx = llm_idx[pid]
            ax[1].scatter(llm_2d[idx, 0], llm_2d[idx, 1],
                         c=c, s=s, edgecolors="black", linewidth=1.5,
                         zorder=5, label=f"{pid} ({sdata['desc'].split('(')[0].strip()})")
    
    # Background: all LLM features from val set
    _, llm_val = get_all_llm_text_features("val")
    if len(llm_val) > 0:
        llm_val_2d = pca_llm.transform(llm_val.astype(np.float32))
        ax[1].scatter(llm_val_2d[:, 0], llm_val_2d[:, 1],
                     c="lightgray", s=20, alpha=0.5, label="Other val participants")
    
    ax[1].set_title(f"LLM Mistral Text Features (4096D → PCA)", 
                    fontsize=10, fontweight="bold")
    ax[1].set_xlabel(f"PC1 ({pca_llm.explained_variance_ratio_[0]:.1%})")
    ax[1].set_ylabel(f"PC2 ({pca_llm.explained_variance_ratio_[1]:.1%})")
    ax[1].legend(fontsize=6, loc="best")
    ax[1].grid(True, alpha=0.3)


def create_feature_comparison_bars(ax, sample_data):
    """Show feature vector statistics for classical vs LLM per sample."""
    bar_width = 0.15
    x = np.arange(len(SAMPLE_IDS))
    
    metrics = ["mean", "std", "max_abs"]
    metric_labels = ["Mean", "Std Dev", "Max |Value|"]
    
    for m_idx, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        classical_vals = []
        llm_vals = []
        
        for pid in SAMPLE_IDS:
            # Classical
            cf = load_classical_features(pid)
            c_text = cf.get("text", np.zeros(CLASSICAL_DIMS["text"]))
            if metric == "mean":
                classical_vals.append(float(c_text.mean()))
            elif metric == "std":
                classical_vals.append(float(c_text.std()))
            elif metric == "max_abs":
                classical_vals.append(float(np.abs(c_text).max()))
            
            # LLM Mistral
            lf = load_llm_features(pid, "L1", "text")
            if lf is not None:
                lf = lf.astype(np.float32)
                if metric == "mean":
                    llm_vals.append(float(lf.mean()))
                elif metric == "std":
                    llm_vals.append(float(lf.std()))
                elif metric == "max_abs":
                    llm_vals.append(float(np.abs(lf).max()))
            else:
                llm_vals.append(0)
        
        sub_ax = ax[m_idx]
        sub_ax.bar(x - bar_width/2, classical_vals, bar_width, 
                    label="Classical (384D)", color="steelblue", alpha=0.8)
        sub_ax.bar(x + bar_width/2, llm_vals, bar_width,
                    label="LLM Mistral (4096D)", color="coral", alpha=0.8)
        sub_ax.set_xticks(x)
        sub_ax.set_xticklabels(SAMPLE_IDS, fontsize=8)
        sub_ax.set_title(f"Text Feature {mlabel}", fontsize=9, fontweight="bold")
        sub_ax.grid(True, alpha=0.3, axis="y")
        if m_idx == 0:
            sub_ax.legend(fontsize=7)


def create_audio_feature_panel(ax, sample_data):
    """Compare audio features: classical COVAREP vs CLAP."""
    # Classical audio: mean of COVAREP features
    classical_audios = []
    llm_audios = []
    
    for pid in SAMPLE_IDS:
        cf = load_classical_features(pid)
        audio_cov = cf.get("audio_cov", np.zeros(CLASSICAL_DIMS["audio"]))
        classical_audios.append(audio_cov)
        
        la = load_llm_features(pid, "L3", "audio")
        llm_audios.append(la if la is not None else np.zeros(512))
    
    # Bar chart: mean activation
    bar_width = 0.15
    x = np.arange(len(SAMPLE_IDS))
    
    c_means = [float(a.mean()) for a in classical_audios]
    c_stds = [float(a.std()) for a in classical_audios]
    l_means = [float(a.mean()) for a in llm_audios]
    l_stds = [float(a.std()) for a in llm_audios]
    
    ax[0].bar(x - bar_width/2, c_means, bar_width, 
              yerr=c_stds, label="Classical (74D)", color="steelblue", alpha=0.8, capsize=3)
    ax[0].bar(x + bar_width/2, l_means, bar_width,
              yerr=l_stds, label="LLM CLAP (512D)", color="coral", alpha=0.8, capsize=3)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(SAMPLE_IDS, fontsize=8)
    ax[0].set_title("Audio Feature Mean ± Std", fontsize=9, fontweight="bold")
    ax[0].legend(fontsize=7)
    ax[0].grid(True, alpha=0.3, axis="y")
    
    # Dimensionality comparison
    ax[1].bar(0, CLASSICAL_DIMS["audio"], 0.4, label="Classical", color="steelblue")
    ax[1].bar(1, 512, 0.4, label="CLAP", color="coral")
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["Classical\n(74D)", "CLAP\n(512D)"], fontsize=8)
    ax[1].set_title("Audio Feature Dimension", fontsize=9, fontweight="bold")
    ax[1].set_ylabel("Dimensions")
    ax[1].grid(True, alpha=0.3, axis="y")
    ax[1].legend(fontsize=7)


def create_video_feature_panel(ax, sample_data):
    """Compare video features: classical OpenFace vs LLaVA."""
    classical_vids = []
    llm_vids = []
    
    for pid in SAMPLE_IDS:
        cf = load_classical_features(pid)
        vis_parts = []
        for k in ["vis_au", "vis_eg", "vis_hp"]:
            if k in cf:
                vis_parts.append(cf[k])
        if vis_parts:
            combined = np.concatenate(vis_parts)
        else:
            combined = np.zeros(CLASSICAL_DIMS["video"])
        classical_vids.append(combined)
        
        lv = load_llm_features(pid, "L4", "video")
        llm_vids.append(lv if lv is not None else np.zeros(4096))
    
    bar_width = 0.15
    x = np.arange(len(SAMPLE_IDS))
    
    c_means = [float(a.mean()) for a in classical_vids]
    c_stds = [float(a.std()) for a in classical_vids]
    l_means = [float(a.mean()) for a in llm_vids]
    l_stds = [float(a.std()) for a in llm_vids]
    
    ax[0].bar(x - bar_width/2, c_means, bar_width,
              yerr=c_stds, label="Classical (31D)", color="steelblue", alpha=0.8, capsize=3)
    ax[0].bar(x + bar_width/2, l_means, bar_width,
              yerr=l_stds, label="LLM LLaVA (4096D)", color="coral", alpha=0.8, capsize=3)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(SAMPLE_IDS, fontsize=8)
    ax[0].set_title("Video Feature Mean ± Std", fontsize=9, fontweight="bold")
    ax[0].legend(fontsize=7)
    ax[0].grid(True, alpha=0.3, axis="y")
    
    # Dimensionality comparison
    ax[1].bar(0, CLASSICAL_DIMS["video"], 0.4, label="Classical", color="steelblue")
    ax[1].bar(1, 4096, 0.4, label="LLaVA", color="coral")
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["Classical\n(31D)", "LLaVA\n(4096D)"], fontsize=8)
    ax[1].set_title("Video Feature Dimension", fontsize=9, fontweight="bold")
    ax[1].set_ylabel("Dimensions")
    ax[1].grid(True, alpha=0.3, axis="y")
    ax[1].legend(fontsize=7)


def create_transcript_excerpt_table(ax, sample_data):
    """Show transcript excerpts for each participant."""
    ax.axis("off")
    
    col_widths = [0.8, 3.5, 3.5, 3.5, 3.5]
    rows = []
    
    # Header
    rows.append(["", *SAMPLE_IDS])
    
    # PHQ-8 row
    rows.append(["PHQ-8", *[str(s["phq8"]) for s in sample_data]])
    
    # Label row
    lbls = ["DEP" if s["label"] == 1 else "CTRL" for s in sample_data]
    rows.append(["Label", *lbls])
    
    # Transcript excerpts (first 3 participant turns)
    excerpts_row = ["Transcript\nExcerpt"]
    for pid in SAMPLE_IDS:
        transcript = [s["transcript"] for s in sample_data if s.get("id") == pid]
        if transcript:
            t = transcript[0]
            lines = t.strip().split("\n")
            p_lines = [l for l in lines if "Participant" in l][:3]
            excerpts = []
            for pl in p_lines:
                parts = pl.split("\t")
                if len(parts) >= 4:
                    excerpts.append(parts[3].strip())
            excerpt_text = "\n".join([f"• {e[:70]}" for e in excerpts])
        else:
            excerpt_text = "N/A"
        excerpts_row.append(excerpt_text)
    rows.append(excerpts_row)
    
    # Create table
    table = ax.table(cellText=rows[1:], colLabels=rows[0],
                     loc="center", cellLoc="left",
                     colWidths=col_widths)
    
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.8)
    
    # Style header
    for j in range(len(rows[0])):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold", fontsize=8)
        cell.set_facecolor("#e8e8e8")
    
    # Color code PHQ-8 row (row 1 in table = index 1)
    for j in range(1, len(rows[0])):
        cell = table[1, j]
        phq = float(sample_data[j-1].get("phq8", 0) or 0)
        if phq >= 15:
            cell.set_facecolor("#f8d7da")
        elif phq >= 10:
            cell.set_facecolor("#fff3cd")
        elif phq > 0:
            cell.set_facecolor("#d4edda")
        else:
            cell.set_facecolor("#e2e3e5")
    
    # Color label row
    for j in range(1, len(rows[0])):
        cell = table[2, j]
        if sample_data[j-1]["label"] == 1:
            cell.set_facecolor("#f8d7da")
        else:
            cell.set_facecolor("#d4edda")
    
    # Style excerpt cells
    transcript_row = len(rows) - 1  # last row
    for j in range(1, len(rows[0])):
        cell = table[transcript_row, j]
        cell.set_text_props(fontsize=6)
    
    ax.set_title("Sample Participant Overview", fontsize=11, fontweight="bold", pad=20)


def create_feature_evolution_summary(ax, sample_data):
    """Summary comparing feature spaces across all levels."""
    ax.axis("off")
    
    # Build summary text
    summary_lines = []
    summary_lines.append("Feature Evolution Summary (L0 → L1/L3/L4)")
    summary_lines.append("=" * 55)
    summary_lines.append("")
    
    # Text (L0→L1)
    summary_lines.append("[TEXT] (L0 -> L1): Classical -> Mistral 7B")
    summary_lines.append(f"  Dimension: 384D → 4096D ({4096/384:.0f}× larger)")
    summary_lines.append(f"  Model size: ~125M → 7.2B params")
    summary_lines.append(f"  DAIC AUROC gain: +{0.6775-0.5471:+.1%}")
    summary_lines.append("")
    
    # Audio (L0→L3) 
    summary_lines.append("[AUDIO] (L0 -> L3): WavLM -> CLAP")
    summary_lines.append(f"  Dimension: 74D → 512D ({512/74:.0f}× larger)")
    summary_lines.append(f"  DAIC AUROC gain: +{0.6522-0.5471:+.1%}")
    summary_lines.append("")
    
    # Video (L0→L4)
    summary_lines.append("[VIDEO] (L0 -> L4): OpenFace -> LLaVA-7B")
    summary_lines.append(f"  Dimension: 31D → 4096D ({4096/31:.0f}× larger)")
    summary_lines.append(f"  DAIC AUROC gain: +{0.6341-0.5471:+.1%}")
    summary_lines.append("")
    
    # Full stack (L5)
    summary_lines.append("[FULL STACK] (L5): Mistral + CLAP + LLaVA")
    summary_lines.append(f"  DAIC AUROC gain: +{0.6667-0.5471:+.1%} (vs L0)")
    summary_lines.append(f"  MOSEI Sent CCC: 0.6223")
    summary_lines.append(f"  FI Avg CCC: 0.5195")
    summary_lines.append("")
    
    summary_lines.append("═" * 55)
    summary_lines.append("Key finding: LLM text (Mistral alone, L1) is the")
    summary_lines.append("single best modality upgrade (+13.0% AUROC).")
    summary_lines.append("Full stack (L5) underperforms L1 alone on DAIC")
    summary_lines.append("but provides more balanced multi-task results.")
    
    text = "\n".join(summary_lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=8, fontfamily="monospace",
            verticalalignment="top", linespacing=1.3)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 8 — Sample-Level Feature Evolution Visualization")
    print("=" * 60)
    
    # Load data for samples
    sample_data = []
    for pid in SAMPLE_IDS:
        print(f"Loading participant {pid}...")
        transcript = load_transcript(pid)
        classical = load_classical_features(pid)
        
        sd = {
            "id": pid,
            **SAMPLE_LABELS[pid],
            "transcript": transcript,
            "classical": classical,
            "llm_text": load_llm_features(pid, "L1", "text"),
            "llm_audio": load_llm_features(pid, "L3", "audio"),
            "llm_video": load_llm_features(pid, "L4", "video"),
        }
        sample_data.append(sd)
    
    # ── Figure 1: Main feature evolution (multi-panel) ──
    print("Creating main sample feature evolution figure...")
    fig = plt.figure(figsize=(20, 22))
    
    # Grid: 
    # Row 0: Data cards (spanning full width)
    # Row 1: Text feature PCA (2 columns)
    # Row 2: Audio comparison (2 columns)
    # Row 3: Video comparison (2 columns)
    # Row 4: Feature comparison bars (3 columns)
    # Row 5: Summary (full width)
    
    gs = fig.add_gridspec(6, 6, hspace=0.35, wspace=0.3,
                          height_ratios=[1.0, 1.2, 1.0, 1.0, 0.8, 1.2])
    
    # Row 0: Data cards
    ax_cards = fig.add_subplot(gs[0, :])
    create_sample_data_cards(ax_cards, sample_data)
    
    # Row 1: Text PCA
    ax_text_pca = [fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:])]
    create_text_feature_panel(ax_text_pca, sample_data)
    
    # Row 2: Audio
    ax_audio = [fig.add_subplot(gs[2, 0:3]), fig.add_subplot(gs[2, 3:])]
    create_audio_feature_panel(ax_audio, sample_data)
    
    # Row 3: Video
    ax_video = [fig.add_subplot(gs[3, 0:3]), fig.add_subplot(gs[3, 3:])]
    create_video_feature_panel(ax_video, sample_data)
    
    # Row 4: Feature comparison bars
    ax_bars = [fig.add_subplot(gs[4, 0:2]), fig.add_subplot(gs[4, 2:4]), fig.add_subplot(gs[4, 4:])]
    create_feature_comparison_bars(ax_bars, sample_data)
    
    # Row 5: Summary
    ax_summary = fig.add_subplot(gs[5, :])
    create_feature_evolution_summary(ax_summary, sample_data)
    
    # Suptitle
    fig.suptitle("DAIC Sample-Level Feature Evolution Across LLM Levels (Phase 8)",
                 fontsize=14, fontweight="bold", y=0.98)
    
    # Save
    path = FIG_DIR / "sample_feature_evolution.png"
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    
    # ── Figure 2: Transcript excerpts table ──
    print("Creating transcript excerpt table...")
    fig2, ax2 = plt.subplots(figsize=(16, 5))
    create_transcript_excerpt_table(ax2, sample_data)
    path2 = FIG_DIR / "sample_transcript_table.png"
    fig2.savefig(str(path2), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  Saved: {path2}")
    
    print("\n✅ Sample visualization complete!")
    print(f"  Figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
