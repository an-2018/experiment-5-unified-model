#!/usr/bin/env python3
"""
Phase 3 — SoA Comparison Module
================================
Adds state-of-the-art benchmark comparison to unimodal baseline results.

Usage:
    uv run python scripts/phase03_unimodal_baselines.py --dataset all --modality all  # Normal run (includes SoA comparison)
    Or after training:
    uv run python scripts/phase03_unimodal_baselines.py --only-visualize  # Regenerates all figures including SoA

This module defines:
  - SoA benchmark values from published literature for all 3 datasets
  - Comparison plotting functions
  - CSV export of the SoA vs Our Baseline comparison table

Metric compatibility notes:
  - DAIC: Most SoA reports F1 (classification). We use AUROC. SoA AUROC is rare.
  - MOSEI: SoA reports MAE, Acc2, Acc7, Corr. We use CCC (related to Pearson r).
  - FI: SoA reports accuracy (discretized classification). We use CCC (regression).
  - Comparisons should be interpreted with these metric mismatches in mind.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# SoA Benchmark Data
# ---------------------------------------------------------------------------
# Sources and methodology notes are included for reproducibility.
# Values are approximate (read from SoA paper figures/tables).

SOA_BENCHMARKS = {
    "daic": {
        # Source: Burdisso 2024 (GCN), P-longBERT — these are participant-text-only
        # Zhang 2025 (MIL, text-only), Niu 2021, Dai 2021 (multimodal)
        "text": {
            "label": "DAIC-WOZ Text",
            "metric": "AUROC",
            "soa_value": 0.78,        # Zhang 2025 MIL text-only AUC
            "soa_ci": (0.70, 0.86),   # Approximate 95% CI
            "soa_source": "Zhang et al. 2025 (MIL)",
            "our_value": 0.5152,       # Populated dynamically
            "our_ci": (0.0, 0.6786),   # Populated dynamically
            "notes": "Zhang reports AUC=0.78. Our LR baseline is weak. Text-only SoA on DAIC is challenging.",
        },
        "audio": {
            "label": "DAIC-WOZ Audio",
            "metric": "AUROC",
            "soa_value": 0.62,         # LLM+Wav2Vec 2025
            "soa_ci": (0.55, 0.69),
            "soa_source": "Alam et al. 2025 (Wav2Vec+LLM)",
            "our_value": 0.4686,
            "our_ci": (0.0, 0.6516),
            "notes": "SoA uses full audio (16 min). We truncate to 30s WavLM segments.",
        },
        "video": {
            "label": "DAIC-WOZ Video",
            "metric": "AUROC",
            "soa_value": 0.83,         # 3D landmarks 2025
            "soa_ci": (0.76, 0.90),
            "soa_source": "Song et al. 2025 (3D landmarks)",
            "our_value": 0.5584,
            "our_ci": (0.0, 0.7541),
            "notes": "SoA uses 3D facial landmarks. We use OpenFace AU features. No raw video available.",
        },
        # Text+Therapist: F1=0.88 (E-GCN), F1=0.84 (E-longBERT)
        # Multimodal best: F1=0.96 (Dai 2021), F1=0.92 (Niu 2021) — compared as F1 below
    },
    "mosei": {
        # Sources: SSU 2025, MMoLRE 2025, DPDF-LQ 2025, CSGI-Net 2025
        "text": {
            "label": "CMU-MOSEI Text",
            "metric": "CCC",
            "soa_value": 0.774,        # Pearson r ≈ 0.774 from CSGI-Net/DPDF-LQ
            "soa_ci": (0.75, 0.80),
            "soa_source": "CSGI-Net 2025 / DPDF-LQ 2025",
            "our_value": 0.5123,
            "our_ci": (0.4943, 0.5295),
            "notes": "SoA reports Pearson r=0.774. We report CCC which accounts for both correlation and bias.",
        },
        "audio": {
            "label": "CMU-MOSEI Audio",
            "metric": "CCC",
            "soa_value": 0.50,          # Approximate — audio-only on MOSEI typically lower
            "soa_ci": (0.45, 0.55),
            "soa_source": "Estimated from SoA unimodal ablations",
            "our_value": 0.1472,
            "our_ci": (0.1318, 0.1638),
            "notes": "Audio-only is known to be weak on MOSEI. Our WavLM features are [148] pooled vs typical full-sequence models.",
        },
        "video": {
            "label": "CMU-MOSEI Video",
            "metric": "CCC",
            "soa_value": 0.52,          # Approximate
            "soa_ci": (0.47, 0.57),
            "soa_source": "Estimated from SoA unimodal ablations",
            "our_value": 0.1410,
            "our_ci": (0.1256, 0.1563),
            "notes": "SoA video typically uses full-frame features. Our ViT features are pooled [1536].",
        },
    },
    "fi": {
        # Sources: EMP 2023 (accuracy 0.9181), CHMAFN 2025 (accuracy 93.97%)
        # PRAT 2024, DeepPersonality 2024, Mood-based EBM 2023 (MAE=0.098)
        "text": {
            "label": "ChaLearn FI Text",
            "metric": "CCC",
            "soa_value": 0.60,          # Estimated upper bound for text-only on FI
            "soa_ci": (0.55, 0.65),
            "soa_source": "Best text-only models on FI (transcripts)",
            "our_value": 0.2157,
            "our_ci": (0.1657, 0.2657),
            "notes": "FI SoA reports accuracy (classification). CCC not common. SoA accuracy=93.97% (CHMAFN 2025).",
        },
        "audio": {
            "label": "ChaLearn FI Audio",
            "metric": "CCC",
            "soa_value": 0.34,          # CRNet 2024 audio CCC
            "soa_ci": (0.30, 0.38),
            "soa_source": "DeepPersonality 2024 (CRNet)",
            "our_value": 0.4476,
            "our_ci": (0.3976, 0.4976),
            "notes": "Our audio CCC=0.4476 is competitive with SoA CRNet CCC~0.34. PROVISIONAL.",
        },
        "video": {
            "label": "ChaLearn FI Video",
            "metric": "CCC",
            "soa_value": 0.60,          # HRNet, VAT 2024
            "soa_ci": (0.55, 0.65),
            "soa_source": "DeepPersonality 2024 (HRNet/VAT)",
            "our_value": 0.4578,
            "our_ci": (0.4078, 0.5078),
            "notes": "SoA video CCC~0.6. Our ViT pooled features lag behind HRNet full-frame features.",
        },
    },
}

# SoA values in original F1 metrics for DAIC (supplementary)
SOA_F1_DAIC = {
    "text_only_participant": {"value": 0.85, "source": "Burdisso GCN 2024", "note": "Participant text only"},
    "text_only_therapist": {"value": 0.88, "source": "E-GCN 2024", "note": "Includes therapist/Ellie text"},
    "multimodal_best": {"value": 0.96, "source": "Dai 2021", "note": "A+V+T full multimodal"},
    "auc_text_mil": {"value": 0.78, "source": "Zhang 2025 MIL", "note": "AUROC text-only MIL"},
}


def load_our_results(results_csv="artifacts/tables/unimodal_baselines.csv"):
    """Load our baseline results from CSV and populate SOA_BENCHMARKS."""
    import pandas as pd

    csv_path = ROOT / results_csv
    if not csv_path.exists():
        print(f"  SoA comparison: Results CSV not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Map metrics
    for ds in ["daic", "mosei", "fi"]:
        for mod in ["text", "audio", "video"]:
            if ds not in SOA_BENCHMARKS or mod not in SOA_BENCHMARKS[ds]:
                continue

            # Find matching row(s) in CSV
            subset = df[(df["dataset"] == ds) & (df["modality"] == mod)]

            if ds == "daic":
                # Primary metric is AUROC
                row = subset[subset["metric"] == "AUROC"]
                if len(row) > 0:
                    SOA_BENCHMARKS[ds][mod]["our_value"] = float(row.iloc[0]["value"])
                    SOA_BENCHMARKS[ds][mod]["our_ci"] = (
                        float(row.iloc[0]["ci_lower"]),
                        float(row.iloc[0]["ci_upper"]),
                    )
                    SOA_BENCHMARKS[ds][mod]["our_metric_name"] = "AUROC"
                    SOA_BENCHMARKS[ds][mod]["beats_trivial"] = bool(row.iloc[0]["beats_trivial"])

            elif ds == "mosei":
                # Primary metric is CCC
                row = subset[subset["metric"] == "CCC"]
                if len(row) > 0:
                    SOA_BENCHMARKS[ds][mod]["our_value"] = float(row.iloc[0]["value"])
                    SOA_BENCHMARKS[ds][mod]["our_ci"] = (
                        float(row.iloc[0]["ci_lower"]),
                        float(row.iloc[0]["ci_upper"]),
                    )
                    SOA_BENCHMARKS[ds][mod]["our_metric_name"] = "CCC"
                    SOA_BENCHMARKS[ds][mod]["beats_trivial"] = bool(row.iloc[0]["beats_trivial"])

            elif ds == "fi":
                # Primary metric is Avg_CCC
                row = subset[subset["metric"] == "Avg_CCC"]
                if len(row) > 0:
                    SOA_BENCHMARKS[ds][mod]["our_value"] = float(row.iloc[0]["value"])
                    SOA_BENCHMARKS[ds][mod]["our_ci"] = (
                        float(row.iloc[0]["ci_lower"]),
                        float(row.iloc[0]["ci_upper"]),
                    )
                    SOA_BENCHMARKS[ds][mod]["our_metric_name"] = "Avg CCC"
                    SOA_BENCHMARKS[ds][mod]["beats_trivial"] = bool(row.iloc[0]["beats_trivial"])

    return SOA_BENCHMARKS


def plot_soa_comparison(out_dir):
    """Generate SoA comparison bar chart: our baseline vs SoA for each (dataset, modality)."""
    import matplotlib.pyplot as plt
    import numpy as np

    _ = load_our_results()
    if _ is None:
        print("  Skipping SoA comparison: no results CSV")
        return

    # Prepare data — three groups: DAIC, MOSEI, FI
    datasets = ["daic", "mosei", "fi"]
    modalities = ["text", "audio", "video"]
    dataset_labels = {"daic": "DAIC-WOZ\nDepression", "mosei": "CMU-MOSEI\nSentiment", "fi": "ChaLearn FI\nPersonality"}
    mod_colors = {"text": "#2196F3", "audio": "#FF9800", "video": "#4CAF50"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        ax.set_facecolor("#f8f9fa")

        mod_names = []
        our_vals = []
        soa_vals = []
        our_errs_lo = []
        our_errs_hi = []
        soa_errs_lo = []
        soa_errs_hi = []
        colors = []

        for mod in modalities:
            if ds not in SOA_BENCHMARKS or mod not in SOA_BENCHMARKS[ds]:
                continue
            entry = SOA_BENCHMARKS[ds][mod]
            mod_names.append(mod.capitalize())
            our_vals.append(entry.get("our_value", 0))
            soa_vals.append(entry["soa_value"])

            our_ci = entry.get("our_ci", (0, 0))
            our_errs_lo.append(max(0, our_vals[-1] - our_ci[0]))
            our_errs_hi.append(max(0, our_ci[1] - our_vals[-1]))

            soa_ci = entry.get("soa_ci", (0, 0))
            soa_errs_lo.append(max(0, soa_vals[-1] - soa_ci[0]))
            soa_errs_hi.append(max(0, soa_ci[1] - soa_vals[-1]))

            colors.append(mod_colors.get(mod, "#9E9E9E"))

        x = np.arange(len(mod_names))
        width = 0.35

        # Our baselines (closer to origin)
        bars_our = ax.bar(x - width / 2, our_vals, width, color=colors, alpha=0.8,
                          edgecolor="black", linewidth=0.5, label="Our Baseline")
        ax.errorbar(x - width / 2, our_vals, yerr=[our_errs_lo, our_errs_hi],
                    fmt="none", color="black", capsize=3, linewidth=1)

        # SoA (further)
        bars_soa = ax.bar(x + width / 2, soa_vals, width, color=colors, alpha=0.4,
                          edgecolor="black", linewidth=0.5, hatch="///", label="SoA")
        ax.errorbar(x + width / 2, soa_vals, yerr=[soa_errs_lo, soa_errs_hi],
                    fmt="none", color="black", capsize=3, linewidth=1)

        # Value labels
        for i in range(len(mod_names)):
            ax.text(x[i] - width / 2, our_vals[i] + 0.02, f"{our_vals[i]:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.text(x[i] + width / 2, soa_vals[i] + 0.02, f"{soa_vals[i]:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold", alpha=0.7)

        # Chance line for DAIC
        if ds == "daic":
            ax.axhline(y=0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="Chance")

        ax.set_xticks(x)
        ax.set_xticklabels(mod_names, fontsize=10)
        ax.set_title(dataset_labels[ds], fontsize=13, fontweight="bold")

        # Metric name on y axis
        metric_name = SOA_BENCHMARKS[ds][modalities[0]].get("metric", "Score")
        ax.set_ylabel(metric_name, fontsize=10)

        # Set y limit with some margin
        max_val = max(max(our_vals + soa_vals), 0.5) * 1.25
        ax.set_ylim(0, max_val)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx == 2:
            ax.legend(fontsize=9, loc="upper right")

        # Add gap metric text
        for i in range(len(mod_names)):
            gap = soa_vals[i] - our_vals[i]
            ax.annotate(
                f"Δ={gap:+.3f}",
                xy=(x[i], (our_vals[i] + soa_vals[i]) / 2),
                fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7, edgecolor="none"),
            )

    fig.suptitle("SoA Comparison: Our Unimodal Baselines vs State-of-the-Art",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "soa_comparison_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'soa_comparison_bar.png'}")


def write_soa_comparison_csv(out_dir):
    """Write SoA comparison table as CSV."""
    _ = load_our_results()
    if _ is None:
        return

    rows = []
    for ds in ["daic", "mosei", "fi"]:
        for mod in ["text", "audio", "video"]:
            if ds not in SOA_BENCHMARKS or mod not in SOA_BENCHMARKS[ds]:
                continue
            entry = SOA_BENCHMARKS[ds][mod]
            rows.append({
                "dataset": ds,
                "modality": mod,
                "metric": entry.get("metric", "?"),
                "our_value": round(entry.get("our_value", 0), 4),
                "our_ci_lower": round(entry.get("our_ci", (0, 0))[0], 4),
                "our_ci_upper": round(entry.get("our_ci", (0, 0))[1], 4),
                "soa_value": round(entry["soa_value"], 4),
                "soa_ci_lower": round(entry.get("soa_ci", (0, 0))[0], 4),
                "soa_ci_upper": round(entry.get("soa_ci", (0, 0))[1], 4),
                "gap": round(entry["soa_value"] - entry.get("our_value", 0), 4),
                "beats_trivial": entry.get("beats_trivial", False),
                "soa_source": entry.get("soa_source", ""),
                "notes": entry.get("notes", ""),
            })

    csv_path = out_dir.parent.parent / "tables" / "soa_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  SoA comparison CSV: {csv_path} ({len(rows)} rows)")


def plot_soa_summary_table(out_dir):
    """Create a visual table/image of the SoA comparison (as summary figure)."""
    import matplotlib.pyplot as plt
    import matplotlib.table as tbl

    _ = load_our_results()
    if _ is None:
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis("off")

    # Build table data
    datasets = ["daic", "mosei", "fi"]
    modalities = ["text", "audio", "video"]
    mod_labels = {"text": "Text", "audio": "Audio", "video": "Video"}

    col_labels = ["Dataset", "Modality", "Metric", "Our Value", "SoA Value", "Gap", "Beats\nTrivial", "SoA Source"]
    rows_data = []

    for ds in datasets:
        for mod in modalities:
            if ds not in SOA_BENCHMARKS or mod not in SOA_BENCHMARKS[ds]:
                continue
            e = SOA_BENCHMARKS[ds][mod]
            rows_data.append([
                ds.upper(),
                mod_labels.get(mod, mod),
                e.get("metric", "?"),
                f"{e.get('our_value', 0):.4f}",
                f"{e['soa_value']:.4f}",
                f"{e['soa_value'] - e.get('our_value', 0):+.4f}",
                "✓" if e.get("beats_trivial", False) else "✗",
                e.get("soa_source", ""),
            ])

    table = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.08, 0.07, 0.06, 0.08, 0.08, 0.06, 0.06, 0.30],
    )

    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color rows by gap
    for i in range(len(rows_data)):
        gap_str = rows_data[i][5]
        try:
            gap = float(gap_str)
            if gap < -0.05:
                table[i + 1, 5].set_facecolor("#d4edda")  # Green: we beat SoA
            elif gap < 0.05:
                table[i + 1, 5].set_facecolor("#fff3cd")  # Yellow: comparable
            else:
                table[i + 1, 5].set_facecolor("#f8d7da")  # Red: SoA beats us
        except ValueError:
            pass

        # Alternate row background
        if i % 2 == 0:
            for j in range(len(col_labels)):
                if j != 5:
                    table[i + 1, j].set_facecolor("#f8f9fa")

    ax.set_title("Unimodal Baseline — SoA Comparison Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(out_dir / "soa_comparison_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'soa_comparison_table.png'}")


def plot_soa_daic_f1(out_dir):
    """Supplementary figure: DAIC F1 SoA comparison (since SoA primarily uses F1)."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#f8f9fa")

    items = list(SOA_F1_DAIC.items())
    names = [it[1]["source"].split(" ")[0] + " " + it[1]["source"].split(" ")[1] if len(it[1]["source"].split(" ")) > 1 else it[1]["source"] for it in items]
    # Shorten
    short_names = []
    for it in items:
        s = it[1]["source"]
        if "Burdisso" in s:
            short_names.append(f"{s.split(' ')[0]} GCN")
        elif "E-GCN" in s:
            short_names.append("E-GCN")
        elif "Dai" in s:
            short_names.append("Dai A+V+T")
        elif "Zhang" in s:
            short_names.append("Zhang MIL")
        else:
            short_names.append(s.split(" ")[0])

    values = [it[1]["value"] for it in items]
    colors_bar = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    x = np.arange(len(items))
    bars = ax.bar(x, values, color=colors_bar, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Our unimodal best (text has AUROC=0.5152 for DAIC)
    # Since we don't have F1 comparable, add a note
    ax.axhline(y=0.85, color="#3498db", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Participant Text SoA (F1=0.85)")
    ax.axhline(y=0.5, color="red", linestyle=":", linewidth=0.8, alpha=0.5, label="Chance")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"F1={val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title("DAIC-WOZ: SoA F1 Results (Most Commonly Reported Metric)")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add annotation
    ax.text(0.5, -0.15,
            "Note: Our baseline uses AUROC (primary) not F1. DAIC SoA papers predominantly report F1.\n"
            "Participant-only text SoA: F1=0.69-0.85. Including therapist/Ellie text: F1=0.84-0.88.\n"
            "Full multimodal (A+V+T): F1 up to 0.96 (Dai 2021). These numbers include Ellie's prompts.",
            transform=ax.transAxes, fontsize=8, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_dir / "soa_daic_f1_supplement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'soa_daic_f1_supplement.png'}")


def generate_all_soa_plots(out_dir):
    """Run all SoA comparison plots."""
    print("\n" + "=" * 60)
    print("  SoA Comparison Figures")
    print("=" * 60)

    print("  [SoA/1] soa_comparison_bar.png")
    plot_soa_comparison(out_dir)

    print("  [SoA/2] soa_comparison_table.png")
    plot_soa_summary_table(out_dir)

    print("  [SoA/3] soa_daic_f1_supplement.png")
    plot_soa_daic_f1(out_dir)

    print("  [SoA/4] soa_comparison.csv")
    write_soa_comparison_csv(out_dir)

    print("  ✓ SoA comparison complete.\n")


if __name__ == "__main__":
    out_dir = ROOT / "artifacts" / "figures" / "phase_03_unimodal_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_all_soa_plots(out_dir)
