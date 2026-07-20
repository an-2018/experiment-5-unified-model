#!/usr/bin/env python3
"""Regenerate the supplementary 'Full Ablation Ladder' table mechanically from
the current source-of-truth CSVs, so every cell is traceable to a file instead
of hand-transcribed (several hand-transcribed cells were found to have drifted
from their sources during a reference/results audit — see review3.md)."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "artifacts" / "tables"


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def get(rows, **filters):
    for r in rows:
        if all(r.get(k) == v for k, v in filters.items()):
            return r
    return None


def main():
    unimodal = load_csv(TABLES / "unimodal_baselines.csv")
    fusion = load_csv(TABLES / "fusion_baselines.csv")
    mmoe = load_csv(TABLES / "mmoe_ex_results.csv")
    knn = load_csv(TABLES / "knn_voting_results.csv")
    ggmoe = load_csv(TABLES / "ggmoe_results.csv")

    def mmoe_val(metric):
        r = get(mmoe, metric=metric)
        return float(r["value"]) if r else None

    def knn_val(metric):
        r = get(knn, metric=metric)
        return float(r["value"]) if r else None

    def ggmoe_val(variant, col):
        r = get(ggmoe, variant=variant)
        return float(r[col]) if r else None

    def best_unimodal(dataset, metric):
        best_mod, best_v = None, None
        for r in unimodal:
            if r["dataset"] == dataset and r["metric"] == metric:
                v = float(r["value"])
                if best_v is None or v > best_v:
                    best_v, best_mod = v, r["modality"]
        return best_v, best_mod

    def gated_fusion(dataset, metric):
        r = get(fusion, dataset=dataset, fusion_type="gated", metric=metric)
        return float(r["value"]) if r else None

    daic_uni, daic_uni_mod = best_unimodal("daic", "AUROC")
    mosei_uni, mosei_uni_mod = best_unimodal("mosei", "CCC")
    fi_uni, fi_uni_mod = best_unimodal("fi", "Avg_CCC")

    rows = [
        ("0", "Trivial (majority class)", 0.500, 0.000, 0.500, 0.000),
        ("1", f"+ Unimodal (best modality: {daic_uni_mod}/{mosei_uni_mod}/{fi_uni_mod})",
         daic_uni, mosei_uni, None, fi_uni),
        ("2", "+ Gated late fusion",
         gated_fusion("daic", "AUROC"), gated_fusion("mosei", "CCC"),
         gated_fusion("mosei_emotion", "Avg Emotion AUROC"), gated_fusion("fi", "Avg CCC")),
        ("3", "+ MMoEEx (no graph)",
         mmoe_val("daic_auroc"), mmoe_val("mosei_sentiment_ccc"),
         mmoe_val("mosei_emotion_auc"), mmoe_val("fi_avg_ccc")),
        ("4", "+ KNN voting (no learned router)",
         knn_val("daic_auroc"), knn_val("mosei_sentiment_ccc"),
         knn_val("mosei_emotion_auc"), knn_val("fi_avg_ccc")),
        ("5", "+ Graph router (V0, inductive $K{=}10$)",
         ggmoe_val("V0", "daic_auroc"), ggmoe_val("V0", "mosei_sentiment_ccc"),
         ggmoe_val("V0", "mosei_emotion_auc"), ggmoe_val("V0", "fi_avg_ccc")),
        ("6", "+ Graph router (V3, inductive $K{=}15$)",
         ggmoe_val("V3", "daic_auroc"), ggmoe_val("V3", "mosei_sentiment_ccc"),
         ggmoe_val("V3", "mosei_emotion_auc"), ggmoe_val("V3", "fi_avg_ccc")),
    ]

    print(f"{'idx':<4}{'row':<45}{'DAIC AUROC':<12}{'MOSEI CCC':<12}{'MOSEI Emo':<12}{'FI CCC':<10}")
    for idx, label, daic, mosei_s, mosei_e, fi in rows:
        def fmt(v):
            return f"{v:.3f}" if v is not None else "--"
        print(f"{idx:<4}{label:<45}{fmt(daic):<12}{fmt(mosei_s):<12}{fmt(mosei_e):<12}{fmt(fi):<10}")

    print("\nLaTeX rows:")
    for idx, label, daic, mosei_s, mosei_e, fi in rows:
        def fmt(v):
            return f"{v:.3f}" if v is not None else "--"
        print(f"{idx} & {label} & {fmt(daic)} & {fmt(mosei_s)} & {fmt(mosei_e)} & {fmt(fi)} \\\\")


if __name__ == "__main__":
    main()
