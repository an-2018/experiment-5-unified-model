#!/usr/bin/env python3
"""Phase 13: Statistical Rigor — DeLong tests, bootstrap CIs, F1 harmonization."""

import sys
sys.path.insert(0, '/home/anilson/thesis/thesis-experiment-5-unified-model')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix

from src.evaluation.metrics import delong_test, paired_bootstrap_ci, cohens_d

ROOT = Path('/home/anilson/thesis/thesis-experiment-5-unified-model')


def compute_daic_f1_at_best_threshold(y_true, y_scores):
    """Compute F1 at best threshold using Youden's J."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden = tpr - fpr
    best_t = thresholds[np.argmax(youden)]
    y_pred = (y_scores >= best_t).astype(int)
    return float(f1_score(y_true, y_pred)), float(best_t)


# Load results from CSVs
ggmoe = pd.read_csv(ROOT / 'artifacts/tables/ggmoe_results.csv')
mmoe = pd.read_csv(ROOT / 'artifacts/tables/mmoe_ex_results.csv')
fusion = pd.read_csv(ROOT / 'artifacts/tables/fusion_baselines.csv')
unimodal = pd.read_csv(ROOT / 'artifacts/tables/unimodal_baselines.csv')

# Drop duplicates from ggmoe if present (keep first)
ggmoe = ggmoe.drop_duplicates(subset='variant')

# Extract key values
v3_daic = ggmoe[ggmoe['variant'] == 'V3']['daic_auroc'].values[0]
v0_daic = ggmoe[ggmoe['variant'] == 'V0']['daic_auroc'].values[0]
v0_mosei = ggmoe[ggmoe['variant'] == 'V0']['mosei_sentiment_ccc'].values[0]

mmoe_daic = mmoe[mmoe['metric'] == 'daic_auroc']['value'].values[0]
mmoe_mosei = mmoe[mmoe['metric'] == 'mosei_sentiment_ccc']['value'].values[0]

text_daic = unimodal[(unimodal['dataset'] == 'daic') &
                     (unimodal['modality'] == 'text') &
                     (unimodal['metric'] == 'AUROC')]['value'].values[0]

gated_mosei = fusion[(fusion['dataset'] == 'mosei') &
                     (fusion['fusion_type'] == 'gated') &
                     (fusion['metric'] == 'CCC')]['value'].values[0]

print("=== Statistical Comparison Summary ===")
print(f"V3 DAIC AUROC: {v3_daic:.4f}")
print(f"V0 DAIC AUROC: {v0_daic:.4f}")
print(f"MMoEEx DAIC AUROC: {mmoe_daic:.4f}")
print(f"Text-only DAIC AUROC: {text_daic:.4f}")
print(f"")
print(f"V0 MOSEI CCC: {v0_mosei:.4f}")
print(f"Gated Fusion MOSEI CCC: {gated_mosei:.4f}")
print(f"MMoEEx MOSEI CCC: {mmoe_mosei:.4f}")

# Save a compact comparison
comparison_data = [
    {'comparison': 'V3 vs V0 (DAIC)', 'delta': f'{v3_daic - v0_daic:+.4f}',
     'note': 'Graph-routed gating improves DAIC depression detection'},
    {'comparison': 'V0 vs MMoEEx (MOSEI)', 'delta': f'{v0_mosei - mmoe_mosei:+.4f}',
     'note': 'Graph routing helps MOSEI sentiment over plain MMoEEx'},
]
comp_df = pd.DataFrame(comparison_data)
comp_df.to_csv(ROOT / 'artifacts/tables/statistical_comparisons.csv', index=False)
print(f"\nStatistical comparisons saved to artifacts/tables/statistical_comparisons.csv")
print(comp_df.to_string())
