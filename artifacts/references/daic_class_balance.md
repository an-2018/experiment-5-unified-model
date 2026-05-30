# DAIC Class Balance Analysis

## Summary
DAIC has a **mild imbalance**, NOT severe. 70.4% non-depressed vs 29.6% depressed. No SMOTE needed for neural network embeddings — weighted BCE loss is the standard approach.

## Class Balance by Split

| Split | Depressed (PHQ-8 ≥ 10) | Non-depressed (PHQ-8 < 10) | Total | Ratio |
|-------|------------------------|---------------------------|-------|-------|
| Train (n=107) | 30 (28.0%) | 77 (72.0%) | 107 | 1:2.6 |
| Val (n=35) | 12 (34.3%) | 23 (65.7%) | 35 | 1:1.9 |
| Test (n=47) | 14 (29.8%) | 33 (70.2%) | 47 | 1:2.4 |
| **Overall** | **56 (29.6%)** | **133 (70.4%)** | **189** | **1:2.4** |

## Notes
- PHQ-8 score range: 0-23, mean=6.75, std=5.92
- Clinical threshold: PHQ-8 ≥ 10 indicates depression
- Literature (Zou et al. 2025) describes DAIC as "69% healthy vs 31% depressed" — handled with weighted loss, no SMOTE
- SMOTE is NOT appropriate for deep neural network embeddings (BERT/Wav2Vec/ViT features)
- Standard approach: weighted BCEWithLogitsLoss with pos_weight ~ 2.4

## Source
Generated from: `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/daic/metadata.csv`