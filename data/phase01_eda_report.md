# Phase 1 EDA Report - Dataset Acquisition and Exploration

## Executive Summary

This report documents the exploratory data analysis for the Unified Multimodal Graph-Gated MoE Experiment (Experiment 5). Three datasets are used: **DAIC-WOZ** (clinical depression), **CMU-MOSEI** (sentiment), and **ChaLearn FI** (apparent personality).

## Dataset Counts Summary

| Dataset | Train | Val | Test | Total |
|---------|-------|-----|------|-------|
| DAIC-WOZ | 107 | 35 | 47 | 189 |
| CMU-MOSEI | 16,265 | 1,869 | 4,643 | 22,777 |
| ChaLearn FI | 6000 | 2000 | 2000 | 10000 |

## Key Findings

### 1. DAIC-WOZ (Depression - Primary Clinical Task)

- **Unit**: Session (single long interview per participant)
- **Evaluation**: Participant-level
- **Labels**: PHQ-8 score (0-24) and binary depression (PHQ-8 ≥ 10)
- **Depression rate**: 29.6% across all splits
- **Modalities**: Audio, Video, Text (all available for all participants)
- **Split verification**: ✅ Subject-independent - no participant ID overlap between train/val/test

### 2. CMU-MOSEI (Sentiment - Auxiliary Supervision)

- **Unit**: Utterance (~3-30 seconds each)
- **Evaluation**: Utterance-level
- **Labels**: Sentiment score (-3 to +3)
- **⚠️ CRITICAL**: MOSEI has 22,777 utterances vs DAIC's 189 sessions
- **Dominance ratio**: 120.5x larger than DAIC
- **Mitigation required**: Temperature-balanced or task-balanced sampling

### 3. ChaLearn FI (Apparent Personality - Auxiliary Supervision)

- **Unit**: Video clip (~15 seconds)
- **Evaluation**: Clip-level
- **Labels**: Big-Five personality traits (openness, conscientiousness, extraversion, agreeableness, neuroticism) normalized to [0, 1]
- **⚠️ IMPORTANT**: Apparent personality ≠ clinical depression. These are auxiliary supervision signals only.
- **Modalities**: Video (primary), Audio, Text

## Leakage Checks

| Check | Status | Details |
|-------|--------|---------|
| DAIC subject-independent splits | ✅ PASS | No participant ID overlap between splits |
| DAIC session-level aggregation | ✅ PASS | Labels inherited from session PHQ-8 |
| MOSEI utterance independence | ✅ PASS | Each utterance is independent |
| FI clip independence | ✅ PASS | Each clip is independent |

## MOSEI Dominance Concern

**YES - MOSEI dominance is a significant risk.**

- MOSEI: 22,777 utterances
- DAIC: 189 sessions
- Ratio: **120.5x**

**Recommended mitigation strategies:**

1. **Temperature-balanced sampling**: Oversample DAIC sessions, undersample MOSEI utterances
2. **Task-balanced sampling**: Ensure each batch has balanced representation from all datasets
3. **Weighted loss**: Give higher weight to DAIC samples in loss computation

## Missing Modality Analysis

- **DAIC**: 100% modality coverage (all participants have audio, video, text)
- **MOSEI**: 100% modality coverage (all utterances have all modalities)
- **FI**: ~95% modality coverage (estimated, some clips may have missing data)

## Label Distributions

### DAIC Binary Depression
- No Depression (PHQ-8 < 10): 133 (70.4%)
- Depression (PHQ-8 ≥ 10): 56 (29.6%)

### MOSEI Sentiment
- Positive (> 0): ~50%
- Neutral (= 0): ~10%
- Negative (< 0): ~40%

## Figures Generated

1. `01_label_distributions.png` - Overview of all label distributions
2. `02_daic_phq_analysis.png` - DAIC PHQ-8 histogram and binary class
3. `03_mosei_sentiment_analysis.png` - MOSEI sentiment distributions
4. `04_fi_big_five_analysis.png` - FI personality trait distributions and correlations
5. `05_duration_distributions.png` - Audio/video duration distributions
6. `06_transcript_lengths.png` - Transcript length distributions
7. `07_missing_modality_heatmap.png` - Missing modality patterns
8. `08_split_distributions.png` - Split verification plots

## Conclusion

All datasets are accessible and properly formatted. Subject-independent splits are verified for DAIC. The MOSEI dominance concern is significant and requires careful sampling strategy during training. The dataset contract is saved to `configs/dataset_contract.yaml`.

---
*Report generated: Phase 1 EDA for Unified Multimodal Graph-Gated MoE Experiment*
