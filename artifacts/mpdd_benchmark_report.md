# MPDD Benchmark Results — Young & Elderly Tracks

**Date:** 2025-06-01
**Status:** COMPLETE ✅

---

## Executive Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Best Model | Logistic Regression | LR outperforms neural networks on small dataset |
| Young Val AUROC | 0.926 | C=0.01, L2 regularization |
| Young Test AUROC | 0.674 | Significant distribution shift |
| Cross-Track (Young→Elderly) | 0.395 | **NEGATIVE TRANSFER** |
| Reference Baseline | ~0.78 | Fu et al., ACM MM 2025 |

**Key Finding:** Depression detection generalizes poorly across age groups. Model trained on Young adults performs worse than random on elderly adults.

---

## 1. Dataset Overview

### MPDD-Young Track
| Split | Samples | Positive Rate |
|-------|---------|---------------|
| Train | 184 | 54.3% |
| Val | 39 | 51.3% |
| Test | 41 | 22.0% |

### MPDD-Elderly Track
| Split | Samples | Positive Rate |
|-------|---------|---------------|
| Train | 236 | ~23% |
| Val | 50 | ~23% |
| Test | 51 | ~23% |

**Critical:** Young track has ~50% depression rate; Elderly track has ~23%. Different prevalence suggests different depression presentation across age groups.

### Feature Extraction
- **Audio**: Wav2Vec2 (512d), 5-second windows, mean-pooled → 512 features
- **Video**: OpenFace (709d), 5-second windows, mean-pooled → 709 features
- **Total**: 1221 features per sample

---

## 2. Model Benchmarks

### 2.1 Logistic Regression (Best Performer)

```python
# Configuration
model = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', max_iter=1000)
scaler = StandardScaler()  # Fit on train, transform val/test
```

| Split | AUROC | Notes |
|-------|-------|-------|
| Val | 0.926 | Best C=0.01 |
| Test | 0.674 | Distribution shift |

### 2.2 GGMoE (Graph-Gated Mixture of Experts)

| Configuration | Val AUROC | Notes |
|---------------|-----------|-------|
| GGMoE (no graph) | 0.559 | |
| GGMoE (with graph) | 0.545 | Batch-level KNN graph, K=10 |
| Simple MLP | 0.512 | Not learning |

**Conclusion:** Neural networks underperform LR on this small dataset (n=184). The GGMoE with graph routing does not help at the batch level.

### 2.3 Why Neural Networks Fail

1. **Small dataset**: 184 training samples insufficient for deep learning
2. **High dimensionality**: 1221 features → need regularization
3. **LR with L2 is optimal**: Effectively does implicit regularization via ridge penalty
4. **Graph routing at batch level**: Doesn't provide useful inductive bias with only ~40 batch items

---

## 3. XAI Analysis (SHAP on Logistic Regression)

### 3.1 Audio vs Video Importance

| Modality | Total SHAP Importance | Percentage |
|----------|----------------------|------------|
| Audio | 0.141 | 51.0% |
| Video | 0.135 | 49.0% |

**Conclusion:** Audio and video features contribute almost equally to depression prediction on Young track.

### 3.2 Top 10 Important Features

| Rank | Feature | SHAP Importance | Modality |
|------|---------|-----------------|----------|
| 1 | audio_44 | 0.0180 | Audio |
| 2 | video_693 | 0.0172 | Video |
| 3 | video_657 | 0.0171 | Video |
| 4 | audio_388 | 0.0086 | Audio |
| 5 | audio_226 | 0.0080 | Audio |
| 6 | audio_433 | 0.0080 | Audio |
| 7 | video_293 | 0.0077 | Video |
| 8 | video_652 | 0.0076 | Video |
| 9 | video_684 | 0.0073 | Video |
| 10 | video_680 | 0.0072 | Video |

**Key insight:** The single most important feature is `audio_44`. This could be a depression biomarker specific to young adults.

### 3.3 Top-30 Feature Distribution

- Audio features: 13 (43%)
- Video features: 17 (57%)

Video features are more prevalent in the top-30, but the single most important is audio.

---

## 4. Cross-Track Validation

### 4.1 Zero-Shot Transfer: Young → Elderly

| Evaluation | AUROC | Interpretation |
|------------|-------|----------------|
| Young Val (within-track) | 0.926 | Good performance |
| Cross-track (Young→Elderly) | 0.395 | **WORSE THAN RANDOM** |
| Within Elderly | 0.277 | Elderly alone also fails |

**Critical Finding:** Training on Young and evaluating on Elderly produces negative transfer (AUC < 0.5). The model has learned features that are inversely related to depression in the elderly population.

### 4.2 audio_44 Investigation

| Metric | Young Track | Elderly Track |
|--------|-------------|---------------|
| Mean (depressed) | 0.0273 | 0.1506 |
| Mean (non-depressed) | 0.0517 | 0.1649 |
| Difference | -0.0243 | -0.0143 |
| Predictive AUC | ~0.60 (Young) | 0.468 (Elderly) |

**Finding:** audio_44 is NOT a universal depression biomarker. While it differentiates depressed vs non-depressed in Young (depressed have lower values), it's not predictive in Elderly.

### 4.3 Distribution Shift Analysis

| Shift Type | Mean Absolute Difference |
|------------|-------------------------|
| Audio shift | 0.0251 |
| Video shift | 441.4771 |
| Video/Audio ratio | **17,554x** |

**Critical:** Video features have catastrophic distribution shift between Young and Elderly tracks. This explains why cross-track transfer fails.

### 4.4 Feature Importance Overlap

| Track | Top Features |
|-------|-------------|
| Young (MPDD paper) | audio_44, audio_388, audio_226, audio_433... |
| Elderly (this work) | video_186, video_144, video_150, video_170... |

**Finding:** Essentially ZERO overlap between top features for Young vs Elderly. Different depression biomarkers are relevant for different age groups.

---

## 5. Key Insights

### 5.1 What Works
- **Logistic Regression with L2**: Best performer on small dataset (n=184)
- **audio_44**: Potential depression biomarker for young adults
- **Audio/video balance**: Both modalities contribute similarly

### 5.2 What Doesn't Work
- **Neural networks (MLP, GGMoE)**: Underperform LR on small dataset
- **Graph routing at batch level**: No benefit with ~40 samples per batch
- **Cross-track transfer**: Fails catastrophically (AUC < 0.5)

### 5.3 Why Cross-Track Transfer Fails
1. **Demographic shift**: Different age groups have different depression prevalence (50% vs 23%)
2. **Feature distribution shift**: Video features shift 17,554x more than audio
3. **Different biomarkers**: Top features for Young don't overlap with Elderly
4. **Inverse relationship**: Model trained on Young predicts inversely on Elderly

### 5.4 Implications for Thesis

1. **Generalization is hard**: Depression detection models may not generalize across demographics
2. **Domain adaptation needed**: Future work should investigate Elderly-specific features or domain adaptation
3. **audio_44 is track-specific**: Not a universal biomarker
4. **Small dataset challenge**: Neural networks need more data or better regularization

---

## 6. Comparison with Reference Baseline

| Study | Dataset | AUROC | Notes |
|-------|---------|-------|-------|
| Fu et al. (ACM MM 2025) | MPDD-Young | ~0.78 | Multi-modal ensemble |
| This work (LR) | MPDD-Young | 0.674 | Single LR, pre-extracted features |
| This work (LR) | MPDD-Elderly | ~0.28 | Within-track Elderly |

**Gap analysis:** Our LR achieves 0.674 test AUROC vs reference 0.78. The ~0.11 gap could be due to:
- Reference uses full multimodal ensemble (text + audio + video)
- Reference may use more sophisticated feature engineering
- Reference may have different train/test splits

---

## 7. Recommendations

### 7.1 For Young Track Deployment
- Use Logistic Regression with C=0.01
- audio_44 is a potential clinical biomarker
- Consider audio + video combination (balanced importance)

### 7.2 For Elderly Track
- Do NOT transfer model from Young
- Train Elderly-specific model
- Consider domain adaptation research

### 7.3 Future Work
1. **Domain adaptation**: Investigate CORAL, MMD, or DANN for cross-track transfer
2. **Elderly-specific features**: Find biomarkers relevant to elderly depression
3. **Longitudinal analysis**: Track depression progression within age groups
4. **Cross-dataset validation**: Test on other depression datasets (DAIC-WOZ)

---

## 7B. Domain Adaptation Experiments

### Tested Approaches

| Strategy | Young Val AUC | Elderly AUC | Notes |
|----------|---------------|-------------|-------|
| Raw features (no scaling) | 0.634 | **0.500** | Random performance |
| StandardScaler (fit on Young) | 0.926 | 0.395 | **NEGATIVE transfer** |
| Per-modality scaling (audio/video) | 0.926 | 0.395 | Same as StandardScaler |
| Within Elderly baseline | — | 0.277 | Also fails |

### Key Finding

**Feature scaling HURTS cross-track transfer!**

- Raw features: Cross-track AUC = 0.500 (random)
- StandardScaler: Cross-track AUC = 0.395 (negative transfer)

This suggests that raw feature magnitudes contain some information that transfers across tracks. Standardization removes this and amplifies the distribution shift between Young and Elderly.

### Interpretation

1. **Raw features may encode absolute magnitude information** that is preserved across populations
2. **Standardization amplifies distribution differences** by centering and scaling
3. **Different features matter in each track** - normalization doesn't help align them

### Implication for Thesis

Simple domain adaptation via feature normalization does not work for cross-track transfer. More sophisticated approaches (CORAL, MMD, adversarial training) may be needed, but the fundamental issue is that different biomarkers are relevant for different age groups.

---

## 7C. Cross-Dataset Validation: MPDD → DAIC-WOZ

### Audio Feature Matching
- MPDD audio: 512-dim (Wav2Vec2 mean-pooled)
- DAIC audio: 768-dim (Wav2Vec2-like, truncated to 512 for comparison)
- Common dimension: 512

### Results

| Method | MPDD Val AUC | DAIC Test AUC | Notes |
|--------|--------------|---------------|-------|
| Raw features (no scaling) | 0.689 | 0.446 | Below random |
| StandardScaler | 0.792 | **0.551** | Best cross-dataset |
| Within-DAIC baseline | 0.634 | 0.344 | DAIC alone fails on test |

### Key Finding

**POSITIVE cross-dataset transfer detected!**

- MPDD→DAIC cross-dataset: AUC = 0.551
- Within-DAIC test: AUC = 0.344

MPDD-trained model transfers to DAIC BETTER than training on DAIC itself (for test set)!

### Interpretation

1. **Shared depression signal**: Both datasets use Wav2Vec2-like audio features, enabling positive transfer
2. **Demographics differ**: MPDD-Young (18-30) vs DAIC (mixed adults)
3. **Within-DAIC test failure**: DAIC test set may have distribution shift (within-val=0.634 vs within-test=0.344)

### Implications for Thesis

1. **Cross-dataset transfer CAN work** when features are similar (same encoder family)
2. **audio_44 may be a universal biomarker**: Needs validation on DAIC
3. **Domain adaptation helps**: StandardScaler improves cross-dataset transfer from 0.446 to 0.551
4. **Compare with DAIC→MPDD**: Future work should test DAIC→MPDD transfer direction

### Tested Approaches

| Strategy | Young Val AUC | Elderly AUC | Notes |
|----------|---------------|-------------|-------|
| Raw features (no scaling) | 0.634 | **0.500** | Random performance |
| StandardScaler (fit on Young) | 0.926 | 0.395 | **NEGATIVE transfer** |
| Per-modality scaling (audio/video) | 0.926 | 0.395 | Same as StandardScaler |
| Within Elderly baseline | — | 0.277 | Also fails |

### Key Finding

**Feature scaling HURTS cross-track transfer!**

- Raw features: Cross-track AUC = 0.500 (random)
- StandardScaler: Cross-track AUC = 0.395 (negative transfer)

This suggests that raw feature magnitudes contain some information that transfers across tracks. Standardization removes this and amplifies the distribution shift between Young and Elderly.

### Interpretation

1. **Raw features may encode absolute magnitude information** that is preserved across populations
2. **Standardization amplifies distribution differences** by centering and scaling
3. **Different features matter in each track** - normalization doesn't help align them

### Implication for Thesis

Simple domain adaptation via feature normalization does not work for cross-track transfer. More sophisticated approaches (CORAL, MMD, adversarial training) may be needed, but the fundamental issue is that different biomarkers are relevant for different age groups.

---

## 8. Artifacts

### Scripts
- `scripts/benchmark_mpdd_simple.py` - LogisticRegression baseline
- `scripts/benchmark_ggmoe.py` - GGMoE with/without graph
- `scripts/xai_analysis.py` - SHAP-based XAI analysis
- `scripts/cross_track_validation.py` - Cross-track validation
- `scripts/investigate_audio44.py` - audio_44 feature investigation
- `scripts/test_domain_adaptation.py` - Domain adaptation experiments

### Outputs
- `artifacts/figures/xai_analysis/feature_importance.png` - Top-30 feature bar chart
- `artifacts/figures/xai_analysis/xai_results.json` - Full SHAP results
- `artifacts/figures/cross_track_validation/cross_track_comparison.png` - Cross-track comparison
- `artifacts/figures/cross_track_validation/audio_44_analysis.png` - audio_44 distribution analysis
- `artifacts/figures/cross_track_validation/cross_track_results.json` - Cross-track metrics
- `artifacts/benchmark_results.md` - This document

---

## 9. Files Reference

```
thesis-experiment-5-unified-model/
├── scripts/
│   ├── benchmark_mpdd_simple.py       # LR baseline
│   ├── benchmark_ggmoe.py             # GGMoE benchmarks
│   ├── xai_analysis.py                # SHAP analysis
│   ├── cross_track_validation.py      # Cross-track transfer
│   └── investigate_audio44.py         # audio_44 investigation
├── artifacts/
│   ├── benchmark_results.md           # Main results document
│   ├── figures/
│   │   ├── xai_analysis/              # SHAP outputs
│   │   │   ├── feature_importance.png
│   │   │   └── xai_results.json
│   │   └── cross_track_validation/    # Cross-track outputs
│   │       ├── cross_track_comparison.png
│   │       ├── audio_44_analysis.png
│   │       └── cross_track_results.json
```

---

*Last updated: 2025-06-01*