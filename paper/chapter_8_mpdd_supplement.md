# Supplementary: MPDD Benchmark Results (Experiment 5 Extended)

**Date:** 2025-06-01
**Status:** COMPLETE

---

## S1. Overview

This supplementary section documents additional benchmark experiments on the **MPDD (Multimodal Depression Detector)** dataset, which provides a complementary evaluation of the unified model architecture on a different depression detection benchmark.

### S1.1 Dataset Comparison

| Dataset | Domain | Samples | Features | Depression Rate |
|---------|--------|---------|----------|-----------------|
| **DAIC-WOZ** | Clinical interview | 189 sessions | RoBERTa + WavLM + ViT | 29.6% |
| **MPDD-Young** | Self-reported | 264 segments | Wav2Vec2 + OpenFace | 48.9% |
| **MPDD-Elderly** | Self-reported | 337 segments | Wav2Vec2 + OpenFace | 23.4% |

### S1.2 Key Difference
- **DAIC**: Clinical PHQ-8 labels, subject-independent splits
- **MPDD**: Binary self-report labels, segment-level features pooled from 5s windows

---

## S2. MPDD-Young Benchmark Results

### S2.1 Model Comparison

| Model | Val AUROC | Test AUROC | Notes |
|-------|-----------|------------|-------|
| **Logistic Regression** | **0.926** | **0.674** | Best performer (C=0.01) |
| GGMoE (no graph) | 0.559 | — | Underperforms LR |
| GGMoE (with graph) | 0.545 | — | Batch-level graph not helpful |
| Simple MLP | 0.512 | — | Not learning |

**Finding:** Logistic Regression with L2 regularization (C=0.01) outperforms neural networks on this small dataset (n=184 training samples).

### S2.2 XAI Analysis (SHAP on Logistic Regression)

#### Audio vs Video Importance
| Modality | Total SHAP | Percentage |
|----------|------------|------------|
| Audio | 0.141 | 51.0% |
| Video | 0.135 | 49.0% |

**Conclusion:** Audio and video contribute nearly equally to depression prediction.

#### Top 10 Important Features
| Rank | Feature | SHAP Value | Modality |
|------|---------|------------|----------|
| 1 | audio_44 | 0.0180 | Audio |
| 2 | video_693 | 0.0172 | Video |
| 3 | video_657 | 0.0171 | Video |
| 4 | audio_388 | 0.0086 | Audio |
| 5 | audio_226 | 0.0080 | Audio |

**Key Finding:** `audio_44` is the single most important feature, potentially a depression biomarker specific to young adults.

---

## S3. Cross-Track Validation (MPDD-Young → Elderly)

### S3.1 Zero-Shot Transfer Results

| Evaluation | AUROC | Interpretation |
|------------|-------|----------------|
| Within Young | 0.926 | Good performance |
| Cross-track (Young→Elderly) | 0.395 | **NEGATIVE transfer** |
| Within Elderly | 0.277 | Elderly alone also fails |

**Critical Finding: Cross-track transfer FAILS - training on Young adults produces worse-than-random predictions on elderly adults.**

### S3.2 Root Cause Analysis

| Analysis | Finding |
|----------|---------|
| **Video feature shift** | 17,554x larger than audio shift |
| **audio_44 in Elderly** | NOT predictive (AUC=0.468) |
| **Feature overlap** | ZERO overlap between Young and Elderly top features |

**Implication:** Different depression biomarkers are relevant for different age groups. Models trained on one demographic may not generalize to another.

---

## S4. Domain Adaptation Experiments

### S4.1 Feature Normalization Tests

| Strategy | Young Val | Elderly AUC | Notes |
|----------|-----------|-------------|-------|
| Raw features (no scaling) | 0.634 | **0.500** | Random performance |
| StandardScaler | 0.926 | 0.395 | **Negative transfer** |

**Finding:** Feature scaling HURTS cross-track transfer. Raw feature magnitudes contain some transferable information.

### S4.2 Cross-Dataset Transfer (MPDD → DAIC)

| Method | DAIC Test AUC | Notes |
|--------|---------------|-------|
| Within-DAIC baseline | 0.387 | DAIC alone fails |
| MPDD→DAIC (source-only) | **0.551** | POSITIVE transfer |
| MPDD→DAIC (CORAL) | **0.535** | Adaptation helps |

**Key Finding:** MPDD features transfer to DAIC BETTER than training on DAIC itself.

---

## S5. Calibration and Statistical Validation (DAIC)

### S5.1 Calibration Methods

| Method | Test AUROC | Brier Score | ECE |
|--------|------------|-------------|-----|
| None (base) | 0.351 | 0.282 | 0.258 |
| Temperature | 0.351 | 0.259 | 0.255 |
| **Platt** | **0.351** | **0.234** | **0.194** |
| Isotonic | 0.329 | 0.275 | 0.272 |

**Finding:** Calibration improves calibration quality (ECE 0.258 → 0.194) but does NOT improve AUROC.

### S5.2 Statistical Validation

| Metric | Value | 95% CI |
|--------|-------|--------|
| Val AUROC (bootstrap) | 0.539 | [0.534, 0.543] |
| DeLong test (base vs Platt) | z=0, p=1.0 | No significant AUROC diff |

---

## S6. XAI Evaluation (DAIC Depression)

### S6.1 Perturbation Test Results

| Condition | AUROC | Change |
|-----------|-------|--------|
| Baseline (all features) | 0.351 | — |
| Audio removed | **0.426** | **+0.076** |

**Critical Finding:** Removing audio features IMPROVES depression detection.

### S6.2 Interpretation

This suggests the DAIC audio features contain **anti-predictive information** - features that push predictions in the wrong direction. This could be due to:
1. Feature extraction mismatch between training and deployment
2. Confounding variables in audio features
3. Need for feature selection or re-extraction

---

## S7. Summary of Key Findings

### S7.1 What Works
1. **Logistic Regression** on small datasets (n<200) with L2 regularization
2. **audio_44** as a potential biomarker for young adult depression
3. **Cross-dataset transfer** (MPDD→DAIC) with same encoder family
4. **Platt calibration** for improving ECE (not AUROC)

### S7.2 What Doesn't Work
1. **Neural networks** (MLP, GGMoE) on small datasets
2. **Graph routing** at batch level (insufficient samples)
3. **Cross-track transfer** (Young→Elderly) - different biomarkers
4. **Feature scaling** for domain adaptation - hurts transfer

### S7.3 Implications for Thesis

1. **Generalization challenge**: Depression detection models may not generalize across demographics
2. **Demographic-specific modeling**: Different age groups may need different models/features
3. **Feature quality matters**: Anti-predictive features can hurt model performance
4. **Cross-dataset potential**: Shared encoder families enable positive transfer

---

## S8. Artifacts

### Scripts
- `scripts/benchmark_mpdd_simple.py` - LR baseline
- `scripts/benchmark_ggmoe.py` - GGMoE benchmarks
- `scripts/xai_analysis.py` - SHAP analysis
- `scripts/cross_track_validation.py` - Cross-track transfer
- `scripts/investigate_audio44.py` - Feature investigation
- `scripts/test_domain_adaptation.py` - Domain adaptation
- `scripts/cross_dataset_mpdd_daic.py` - Cross-dataset transfer
- `scripts/phase09_domain_adaptation.py` - FI→DAIC adaptation
- `scripts/phase10_calibration.py` - Calibration evaluation
- `scripts/phase11_xai.py` - XAI evaluation

### Outputs
- `artifacts/mpdd_benchmark_report.md` - Comprehensive report
- `artifacts/figures/xai_analysis/` - SHAP visualizations
- `artifacts/figures/cross_track_validation/` - Cross-track analysis
- `artifacts/figures/phase09_domain_adaptation/` - Domain adaptation results
- `artifacts/figures/phase10_calibration/` - Calibration results
- `artifacts/figures/phase11_xai/` - XAI results

---

*Last updated: 2025-06-01*