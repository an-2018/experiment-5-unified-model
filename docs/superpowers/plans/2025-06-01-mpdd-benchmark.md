# MPDD Benchmark Implementation Plan
## Unified Graph-Gated Mixture of Experts (GGMoE) on Multimodal Personality-aware Depression Detection

**Date:** 2025-06-01  
**Project:** Thesis Experiment 5 - Unified Multimodal Graph-Gated MoE Architecture  
**Benchmark Dataset:** MPDD (Multimodal Personality-aware Depression Detection)

---

## 1. Executive Summary

This implementation plan benchmarks the unified GGMoE architecture on the MPDD dataset, comparing results against the baseline from the ACM MM 2025 MPDD Challenge paper.

**MPDD Dataset Characteristics:**
- Two tracks: MPDD-Young and MPDD-Elderly
- Pre-extracted features: Wav2Vec audio (15, 512) and OpenFace video (15, 709)
- Labels: PHQ-9 scores, binary depression, Big Five personality traits
- No text modality (audio + video only)

**Tasks:**
1. **Depression Binary Classification** (primary) - AUROC, AUPRC, F1
2. **PHQ-9 Regression** (auxiliary) - MAE, RMSE, Pearson correlation
3. **Big Five Personality Prediction** (auxiliary) - MAE, CCC per trait

---

## 2. Data Summary

### 2.1 MPDD Dataset Structure

| Track | Samples | Train | Val | Test | Depressed | Non-Depressed |
|-------|---------|-------|-----|------|-----------|---------------|
| Young | 264 | 184 | ~40 | ~40 | ~129 | ~135 |
| Elderly | 337 | ~236 | ~50 | ~50 | ~79 | ~258 |
| Combined | 601 | 419 | 89 | 93 | 208 | 393 |

### 2.2 Feature Dimensions

- **Audio:** (15 timesteps × 512 Wav2Vec features) = 7,680 dims total, pooled to 512
- **Video:** (15 timesteps × 709 OpenFace features) = 10,635 dims total, pooled to 512
- **Fused:** 512 + 512 = 1024 → projected to 512

### 2.3 Modality Configuration

```
modality_mask: (text=False, audio=True, video=True)  # MPDD has no text
task_mask: (depression=True, phq=True, personality=True)  # All tasks available
```

---

## 3. Architecture

### 3.1 GGMoE Model Architecture

```python
MPDDGGMoE(
    # Modality Projection
    modality_projector: MPDDFeatureProjector
        audio_projector: Linear(512 → 512) + LayerNorm + GELU
        video_projector: Linear(709 → 512) + LayerNorm + GELU
    
    # Fusion
    fusion_proj: Sequential(Linear(1024 → 512), LayerNorm, GELU, Dropout(0.1))
    
    # MMoEEx
    mmoe: MMoEEx(
        input_dim: 512
        num_experts: 8 (2 shared, 6 task-exclusive)
        expert_dim: 256
        num_tasks: 3
        expert_isolation: True
        graph_router: GraphSAGE (optional)
    )
    
    # Task Heads
    depression_head: Linear(256, 1)
    phq_head: Linear(256, 1)
    personality_head: Linear(256, 5)
    
    # Uncertainty weighting
    log_task_weights: Parameter(3)
)
```

### 3.2 Baseline Variants

| Variant | Description | Purpose |
|---------|-------------|---------|
| Audio-only | Use only Wav2Vec features | Modality contribution |
| Video-only | Use only OpenFace features | Modality contribution |
| Fusion (no MoE) | Concatenate + project, single head | Fusion contribution |
| MMoEEx | MMoE without graph | Controlled sharing |
| GGMoE | MMoEEx + GraphSAGE router | Graph routing contribution |

---

## 4. Implementation Tasks

### Task 1: Environment Setup
- [x] Create symlink to MPDD data
- [ ] Install dependencies (pytorch-lightning, scikit-learn, etc.)
- [ ] Verify GPU availability

### Task 2: MPDD Data Loader
- [x] Create `src/data/mpdd_loader.py`
- [x] Load from zip files (lazy loading)
- [x] Handle train/val/test splits
- [x] Map subject-level labels to segment-level samples
- [ ] Verify label alignment

### Task 3: Dataset Integration
- [ ] Create PyTorch Dataset class for MPDD
- [ ] Add to unified dataset framework
- [ ] Verify modality masks

### Task 4: Unimodal Baselines
- [ ] Audio-only baseline
- [ ] Video-only baseline
- [ ] Train and evaluate
- [ ] Save results

### Task 5: Fusion Baseline
- [ ] Audio + video concatenation
- [ ] Single task head
- [ ] Train and evaluate

### Task 6: MMoEEx (no graph)
- [ ] 8 experts, 3 tasks
- [ ] Expert isolation enabled
- [ ] Uncertainty-weighted loss
- [ ] Train and evaluate

### Task 7: GGMoE (with GraphSAGE)
- [ ] Build KNN graph from embeddings
- [ ] GraphSAGE router integration
- [ ] Combined MMoE + graph routing
- [ ] Train and evaluate

### Task 8: Statistical Validation
- [ ] Bootstrap 95% CIs
- [ ] DeLong test for AUROC
- [ ] Paired permutation tests
- [ ] Effect sizes (Cohen's d)

### Task 9: XAI Analysis
- [ ] SHAP for modality attribution
- [ ] Integrated Gradients for feature importance
- [ ] GNNExplainer for graph routing
- [ ] Counterfactual validation

### Task 10: Visualization
- [ ] Training curves
- [ ] Modality importance bars
- [ ] Expert routing heatmaps
- [ ] Calibration plots
- [ ] XAI case studies

### Task 11: QA Validation
- [ ] Validate each phase against acceptance criteria
- [ ] Check for data leakage
- [ ] Verify reproducibility

### Task 12: Paper Update
- [ ] Compare with MPDD challenge baselines
- [ ] Add results to thesis
- [ ] Generate LaTeX tables

---

## 5. Reference Baselines (from MPDD Paper)

From Fu et al., ACM MM 2025:

| Method | AUROC | F1 | Notes |
|--------|-------|-----|-------|
| Audio-only | ~0.72 | ~0.65 | Wav2Vec features |
| Video-only | ~0.68 | ~0.62 | OpenFace features |
| Multimodal Fusion | ~0.78 | ~0.71 | Audio + Video + Personality |
| MPDD Challenge Winner | TBD | TBD | To be determined |

**Note:** These are approximate values from the paper abstract. Actual results to be obtained.

---

## 6. Statistical Testing Protocol

### 6.1 Metrics to Report

**Depression Classification:**
- AUROC with 95% bootstrap CI
- AUPRC with 95% bootstrap CI
- F1 with 95% bootstrap CI
- Sensitivity, Specificity

**PHQ Regression:**
- MAE with 95% CI
- RMSE with 95% CI
- Pearson correlation with significance test

**Personality Prediction:**
- Mean MAE across 5 traits
- Mean CCC across 5 traits

### 6.2 Comparison Tests
- DeLong test for AUROC comparisons
- Paired t-test or Wilcoxon for other metrics
- Bonferroni correction for multiple comparisons

---

## 7. XAI Methods

### 7.1 Modality Attribution
- **SHAP:** Compute SHAP values for audio and video contributions
- **Integrated Gradients:** Track importance through the network

### 7.2 Graph Explanations
- **GNNExplainer:** Identify influential edges and neighbors
- **Subgraph visualization:** Show local neighborhood for selected cases

### 7.3 Validation
- **Counterfactual tests:** Remove top features/edges, measure prediction change
- **Faithfulness check:** Ensure explanations correlate with model decisions

---

## 8. Expected Timeline

| Task | Time | Total |
|------|------|-------|
| Environment Setup | 15 min | 15 min |
| Data Loader | 30 min | 45 min |
| Dataset Integration | 30 min | 1h 15min |
| Unimodal Baselines | 1 hour | 2h 15min |
| Fusion Baseline | 45 min | 3h |
| MMoEEx | 1 hour | 4h |
| GGMoE | 1 hour | 5h |
| Statistical Validation | 30 min | 5h 30min |
| XAI Analysis | 1 hour | 6h 30min |
| Visualization | 30 min | 7h |
| QA Validation | 1 hour | 8h |
| Paper Update | 1 hour | 9h |

**Total estimated: ~9 hours**

---

## 9. Files Created/Modified

### Created Files
- `src/data/mpdd_loader.py` - MPDD data loader
- `scripts/benchmark_mpdd.py` - Benchmark script
- `docs/superpowers/plans/2025-06-01-mpdd-benchmark.md` - This plan

### Modified Files
- `data/raw/mpdd` → symlink to source data
- `src/data/__init__.py` - To add MPDD loader export

---

## 10. Key Decisions

1. **MPDD only** - Skipped PDCH due to Chinese language complexity
2. **English-only** - MPDD is English-based
3. **Audio + Video only** - MPDD has no text transcripts
4. **Pre-extracted features** - Wav2Vec (512d) and OpenFace (709d)
5. **Segment-level training** - Subject-level aggregation for evaluation
6. **GraphSAGE router** - Optional, compared against no-graph baseline

---

## 11. References

1. Fu et al., "The First MPDD Challenge: Multimodal Personality-aware Depression Detection", ACM MM 2025
   - Paper: https://arxiv.org/abs/2505.10034
   - Code: https://github.com/hacilab/MPDD
   - Challenge: https://hacilab.github.io/MPDDChallenge.github.io

2. Cao et al., "A Multimodal Depression Consultation Dataset of Speech and Text with HAMD-17 Assessments", Scientific Data 2025
   - Paper: https://www.nature.com/articles/s41597-025-05817-9
   - Dataset: https://doi.org/10.57760/sciencedb.27818

---

*Implementation plan created for Thesis Experiment 5*