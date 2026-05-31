# State of the Art Tracking — Experiment 5

**Last updated:** 2026-05-31
**Maintained by:** @paper-researcher

---

## DAIC-WOZ Depression Detection

| Method | Year | Modality | Metric | Value | Our Best Result | Metric | vs Ours | Comparable? |
|--------|------|----------|--------|-------|-----------------|--------|---------|-------------|
| Majority class | — | — | AUROC | 0.500 | — | — | baseline | ✅ |
| RoBERTa text-only (ours, Phase 3 unimodal baseline) | 2026 | Text | AUROC | 0.5346 | 0.5346 | AUROC | baseline | ✅ SoA |
| Burdisso GCN | 2024 | Text | F1 | 0.85 | 0.5346 | AUROC | not comparable | ❌ |
| Zhang MIL | 2025 | Text | AUROC | 0.78 | 0.5346 | AUROC | −0.245 | ✅ comparable |
| E-longBERT (incl. Ellie) | 2024 | Text+Therapist | F1 | 0.84 | — | — | — | ❌ |
| 3D Landmarks | 2025 | Video | F1 | 0.74, AUC | 0.83 | 0.5346 | AUROC | not comparable | ❌ |
| Niu multimodal | 2021 | A+V+T | F1 | 0.92 | 0.5346 | AUROC | not comparable | ❌ |
| Dai full fusion | 2021 | A+V+T | F1 | 0.96 | 0.5346 | AUROC | not comparable | ❌ |
| MMFformer | 2026 | A+V+T | F1 | 0.79 | 0.5346 | AUROC | not comparable | ❌ |
| **V3 graph (ours, Phase 6 best)** | **2026** | **T+A+V** | **AUROC** | **0.8967** | **0.8967** | **AUROC** | **+0.117 over Zhang 2025** | **✅ comparable** |

**Note:** Most SoA papers report F1 for DAIC, not AUROC. Zhang 2025 (AUROC=0.78) is the most directly comparable comparison metric. Our Phase 3 unimodal text baseline (AUROC=0.5346) underperforms Zhang 2025 by 0.245 AUROC, but our Phase 6 V3 graph model (AUROC=0.8967) outperforms Zhang 2025 by 0.117 AUROC on the same AUROC metric. The Phase 8 L0 text-only run (AUROC=0.6991) is a separate experiment with different preprocessing and model scale, not a unimodal baseline.

---

## CMU-MOSEI Sentiment Analysis

| Method | Year | Modality | Metric | Value | Our Best Result | Metric | vs Ours | Comparable? |
|--------|------|----------|--------|-------|-----------------|--------|---------|-------------|
| Majority class | — | — | CCC | ~0 | — | — | baseline | ✅ |
| RoBERTa text-only (ours) | 2026 | Text | CCC | 0.5123 | 0.5123 | CCC | baseline | ✅ SoA |
| Gated fusion (ours) | 2026 | T+A+V | CCC | 0.6229 | 0.6229 | CCC | +0.1106 | ✅ SoA |
| **V0 graph routing (ours, Phase 6 best)** | **2026** | **T+A+V+Graph** | **CCC** | **0.6803** | **0.6803** | **CCC** | **+0.1680** | **✅ SoA** |
| SSU | 2025 | Full | Acc2 | 87.93% | 0.6803 | CCC | not comparable | ❌ |
| DPDF-LQ | 2025 | Full | Corr | 0.774 | 0.6803 | CCC | −0.094 | ⚠️ |
| CSGI-Net | 2025 | Full | Corr | 0.774 | 0.6803 | CCC | −0.094 | ✅ comparable |
| MMoLRE | 2025 | Full | Corr | 0.797 | 0.6803 | CCC | −0.117 | ✅ comparable |
| PAMoE-MSA (MoE) | 2025 | Full | — | — | 0.6803 | CCC | — | — |

**Note:** CCC ≈ Pearson r when means and variances match. SoA Corr=0.774–0.797 is the upper bound. Our V0 graph model (CCC=0.6803) narrows the gap to 0.094–0.117 from the previous gap of 0.174 (gated fusion only).

---

## ChaLearn First Impressions (Apparent Personality)

| Method | Year | Modality | Metric | Value | Our Best Result | Metric | vs Ours | Comparable? |
|--------|------|----------|--------|-------|-----------------|--------|---------|-------------|
| Majority class | — | — | Avg CCC | ~0 | — | — | baseline | ✅ |
| ViT video-only (ours, Phase 3) | 2026 | Video | Avg CCC | 0.4578 | 0.4578 | CCC | baseline | ✅ SoA |
| **MMoEEx (ours, Phase 5 best)** | **2026** | **T+A+V** | **Avg CCC** | **0.5793** | **0.5793** | **CCC** | **+0.1215** | **✅ SoA** |
| DeepPersonality HRNet | 2024 | Video | CCC | ~0.60 | 0.5793 | CCC | −0.021 | ✅ comparable |
| EMP Ensemble | 2023 | Full | Avg Acc | 91.81% | 0.5793 | CCC | not comparable | ❌ |
| CHMAFN | 2025 | Full | Avg Acc | 93.97% | 0.5793 | CCC | not comparable | ❌ |

**Note:** SoA for FI reports accuracy (classification), not CCC (regression). These metrics are not comparable. Our MMoEEx Avg CCC=0.5793 approaches the DeepPersonality 2024 CCC~0.60 on video-only, with a gap of only −0.021.

---

## Architectural Components SoA

### Multimodal Fusion
| Method | Year | Metric | Notes |
|--------|------|--------|-------|
| GatedLateFusion (ours) | 2026 | DAIC: AUROC=0.4632, MOSEI: CCC=0.6229 | Small dataset friendly |
| LMF (ours) | 2026 | MOSEI: CCC=0.5313 | Mid-size dataset |
| Cross-Attention (ours) | 2026 | MOSEI: CCC=0.5397 | FAILED on all datasets |

### Mixture of Experts
| Method | Year | Task | Metric |
|--------|------|------|--------|
| MMoEEx (ours, Phase 5) | 2026 | DAIC/MOSEI/FI | TBD after training |
| PAMoE-MSA | 2025 | MOSEI sentiment | SoA Corr |
| MoE-Health (Wang 2025) | 2025 | MIMIC-IV mortality | AUROC=0.818 |

### Graph Routing
| Method | Year | Dataset | Metric |
|--------|------|---------|--------|
| GraphSAGE routing (ours, Phase 6) | 2026 | All | TBD after Phase 6 |
| GNNExplainer (ours, Phase 11) | 2026 | All | TBD |

---

## Missing SoA Data Points

- [x] DAIC multimodal AUROC comparison — Zhao 2025 MLLM F1=0.844 (not comparable)
- [x] LLM-enhanced depression detection — found Zhao 2025, Li 2025, Hu 2025 JMIR
- [x] MoE for healthcare — found Wang 2025 MoE-Health (AUROC=0.818 MIMIC-IV)
- [x] GraphXAIN for narrative GNN explanations — found Cedro 2024
- [x] CORAL/DANN domain adaptation — found Sun 2016, Deep CORAL, Ganin 2015
- [ ] MOSEI full ablation with CCC comparison across all methods
- [ ] FI CCC comparison — need to find papers reporting CCC instead of accuracy
- [ ] Calibration methods for healthcare ECE comparison

---

## Citation Quality Issues

- **DAIC**: Cannot directly compare AUROC vs F1 with most SoA papers. Need to either (a) convert our AUROC to F1 using optimal threshold, or (b) only compare with Zhang 2025 which reports AUROC.
- **MOSEI**: CCC is a strict metric. SoA papers reporting Pearson r are comparable if means/variances align. Acc2/Acc7 are NOT comparable.
- **FI**: Accuracy vs CCC is a fundamental metric mismatch. Report both, note incomparability.