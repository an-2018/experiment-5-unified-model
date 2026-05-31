# Chapter 8 Progress Tracker — Unified Multimodal Graph-Gated MoE

**Paper lead:** @paper-lead
**Last updated:** 2026-05-31

## Writing Waves Status

| Wave | Phase(s) | Section | Status | Notes |
|------|----------|---------|--------|-------|
| Wave 0 | Phases 0–2 | Abstract, Intro, RQs, Dataset | ✅ DONE | Written in chapter_8.tex |
| Wave 1 | Phase 3 | Unimodal baselines methods+results | ✅ DONE | Table 3, Figure referenced |
| Wave 2 | Phase 4 | Fusion baselines methods+results | ✅ DONE | Table 4, cross-attention REJECTED |
| Wave 3 | Phase 5 | MMoEEx architecture + results | ✅ DONE | Table 5, per-dataset routing policy |
| Wave 4 | Phases 6–7 | Graph routing + joint training | ✅ DONE | Table 6 (V0-V4), Table 7 (ablation ladder) |
| Wave 5 | Phase 8 | LLM ablations (L0 baseline only) | ✅ PARTIAL | L0 baseline written; L1-L9 pending |
| Wave 6 | Phases 9–11 | Stats, calibration, XAI | ⏳ STUB | Sections 8.6-8.7 written as [AWAITING] |
| Wave 7 | Phase 12 | Discussion, limitations, conclusion | ✅ PARTIAL | Discussion written; conclusion stub |

## Section Completion Checklist

### 8.1 Introduction ✅
- [x] Problem statement (depression detection, small N, multimodal)
- [x] Motivation (shared representations, graph routing, explainability)
- [x] Contributions bullet list (5 contributions)

### 8.2 Background and Related Work ✅
- [x] Multimodal fusion (gated, LMF, cross-attention)
- [x] Mixture of Experts (MMoE, MMoEEx, PAMoE-MSA)
- [x] Graph neural networks (GraphSAGE, GAT, GNNExplainer)
- [x] LLM-enhanced modalities (ablation track noted)
- [x] Clinical depression detection (DAIC literature)

### 8.3 Dataset and Preprocessing [Phases 1–2] ✅
- [x] 8.3.1 DAIC-WOZ (189 sessions, PHQ-8, subject-independent splits)
- [x] 8.3.2 CMU-MOSEI (22,777 utterances, sentiment+emotion)
- [x] 8.3.3 ChaLearn First Impressions (10,000 clips, Big-Five)
- [x] 8.3.4 Data contract and leakage protocol

### 8.4 Architecture [Phases 3–7] ✅
- [x] 8.4.1 Modality Encoders (RoBERTa, WavLM, ViT → 256d)
- [x] 8.4.2 GatedLateFusion (rejected: CrossAttention, LMF)
- [x] 8.4.3 MMoEEx Expert Bank (8 experts, 2 shared, 6 exclusive)
- [x] 8.4.4 KNN Graph + GraphSAGE Router (inductive/split-local/transductive)
- [x] 8.4.5 Joint Uncertainty-Weighted Multitask Learning (NLL loss, log_sigma)

### 8.5 Experimental Setup ✅
- [x] Training setup (AdamW, lr=1e-3, cosine annealing, early stopping)
- [x] Temperature-balanced sampling (T=2.0) for MOSEI dominance
- [x] Evaluation protocol (bootstrap CIs, DeLong test, paired bootstrap)

### 8.6 Results [Phases 3–8] ✅ (majority)
- [x] 8.6.1 Unimodal Baselines (Table 3)
- [x] 8.6.2 Fusion Ablation (Table 4, cross-attention REJECTED)
- [x] 8.6.3 MMoEEx vs Hard Sharing (Table 5)
- [x] 8.6.4 Graph Routing Ablation (Table 6: V0-V4, V0 best MOSEI, V3 best DAIC)
- [x] 8.6.5 LLM Modality Ablations (L0 baseline only, Table 7)
- [ ] 8.6.6 Domain Adaptation (PENDING - Phase 9)

### 8.7 Calibration and Statistical Validation [Phase 10] ⏳ STUB
- [ ] ECE, Brier scores, DeLong tests, bootstrap CIs (1000 iterations)
- [ ] Reliability diagrams before/after temperature scaling

### 8.8 Explainability [Phase 11] ⏳ STUB
- [ ] SHAP modality attribution
- [ ] GNNExplainer subgraphs (3 case studies per dataset)
- [ ] GraphXAIN narratives

### 8.9 Discussion ✅ (majority)
- [x] What worked (graph routing, gated fusion, MMoEEx for FI)
- [x] What did not (cross-attention, MMoEEx on small N)
- [x] Limitations (DAIC n=107, MOSEI audio/video incomplete, LLM ablations pending)
- [x] Negative results as contributions (cross-attention literature claim REJECTED)

### 8.10 Conclusion and Future Work ⏳ PARTIAL
- [ ] Full conclusion (pending Phase 12)
- [x] Future work bullets (LLM track, domain adaptation, GraphXAIN validation, calibration)

## Diagrams Generated

| Diagram | File | Status | Used in Section |
|---------|------|--------|----------------|
| arch_unified_model | paper/diagrams/arch_unified_model.mmd + .png | ✅ DONE | 8.4 |
| arch_process_flow | paper/diagrams/arch_process_flow.mmd + .png | ✅ DONE | 8.5 |
| arch_visualization_map | paper/diagrams/arch_visualization_map.mmd + .png | ✅ DONE | 8.7 |
| results_fusion_comparison | paper/diagrams/results_fusion_comparison.mmd | ✅ DONE | 8.6.2 |
| results_graph_ablation | paper/diagrams/results_graph_ablation.mmd | ✅ DONE | 8.6.4 |
| results_unimodal_bar | paper/diagrams/results_unimodal_bar.mmd | ✅ DONE | 8.6.1 |

## Tables Generated

| Table | File | Used in Section |
|-------|------|----------------|
| Dataset Summary | chapter8_dataset_summary.tex | 8.3 |
| Architecture Summary | chapter8_architecture.tex | 8.4 |
| Hyperparameters | chapter8_hyperparameters.tex | 8.4 |
| Evaluation Protocol | chapter8_evaluation_protocol.tex | 8.5 |
| Unimodal Results | chapter8_unimodal_results.tex | 8.6.1 |
| Fusion Results | chapter8_fusion_results.tex | 8.6.2 |
| MMoEEx Results | chapter8_mmoeex_results.tex | 8.6.3 |
| Graph Results (V0-V4) | chapter8_graph_results.tex | 8.6.4 |
| Ablation Ladder | chapter8_ablation_ladder.tex | 8.6.4 |
| Graph Stats | chapter8_graph_stats.tex | 8.4 |

## Key Decisions Documented

### Already Made (Phase 0–6) ✅
1. **Cross-attention REJECTED**: Literature claim (+0.041 AUC) does NOT replicate. CrossAttn has 65K-2.8M params vs 6K-827K for gated → overparameterized for small DAIC (n=107) and even MOSEI. See Table 4.
2. **Per-dataset routing**: DAIC→text_only (fusion fails at n=107), MOSEI→multimodal (gated CCC=0.6229), FI→video_only (MSE loss collapse → NLL resolves).
3. **NLL loss for regression**: Fixes FI constant-prediction collapse from MSE loss.
4. **Temperature-balanced sampling (T=2.0)**: Mitigates MOSEI dominance (120x larger than DAIC).
5. **V0 (inductive, K=10) is best for MOSEI**: CCC=0.6803 (+0.18 vs MMoEEx).
6. **V3 (inductive, K=15) is best for DAIC**: AUROC=0.8967 (+0.40 vs MMoEEx).
7. **V4 (split-local, K=15) is best for FI**: Avg CCC=0.5032 (most conservative protocol).

### Pending Decisions ⏳
- LLM encoder choice (Phase 8, L1-L9)
- Domain adaptation method (Phase 9)
- Calibration method (Phase 10)
- GAT vs GraphSAGE router comparison (Phase 6)