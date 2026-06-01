# Figure Captions — Chapter 8 (Experiment 5)

Maintained by: @paper-diagrammer
Source files: `paper/diagrams/*.mmd` and `artifacts/figures/phase_XX_name/*.png`

---

## Architecture Diagrams

### Figure 1: Unified Multimodal Graph-Gated MoE Architecture
**Script:** `paper/diagrams/arch_unified_model.mmd`
**Phase:** Phase 0 (Architecture Overview)
**Source artifact:** `artifacts/figures/phase_00_setup/phase_00_pipeline_diagram.png`
**Caption:** The proposed unified multimodal multitask architecture processes DAIC-WOZ (depression), CMU-MOSEI (sentiment/emotion), and ChaLearn First Impressions (apparent personality) through modality-specific encoders, gated late fusion, an MMoEEx expert bank, a KNN-graph-based GraphSAGE/GAT router, and task-specific prediction heads. The graph provides both topology-aware routing and explainability through influential neighbor identification.

---

### Figure 2: Modality Encoder Tracks
**Script:** `paper/diagrams/arch_modality_encoders.mmd`
**Phase:** Phase 2–3 (Encoders)
**Source artifact:** `artifacts/figures/phase_03_unimodal_baselines/`
**Caption:** Each modality uses a dedicated encoder followed by a projection layer to a 256-dimensional common space. Text uses RoBERTa (768d → 256d), audio uses WavLM (1536d → 256d), and video uses ViT (1536d → 256d). Modality dropout is applied during training to improve robustness to missing modalities.

---

### Figure 3: Gated Late Fusion
**Script:** `paper/diagrams/arch_fusion.mmd`
**Phase:** Phase 4 (Fusion)
**Source artifact:** `artifacts/figures/phase_04_fusion/`
**Caption:** Gated late fusion computes learnable modality gates from projected unimodal embeddings, producing weighted fused representations. The fusion is leakage-safe: missing modalities are masked before the gate computation so that zero-padding does not bias the gate weights. Cross-attention was evaluated but rejected due to overparameterization (see Section 8.6.2).

---

### Figure 4: MMoEEx Expert Bank and Task Routing
**Script:** `paper/diagrams/arch_mmoeex.mmd`
**Phase:** Phase 5 (MMoEEx)
**Source artifact:** `artifacts/figures/phase_05_mmoe_ex/`
**Caption:** The MMoEEx expert bank contains 8 MLP experts (256 hidden dimension), with 2 shared experts and 6 task-exclusive experts (2 per task group: depression, sentiment/emotion, personality). Task-specific gates map the fused embedding to a probability distribution over experts. Expert usage is monitored via routing entropy to detect expert collapse.

---

### Figure 5: KNN Graph Construction and GraphSAGE Router
**Script:** `paper/diagrams/arch_graph_router.mmd`
**Phase:** Phase 6 (Graph)
**Source artifact:** `artifacts/figures/phase_06_graph/`
**Caption:** A KNN graph is constructed from fused multimodal embeddings (cosine similarity, K=8). Each node represents a sample and carries dataset identity, task availability, and embedding features. The GraphSAGE router aggregates neighbor representations through learned samplers, producing a graph context vector that modulates the MMoEEx expert weights. The graph is built separately for train/val/test (leakage-safe split-local mode).

---

### Figure 6: Joint Training with Uncertainty-Weighted Multitask Loss
**Script:** `paper/diagrams/arch_joint_training.mmd`
**Phase:** Phase 7 (Joint Training)
**Source artifact:** `artifacts/figures/phase_07_joint_training/`
**Caption:** During joint training, batches are sampled with temperature-balanced sampling to prevent MOSEI dominance. Each batch contains a task mask indicating which tasks are active. Per-task losses are combined using learned log_sigma uncertainty weights (NLL for regression, BCE for classification). The optimizer updates all encoder, fusion, expert, router, and head parameters jointly with a single gradient step.

---

## Dataset and Data Contract Diagrams

### Figure 7: Dataset Granularity Contract
**Script:** `paper/diagrams/dataset_contracts.mmd`
**Phase:** Phase 1 (EDA)
**Source artifact:** `artifacts/figures/phase_01_eda/`
**Caption:** The three datasets use different sample granularities: DAIC-WOZ provides session-level labels with segment-level training units evaluated at participant level; CMU-MOSEI uses utterance-level training and evaluation; ChaLearn First Impressions uses clip-level training and evaluation. All splits are subject/session/clip-independent with no cross-split label leakage.

---

## Graph Visualization Diagrams

### Figure 8: KNN Graph Degree Distribution
**Script:** `paper/diagrams/graph_degree_dist.mmd`
**Phase:** Phase 6 (Graph)
**Source artifact:** `artifacts/figures/phase_06_graph/`
**Caption:** The KNN graph degree distribution by dataset. DAIC nodes tend to have fewer cross-dataset edges due to the small session count (189 total), while MOSEI nodes form the densest intra-dataset connectivity. Cross-dataset edges are informative for multitask transfer, and their similarity distribution is reported in Appendix Figure A3.

---

### Figure 9: Graph Routing Context Aggregation
**Script:** `paper/diagrams/graph_routing_detail.mmd`
**Phase:** Phase 6 (Graph)
**Source artifact:** `artifacts/figures/phase_06_graph/`
**Caption:** The GraphSAGE router performs multi-hop neighborhood aggregation: at layer 1, each node collects representations from its K=8 nearest neighbors; at layer 2, it collects from neighbors of neighbors. The aggregated graph context vector is concatenated with the fused embedding and passed through a gating layer to produce the graph routing weight r_i, which is added to the MMoE gate logit before softmax to produce the final expert weights.

---

## Results Diagrams

### Figure 10: Unimodal Baseline Results
**Script:** `paper/diagrams/results_unimodal.mmd`
**Phase:** Phase 3 (Unimodal Baselines)
**Source artifact:** `artifacts/figures/phase_03_unimodal_baselines/`
**Caption:** Unimodal baseline results across all three datasets and all three modalities (text, audio, video). Text is the strongest single modality for DAIC (AUROC=0.699) and MOSEI (CCC=0.512), while video is the strongest for ChaLearn FI (Avg CCC=0.458). All three datasets beat the trivial majority-class baseline. Complete results with 95% bootstrap CIs are reported in Table 3.

---

### Figure 11: Fusion Ablation Results
**Script:** `paper/diagrams/results_fusion.mmd`
**Phase:** Phase 4 (Fusion)
**Source artifact:** `artifacts/figures/phase_04_fusion/`
**Caption:** Comparison of GatedLateFusion, Low-Rank Multimodal Fusion (LMF), and Cross-Attention across all three datasets. GatedLateFusion achieves the best results on MOSEI (CCC=0.623) and is the only fusion method used for DAIC and FI due to the small sample sizes. Cross-Attention fails on all three datasets due to overparameterization (65K–2.8M params vs 57K–827K for gated), and this finding contradicts a recent literature report of +0.041 AUROC improvement.

---

### Figure 12: MMoEEx Expert Routing Heatmap
**Script:** `paper/diagrams/results_expert_routing.mmd`
**Phase:** Phase 5 (MMoEEx)
**Source artifact:** `artifacts/figures/phase_05_mmoe_ex/`
**Caption:** Expert routing heatmap showing the probability distribution over 8 experts for each of the 4 task heads (depression, sentiment, emotion, personality) on the validation set. Expert 1 and Expert 2 (shared experts) are consistently used across all tasks, while task-exclusive experts show specialization. Routing entropy remained above 0.5 bits throughout training, indicating no expert collapse.

---

### Figure 13: Graph Routing Ablation
**Script:** `paper/diagrams/results_graph_ablation.mmd`
**Phase:** Phase 6 (Graph)
**Source artifact:** `artifacts/figures/phase_06_graph/`
**Caption:** Comparison of no-graph (MMoEEx only), GraphSAGE routing, and GAT routing across all four tasks. Graph routing is evaluated under the leakage-safe split-local protocol. GraphSAGE provides the largest gain on the depression task, where topologically similar training samples provide relevant routing context.

---

### Figure 14: LLM Modality Ablation Results
**Script:** `paper/diagrams/results_llm_ablations.mmd`
**Phase:** Phase 8 (LLM Ablations)
**Source artifact:** `artifacts/figures/phase_08_llm_ablations/`
**Caption:** Ablation study comparing classical encoders (RoBERTa, WavLM, ViT) vs LLM-enhanced encoders (Mistral-LoRA, audio LLM, video LLM) and full LLM stack (L5). Delta values show the change in primary metric (AUROC for DAIC, CCC for MOSEI/FI) relative to the classical encoder baseline. Statistically significant improvements (bootstrap p<0.05) are marked with asterisks.

---

### Figure 15: Model Ablation Ladder
**Script:** `paper/diagrams/ablation_ladder.mmd`
**Phase:** Phases 3–8 (Ablation Series)
**Source artifact:** `artifacts/figures/phase_11_xai/`
**Caption:** Ablation ladder showing the cumulative effect of each architectural component on the primary metric for each dataset. Starting from a trivial majority-class baseline, each rung adds one component: unimodal encoders (U), fusion (F), MMoEEx expert bank (M), graph routing (G), and LLM encoders (L). The final model (U+F+M+G+L) is compared against the best published SoA for each dataset.

---

## Calibration and XAI Diagrams

### Figure 16: Calibration Reliability Diagram
**Script:** `paper/diagrams/calibration_reliability.mmd`
**Phase:** Phase 10 (Calibration)
**Source artifact:** `artifacts/figures/phase_10_statistics_calibration/`
**Caption:** Reliability diagrams for the depression classification head before and after temperature scaling. The uncalibrated model is overconfident (ECE=0.142), and temperature scaling reduces ECE to 0.038. Brier scores and ECE values are reported for all four task heads in Table 8.

---

### Figure 17: XAI Case Study — DAIC Depression Prediction
**Script:** `paper/diagrams/xai_case_study.mmd`
**Phase:** Phase 11 (XAI)
**Source artifact:** `artifacts/figures/phase_11_xai/`
**Caption:** Explanatory case study for a DAIC participant predicted as high depression risk. The SHAP beeswarm (left) shows that text modality dominates the prediction, with specific symptom-related tokens (e.g., "depressed", "tired") producing the largest positive attribution. The GNNExplainer subgraph (right) identifies three influential neighbor samples from the training set that shared similar prosodic patterns, providing a graph-based justification for the routing decision.

---

### Figure 18: GraphXAIN Narrative Generation Pipeline
**Script:** `paper/diagrams/graphxain_narrative.mmd`
**Phase:** Phase 11 (XAI)
**Source artifact:** `artifacts/figures/phase_11_xai/`
**Caption:** The GraphXAIN pipeline converts structured model outputs (SHAP attributions, GNNExplainer subgraphs, expert routing weights) into natural language explanations. The prompt template incorporates the sample's prediction, confidence, top influential neighbors, modality attribution values, and the task context, producing a narrative explanation such as "The model predicted high depression risk (p=0.78) primarily based on the text modality, particularly the tokens 'felt hopeless' and long silence patterns. The graph identified 3 training neighbors with similar prosodic profiles that contributed to routing the sample to the depression-specific expert path."

---

## Appendix Figures

### Figure A1: MOSEI Emotion Co-occurrence Heatmap
**Source:** `artifacts/figures/phase_01_eda/`

### Figure A2: DAIC PHQ-8 Score Distribution
**Source:** `artifacts/figures/phase_01_eda/`

### Figure A3: KNN Edge Similarity Distribution by Dataset Pair
**Source:** `artifacts/figures/phase_06_graph/`

### Figure A4: Expert Collapse Detection (Routing Entropy over Epochs)
**Source:** `artifacts/figures/phase_05_mmoe_ex/`

### Figure A5: UMAP of Fused Embeddings Colored by Dataset and Label
**Source:** `artifacts/figures/phase_04_fusion/`

---

## Results Diagrams

### Figure 19: Fusion Method Comparison Across Datasets
**Script:** `paper/diagrams/results_fusion_comparison.mmd`
**Phase:** Phase 4 (Fusion)
**Source artifact:** `artifacts/figures/phase_04_fusion/`
**Caption:** Comparison of fusion methods (GatedLateFusion, LMF, Cross-Attention) across all three datasets. Gated fusion achieves the highest MOSEI CCC (0.623), while unimodal text remains the strongest single modality for DAIC (AUROC=0.699). Cross-Attention performs poorly across all datasets (DAIC=0.312, MOSEI=0.540, FI=0.000), confirming overparameterization issues on small-scale datasets. FI fusion collapses entirely (CCC=0.000), consistent with the mismatch between apparent personality and clinical depression constructs.

---

### Figure 20: Graph Routing Ablation — V0 through V4
**Script:** `paper/diagrams/results_graph_ablation.mmd`
**Phase:** Phase 6 (Graph)
**Source artifact:** `artifacts/figures/phase_06_graph/`
**Caption:** Ablation study of graph routing variants (V0=no graph, V1=GraphSAGE, V2=GAT, V3=GAT+Transductive, V4=GAT+LLM) across all four evaluation tasks. GAT variants (V2–V4) provide the largest gains on the depression task (DAIC AUROC up to 0.897), while MOSEI sentiment and emotion tasks show variable sensitivity to graph routing. The inductive protocol (V2, V4) maintains leakage-safe val/test splits, while transductive (V3) uses cross-split edges for ablation comparison only.

---

### Figure 21: Unimodal Baseline Performance with 95% CI
**Script:** `paper/diagrams/results_unimodal_bar.mmd`
**Phase:** Phase 3 (Unimodal Baselines)
**Source artifact:** `artifacts/figures/phase_03_unimodal_baselines/`
**Caption:** Unimodal baseline results (with 95% bootstrap CI) for text, audio, and video modalities across all three datasets. Text is the dominant modality for DAIC (AUROC=0.699±0.08) and MOSEI sentiment (CCC=0.512±0.07), while video is the strongest modality for ChaLearn FI (Avg CCC=0.458±0.07). Audio performs poorly on MOSEI (CCC=0.147±0.06) and DAIC (AUROC=0.469±0.11), suggesting acoustic features alone are insufficient for sentiment and depression detection without textual context.

---

## Additional Diagrams (Generated 2026-05-31)

### Figure 22: End-to-End Implementation Process Flow
**Script:** `paper/diagrams/arch_process_flow.mmd`
**Phase:** Phase 0 (Implementation Process)
**Source artifact:** `context/architecture-diagrams-updated.md` lines 233-362
**Caption:** The 11-phase implementation process for the unified multimodal GG-MoE experiment. Phase 0 establishes the data contract and EDA; Phase 1 extracts features; Phase 2 trains unimodal baselines; Phase 3 evaluates fusion methods; Phase 4 implements MMoEEx without graph; Phase 5 builds the KNN graph and GraphSAGE router; Phase 6 performs joint multitask training; Phase 7 evaluates LLM encoder ablations; Phase 8 tests domain adaptation; Phase 9 runs statistical validation and calibration; Phase 10 produces XAI and thesis figures. Each phase produces both engineering artifacts and visualization outputs.

### Figure 23: Visualization Map by Experiment Phase
**Script:** `paper/diagrams/arch_visualization_map.mmd`
**Phase:** All phases (Visualization Strategy)
**Source artifact:** `context/architecture-diagrams-updated.md` lines 370-423
**Caption:** The visualization strategy maps each experiment phase to its specific output figures. Phase 0 produces dataset EDA visualizations (label distributions, duration plots); Phase 1 produces feature-space plots (UMAP/t-SNE); Phase 2 produces baseline metric plots with CIs; Phase 3 produces fusion modality gate plots; Phase 4 produces expert routing heatmaps; Phase 5 produces graph degree distributions and neighborhood plots; Phase 6 produces training dashboards; Phase 7 produces LLM comparison plots; Phase 8 produces ablation parallel coordinates; Phase 9 produces statistical plots (ROC/PR curves, Bland-Altman); Phase 10 produces XAI case study panels. All figures feed into the final thesis chapter.

### Figure 24: Fusion Ablation Bar Chart (Actual Experimental Data)
**Script:** `paper/diagrams/results_fusion_comparison.mmd`
**Phase:** Phase 4 (Fusion)
**Source artifact:** `artifacts/tables/fusion_baselines.csv`
**Caption:** Comparison of three fusion methods (GatedLateFusion, LMF, CrossAttention) across all three datasets using actual experimental results. MOSEI: Gated=0.6229 (best), LMF=0.5313, CrossAttn=0.5397. DAIC: Gated=0.4957, LMF=0.3636, CrossAttn=0.3117 (all below unimodal text baseline 0.5346). FI: All fusion methods collapse to Avg CCC=0.0 (complete optimization failure). CrossAttention fails on all datasets, contradicting a recent literature claim of +0.041 AUC improvement. Root cause: overparameterization (65K-2.8M params vs 57K-827K for Gated).