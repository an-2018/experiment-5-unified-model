# Literature: DAIC-WOZ Improvements (2024-2025)

## State-of-the-Art Results on DAIC-WOZ

| Paper | Technique | DAIC Result | Notes |
|-------|-----------|-------------|-------|
| Zhang et al. 2025 (MIL) | Multi-instance learning + MT5/RoBERTa ensemble | **F1=0.88** | First to apply MIL to text interviews; handles class imbalance via MIL bag-level predictions |
| Zhuang et al. 2024 | LLM-empowered structural element graph | F1=0.85 (text-only) | GCN-based, leverages LLM for structural elements |
| Idiap bias paper (arxiv:2404.14463) | Ellie therapist prompts analysis | **F1=0.88 (Ellie-only)** vs 0.72 (participant-only) | CRITICAL: Ellie prompts carry bias signal; models exploit therapist questioning patterns |
| Zou et al. 2025 | Wav2Vec 2.0 + BERT + BiLSTM + adaptive pooling | F1=0.85, MAE=4.48 | End-to-end fusion; audio outperforms text on clustering |
| MDD (TechScience 2024) | T5 + WaveNet + CCA fusion | **Acc=92.75%**, Precision=92.05%, Recall=92.22% | CCA for correlated projections, neural network classifier |

## Key Techniques for DAIC Improvement

### 1. Multi-Instance Learning (MIL)
- Treats each interview as a bag of utterances (instances)
- Bag-level predictions for depression classification
- Addresses sample imbalance by considering multiple instances per participant
- Reference: Zhang, X., Li, C., Chen, W. et al. Sci Rep 15, 6637 (2025)

### 2. Therapist Prompt Analysis
- Ellie (virtual interviewer) prompts contain discriminative bias
- Models using only Ellie's questions achieve F1=0.88
- Participant responses alone achieve F1=0.72
- Ensemble (Ellie + participant) achieves F1=0.90
- **Implication**: Our model should be evaluated on participant responses only
- Reference: https://arxiv.org/html/2404.14463

### 3. LLM Integration
- MT5 and RoBERTa ensemble for feature extraction
- Knowledge injection from psychology texts
- Multi-scale convolutional layers + BiLSTM for temporal correlations
- Reference: Zhuang et al. 2024, Zhang et al. 2025

### 4. Weighted Loss for Imbalance
- BCEWithLogitsLoss with pos_weight accounting for 70/30 split
- No SMOTE needed for neural embedding features
- Reference: Zou et al. 2025

## Our Results Context

| Phase | Approach | DAIC AUROC | Notes |
|-------|----------|------------|-------|
| Phase 3 (unimodal text) | text-only baseline | **0.6991** | Single-task, isolated |
| Phase 5 (joint MMoEEx) | text_only routing + joint training | **0.5145** | Near random — MOSEI dominance destroys DAIC |

## Action Items
1. Isolate DAIC expert tower from MOSEI/FI sharing (current sharing causes regression)
2. Consider MIL-based approach for DAIC in Phase 6+
3. Document Ellie prompt bias in thesis limitations
4. Increase temperature for stronger DAIC upweighting (T=3.0 or T=4.0)

## References

1. Zhang, X., Li, C., Chen, W. et al. Optimizing depression detection in clinical doctor-patient interviews using a multi-instance learning framework. Sci Rep 15, 6637 (2025). https://doi.org/10.1388-025-90117-w

2. Zhuang, C., Mao, K. & Chen, J. A Multimodal Approach for Detection and Assessment of Depression Using Text, Audio and Video. Phenomics 4, 234–249 (2024). https://doi.org/10.1007/s43657-023-00152-8

3. Anonymous. DAIC-WOZ: On the Validity of Using the Therapist's prompts in Automatic Depression Detection from Clinical Interviews. arXiv:2404.14463 (2024). https://arxiv.org/html/2404.14463

4. Zou, Z., Gao, Y. & Wang, F. Depression detection methods based on multimodal fusion of voice and text. Sci Rep 15, 21907 (2025). https://doi.org/10.1038/s41598-025-03524-4

5. Anonymous. MDD: A Unified Multimodal Deep Learning Approach for Depression Diagnosis Based on Text and Audio Speech. Computers, Materials & Continua 81(3), 4125-4147 (2024). https://doi.org/10.32604/cmc.2024.056666