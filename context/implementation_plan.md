# Unified Multimodal Graph-Gated MoE (Experiment 5)

Provide a detailed architecture and evaluation framework for the Unified Model proposed in Chapter 8 of the thesis. This implementation fuses text, audio, and visual modalities to simultaneously model depression, sentiment, emotion, and personality using a Graph-Gated Mixture of Experts (GG-MoE). It applies state-of-the-art Graph Neural Networks (GNNs), Explainable AI (XAI) features, and rigorous statistical validation.

## User Review Required

> [!IMPORTANT]
> **Dataset Selection & Acquisition**: The plan targets **DAIC-WOZ** (Depression), **CMU-MOSEI** (Sentiment/Emotion), and **ChaLearn FI** (Personality) to cover all thesis tasks. Please confirm if you have local access to these datasets preprocessed, or if we need to write automated download and extraction pipelines.
> 
> **Fusion Dimensionality**: To solve the TFN parameter explosion issue (16.8M elements → 8.6B params) noted in your feedback, I propose replacing the standard Tensor Fusion Network (TFN) with **Low-Rank Multimodal Fusion (LMF)**. This is a SoA approach that drastically reduces parameters while maintaining representation power. Please approve this architectural change.

## Open Questions

> [!WARNING]
> 1. **Graph Construction Paradigm**: Following CS224W concepts, should the graph be constructed as a **Conversational/Temporal Graph** (nodes = utterances, edges = speaker context & time flow, best for datasets like MELD/IEMOCAP) or a **Global Similarity Graph** (nodes = patients/clips, edges = multimodal cosine similarity, best for isolated clips like ChaLearn FI)?
> 2. **Compute Constraints**: Training a unified multimodal GNN across three large datasets is computationally expensive. Should we freeze the base feature extractors (e.g., DistilBERT, Wav2Vec 2.0, OpenFace) and only train the fusion, router, and MoE layers?

## Proposed Changes

---

### Data and Graph Construction
Implementation of the multimodal data loading and heterogeneous graph definition.

#### [NEW] `src/data/multimodal_dataset.py`
PyTorch Dataset class for handling time-aligned text tokens, audio features (eGeMAPS), and video features (OpenFace). Handles dynamic padding and sequence alignment.

#### [NEW] `src/data/graph_builder.py`
Implements the graph construction logic using `PyTorch Geometric`. Creates a KNN graph based on multimodal feature similarity (IMP-10) where edges connect text-audio-video representations to enable context-aware routing.

### Multimodal GG-MoE Architecture
Implementation of the core unified model defined in Chapter 8.

#### [NEW] `src/models/encoders.py`
- Text: DistilBERT → 256-d BiLSTM
- Audio: CNN + BiLSTM (eGeMAPS/Wav2Vec)
- Video: BiLSTM (OpenFace/ViT)

#### [NEW] `src/models/fusion.py`
Implementation of Low-Rank Multimodal Fusion (LMF) to project the three modalities into a shared joint representation efficiently.

#### [NEW] `src/models/gnn_router.py`
A state-of-the-art **GraphSAGE / GAT** router that operates on the multimodal graph. Aggregates neighborhood context to output stable routing weights for the experts (addressing IMP-5 routing stability).

#### [NEW] `src/models/unified_moe.py`
The MoE layer with specialized experts and four multi-task heads: Depression (binary + severity), Sentiment (regression), Emotion (multi-label), and Personality (Big-Five regression).

### Training and Calibration
#### [NEW] `src/training/trainer.py`
PyTorch Lightning module utilizing **homoscedastic uncertainty weighting** to dynamically balance the disparate loss scales of the four tasks during multi-task learning.

#### [NEW] `src/training/calibration.py`
Post-hoc calibration algorithms including **Temperature Scaling** and **Platt Scaling** to improve the Brier Score and ECE of the GG-MoE predictions (IMP-3).

### XAI and Validation Framework
Rigorous statistical tests and explainability as required for thesis defense.

#### [NEW] `src/evaluation/xai_engine.py`
- **GNNExplainer**: Extracts the most influential subgraph that led the GraphSAGE router to its decision.
- **SHAP Integration**: Feature-level and modality-level attribution to understand whether text, audio, or video was most relied upon for a specific prediction.

#### [NEW] `src/evaluation/statistics.py`
Implements bootstrapping (1,000 iterations) and Delong's test to generate robust 95% Confidence Intervals for AUROC and F1 scores (IMP-4). 

#### [NEW] `src/evaluation/visualizations.py`
Generates publication-ready SoA plots:
- Routing weight heatmaps.
- XAI modality attribution bar charts.
- Subgraph visualizations (using `networkx`).
- UMAP projections of the multimodal embedding space.

### Thesis Report & Paper
#### [NEW] `paper/experiment_5_unified_model.tex`
A standalone, fully-formatted LaTeX paper acting as a comprehensive report for Experiment 5. Includes methodology, results tables (with CI bounds), XAI visual analysis, and conclusions. This will be structurally compliant with the thesis for easy integration into Chapter 8.

## Verification Plan

### Automated Tests
- **Shape and Parameter Check**: Assert that the LMF fusion layer correctly outputs the expected embedding dimensions without exceeding parameter budgets.
- **Statistical Testing CI**: Validate the bootstrapping script using dummy data to ensure the 95% CI calculation and p-values match `scipy.stats` reference implementations.
- **Multi-task Convergence**: Run a small epoch (overfit on a mini-batch) to verify that all four task losses decrease simultaneously without NaN instabilities.

### Manual Verification
- **XAI Validity**: Inspect the output of the `GNNExplainer` and SHAP summary plots to ensure the explanations are clinically meaningful (e.g., verifying that the depression head relies heavily on acoustic characteristics or specific sentiment words).
- **Paper Quality**: The user will review the generated `experiment_5_unified_model.tex` for academic rigor, correct LaTeX formatting, and alignment with the thesis narrative.
