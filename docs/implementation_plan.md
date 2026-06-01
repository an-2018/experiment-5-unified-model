# Experiment 5 Implementation Validation & Improvements

This document outlines the findings from validating the Phase 2 (Preprocessing) and Phase 3 (Unimodal Baselines) implementations, specifically focusing on the visualization artifacts, logic bugs, and proposing state-of-the-art (SOTA) improvements.

## User Review Required

> [!IMPORTANT]
> The validation revealed critical issues in how the visualizations are constructed, particularly the UMAP feature spaces, which currently display artificial clustering due to zero-padding. Please review the proposed fixes and SOTA architecture improvements before we proceed to apply them.

## 1. Visualizations & Implementation Issues Found

### Phase 2: Preprocessing Artifacts
1. **UMAP Artificial Clustering (Padding Bug)**: In `scripts/phase02_preprocess.py`, the feature dimensions vary by dataset (e.g., DAIC uses 768-dim RoBERTa, while MOSEI uses 600-dim GloVe). To concatenate these for UMAP, the script pads smaller arrays with zeros (`np.pad`). This introduces massive artificial variance, causing UMAP to perfectly cluster datasets based on the presence of zeros rather than the underlying semantics. 
2. **Spectrogram Misrepresentation**: The `librosa.display.specshow` function is being applied to raw hidden states of WavLM (768-dim) or eGeMAPS (88-dim). While it works for traditional frequency spectrograms, it misrepresents latent embeddings as time-frequency plots.
3. **Empty/Flat-line Plots**: For missing video/audio data, the fallback creates a `torch.zeros()` tensor. The AU time-series visualization blindly plots these as straight lines at zero, which gives the false impression of a static, emotionless face rather than missing data.

### Phase 3: Unimodal Baselines Artifacts
1. **Misaligned Trivial Baselines**: In `artifacts/tables/unimodal_baselines.csv`, DAIC text AUROC is reported as `0.5346`, with `trivial_value` at `0.7196` (the majority class accuracy), yet `beats_trivial` is set to `True`. The logic correctly compares AUROC against 0.5 (random chance), but the CSV structure incorrectly maps the accuracy trivial baseline to the AUROC row, causing severe reporting ambiguity.
2. **Synthetic FI Scatter Plots**: Based on the codebase history, the ChaLearn FI scatter plots use synthetic CCC-correlated gauges to bypass missing true distributions, which invalidates their scientific utility.

## 2. Web Search Results: SOTA Approaches

Research into the current state-of-the-art for Multimodal Depression Detection indicates a transition away from standard late/early fusion toward dynamic, context-aware mechanisms:

1. **Intermediate/Hybrid Fusion with Attention**: Cross-modal transformers and co-attention mechanisms are widely considered SOTA. They allow the model to dynamically weigh the importance of text versus audio (e.g., paying more attention to linguistic markers when audio is noisy).
2. **Dynamic Gating Networks (DGN)**: These architectures dynamically adjust the contribution of each modality based on data quality or context. This is highly effective in clinical settings (like DAIC) where one modality might be unreliable or missing.
3. **Overfitting on Small Datasets**: Standard cross-attention fails on small datasets (like DAIC, n=107). To use attention without overfitting, the community recommends utilizing Parameter-Efficient Fine-Tuning (e.g., LoRA) or constrained cross-attention (where only key/value pairs are exchanged in low-dimensional space).

## 3. Proposed Changes

### Fix Visualizations (`scripts/phase02_preprocess.py` & `scripts/phase03_unimodal_baselines.py`)
- **[MODIFY]** `scripts/phase02_preprocess.py`: Replace zero-padding before UMAP with PCA projection. All datasets will be individually reduced to the minimum common dimension (e.g., 50) using PCA *before* being concatenated and passed to UMAP.
- **[MODIFY]** `scripts/phase02_preprocess.py`: Add an explicit filter to skip `all-zero` tensors from fallback when generating AU time-series and spectrogram plots.
- **[MODIFY]** `scripts/phase03_unimodal_baselines.py`: Fix the CSV reporting logic so `trivial_value` accurately reflects the metric being evaluated (e.g., 0.5 for AUROC, majority accuracy for Acc).

### Architecture Improvements
Given that Phase 4 previously identified standard Cross-Attention as too parameter-heavy for DAIC (causing overfitting), we will:
- Implement a **Low-Rank Dynamic Gating Network (LR-DGN)**. By applying a low-rank bottleneck before computing the gate weights, we can achieve the dynamic modality balancing of SOTA approaches while drastically reducing the parameter count (preventing the DAIC collapse).

### Phase 6: Graph Construction & Routing
1. **Local vs Global Graph Logic Flaw**: `phase06_graph.py` computes global graph edges properly in `build_split_local_graph` to prevent leakage. However, during the quick-test model training (`collate_batch`), the code ignores this global graph and dynamically builds a *local mini-batch* k-NN graph using `cos_sim = torch.mm(x_norm, x_norm.T)`. This means the GraphSAGE router never actually sees the globally structured dataset connections during training, undermining the primary graph routing thesis.

### Phase 9: Domain Adaptation
1. **Mock Data Anti-Pattern**: The `load_mosei_features()` function inside `phase09_domain_adaptation.py` relies on a fallback mechanism that states "Fallback: create synthetic labels for demonstration". If the external `.pkl` is missing, it returns `None`, meaning the MOSEI domain adaptation part of the experiment is effectively skipped/stubbed out. This violates the `@qa-validator` anti-mock check requirement.

### Phase 11: XAI (Explainability)
1. **Truncated Modality Evaluation**: In `phase11_xai.py`, when calculating the SHAP modality importance for Audio, it arbitrarily slices `shap_values[:, :512]`. However, DAIC-WOZ audio features (WavLM) have 768 dimensions. This ignores the last 256 dimensions of the audio embedding, producing an inaccurate modality dominance score.
2. **Incomplete Perturbation Test**: Similar to the SHAP slice, the perturbation test (`X_perturbed[:, :512] = 0`) only zeroes out the first 512 features instead of the entire 768-feature audio vector.

## 3. Proposed Changes (Continued)

### Fix Graph Routing (`scripts/phase06_graph.py`)
- **[MODIFY]** `scripts/phase06_graph.py`: Update the PyTorch Geometric data loader (`collate_batch`) to correctly sample subgraphs from the *global* `edge_index` (using PyG's `NeighborSampler` or `NeighborLoader`) instead of dynamically building isolated, meaningless mini-batch graphs.

### Fix Domain Adaptation (`scripts/phase09_domain_adaptation.py`)
- **[MODIFY]** `scripts/phase09_domain_adaptation.py`: Remove the synthetic fallback logic. Properly integrate the MOSEI dataset ingestion pipeline used in Phase 2 so that real sentiment features are used for the DA evaluation.

### Fix XAI Truncation (`scripts/phase11_xai.py`)
- **[MODIFY]** `scripts/phase11_xai.py`: Update all feature dimension hardcoding (e.g., `512` to `768` or dynamically infer from shape) to ensure the full audio embeddings are evaluated in both SHAP and Perturbation tests.

## Open Questions

1. Should we re-generate all Phase 2 and Phase 3 artifacts to correct the visual and CSV errors, or simply apply these fixes moving forward into the unified model reporting?
2. Do you approve the addition of the Low-Rank Dynamic Gating Network to mitigate the overfitting issue seen in the standard Cross-Attention implementation?
3. For Phase 6 Graph Routing, would you prefer using PyTorch Geometric's `NeighborLoader` to properly sample the global graph during training?
