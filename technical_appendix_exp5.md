# Technical Appendix: Unified Multimodal Graph-Gated MoE (Experiment 5)

## 1. High-Level System Overview
The Experiment 5 codebase implements a unified, multimodal multi-task learning system designed for computational mental health assessment. The primary objective is clinical depression detection (DAIC-WOZ), supported by auxiliary tasks including sentiment and emotion recognition (CMU-MOSEI) and personality trait prediction (ChaLearn FI). 

The system leverages a **Graph-Gated Mixture-of-Experts (GG-MoE) architecture** named `MMoEEx` to map raw feature embeddings (Text, Audio, Video) into specialized expert sub-networks, preventing negative transfer. Graph Neural Networks (GraphSAGE or GAT) route samples based on feature similarity (K-Nearest Neighbors).

- **Data Flow**: Extracted features $\to$ `GatedLateFusion` $\to$ `MMoEEx` (with `GraphSAGERouter`) $\to$ Task-Specific Heads.
- **Orchestration**: Coordinated via `scripts/run_full_pipeline.py`. Phase 7 (`scripts/phase07_joint_training.py`) executes the core joint training loop.
- **Data Organization**: Raw data is preprocessed into explicit dimension-bounded tensors tracked by `data/features/manifest.json`.

---

## 2. Detailed Model Architecture Explanation

### Modality Projectors and Fusion (`src/models/fusion.py`)
Each modality (Text: RoBERTa $D=768$, Audio: WavLM $D=768$, Video: ViT/OpenFace $D=1536$) is projected into a common hidden dimension $H=256$.
Projectors consist of: $\text{Linear}(D, H) \to \text{LayerNorm}(H) \to \text{GELU}$.

**GatedLateFusion** computes dynamic modality gates to handle missing modalities without zero-padding:
$$ g_m = \sigma(W_{m} h_m + b_m) \quad \text{where } m \in \{t, a, v\} $$
The fused representation is:
$$ h_{fused} = (g_t \odot h_t) + (g_a \odot h_a) + (g_v \odot h_v) $$
*Implementation Note:* If a sample lacks a modality (e.g., video in DAIC text-only routing), its binary mask explicitly sets $g_m = 0$ prior to the summation step, ensuring the network doesn't process noisy zero-padded vectors.

### Graph-Gated MMoEEx (`src/models/unified_moe.py`)
The Multi-Task Multi-Expert (`MMoEEx`) framework utilizes 8 expert networks (`num_experts=8`). Each expert is a Multi-Layer Perceptron with a skip connection:
$$ E_i(x) = x + \text{Linear}_{256 \to 256}(\text{Dropout}(\text{GELU}(\text{Linear}_{256 \to 256}(x)))) $$

**Routing Mechanism (`src/models/gnn_router.py`)**:
Instead of standard Multi-gate Mixture of Experts, GG-MoE incorporates a neighborhood-aware graph router. 
The standard task gate $g_k(x)$ produces logits for routing. Simultaneously, the GraphSAGE router processes the batch using an adjacency edge index to produce graph-aware probabilities $r(x, E)$.
The combined routing weights are explicitly calculated as:
$$ W_{routing} = \text{Softmax}(\log(g_k(x)) + \lambda \log(r(x, E))) $$
where $\lambda$ maps to `self.graph_weight` (default 0.5).

### Task Heads (`src/models/task_heads.py`)
Task heads map the fused expert representation (Batch, 256) to final predictions:
- **DepressionHead** (DAIC): `Linear(256->128) -> ReLU -> Dropout -> Linear(128->1)`
- **SentimentHead** (MOSEI): Identical sequential structure as DepressionHead.
- **EmotionMultiLabelHead** (MOSEI): `Linear(256->128) -> ReLU -> Dropout -> Linear(128->6)`
- **PersonalityHead** (FI): 5 parallel branches of `Linear(256->64) -> ReLU -> Dropout -> Linear(64->1)`

---

## 3. Training, Optimisation, and Experiment Logic

The core training logic is manually orchestrated in `scripts/phase07_joint_training.py`.

### Progressive Unfreezing
To prevent catastrophic overfitting on the severely limited DAIC dataset (107 training samples), the unimodal projectors are frozen for the first 20 epochs (`FREEZE_EPOCHS = 20`). 
At Epoch 20, `unfreeze_top_layers(num_layers=2)` is invoked, making the final $\text{Linear}$ and $\text{LayerNorm}$ components trainable.

### Loss Calculation and Optimisation
The optimizer is `AdamW` ($\text{lr}=3\times 10^{-4}$, weight decay=$10^{-4}$) paired with a `CosineAnnealingLR` scheduler ($T_{max}=50$).

The joint training uses an **Uncertainty-Weighted Multi-Task Loss** (`TaskLosses` imported from `phase05_mmoe_ex.py`). Each task $k$ is assigned a learnable homoscedastic uncertainty parameter $\log(\sigma_k^2)$.
The joint loss optimizes:
$$ L_{total} = \sum_k \left( \exp(-\log(\sigma_k^2)) L_k + \frac{1}{2} \log(\sigma_k^2) \right) $$
- **DAIC & Emotion**: Binary Cross Entropy with Logits Loss (`BCEWithLogitsLoss`).
- **Sentiment & Personality**: Mean Squared Error (`MSELoss`) or Mean Absolute Error (`L1Loss`).

### Negative Transfer Monitor
At the conclusion of each optimizer step/epoch, the `NegativeTransferMonitor` class verifies if the current task metrics drop below 95% of their isolated unimodal baseline performance (e.g., DAIC AUROC $0.6991 \times 0.95$). If triggered, it logs an explicit alert to prevent gradient dominance from larger tasks (like MOSEI).

---

## 4. Tools, Libraries, and Frameworks

- **PyTorch (2.x)**: Underpins all core tensor computations and neural network graphing.
- **PyTorch Geometric (PyG)**: Heavily utilized for `GraphSAGERouter` and `GATRouter`. It manages the dynamic node aggregation across `edge_index` structures representing patient similarities.
- **Transformers (HuggingFace)**: Essential for extracting RoBERTa text features during Phase 2.
- **Librosa / PySoundFile**: Employed for audio processing. (Logs show deliberate fallbacks to `audioread` when `PySoundFile` fails on specific corrupt files).
- **UV Package Manager**: Mandated by `AGENTS.md` to ensure reproducible, fast dependency resolution, specifically avoiding `pip` or `conda` conflicts with CUDA binaries.
- **Captum & SHAP**: Slated for Phase 11 (XAI) to perform integrated gradients and feature attribution across the fusion layers.

---

## 5. Inputs, Outputs, and Assets

- **Expected Inputs**: Raw arrays generated in Phase 2 are cached into `data/features/`.
  - Shape boundaries: Text $(N, 768)$, Audio $(N, 768)$, Video $(N, 1536)$.
- **Intermediate Artifacts (Graphs)**: `scripts/phase06_graph.py` computes K-Nearest Neighbors (KNN) and outputs explicit adjacency matrices (`edge_index`, `edge_weight`). The code explicitly supports `split-local`, `inductive`, and `transductive` generation topologies.
- **Final Outputs**:
  - Lightning Checkpoints (`.ckpt`) stored in `lightning_logs/`.
  - Analytical figures and routing mechanism visualizations are rigorously routed to `artifacts/figures/phase_XX_name/`.
  - Console logs are routed via standard streams and `nohup` files (e.g., `logs/phase08_full_run...`).

---

## 6. Logs, Results, and Visualisations Analysis

Review of execution logs (e.g., `phase02-logs.log`, `nohup.out`) reveals specific data characteristics:
- **DAIC Imbalance**: The preprocessing logs confirm exactly 107 DAIC training samples vs. an overwhelmingly larger presence for MOSEI.
  - *Architectural Mapping*: This imbalance fundamentally justifies the `GraphEnhancedDataset` dynamically scaling subsets using `temperature=3.0` during probability sampling to avoid erasing DAIC representations.
- **Evaluation Metrics**:
  - DAIC utilizes **AUROC** as the primary medical diagnostic threshold.
  - Personality relies on the **Concordance Correlation Coefficient (CCC)**, tracking temporal alignment of continuous traits.
- **Conclusion Derivations**: When the `NegativeTransferMonitor` alerts a drop in DAIC AUROC while MOSEI CCC rises, it empirically proves the "gradient dominance" effect, proving the necessity of the expert isolation mapping (`TASK_TO_EXPERTS`).

---

## 7. Limitations and Implementation Gaps

- **Audio Extraction Fallbacks**: As confirmed by `phase02-logs.log`, standard OpenSMILE features fail in this environment, forcing the system to fall back to `librosa` spectral derivations for `eGeMAPS`. This is an explicit implementation gap that alters baseline reproducibility compared to literature using strict OpenSMILE sets.
- **Hardware Memory Constraints**: The `nohup.out` from Phase 8 documents that Mistral/LLaVA modality ablations require an NVIDIA RTX A6000 (48GB VRAM). The code contains explicit fail-safes to use "classical" fallback encoders if `peft` and memory allocations fail.
- **Transductive Graph Leakage Risk**: The function `build_multimodal_graph(cross_dataset_edges=True)` links nodes indiscriminately across Train/Val/Test splits. The implementation flags this entirely as an `(ABLATION)`. Running primary metrics on transductive graphs would strictly violate the "Subject-independent splits" rule in `AGENTS.md`. Valid clinical evaluations must rely strictly on `build_inductive_graph()`.
