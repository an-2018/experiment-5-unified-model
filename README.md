# Unified Multimodal Graph-Gated MoE Experiment (Experiment 5)

**Thesis experiment** for multimodal mental health assessment across DAIC-WOZ (depression), CMU-MOSEI (sentiment/emotion), and ChaLearn FI (apparent personality).

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| CPU | Any modern multi-core (used for text, eGeMAPS, OpenFace) |
| GPU | **4x NVIDIA RTX A6000 (48GB VRAM each)** required for Phase 8 LLM ablations (L1–L5) |
| RAM | 64GB+ recommended for dataset loading |
| Storage | ~50GB for feature cache, logs, and artifacts |

> **Note:** Phase 8 (LLM Ablations) requires 4x A6000 for Mistral-7B-Instruct + LLaVA feature extraction. Classical encoders (RoBERTa, WavLM, ViT) run on any GPU or CPU. Without A6000 hardware, use `--skip_extraction` to run Phase 8 with cached features or use the classical fallback (L0).

## Quick Start

```bash
# Install dependencies
uv sync

# Verify environment
uv run python scripts/test_verify.py

# Run full pipeline (all phases 0→12)
uv run python scripts/run_full_pipeline.py

# Run specific phase range
uv run python scripts/run_full_pipeline.py --start_phase 2 --stop_phase 7

# Dry run — print commands without executing
uv run python scripts/run_full_pipeline.py --dry_run
```

---

## Phase-by-Phase Commands

### Phase 0 — Environment Verification

```bash
uv run python scripts/test_verify.py
```

### Phase 1 — Dataset EDA and Contract

```bash
uv run python scripts/phase01_eda.py
```
No arguments — runs all datasets. Outputs: 8 EDA figures in `artifacts/figures/phase_01_eda/`

### Phase 2 — Feature Extraction

Extract features for all 3 datasets across all modalities. Uses **OpenSMILE eGeMAPSv02** (not librosa) for audio:

```bash
# Text (RoBERTa) — CPU, parallel
for dataset in daic mosei fi; do
    uv run python scripts/phase02_preprocess.py --dataset $dataset --encoder roberta --parallel 16 --device cpu
done

# Audio eGeMAPS (OpenSMILE eGeMAPSv02, Functionals) — CPU, parallel
for dataset in daic mosei fi; do
    uv run python scripts/phase02_preprocess.py --dataset $dataset --encoder egemaps --parallel 16 --device cpu
done

# Audio WavLM — GPU, --parallel 1 (CUDA OOM on >1)
for dataset in daic mosei fi; do
    uv run python scripts/phase02_preprocess.py --dataset $dataset --encoder wavlm --parallel 1 --device cuda
done

# Video OpenFace — DAIC only
uv run python scripts/phase02_preprocess.py --dataset daic --encoder openface --parallel 16 --device cpu

# Video ViT — CPU
uv run python scripts/phase02_preprocess.py --dataset mosei --encoder vit --parallel 16 --device cpu
uv run python scripts/phase02_preprocess.py --dataset fi --encoder vit --parallel 8 --device cpu

# Rebuild manifest + regenerate visualizations (no re-extraction)
uv run python scripts/rebuild_manifest.py
uv run python scripts/phase02_preprocess.py --only-visualize
```

Outputs: Feature cache in `data/features/` + figures in `artifacts/figures/phase_02_preprocessing/`

### Phase 3 — Unimodal Baselines

```bash
# All 9 (dataset × modality) combos
uv run python scripts/phase03_unimodal_baselines.py --dataset all --modality all --device cuda

# Regenerate figures from cached results (no retraining)
uv run python scripts/phase03_unimodal_baselines.py --only-visualize
```

Key finding: DAIC text achieves AUROC=0.5346 (fails vs majority-class trivial=0.7196). FI video Avg CCC=0.4578 is best modality. SoA comparison included.

### Phase 4 — Fusion Baselines

```bash
# Gated fusion for all datasets
for dataset in daic mosei fi; do
    uv run python scripts/phase04_fusion.py --dataset $dataset --fusion gated --device cuda
done

# LMF for MOSEI
uv run python scripts/phase04_fusion.py --dataset mosei --fusion lmf --device cuda

# LR-DGN (Low-Rank Dynamic Gating Network) for DAIC — r=16 default
uv run python scripts/phase04_fusion.py --dataset daic --fusion lrdgn --device cuda
# Maximum regularization variant
uv run python scripts/phase04_fusion.py --dataset daic --fusion lrdgn --lrdgn_rank 8 --device cuda
```

LR-DGN replaces cross-attention for small clinical datasets (DAIC n=107): r=16 → 57.9K params vs 64.8K cross-attention.

### Phase 5 — MMoEEx (no graph)

```bash
uv run python scripts/phase05_mmoe_ex.py \
    --dataset all \
    --tasks depression,sentiment,emotion,personality \
    --device cuda
```

### Phase 6 — Graph Construction

```bash
# Primary result: split-local graph (leakage-safe, no cross-split edges)
uv run python scripts/phase06_graph.py --graph_type split-local --k 10 --device cuda

# Inductive graph (for final evaluation — test nodes connect only to train)
uv run python scripts/phase06_graph.py --graph_type inductive --k 10 --device cuda

# Ablation matrix (V0=no graph, V1=GraphSAGE, V2=GAT) — 150 epochs
uv run python scripts/phase06_graph.py --run_ablation --graph_type split-local --k 10 --epochs 150 --device cuda
```

Outputs: `artifacts/figures/phase_06_graph/`

### Phase 7 — Joint Multitask Training

```bash
# Quick test (3 epochs, no graph)
uv run python scripts/phase07_joint_training.py \
    --epochs 3 --quick_test --router none --graph_type split-local \
    --temperature 2.0 --device cuda

# Primary result: GraphSAGE router, 150 epochs
uv run python scripts/phase07_joint_training.py \
    --epochs 150 --router graphsage --graph_type split-local \
    --k 10 --graph_weight 0.5 --freeze_epochs 20 \
    --temperature 2.0 --batch_size 32 --lr 1e-3 \
    --device cuda 2>&1 | tee logs/phase07_full_run.log

# Router ablation (V0=none, V1=graphsage, V2=gat)
for router in none graphsage gat; do
    uv run python scripts/phase07_joint_training.py \
        --epochs 150 --router $router --graph_type split-local \
        --k 10 --graph_weight 0.5 --freeze_epochs 20 \
        --temperature 2.0 --batch_size 32 --lr 1e-3 --device cuda
done

# Resume from checkpoint
uv run python scripts/phase07_joint_training.py \
    --resume artifacts/tables/phase07_best.pt \
    --epochs 150 --router graphsage --graph_type split-local \
    --k 10 --graph_weight 0.5 --freeze_epochs 20 \
    --temperature 2.0 --batch_size 32 --lr 1e-3 --device cuda
```

Outputs: `artifacts/figures/phase_07_joint_training/`

Key finding: Joint training yields DAIC AUROC=0.5543, MOSEI CCC=0.4874, MOSEI Emo AUC=0.6858, FI Avg CCC=0.4258.

### Phase 8 — LLM Modality Ablations (L1–L5)

> **Requires 4x NVIDIA RTX A6000 (48GB VRAM each)**
> Auto-detects GPUs and distributes Mistral/LLaVA via `accelerate device_map="auto"`
> After initial extraction (~1h/level), subsequent runs use `--skip_extraction` for fast iteration (~0.2h/level).

```bash
# Full run: L0 (classical) → L1 (Mistral) → L5 (full LLM stack)
# First run: ~2-4 hours (including extraction). After caching: ~0.3 GPU-hours/level.
bash scripts/run_phase08_all.sh --execute --epochs 30 --device cuda

# Skip extraction — use cached features from previous run
bash scripts/run_phase08_all.sh --execute --epochs 30 --device cuda --skip_extraction

# Generate summary report + UMAP visualization from existing results
bash scripts/run_phase08_all.sh --report

# Or directly:
uv run python scripts/phase08_llm_ablations.py --generate_report

# Individual levels (alternative)
uv run python scripts/phase08_llm_ablations.py --ablation L0 --epochs 30          # instant (classical fallback)
uv run python scripts/phase08_llm_ablations.py --ablation L1 --epochs 30 --device cuda  # Mistral-7B frozen text
uv run python scripts/phase08_llm_ablations.py --ablation L2 --epochs 30 --device cuda  # Mistral + LoRA
uv run python scripts/phase08_llm_ablations.py --ablation L3 --epochs 30 --device cuda  # Mistral + CLAP audio
uv run python scripts/phase08_llm_ablations.py --ablation L4 --epochs 30 --device cuda  # Mistral + LLaVA video
uv run python scripts/phase08_llm_ablations.py --ablation L5 --epochs 30 --device cuda  # full LLM stack (text+audio+video)

# L6–L9 require external APIs (ImageBind, LLM teacher, direct prompting, GraphXAIN)
uv run python scripts/phase08_llm_ablations.py --ablation L6  # Requires external API
```

Outputs: `artifacts/figures/phase_08_llm_ablations/` — 5 figures including:
- `llm_delta_bar.png` — DAIC AUROC improvement per LLM level
- `embedding_umap.png` — UMAP of classical (RoBERTa) vs LLM (Mistral) text embeddings
- `cost_performance.png` — GPU hours vs AUROC gain trade-off

### Phase 9 — Domain Adaptation

```bash
# Runs all methods (CORAL, MMD, DANN) by default
uv run python scripts/phase09_domain_adaptation.py
```

Outputs: `artifacts/figures/phase_09_domain_adaptation/`

### Phase 10 — Calibration + Statistical Validation

```bash
for method in temperature isotonic platt; do
    uv run python scripts/phase10_calibration.py --dataset daic --method $method --device cuda
done
```

Outputs: `artifacts/figures/phase_10_evaluation/`

### Phase 11 — XAI (SHAP, GNNExplainer, GraphXAIN)

```bash
# List available samples
# uv run python scripts/phase11_xai.py --list_samples

# Run on a representative test sample
SAMPLE="daic_test_001"
for mode in shap gnn graphxain; do
    uv run python scripts/phase11_xai.py --sample_id $SAMPLE --explain_mode $mode --device cuda
done
```

Outputs: `artifacts/figures/phase_11_xai/`, `artifacts/figures/xai_analysis/`

### Phase 12 — Thesis Chapter

```bash
uv run python scripts/phase12_thesis.py --output_dir paper/
```

Outputs: `paper/chapter_8.tex`, `artifacts/figures/phase_12_thesis/`

---

## Full Pipeline Run

```bash
# All phases 0→12
uv run python scripts/run_full_pipeline.py

# Core architecture only (0→7)
uv run python scripts/run_full_pipeline.py --stop_phase 7

# Skip setup phases
uv run python scripts/run_full_pipeline.py --start_phase 3

# Single phase
uv run python scripts/run_full_pipeline.py --phase 2

# Dry run
uv run python scripts/run_full_pipeline.py --dry_run
```

---

## Project Structure

```
src/
  data/          — Dataset loaders (DAIC, MOSEI, FI), MultimodalDataset, preprocessing, graph_builder
  models/        — Encoders, fusion (LMF/gated/LR-DGN), MMoEEx, GNN routers, task heads
  training/      — Lightning trainers, losses, samplers, calibration
  evaluation/    — Metrics, statistics, visualizations, XAI engine, graph_xai
  utils/         — Seed, logging, registry

data/
  features/      — Cached extracted features (NOT committed — see .gitignore)
  flags/         — Low-quality sample flags

configs/         — Dataset contract, experiment configs

scripts/
  phase00_visualizations.py    — Phase 0 visualization
  test_verify.py               — Phase 0 environment verification
  phase01_eda.py               — Phase 1 EDA
  phase02_preprocess.py        — Phase 2 feature extraction
  phase03_unimodal_baselines.py  — Phase 3 baselines
  phase04_fusion.py            — Phase 4 fusion
  phase05_mmoe_ex.py           — Phase 5 MMoEEx
  phase06_graph.py             — Phase 6 graph construction
  phase07_joint_training.py    — Phase 7 joint training
  phase08_llm_ablations.py     — Phase 8 LLM ablations
  phase08_all.sh               — Phase 8 orchestration (L0–L5)
  phase09_domain_adaptation.py — Phase 9 domain adaptation
  phase10_calibration.py       — Phase 10 calibration
  phase11_xai.py               — Phase 11 XAI
  phase12_thesis.py            — Phase 12 thesis
  run_full_pipeline.py         — Full pipeline orchestrator
  run_full.sh                  — Legacy Phase 2 extraction script

artifacts/
  figures/       — phase_XX_name/ subdirectories for all visualizations
  tables/        — Results CSVs (baselines, SoA comparison)
  references/    — SoA source bibliography

paper/           — Thesis chapter drafts and diagrams
context/         — Implementation plans, technical appendix
docs/            — Specs and plans
```

---

## Visualization Output Directory Map

| Phase | Directory | Key Outputs |
|-------|-----------|-------------|
| 0 | `phase_00_setup/` | Environment verification plots |
| 1 | `phase_01_eda/` | Dataset distributions, modality coverage, leakage diagnostics |
| 2 | `phase_02_preprocessing/` | UMAP projections, audio spectrograms, AU coverage plots |
| 3 | `phase_03_unimodal_baselines/` | AUROC/CCC/F1 bar charts, bootstrap CIs, SoA comparison |
| 4 | `phase_04_fusion/` | Fusion architecture diagrams, per-dataset performance |
| 5 | `phase_05_mmoe_ex/` | Expert utilization, task routing heatmaps |
| 6 | `phase_06_graph/` | KNN graph visualizations, degree distributions, router comparison |
| 7 | `phase_07_joint_training/` | Training curves, expert specialization, final metrics |
| 8 | `phase_08_llm_ablations/` | L0–L5 comparison delta bar, UMAP classical vs LLM embeddings, cost-performance trade-off |
| 9 | `phase_09_domain_adaptation/` | Domain shift visualization, transfer gain analysis |
| 10 | `phase_10_evaluation/` | Calibration curves, reliability diagrams, statistical test tables |
| 11 | `phase_11_xai/` | SHAP importance, GNNExplainer subgraphs, GraphXAIN narratives |
| 12 | `phase_12_thesis/` | Final thesis figures and tables |

---

## Dataset Paths

| Dataset | Path |
|---------|------|
| DAIC-WOZ | `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/daic/raw` |
| CMU-MOSEI | `/home/anilson/projects/posei-dataset/data/CMU-MOSEI` |
| ChaLearn FI | `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/fi/raw` |

---

## Package Manager

Always use `uv` — **never** `pip` or `conda`:

```bash
uv sync                     # Install dependencies from pyproject.toml
uv run python <script.py>   # Run a script with dependencies available
uv add <package>            # Add a dependency
uv run pytest               # Run tests
```

---

## Key Constraints

- **Leakage-safe graph protocol**: Build train/val/test KNN graphs **separately**; `build_multimodal_graph(cross_dataset_edges=False)` is the safe default. Use `build_inductive_graph()` for final evaluation. Transductive graphs (`cross_dataset_edges=True`) are **ABLATION ONLY**.
- **Subject-independent splits**: DAIC splits by participant ID, **never** by segment/turn
- **Modality masks required**: Every sample must declare which modalities are available
- **MOSEI dominance risk**: Use temperature-balanced or task-balanced sampling (MOSEI is ~120× larger than DAIC)
- **Visualization-first**: Every phase outputs ≥1 figure to `artifacts/figures/phase_XX_name/`
- **LLM ablations require A6000**: L1–L5 need 4x RTX A6000 (48GB); L0 (classical) runs on any hardware

---

## Phase Dependencies

```
Phase 0 (setup)  ───────────────────────────────► Phase 1 (EDA)  ──► Phase 2 (preproc)
                                                                          │
                                                                          ▼
Phase 8 (LLM) ◄── Phase 7 (joint) ◄── Phase 6 (graph) ◄── Phase 5 (MMoEEx) ◄── Phase 4 (fusion)
      │                                                             │
      ▼                                                             ▼
Phase 9 (domain adapt)                            Phase 3 (unimodal baselines)
      │
      ▼
Phase 10 (calibration) ──► Phase 11 (XAI) ──► Phase 12 (thesis)
```

**Do not proceed to Phase 3 until Phase 2 preprocessing is complete and verified.**

---

## Current Status

| Phase | Status | Key Outputs |
|-------|--------|-------------|
| 0 — Setup | ✅ Complete | `uv` environment, stub modules verified |
| 1 — EDA | ✅ Complete | 8 EDA figures, dataset contract, leakage checks |
| 2 — Preprocessing | ✅ Complete | OpenSMILE eGeMAPSv02, WavLM, RoBERTa, ViT, OpenFace |
| 3 — Unimodal Baselines | ✅ Complete | 17 figures, SoA comparison, CSV with bootstrap CIs |
| 4 — Fusion | ✅ Complete | Gated/LMF/LR-DGN, LR-DGN r=16 (57.9K params) |
| 5 — MMoEEx | ✅ Complete | DAIC AUROC=0.4928, MOSEI CCC=0.4979, FI CCC=0.5793 (+0.12 over uni) |
| 6 — Graph Construction | ✅ Complete | KNN graphs, 5 variants V0-V4, 8 figures, V0 MOSEI CCC=0.6803 |
| 7 — Joint Training | ✅ Complete | DAIC AUROC=0.5543, training curves + routing entropy |
| 8 — LLM Ablations | ✅ Complete | L0–L5 on 4x A6000, L3(CLAP) best DAIC=0.7210, 5 figures + UMAP |
| 9 — Domain Adaptation | ✅ Complete | CORAL/MMD/DANN/combined, negative transfer detected (10/12 conditions) |
| 10 — Calibration | ✅ Complete | Temperature/Platt/isotonic, BCa bootstrap CIs, DeLong tests |
| 11 — XAI | ✅ Complete | SHAP beeswarm, GNNExplainer, GraphXAIN narratives, 21 figures |
| 12 — Thesis | ✅ Complete | chapter_8.tex, all tables (14) and figures (6) |

---

## Implementation Gap Fixes (June 2026)

Three validated gaps were fixed:

1. **OpenSMILE eGeMAPS** — `AudioPreprocessor` now uses official `opensmile.Smile(FeatureSet.eGeMAPSv02, Functionals)` producing standard 88-dim eGeMAPS features. Librosa fallback preserved as `_extract_egemaps_librosa`.

2. **LLM Ablations on 4x A6000** — Phase 8 uses `accelerate device_map="auto"` for Mistral/LLaVA distribution across all 4 GPUs. Classical fallback preserved for portability.

3. **Safe Graph Default** — `build_multimodal_graph(cross_dataset_edges=False)` is the default. `validate_graph_no_cross_split_leakage()` raises `ValueError` on cross-split edges. Transductive mode clearly marked as ABLATION only.