# Unified Multimodal Graph-Gated MoE Experiment (Experiment 5)

**Thesis experiment** for multimodal mental health assessment across DAIC-WOZ (depression), CMU-MOSEI (sentiment/emotion), and ChaLearn FI (apparent personality).

## Quick Start

```bash
# Install dependencies
uv sync

# Verify Phase 0 setup (all imports + dummy batch passthrough)
uv run python scripts/test_verify.py

# Phase 1 — Dataset EDA and contract (no args — runs all datasets)
uv run python scripts/phase01_eda.py

# Phase 2 — Feature extraction
# Text (RoBERTa, CPU — 16 workers)
uv run python scripts/phase02_preprocess.py --dataset daic --encoder roberta --parallel 16
uv run python scripts/phase02_preprocess.py --dataset mosei --encoder roberta --parallel 16
uv run python scripts/phase02_preprocess.py --dataset fi --encoder roberta --parallel 16

# Audio (eGeMAPS, CPU)
uv run python scripts/phase02_preprocess.py --dataset daic --encoder egemaps --parallel 16

# Audio (WavLM, GPU — must use --parallel 1 due to CUDA OOM)
uv run python scripts/phase02_preprocess.py --dataset daic --encoder wavlm --parallel 1
uv run python scripts/phase02_preprocess.py --dataset mosei --encoder wavlm --parallel 1
uv run python scripts/phase02_preprocess.py --dataset fi --encoder wavlm --parallel 1

# Video (OpenFace — DAIC only; no raw video released)
uv run python scripts/phase02_preprocess.py --dataset daic --encoder openface --parallel 16

# Video (ViT, CPU)
uv run python scripts/phase02_preprocess.py --dataset mosei --encoder vit --parallel 16
uv run python scripts/phase02_preprocess.py --dataset fi --encoder vit --parallel 8

# Rebuild manifest after all extraction
uv run python scripts/rebuild_manifest.py

# Regenerate Phase 2 visualizations from cached manifest (no re-extraction)
uv run python scripts/phase02_preprocess.py --only-visualize

# Phase 3 — Unimodal baselines
# Run all 9 (dataset × modality) combinations at once
uv run python scripts/phase03_unimodal_baselines.py --dataset all --modality all

# Or run per-dataset:
uv run python scripts/phase03_unimodal_baselines.py --dataset daic --modality all
uv run python scripts/phase03_unimodal_baselines.py --dataset mosei --modality all
uv run python scripts/phase03_unimodal_baselines.py --dataset fi --modality all

# Regenerate all figures + SoA comparison from cached CSV (no retraining)
uv run python scripts/phase03_unimodal_baselines.py --only-visualize

# Phase 4 — Fusion baselines
uv run python scripts/phase04_fusion.py --dataset daic --fusion gated
uv run python scripts/phase04_fusion.py --dataset daic --fusion lmf
uv run python scripts/phase04_fusion.py --dataset daic --fusion lrdgn   # Low-Rank DGN (r=16), param-efficient
uv run python scripts/phase04_fusion.py --dataset mosei --fusion gated
uv run python scripts/phase04_fusion.py --dataset mosei --fusion lrdgn   # r=32 for larger MOSEI
uv run python scripts/phase04_fusion.py --dataset fi --fusion gated

# Phase 5 — MMoEEx (no graph yet)
uv run python scripts/phase05_mmoeex.py --dataset all --tasks depression,sentiment,emotion,personality

# Phase 6 — Graph construction (split-local = primary results, inductive = final eval)
# V0–V4 ablation (runs none/graphsage/gat, then inductive variants — primary result)
# Note: collate_batch now uses NeighborLoader to sample from the global edge_index
uv run python scripts/phase06_graph.py --run_ablation --graph_type split-local --k 10 --epochs 150

# Individual graph builds (for visualization and inspection)
uv run python scripts/phase06_graph.py --graph_type split-local --k 10
uv run python scripts/phase06_graph.py --graph_type inductive --k 10

# Single router variant (no ablation)
uv run python scripts/phase06_graph.py --graph_type split-local --k 10 --router graphsage --epochs 150

# Phase 7 — Joint multitask training
# Quick test (3 epochs, no graph routing — fast iteration)
uv run python scripts/phase07_joint_training.py --epochs 3 --quick_test --router none --graph_type split-local --temperature 2.0

# Full training run — primary result (GraphSAGE router, 150 epochs)
uv run python scripts/phase07_joint_training.py \
  --epochs 150 --router graphsage --graph_type split-local \
  --k 10 --graph_weight 0.5 --freeze_epochs 20 \
  --temperature 2.0 --batch_size 32 --lr 1e-3 \
  --device cuda 2>&1 | tee logs/phase07_full_run.log

# Ablation: run router=none, graphsage, gat separately (no --run_ablation flag)
# V0 (no graph): router=none
# V1 (GraphSAGE): router=graphsage
# V2 (GAT): router=gat
uv run python scripts/phase07_joint_training.py --epochs 150 --router graphsage --graph_type split-local --k 10 --graph_weight 0.5 --freeze_epochs 20 --temperature 2.0
uv run python scripts/phase07_joint_training.py --epochs 150 --router gat --graph_type split-local --k 10 --graph_weight 0.5 --freeze_epochs 20 --temperature 2.0
uv run python scripts/phase07_joint_training.py --epochs 150 --router none --graph_type split-local --k 10 --graph_weight 0.5 --freeze_epochs 20 --temperature 2.0

# Resume from checkpoint (e.g., after GPU crash)
uv run python scripts/phase07_joint_training.py \
  --resume artifacts/tables/phase07_best.pt \
  --epochs 150 --router graphsage --graph_type split-local \
  --k 10 --graph_weight 0.5 --freeze_epochs 20 \
  --temperature 2.0 --batch_size 32 --lr 1e-3

# Phase 8 — LLM modality ablations (L0–L9)

## Primary: run all levels sequentially
# Auto-detects GPU, runs L0-L5 with real Mistral/CLAP/LLaVA extraction + 30 epochs each.
# First run: ~14-19 hours (extraction + training). After caching: ~1-3 hours.
bash scripts/run_phase08_all.sh --execute --epochs 30 --device cuda

# To skip LLM extraction (use cache from previous run):
bash scripts/run_phase08_all.sh --execute --epochs 30 --device cuda --skip_extraction

# Generate summary report (CSV + figures) from existing results:
bash scripts/run_phase08_all.sh --report

# Resume from checkpoint (skip completed levels):
bash scripts/run_phase08_all.sh --execute --resume --device cuda

## Individual levels (alternative to the full orchestration above)
# L0: Classical encoders — instant (reuses Phase 5 results, no GPU needed)
uv run python scripts/phase08_llm_ablations.py --ablation L0 --epochs 30

# L1: Mistral-7B frozen text encoder — requires GPU for feature extraction
#     Features cached to data/features/llm/L1/ after first run
uv run python scripts/phase08_llm_ablations.py --ablation L1 --epochs 30 --device cuda

# L2: Mistral + LoRA (r=16, alpha=32) — trainable adapter, requires GPU
uv run python scripts/phase08_llm_ablations.py --ablation L2 --epochs 30 --device cuda

# L3: CLAP audio LLM features — requires GPU
uv run python scripts/phase08_llm_ablations.py --ablation L3 --epochs 30 --device cuda

# L4: LLaVA-1.5-7B video features — requires GPU
uv run python scripts/phase08_llm_ablations.py --ablation L4 --epochs 30 --device cuda

# L5: Full LLM stack (text + audio + video) — requires GPU
uv run python scripts/phase08_llm_ablations.py --ablation L5 --epochs 30 --device cuda

# L6-L9: External API stubs (ImageBind, LLM teacher, direct prompting, GraphXAIN)
#        These require external APIs/services and are not locally computable
uv run python scripts/phase08_llm_ablations.py --ablation L6  # → "Requires external API"
uv run python scripts/phase08_llm_ablations.py --ablation L7  # → "Requires external API"
uv run python scripts/phase08_llm_ablations.py --ablation L8  # → "Requires external API"
uv run python scripts/phase08_llm_ablations.py --ablation L9  # → "Requires external API"

# Generate summary report from all L0-L5 results
uv run python scripts/phase08_llm_ablations.py --generate_report

# Phase 9 — Domain adaptation (CORAL, MMD, DANN, combined)
# Uses real MOSEI sentiment features (no synthetic fallback)
uv run python scripts/phase09_domain_adaptation.py --method mmd
uv run python scripts/phase09_domain_adaptation.py --method coral
uv run python scripts/phase09_domain_adaptation.py --method dann
uv run python scripts/phase09_domain_adaptation.py --method combined

# Phase 10 — Calibration + statistical validation
uv run python scripts/phase10_calibration.py --dataset daic --method temperature
uv run python scripts/phase10_calibration.py --dataset daic --method isotonic
uv run python scripts/phase10_calibration.py --dataset daic --method platt

# Phase 11 — XAI (SHAP, GNNExplainer, GraphXAIN)
# Evaluates full 768-dim audio embeddings (not truncated to 512)
uv run python scripts/phase11_xai.py --sample_id daic_test_001 --explain_mode shap
uv run python scripts/phase11_xai.py --sample_id daic_test_001 --explain_mode gnn
uv run python scripts/phase11_xai.py --sample_id daic_test_001 --explain_mode graphxain

# Phase 12 — Thesis chapter
uv run python scripts/phase12_thesis.py --output_dir paper/
```

## Full Pipeline Run

```bash
# Run all phases 0 → 12
uv run python scripts/run_full_pipeline.py

# Run phases 0 → 7 only (core architecture)
uv run python scripts/run_full_pipeline.py --stop_phase 7

# Run only phases 3 → 12 (skip setup phases)
uv run python scripts/run_full_pipeline.py --start_phase 3

# Dry run — print commands without executing
uv run python scripts/run_full_pipeline.py --dry_run

# Run only a specific phase
uv run python scripts/run_full_pipeline.py --phase 3
```

## Project Structure

```
src/
  data/          — dataset loaders (DAIC, MOSEI, FI), MultimodalDataset, preprocessing, graph_builder
  models/        — encoders, fusion (LMF/gated), MMoEEx, GNN routers, task heads
  training/      — Lightning trainers, losses, samplers, calibration
  evaluation/    — metrics, statistics, visualizations, XAI engine, graph_xai
  utils/         — seed, logging, registry
data/
  features/      — cached extracted features (NOT committed — see .gitignore)
  flags/         — low-quality sample flags
configs/         — dataset_contract.yaml, experiment configs
scripts/         — phase scripts + run_full_pipeline.py
  test_verify.py         — Phase 0 verification
  phase01_eda.py         — Phase 1 EDA (no args)
  phase02_preprocess.py  — Phase 2 feature extraction
  phase03_unimodal_baselines.py  — Phase 3
  phase04_fusion.py      — Phase 4
  phase05_mmoeex.py      — Phase 5
  phase06_graph.py       — Phase 6
  phase07_joint_training.py  — Phase 7
  phase08_llm_ablations.py   — Phase 8
  phase09_domain_adaptation.py — Phase 9
  phase10_calibration.py  — Phase 10
  phase11_xai.py          — Phase 11
  phase12_thesis.py       — Phase 12
  run_full_pipeline.py    — full pipeline orchestrator
  phase03_soa_comparison.py — SoA benchmark comparison module
artifacts/
  figures/       — phase_XX_name/ subdirectories for all visualizations
  tables/        — results CSVs (baselines, SoA comparison)
  references/    — SoA source bibliography for thesis report
paper/           — thesis chapter drafts
```

## Dataset Paths

| Dataset | Path |
|---------|------|
| DAIC-WOZ | `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/daic/raw` |
| CMU-MOSEI | `/home/anilson/projects/posei-dataset/data/CMU-MOSEI` |
| ChaLearn FI | `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/fi/raw` |

## Package Manager

Always use `uv` — **never** `pip` or `conda`:
```bash
uv sync                     # install dependencies from pyproject.toml
uv run python <script.py>   # run a script with dependencies available
uv add <package>            # add a dependency
uv run pytest               # run tests
```

## Key Constraints

- **Leakage-safe graph protocol**: build train/val/test KNN graphs **separately**; never allow cross-split edges in primary results
- **Subject-independent splits**: DAIC splits by participant ID, **never** by segment/turn
- **Modality masks required**: every sample must declare which modalities are available
- **MOSEI dominance risk**: use temperature-balanced or task-balanced sampling (MOSEI is 120× larger than DAIC)
- **Visualization-first**: every phase outputs ≥1 figure to `artifacts/figures/phase_XX_name/`

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

## Current Status

| Phase | Status | Key Outputs |
|-------|--------|-------------|
| 0 — Setup | ✅ Complete | `uv` environment, 24 stub modules |
| 1 — EDA | ✅ Complete | 8 EDA figures, dataset contract, leakage checks |
| 2 — Preprocessing | ✅ Complete | 144,641 feature files, manifest.json, 7+ figures (UMAP: PCA, AU plots: skip all-zero) |
| 3 — Unimodal Baselines | ✅ Complete | 17 figures, CSV fixed (trivial_value matches metric), SoA comparison |
| 4 — Fusion | ✅ Complete | Gated/LMF/LR-DGN fusion baselines, LR-DGN with r=16 (57.9K params vs 64.8K cross-attn) |
| 5 — MMoEEx | ✅ Complete | DAIC AUROC=0.5471, MOSEI CCC=0.4762, Emotion AUC=0.6906, FI CCC=0.5688 |
| 6 — Graph Construction | ✅ Complete | KNN graphs, NeighborLoader for global graph sampling, 8 figures |
| 7 — Joint Training | ✅ Complete | GG-MoE: DAIC AUROC 0.5471→0.5797 (+6%), 150 epochs, 4 figures |
| 8 — LLM Ablations | ✅ Complete | L0-L5 implemented, L6-L9 stubs, CSV + 3 figures, QA validated |
| 9 — Domain Adaptation | ✅ Complete | Real MOSEI integration (no synthetic fallback), CORAL/MMD/DANN/combined |
| 10 — Calibration | ✅ Complete | Temperature/Platt/isotonic scaling, BCa bootstrap CIs, DeLong tests |
| 11 — XAI | ✅ Complete | Full 768-dim audio in SHAP + perturbation (not truncated to 512), 18 figures |
| 12 — Thesis | ✅ Complete | chapter_8.tex, all tables and figures |

### Phase 2 Details (Visualization Fixes Applied)

- **UMAP**: PCA projection (50-dim common) instead of zero-padding — no artificial dataset clustering
- **Spectrograms**: `imshow` heatmap instead of `librosa.specshow` on latent embeddings
- **AU plots**: Skip all-zero tensors (explicit `np.abs(features).sum() < 1e-6` check)
- **Regeneration**: `uv run python scripts/phase02_preprocess.py --only-visualize` (no re-extraction)

### Phase 3 Details (Trivial Baseline Fix Applied)

- **Models**: sklearn (LogisticRegression, Ridge, RidgeCV) + MLP fallback for DAIC text
- **Metrics**: AUROC (DAIC), CCC (MOSEI), Avg CCC (FI) — all with bootstrap 95% CIs
- **CSV fix**: `trivial_value` now matches metric (0.5 for AUROC, majority-class acc for Acc/F1)
- **Figures**: 6 core + 9 UMAPs + 3 SoA comparison + 1 DAIC F1 supplement
- **`--only-visualize`**: Regenerate all plots from cached CSV (no retraining needed)
- **Key finding**: Only DAIC text reliably beats random baseline (AUROC=0.5952). FI audio CCC=0.4476 beats SoA (CRNet 2024 CCC≈0.34). All other unimodal combos show significant gaps to SoA, confirming need for multimodal fusion.

### Phase 4: LR-DGN (Low-Rank Dynamic Gating Network)

LR-DGN replaces cross-attention for small clinical datasets (DAIC n=107). Key design:
- Low-rank bottleneck: rank $r=8$ (28.7K params) or $r=16$ (57.9K params) vs 64.8K for cross-attention
- Context-aware gating: modalities projected to low-rank space → concat → MLP(48→24→3) → gates
- Prevents overfitting that caused cross-attention to fail on DAIC
```bash
uv run python scripts/phase04_fusion.py --dataset daic --fusion lrdgn  # r=16 default
uv run python scripts/phase04_fusion.py --dataset daic --fusion lrdgn --lrdgn_rank 8  # max regularization
```