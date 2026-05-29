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

# Phase 2 — Feature extraction (use --parallel N for N workers)
uv run python scripts/phase02_preprocess.py --dataset daic --encoder all --parallel 8
uv run python scripts/phase02_preprocess.py --dataset mosei --encoder roberta --parallel 8
uv run python scripts/phase02_preprocess.py --dataset fi --encoder roberta --parallel 8

# Phase 3 — Unimodal baselines
uv run python scripts/phase03_unimodal_baselines.py --dataset daic --modality text
uv run python scripts/phase03_unimodal_baselines.py --dataset daic --modality audio
uv run python scripts/phase03_unimodal_baselines.py --dataset daic --modality video
uv run python scripts/phase03_unimodal_baselines.py --dataset mosei --modality text
uv run python scripts/phase03_unimodal_baselines.py --dataset fi --modality video

# Phase 4 — Fusion baselines
uv run python scripts/phase04_fusion.py --dataset daic --fusion gated
uv run python scripts/phase04_fusion.py --dataset daic --fusion lmf
uv run python scripts/phase04_fusion.py --dataset mosei --fusion gated
uv run python scripts/phase04_fusion.py --dataset fi --fusion gated

# Phase 5 — MMoEEx (no graph yet)
uv run python scripts/phase05_mmoeex.py --dataset all --tasks depression,sentiment,emotion,personality

# Phase 6 — Graph construction (split-local = primary results, inductive = final eval)
uv run python scripts/phase06_graph.py --graph_type split-local --k 10
uv run python scripts/phase06_graph.py --graph_type inductive --k 10

# Phase 7 — Joint multitask training
uv run python scripts/phase07_joint_training.py --epochs 50 --batch_size 32 --temperature 2.0

# Phase 8 — LLM modality ablations (L0–L9)
uv run python scripts/phase08_llm_ablations.py --ablation L0   # classical encoders
uv run python scripts/phase08_llm_ablations.py --ablation L1   # Mistral frozen
uv run python scripts/phase08_llm_ablations.py --ablation L3   # audio LLM
uv run python scripts/phase08_llm_ablations.py --ablation L5   # full LLM stack

# Phase 9 — Domain adaptation (MMD, Deep CORAL, DANN)
uv run python scripts/phase09_domain_adaptation.py --method mmd
uv run python scripts/phase09_domain_adaptation.py --method coral
uv run python scripts/phase09_domain_adaptation.py --method dann

# Phase 10 — Calibration + statistical validation
uv run python scripts/phase10_calibration.py --dataset daic --method temperature
uv run python scripts/phase10_calibration.py --dataset daic --method isotonic
uv run python scripts/phase10_calibration.py --dataset daic --method platt

# Phase 11 — XAI (SHAP, GNNExplainer, GraphXAIN)
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
artifacts/
  figures/       — phase_XX_name/ subdirectories for all visualizations
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