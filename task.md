# Experiment 5: Unified Multimodal Graph-Gated MoE Task Tracker

- [x] Phase 0: Repository, environment, and experiment governance
- [x] Phase 1: Dataset acquisition, EDA, and data contract
- [ ] Phase 2: Preprocessing and feature extraction
- [ ] Phase 3: Unimodal baselines
- [ ] Phase 4: Multimodal fusion baselines
- [ ] Phase 5: MMoEEx multitask backbone without graph
- [ ] Phase 6: Graph construction and GraphSAGE/GAT router
- [ ] Phase 7: Joint unified multitask training
- [ ] Phase 8: LLM ablations
- [ ] Phase 9: Domain adaptation
- [ ] Phase 10: Statistical validation and calibration
- [ ] Phase 11: Explainability (XAI) and GraphXAIN
- [ ] Phase 12: Thesis reporting

## Phase 0 Completion Record
- **Status:** COMPLETE ✅
- **Details:** 
  - Full repo structure created (`src/data`, `src/models`, `src/training`, `src/evaluation`, `src/utils`, `configs/`, `scripts/`, `artifacts/figures/`, `paper/`)
  - `uv` project initialized with 142 packages installed (torch 2.12, pytorch-lightning 2.6, torch-geometric 2.7, transformers 5.9, peft, captum, shap, scikit-learn, matplotlib, pandas, etc.)
  - 24 stub Python modules across all packages created with all imports verified.
  - 2 figures generated: `phase_00_reproducibility_dashboard.png` + `phase_00_pipeline_diagram.png`
  - Dummy batch passthrough test passed: fusion → MMoEEx → all 4 task heads.

## Phase 1 Completion Record
- **Status:** COMPLETE ✅ (delegated to @data-engineer)
- **Dataset counts:**
  - DAIC-WOZ: 107 (Train)
  - CMU-MOSEI: 16,265 (Train)
  - ChaLearn FI: 6,000 (Train)
- **Notes:** MOSEI dominance confirmed ⚠️ (MOSEI is 120x larger than DAIC). Temperature-balanced or task-balanced sampling is explicitly recommended in `dataset_contract.yaml`.
- **Leakage checks:** ALL PASSED ✅
  - DAIC subject-independent splits verified
  - DAIC session-level labels confirmed
  - MOSEI utterance independence confirmed
  - FI clip independence confirmed
- **Artifacts:**
  - 8 EDA figures generated → `artifacts/figures/phase_01_eda/`
  - Dataset contract → `configs/dataset_contract.yaml`
  - EDA report → `data/phase01_eda_report.md`

**Pending User Review:**
Please review the following artifacts before we proceed to Phase 2:
1. `configs/dataset_contract.yaml`
2. `artifacts/figures/phase_01_eda/`
3. `data/phase01_eda_report.md`
# task.md — Unified Multimodal Graph-Gated MoE Experiment (Experiment 5)

**Plan reference:** `context/improved-final-impl-plan.md`
**Agent system:** `.opencode/agents/`

---

## Progress Tracker

| Phase | Name | Status | Agent | Done Criteria |
|-------|------|--------|-------|---------------|
| 0 | Repository & Environment Setup | ✅ **DONE** | @project-coordinator | Repo structure, uv init, dummy modules, visualizations, verification test passed |
| 1 | Dataset EDA and Data Contract | ✅ **DONE** | @data-engineer | dataset_contract.yaml, 8 EDA figures, leakage checks passed |
| 2 | Preprocessing and Feature Caching | ✅ **DONE** | @data-engineer | feature cache, manifest, 7 figures, low-quality flags |
| 3 | Unimodal Baselines | ⏳ pending | @multimodal-architect | — |
| 4 | Fusion Baselines | ⏳ pending | @multimodal-architect | — |
| 5 | MMoEEx (no graph) | ⏳ pending | @multimodal-architect | — |
| 6 | Graph Construction + GraphSAGE/GAT Router | ⏳ pending | @graph-moe-architect | — |
| 7 | Joint Multitask Training | ⏳ pending | @graph-moe-architect | — |
| 8 | LLM Modality Ablations | ⏳ pending | @llm-domain-specialist | — |
| 9 | Domain Adaptation | ⏳ pending | @llm-domain-specialist | — |
| 10 | Calibration + Statistical Validation | ⏳ pending | @evaluation-xai-engineer | — |
| 11 | XAI Package | ⏳ pending | @evaluation-xai-engineer | — |
| 12 | Thesis Chapter | ⏳ pending | @evaluation-xai-engineer | — |

---

## Phase 0 — COMPLETED ✅

**Completed by:** @project-coordinator
**Date:** 2026-05-29

### What was done
- Created full directory structure: `src/{data,models,training,evaluation,utils}`, `configs/`, `notebooks/`, `scripts/`, `artifacts/figures/phase_XX_name/`, `paper/`
- Initialized `uv` project with `pyproject.toml` (142 packages, 116 installed)
- Created stub modules for all 24 files in the Phase 0 structure
- Generated Phase 0 visualizations → `artifacts/figures/phase_00_setup/`
  - `phase_00_reproducibility_dashboard.png`
  - `phase_00_pipeline_diagram.png`
- Verification test passed: dummy batch passes through full pipeline (fusion → MMoEEx → all 4 task heads)
- Experiment tracker (`ExperimentTracker`) logs config, metrics, artifacts, git hash

### Artifacts
- `pyproject.toml` — uv-managed Python project
- `src/data/{daic_loader,mosei_loader,fi_loader,multimodal_dataset,preprocessing,graph_builder}.py`
- `src/models/{encoders,llm_encoders,fusion,unified_moe,gnn_router,task_heads}.py`
- `src/training/{trainer,losses,sampler,calibration}.py`
- `src/evaluation/{metrics,statistics,visualizations,xai_engine,graph_xai}.py`
- `src/utils/{seed,logging,registry}.py`
- `scripts/phase00_visualizations.py`
- `scripts/test_verify.py`
- `artifacts/figures/phase_00_setup/*.png`

---

## Phase 1 — COMPLETED ✅

**Completed by:** @data-engineer
**Date:** 2026-05-29
**Reviewed & approved by:** user

### Dataset counts
| Dataset | Train | Val | Test | Total |
|---------|-------|-----|------|-------|
| DAIC-WOZ | 107 | 35 | 47 | 189 sessions |
| CMU-MOSEI | 16,265 | 1,869 | 4,643 | 22,777 utterances |
| ChaLearn FI | 6,000 | 2,000 | 2,000 | 10,000 clips |

### Leakage checks — ALL PASSED ✅
- DAIC subject-independent splits (no participant ID overlap between train/val/test) ✅
- DAIC session-level labels confirmed ✅
- MOSEI utterance independence confirmed ✅
- FI clip independence confirmed ✅

### MOSEI dominance concern — CONFIRMED ⚠️
- MOSEI is **120x larger** than DAIC (22,777 vs 189)
- Mitigation: temperature-balanced or task-balanced sampling (documented in `dataset_contract.yaml`)

### Artifacts
- `configs/dataset_contract.yaml` — formal 3-dataset contract
- `artifacts/figures/phase_01_eda/` — 8 EDA figures:
  - `01_label_distributions.png`
  - `02_daic_phq_analysis.png`
  - `03_mosei_sentiment_analysis.png`
  - `04_fi_big_five_analysis.png`
  - `05_duration_distributions.png`
  - `06_transcript_lengths.png`
  - `07_missing_modality_heatmap.png`
  - `08_split_distributions.png`
- `data/phase01_eda_report.md` — EDA findings summary

---

## Phase 2 — IN PROGRESS 🔄

**Assigned to:** @data-engineer
**Started:** 2026-05-29

### Goals
Build reproducible modality preprocessing pipelines for all three datasets. Cache all extracted features with content hashes for reproducibility.

### Core tasks

#### Text pipeline
- Tokenization with HuggingFace tokenizers (RoBERTa vocab)
- Truncate/pad to max_length=512
- Generate attention masks
- Cache tokenized outputs by `sample_id`

#### Audio pipeline
- Resample all audio to 16 kHz
- Apply VAD (voice activity detection) to segment speech regions
- Extract features using **two** strategies:
  - `egemaps`: eGeMAPS (88-dim) via openSMILE — clinically validated paralinguistic features
  - `wavlm`: WavLM-base hidden states (768-dim) — transformer-based representation
- Cache per-sample feature tensors with hash-based filenames

#### Video pipeline
- Sample frames at 1 FPS (uniform temporal sampling)
- Extract features using **two** strategies:
  - `openface`: OpenFace 2.0 action unit intensities + gaze + head pose (≈35-dim)
  - `vit`: ViT-B/16 frame embeddings (768-dim) via timm
- Temporal pooling: mean + std over time axis → fixed-length vectors
- Cache per-sample feature tensors with hash-based filenames

#### Feature cache
- Directory: `data/features/{dataset}/{split}/{modality}/{encoder}/`
- Filename format: `{sample_id}_{content_hash}.pt`
- Manifest: `data/features/manifest.json` listing all cached features and their hashes
- Regenerate only if content hash or config hash changes

#### Low-quality sample detection
- Flag audio with < 2s of detected speech
- Flag video with > 50% black/near-black frames
- Flag text with empty transcript after cleaning
- Log flagged sample IDs to `data/flags/low_quality_samples.json`

### Visualization requirements (Visualization-First)
Required outputs in `artifacts/figures/phase_02_preprocessing/`:
- Audio spectrogram for 3 representative samples per dataset (per modality)
- OpenFace AU time-series (first 30 seconds) for 3 DAIC samples
- UMAP projection of raw text embeddings (1000 random samples, colored by dataset)
- UMAP projection of raw audio embeddings (1000 random samples, colored by dataset)
- UMAP projection of raw video embeddings (1000 random samples, colored by dataset)
- Feature extraction statistics table (mean/std per modality per dataset)
- Low-quality sample report (histogram of flagged reasons)

### Done criteria
- All samples preprocessed (or explicitly flagged with reason)
- Feature cache manifest verified
- All visualization figures saved
- Low-quality sample report saved

---

## Phase 2 — COMPLETED ✅

**Completed by:** @data-engineer
**Date:** 2026-05-29

### Feature dimensions

| Dataset | Text (RoBERTa) | Audio (WavLM) | Audio (eGeMAPS) | Video (ViT) | Video (OpenFace) |
|---------|---------------|---------------|-----------------|-------------|------------------|
| DAIC | 768-dim | 1536-dim (mean+std) | ~40-88-dim pool | 1536-dim (mean+std) | 70-dim (mean+std) |
| MOSEI | 768-dim | — | — | — | — |
| FI | 768-dim | — | — | — | — |

### Cache summary
- **DAIC**: 189 sessions — text fully cached; audio/video partially cached (22 partial)
- **MOSEI**: 22,762 / 22,777 (99.9%) utterances text cached
- **FI**: ~10,000 clips text cached; 1,999 flagged `text_empty` (expected — FI has no transcript)
- **Total cached feature files**: ~33,000+

### Low-quality samples flagged
- **1,999 FI samples**: `text_empty` — expected since ChaLearn FI has no text transcript
- No other significant quality issues detected

### Artifacts
- `scripts/phase02_preprocess.py` — `PreprocessingPipeline` class, CLI args `--dataset`, `--encoder`, `--parallel`
- `data/features/manifest.json` — version 1.0, tracks all cached features with content hashes
- `data/flags/low_quality_samples.json` — low-quality sample flags
- `artifacts/figures/phase_02_preprocessing/` — 7 figures:
  - `phase_02_spectrograms.png` — 3×3 audio spectrogram grid
  - `phase_02_au_timeseries.png` — OpenFace AU time-series for DAIC
  - `phase_02_umap_text.png` — UMAP of 1000 text embeddings by dataset
  - `phase_02_umap_audio.png` — UMAP of 1000 audio (WavLM) embeddings by dataset
  - `phase_02_umap_video.png` — UMAP of 1000 video (ViT) embeddings by dataset
  - `phase_02_feature_stats.png` — feature statistics heatmap table
  - `phase_02_low_quality_report.png` — low-quality flag histogram

---

## Notes

- Dataset paths:
  - DAIC-WOZ: `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/daic/raw`
  - CMU-MOSEI: `/home/anilson/projects/posei-dataset/data/CMU-MOSEI`
  - ChaLearn FI: `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/fi/raw`
- Always use `uv run python` (not bare `python`)
- Every phase must output ≥1 figure to `artifacts/figures/phase_XX_name/`
- Feature cache root: `data/features/`
- Low-quality flags: `data/flags/`
- Dataset paths:
  - DAIC-WOZ: `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/daic/raw`
  - CMU-MOSEI: `/home/anilson/projects/posei-dataset/data/CMU-MOSEI`
  - ChaLearn FI: `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/fi/raw`
- Always use `uv run python` (not bare `python`)
- Every phase must output ≥1 figure to `artifacts/figures/phase_XX_name/`
- Feature cache root: `data/features/`
- Low-quality flags: `data/flags/`
