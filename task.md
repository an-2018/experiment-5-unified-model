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
| 1 | Dataset EDA and Data Contract | 🔄 **IN PROGRESS** | @data-engineer | — |
| 2 | Preprocessing and Feature Caching | ⏳ pending | @data-engineer | — |
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

## Phase 1 — IN PROGRESS 🔄

**Assigned to:** @data-engineer
**Started:** 2026-05-29

### Goals
Confirm dataset access, splits, sample counts, label formats, and modality availability.

### Core tasks
- Load DAIC-WOZ, CMU-MOSEI, ChaLearn FI metadata
- Build dataset cards for each dataset
- Implement split validation
- Verify subject/session separation
- Define DAIC session/segment aggregation mode
- Store a `dataset_contract.yaml`

### Required outputs
- `configs/dataset_contract.yaml` — formal dataset contract
- `artifacts/figures/phase_01_eda/` — EDA visualizations:
  - Label distribution per dataset
  - DAIC PHQ-8 histogram and binary class imbalance
  - MOSEI sentiment distribution and emotion co-occurrence heatmap
  - FI Big-Five trait distributions and correlation matrix
  - Duration distributions for audio/video
  - Transcript length distributions
  - Missing modality heatmap
  - Split distribution plots
- `data/phase01_eda_report.md` — EDA findings summary

### Done criteria
- Exact counts and label distributions are saved
- Split leakage check passes
- EDA report is generated as HTML/Markdown

---

## Notes

- **HOLD for Phase 2**: Do NOT proceed to Phase 2 until I review Phase 1 EDA outputs and dataset contract.
- Dataset paths:
  - DAIC-WOZ: `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/daic/raw`
  - CMU-MOSEI: `/home/anilson/projects/posei-dataset/data/CMU-MOSEI`
  - ChaLearn FI: `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/fi/raw`
- Always use `uv run python` (not bare `python`)
- Every phase must output ≥1 figure to `artifacts/figures/phase_XX_name/`
