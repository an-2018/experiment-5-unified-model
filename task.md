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
