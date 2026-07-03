# Paper Diagrams — Experiment 5 (Chapter 8)

All diagrams in this directory are fully updated and synchronized with the concrete implementations in `scripts/` and `src/` as well as the metrics in `technical_appendix_exp5.md`.

## Status Table

| Diagram | Status | Description |
|---------|--------|-------------|
| `arch_unified_model.mmd` | ✅ Completed | Concrete GG-MoE model architecture (modality policy, gated late fusion, MMoEEx experts, GNN routers, heads) |
| `arch_process_flow.mmd` | ✅ Completed | Concrete 13-phase experimental pipeline flow (Phase 0 to Phase 12) |
| `arch_visualization_map.mmd` | ✅ Completed | Concrete 13-phase mapping to corresponding visual outputs |
| `results_unimodal_bar.mmd` | ✅ Completed | Unimodal baseline results (DAIC, MOSEI, FI) |
| `results_fusion_comparison.mmd` | ✅ Completed | Fusion comparison results (unimodal, gated late, LMF, cross-attention) |
| `results_graph_ablation.mmd` | ✅ Completed | Graph-routing ablation results (MMoEEx vs V0-V4 variants) |
| `ablation_ladder.mmd` | ✅ Completed | LLM ablation ladder (L0-L5) modality replacement flow |
| `dataset_contracts.mmd` | ✅ Completed | Dataset contracts, granularities, tasks, masks, and risks |
| `graph_knn_construction.mmd` | ✅ Completed | Graph construction protocols (split-local, inductive, transductive) |
| `arch_joint_training.mmd` | ✅ Completed | Joint training loop details and per-epoch monitors |
| `calibration_reliability.mmd` | ✅ Completed | Calibration scaling and statistical validation pipeline |
| `xai_case_study.mmd` | ✅ Completed | Explainability pipeline (SHAP, GNNExplainer, perturbation, GraphXAIN) |

## Naming Convention

Pattern: `<type>_<description>.mmd`

Types:
- `arch_` (architecture)
- `results_` (results)
- `ablation_` (ablation comparison)
- `dataset_` (dataset details)
- `graph_` (graph details)
- `calibration_` (calibration details)
- `xai_` (explainability details)

## Mermaid Style

All diagrams use base theme tailored for dark readability with:
- Primary node fill: `#1a1a2e` (dark blue)
- Data/Feature node fill: `#0f3460` (medium blue)
- Expert node fill: `#2d6a4f` (green)
- Alert/Baseline node fill: `#e63946` (red)
- Task head fill: `#9d4edd` (purple)
- Edge labels in light gray `#c0c0c0`