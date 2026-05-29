# AGENTS.md — Unified Multimodal Graph-Gated MoE Experiment (Experiment 5)

This is a **planning/orchestration repo** for a thesis experiment. No source code exists yet.

## Master Reference

**`context/improved-final-impl-plan.md`** — full roadmap, architecture, leakage protocol, ablation matrix, acceptance criteria. Read before any work.

## Subagents

Defined in `.opencode/agents/`:
- `@project-coordinator` — orchestrates the 12-phase workflow
- `@data-engineer` — Phases 0-2 (data loaders, EDA, preprocessing)
- `@multimodal-architect` — Phases 3-5 (baselines, LMF fusion, MMoEEx)
- `@graph-moe-architect` — Phases 6-7 (KNN graph, GraphSAGE/GAT routing, joint training)
- `@llm-domain-specialist` — Phases 8-9 (LLM encoders, teacher features, domain adaptation)
- `@evaluation-xai-engineer` — Phases 10-12 (statistics, calibration, XAI)

## Package Manager

Use **`uv`** exclusively (not pip, not conda):
```bash
uv init  # creates pyproject.toml
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv add pytorch-lightning torch-geometric transformers peft datasets
uv add captum shap scikit-learn
uv run python script.py
```

## Project Structure (to create)

```
src/
  data/         # dataset classes, loaders, preproc
  models/       # encoders, fusion, MMoEEx, GNN, task heads
  training/     # lightning modules, samplers, losses
  evaluation/   # metrics, calibration, statistical tests
  xai/          # SHAP, GNNExplainer, GraphXAIN
artifacts/
  figures/      # phase_XX_name/ subdirectories
context/        # plans, reports, architecture diagrams (read-only)
```

## Key Technical Constraints

- **Leakage-safe graph protocol**: build train/val/test KNN graphs separately; never cross-split edges in primary results; use inductive inference for final eval.
- **Subject-independent splits**: DAIC splits by participant ID, never by segment/turn.
- **Modality masks required**: every sample declares available modalities.
- **Visualization-first**: every phase outputs ≥1 figure to `artifacts/figures/phase_XX_name/`.
- **MOSEI dominance risk**: use temperature-balanced or task-balanced sampling to prevent utterance-level MOSEI from overwhelming session-level DAIC.

## Dataset Locations

| Dataset | Path |
|---------|------|
| DAIC-WOZ | `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/daic/raw` |
| ChaLearn FI | `/home/anilson/projects/mental-ai-emnlp-2025/daic-first-impressions-experiments/data/fi/raw` |
| CMU-MOSEI | `/home/anilson/projects/posei-dataset/data/CMU-MOSEI` |

## Dataset Granularity Contract

| Dataset | Training unit | Eval unit | Key risk |
|---------|--------------|-----------|----------|
| DAIC-WOZ | session or segment (inherited label) | participant/session | cross-segment label leakage |
| CMU-MOSEI | utterance | utterance | size imbalance |
| ChaLearn FI | clip | clip | apparent ≠ clinical personality |

## Critical Rules

- Depression is the **primary clinical task**; sentiment/emotion/personality are auxiliary supervision.
- Do not claim apparent personality and clinical depression measure the same construct.
- Every headline result: mean + 95% CI + paired statistical test (DeLong for AUROC, bootstrap for F1/CCC/MAE).
- XAI explanations must be validated with perturbation/counterfactual tests.
- LLM-generated features are derived features, not ground truth — keep them separate in reporting.
