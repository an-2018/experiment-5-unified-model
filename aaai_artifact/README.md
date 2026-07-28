# Graph-Guided Mixture-of-Experts for Unified Multimodal Mental Health Assessment

Code and results artifact accompanying the AAAI 2027 submission of the same name

This paper investigates whether graph-guided mixture-of-experts (MoE) routing improves unified multimodal mental health assessment across three public benchmarks — DAIC-WOZ (depression), CMU-MOSEI (sentiment/emotion), and ChaLearn First Impressions (personality) — under a single model with modality-specific encoders, gated late fusion, an MMoEEx expert bank, and a leakage-safe KNN-graph GraphSAGE/GAT router. Across five leakage-safe graph configurations, a $K$-sensitivity sweep, and a learned per-task routing weight — all replicated across 5 seeds — routing does not improve depression detection, and a homophily diagnostic localizes why. Pursuing that diagnostic to its representational source is the paper's central result: a supervised affective-construct projection of the shared representation (sentiment/emotion/personality) is reliably **outperformed** by an unsupervised random projection or PCA of the same features at matched dimensionality, replicated across 5 seeds and a subject-level bootstrap. A second, independent corpus (MPDD) dissociates why — ground-truth Big-Five traits decode depression, but traits *estimated* from the same features do not generalize — localizing the bottleneck to construct **measurement**, not construct **validity**. LLM ablations, calibration, domain adaptation, and XAI case studies are reported separately as single-seed exploratory material only, under an explicit no-claims banner (`analysis/`, `claims/` below).

This folder is a self-contained release of the code, non-proprietary results, and reproduction instructions, including the claim-verification harness (`claims/ledger.yaml` + `claims/verify_claims.py`) that binds every quantitative statement in the paper to a machine-checked artifact — run it yourself (`uv run python claims/verify_claims.py --pre-submission`, see [Reproducing Results](#reproducing-results)) rather than taking the paper's numbers on faith. It excludes raw datasets (redistribution is not permitted by the original data licenses — see [Data Access](#data-access) below) and trained model checkpoints (large binary files; regenerable from the code, see [Reproducing Results](#reproducing-results)).

## Contents

```
aaai_artifact/
├── README.md              — this file
├── pyproject.toml         — dependency manifest (uv-managed)
├── uv.lock                — pinned dependency versions
├── .python-version        — Python 3.13
├── src/                   — core library
│   ├── data/               dataset loaders (DAIC, MOSEI, FI, MPDD), leakage-safe graph builder
│   ├── models/              modality encoders, gated fusion, MMoEEx expert bank, GraphSAGE/GAT routers, task heads
│   ├── training/            Lightning trainers, losses, temperature-balanced sampler, calibration
│   ├── evaluation/          metrics, bootstrap/permutation statistics, SHAP/GNNExplainer/GraphXAIN XAI engine
│   └── utils/               seeding, logging, module registry
├── scripts/                — numbered pipeline scripts (phase01–phase13) + orchestration shell scripts;
│                              phase05/07/13 and diagnose_graph_homophily accept --seed for the 5-seed protocol
├── analysis/                — statistical analysis scripts backing the paper's central result: matched-dimensionality
│                              construct-vs-random-projection comparison, subject-level bootstrap, MPDD ground-truth-
│                              vs-estimated dissociation, degeneracy/in-domain controls, routing-table reportability
├── claims/                  — claim-verification harness
│   ├── ledger.yaml           48 claims, each bound to a source artifact and a tolerance/check
│   └── verify_claims.py      run with --pre-submission for the strict gate used before this paper was submitted
├── configs/
│   └── dataset_contract.yaml — modality/task/split contract enforced across all three datasets
├── tests/                  — unit tests for the graph builder and GNN routers
├── artifacts/               — pre-computed, non-proprietary outputs (no checkpoints, no raw data)
│   ├── tables/               result CSVs/JSON for every phase (baselines, ablations, calibration, XAI case studies)
│   ├── figures/               every figure referenced in the paper and supplementary material
│   ├── predictions/           held-out predictions used for statistical tests
│   ├── profiles/              per-seed DAIC construct-profile parquet files (analysis/e1_e7_profile_gate.py output)
│   ├── stats/                 5-seed aggregates and reportability statistics referenced by claims/ledger.yaml
│   ├── *.md                   diagnostic reports (in-domain control, homophily methodology, reproducibility audits)
│   └── references/
│       └── bibliography.bib   full citation list
└── paper/
    ├── main-conference.tex    submitted paper source (needed for claims/verify_claims.py's text-scan checks)
    ├── supplementary.tex      supplementary material source
    ├── aaai2027.sty           AAAI style file
    ├── main-conference.pdf    the submitted paper
    └── supplementary.pdf      supplementary material
```

## Requirements

| Component | Specification |
|---|---|
| Python | 3.13 (pinned in `.python-version`) |
| Package manager | [`uv`](https://docs.astral.sh/uv/) — **do not use pip/conda directly**, dependency resolution assumes uv |
| GPU | Any CUDA GPU for Phases 3–7, 9–11 (classical encoders). **Phase 8 (LLM ablations, L1–L5)** additionally requires enough VRAM for a frozen Mistral-7B-Instruct text encoder (and CLAP/LLaVA-1.5-7B for L3/L4/L5) — a single NVIDIA RTX A6000 (48GB) is sufficient; L0 (classical baseline) runs on any GPU or CPU |
| RAM | 64GB+ recommended for full-dataset loading |
| Storage | ~50GB free for extracted feature caches once you supply the raw datasets |

## Installation

```bash
cd aaai_artifact
uv sync                              # installs all dependencies from pyproject.toml / uv.lock
uv run python scripts/test_verify.py # sanity-checks the environment (imports, CUDA availability)
```

## Data Access

None of the three primary datasets are redistributed in this artifact — all three require a data-use agreement directly with their custodians:

- **DAIC-WOZ** (clinical depression interviews, PHQ-8 labels): distributed by USC Institute for Creative Technologies under a research data-use agreement.
- **CMU-MOSEI** (YouTube review videos, sentiment + emotion labels): distributed via the CMU MultiComp Lab's CMU-MultimodalSDK.
- **ChaLearn First Impressions** (short video clips, apparent Big-Five personality labels): distributed via the ChaLearn Looking at People (LAP) challenge series.

After obtaining each dataset under its own license, populate a local `data/` directory (created at the repository root, alongside `src/` and `scripts/`) in the layout every script expects by default:

```
data/
├── daic/
│   ├── raw/            — DAIC-WOZ session zips/transcripts (USC ICT distribution format)
│   └── metadata.csv    — participant-level split/label metadata
├── mosei/
│   └── CMU-MOSEI/       — CMU-MultimodalSDK raw pickle files (mosei_senti_data.pkl, etc.)
├── fi/
│   └── raw/             — ChaLearn FI train/val/test annotation pickles + video clips
└── mpdd/                — optional: MPDD Young/Elderly track zips, for the cross-dataset
                             generalization benchmark only
```

All of the above are relative to wherever you run the scripts from (the repository root — i.e., `cd aaai_artifact` first). Once populated, run Phase 1 (`scripts/phase01_eda.py`) to validate the dataset contract (modality availability, subject-independent splits, label ranges) before proceeding to later phases.

The optional MPDD generalization benchmark (cross-dataset transfer results) uses pre-extracted Wav2Vec2 + OpenFace features distributed separately by its own organizers; see `src/data/mpdd_loader.py` for the expected feature format.

> All absolute, machine-specific paths from the original development environment have been replaced with paths relative to each script's own location (or to the `data/` layout above) so this artifact runs on any machine without editing source files — the only setup step is populating `data/` as shown.

## Reproducing Results

Every numbered script corresponds to one stage of the pipeline and writes both a results table (`artifacts/tables/`) and at least one figure (`artifacts/figures/phase_XX_.../`). The table below maps each paper section to the script and pre-computed artifact that produced it — you can inspect the pre-computed CSV/JSON/PNG directly without re-running anything, or regenerate them from raw data with the commands below.

| Paper content | Script | Pre-computed output |
|---|---|---|
| Unimodal baselines | `scripts/phase03_unimodal_baselines.py` | `artifacts/tables/unimodal_baselines.csv` |
| Fusion ablation (gated / cross-attention / LMF) | `scripts/phase04_fusion.py` | `artifacts/figures/phase_04_fusion/` |
| Non-graph MMoEEx baseline (Phase 5) | `scripts/phase05_mmoe_ex.py` | `artifacts/tables/mmoe_ex_results.csv` |
| Leakage-safe graph construction | `scripts/phase06_graph.py` | `artifacts/figures/phase_06_graph/` |
| Graph-routing ablation (V0–V4) + homophily diagnostic | `scripts/phase07_joint_training.py`, `scripts/diagnose_graph_homophily.py` | `artifacts/tables/ggmoe_results.csv`, `artifacts/figures/09_graph_homophily_diagnostic.png` |
| $K$-sensitivity sweep / learned routing weight | `scripts/phase13_graph_sensitivity.py`, `scripts/run_v2_fix_and_learned_lambda.sh` | `artifacts/tables/graph_sensitivity.csv` |
| KNN-voting baseline | `scripts/phase13_knn_voting_baseline.py` | `artifacts/tables/knn_voting_results.csv` |
| LLM encoder ablation (L0–L5) | `scripts/phase08_llm_ablations.py`, `scripts/run_phase08_all.sh` | `artifacts/figures/llm_delta_bar.png` |
| LLM-ablation significance (Cohen's *d*, DeLong *p*) | `scripts/compute_llm_ablation_stats.py` (reads `artifacts/predictions/predictions_L*.npz`) | `artifacts/tables/llm_ablation_statistics.csv` |
| Full ablation ladder (Table, supplementary) | `scripts/generate_ablation_ladder_table.py` | printed table, sourced from the CSVs above |
| Domain adaptation (CORAL/MMD/DANN) | `scripts/phase09_domain_adaptation.py` | `artifacts/figures/phase_09_domain_adaptation/` |
| Cross-dataset/cross-track generalization (MPDD) | `scripts/benchmark_mpdd.py`, `scripts/cross_track_validation.py`, `scripts/cross_dataset_mpdd_daic.py` | `artifacts/figures/feature_importance.png`, `cross_track_comparison.png` |
| Calibration + statistical validation | `scripts/phase10_calibration.py`, `scripts/phase13_statistical_rigor.py` | `artifacts/figures/calibration.png` |
| Explainability (SHAP, GNNExplainer, GraphXAIN, multi-construct profiles) | `scripts/phase11_xai.py` | `artifacts/tables/phase11_xai_results.json`, `artifacts/figures/phase_11_xai/` |
| Leakage audit | `scripts/phase13_leakage_audit.py` | printed report; exits non-zero on any detected leak |

### Quick reproduction of the central result

```bash
# Non-graph MMoEEx baseline (the comparison point for everything below)
uv run python scripts/phase05_mmoe_ex.py --dataset all --tasks depression,sentiment,emotion,personality --device cuda

# Graph-routing ablation: five leakage-safe variants (V0–V4)
uv run python scripts/phase07_joint_training.py --epochs 150 --router graphsage --graph_type split-local \
    --k 10 --graph_weight 0.5 --freeze_epochs 20 --temperature 3.0 --batch_size 32 --lr 3e-4 --device cuda

# Why doesn't the graph help? Homophily diagnostic
uv run python scripts/diagnose_graph_homophily.py
```

### Full pipeline (all phases)

```bash
uv run python scripts/run_full_pipeline.py                       # phases 0→12, in order
uv run python scripts/run_full_pipeline.py --start_phase 5 --stop_phase 7   # a specific range
uv run python scripts/run_full_pipeline.py --dry_run              # print commands without executing
```

### Verifying the claim ledger

```bash
uv run python claims/verify_claims.py                  # draft-mode check (48 claims)
uv run python claims/verify_claims.py --pre-submission # strict gate; also checks quarantined-number placement
```

Each claim in `claims/ledger.yaml` is bound to a source file under `artifacts/` (a scalar value with tolerance, a bootstrap interval, or an assertion — e.g. that no graph-routing variant shows a reportable DAIC delta from baseline, or that a discarded/irreproducible number only appears inside its designated reproducibility-appendix section). This is the same check that gated this paper before submission; running it here reproduces that gate against the pre-computed artifacts already in this folder, without needing to retrain anything.

### Testing

```bash
uv run pytest tests/
```

Covers the leakage-safe graph builder (`tests/data/test_graph_builder.py`) and the GraphSAGE/GAT router implementations (`tests/models/test_gnn_router.py`).

## Reproducibility Notes

- **Trained checkpoints are not included** (each is 30–60MB; excluded to keep this artifact lightweight). Every checkpoint is regenerable from the commands above.
- **Every result the paper treats as evidence is replicated across 5 independently trained seeds** (17, 42, 1337, 2024, 31415), not a single run: `scripts/phase05_mmoe_ex.py`, `scripts/phase07_joint_training.py`, and `scripts/phase13_knn_voting_baseline.py` each accept `--seed`; `scripts/diagnose_graph_homophily.py` accepts `--seed` for its random-projection baseline. An earlier version of this codebase had no seed argument at all in these scripts (train/test splits were fixed, but model initialization and minibatch order were not controlled), which is why the paper reports mean $\pm$ std and a reportability-gated statistic (SPEC-H4-03) rather than a single point estimate for every claim it relies on — see `claims/ledger.yaml` and the reproducibility appendix (`paper/supplementary.tex`) for six documented cases where multi-seed treatment reversed a single-seed result.
- **Leakage safety is enforced, not just claimed**: `scripts/phase13_leakage_audit.py` runs an automated audit (inductive train/test separation, split-local val/test separation, transductive cross-split-edge accounting, and an injected-bug detection check) and exits non-zero if any check fails.
- **All reported numbers in the paper are read directly from the CSV/JSON files in `artifacts/tables/`** — nothing in the paper is a number typed by hand without a corresponding artifact file.
- **The non-graph MMoEEx baseline (`phase05_mmoe_ex.py`) and the LLM ablation (`phase08_llm_ablations.py`) use a `WeightedRandomSampler` over each sample's temperature-balanced weight** so DAIC's 107 training rows aren't swamped by MOSEI's ~32k task-rows in every batch. An earlier version of this codebase computed that weight but never passed it to the `DataLoader` (plain `shuffle=True`), which silently capped the non-graph DAIC baseline at chance-level AUROC (0.493) and made every downstream graph-routing and LLM-ablation comparison against it unreliable. If you see a chance-level DAIC AUROC for the non-graph baseline in your own rerun, check that the `sampler=` argument (not `shuffle=True`) is being used on the training `DataLoader`.

## License

See `LICENSE` in this directory. This covers the code in `src/` and `scripts/` only; the three source datasets remain under their own respective licenses and are not distributed here.

## Citation

If you use this code, please cite the paper once published.
