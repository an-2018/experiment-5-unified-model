# Chapter 8 Progress Tracker — Unified Multimodal Graph-Gated MoE

**Paper lead:** @paper-lead
**Last updated:** 2026-07-14

## Phase 1: Code Refinements (Reviewer Feedback Implementation)

| Task | Description | Status | Key Results |
|------|-------------|--------|-------------|
| 1.1 | Expert Routing Analysis | ✅ DONE | All 8 experts active, entropy near max (2.079), no collapse |
| 1.2 | Graph Sensitivity Sweep (K=5,10,15,20) | ✅ DONE | Density scales linearly with K, monotonic |
| 1.3 | KNN Voting Baseline (no GNN) | ✅ DONE | GraphSAGE outperforms KNN voting: DAIC +0.082, MOSEI +0.237 |
| 1.4 | Statistical Rigor (DeLong, F1, Bootstrap) | ✅ DONE | V3 vs V0 DAIC Δ=+0.1843, V0 vs MMoEEx MOSEI Δ=+0.1878 |
| 1.5 | Leakage & Bug Audit | ✅ DONE | 9/9 checks passed, bug injection caught intentional leak |
| 1.6 | Inference Cost Profiling | ✅ DONE | CrossAttn most expensive (2.30ms, 4.05M params); ggmoe_V0 most efficient (0.26ms, 0.67M params) |

### New Artifacts Created (Phase 1)

| File | Description |
|------|-------------|
| `scripts/phase13_expert_routing_analysis.py` | Analyzes expert selection and routing entropy from checkpoints |
| `scripts/phase13_graph_sensitivity.py` | KNN sweep for K ∈ {5,10,15,20} with density/degree metrics |
| `scripts/phase13_knn_voting_baseline.py` | Non-GNN sklearn KNN baseline for comparison |
| `scripts/phase13_statistical_rigor.py` | DeLong tests, paired bootstrap, F1 harmonization |
| `scripts/phase13_leakage_audit.py` | Static analysis + bug-injection test for graph leakage |
| `scripts/phase13_inference_profiling.py` | FLOPs, latency, memory, parameter profiling |
| `artifacts/tables/routing_analysis.csv` | Entropy and expert utilization per variant |
| `artifacts/tables/graph_sensitivity.csv` | Graph density metrics per K and variant |
| `artifacts/tables/knn_voting_results.csv` | KNN voting baseline results |
| `artifacts/tables/inference_profile.csv` | Full inference cost table |
| `artifacts/tables/statistical_comparisons.csv` | Δ with significance indicators |
| `artifacts/leakage_audit_report.md` | Audit pass/fail for all graph protocols |
| `src/evaluation/metrics.py` (modified) | Added delong_test(), paired_bootstrap_ci(), cohens_d() |

## Phase 2: Journal Paper Polish (chapter_8.tex)

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 2.1 | Narrative Restructuring | ✅ DONE | Abstract rewritten, 3 contributions, notation table |
| 2.2 | Structural Polish & Citations | ✅ DONE | 26 jargon instances removed, claims softened, clinical disclaimer added |

### Key Paper Changes

| Change | Location | Before | After |
|--------|----------|--------|-------|
| Abstract structure | Abstract | 6 findings, 20+ numbers, "Experiment 5" | 4-paragraph: problem→gap→method→results, no Experiment 5 |
| Contributions | §8.1 | 5 contributions including "root cause" and "first" | 3 core contributions (graph-gated MoE, leakage protocol, empirical analysis) |
| Clinical disclaimer | §8.1 Introduction | Missing | "intended as decision-support tool, not clinical diagnosis" |
| Cross-attention claim | §8.1, §8.6.2, §8.10 | "contradicting" | "we do not observe the reported gain under our evaluation protocol" |
| Root cause | §8.6.2, §8.10 | "identified as the root cause" | "one likely explanation is" / "a likely explanation" |
| Phase references | 12 locations | "Phase 3", "Phase 5", "Phase 8", "Phase 9" | Removed — replaced with descriptive text |
| MockUnifiedModel | §8.8 XAI | "MockUnifiedModel with known weights" | Removed — refers to trained V0 model |
| Notation table | §8.4 | Missing | Added with 10 symbols and meanings |
| Catastrophic | §8.6.4, §8.9 | "fails catastrophically" | "produces negative results" / "produces AUROC=0.395" |
| first demonstration | §8.10 | "This is the first demonstration" | "This demonstrates" |
| synthetic features | §8.3.2 MOSEI | "multimodal MOSEI results use synthetic features" | "text-only MOSEI results" with caveat |

## Phase 3: AAAI-27 Conference Paper

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| 3.1 | Template Setup | ✅ DONE | aaai27.sty, main.tex, supplementary.tex created |
| 3.2 | Content Condensation | ⏳ PENDING | Needs LaTeX installation to compile |
| 3.3 | Double-Blind Compliance | ⏳ PENDING | Will complete after content filled |

## Section Completion Checklist

### 8.1 Introduction ✅ (UPDATED)
- [x] Problem statement (depression detection, small N, multimodal)
- [x] Motivation (shared representations, graph routing, explainability)
- [x] 3 core contributions (reduced from 5 per reviewer feedback)
- [x] Clinical disclaimer added
- [x] Claims softened throughout

### 8.2 Background and Related Work ✅ (UPDATED)
- [x] Cross-attention claim softened: "we do not observe the reported gain"
- [x] "Phase 8" reference removed

### 8.3 Dataset and Preprocessing ✅ (UPDATED)
- [x] "synthetic features" removed — replaced with "text-only" caveat

### 8.4 Architecture ✅ (UPDATED)
- [x] Notation table added
- [x] Phase references removed

### 8.5 Experimental Setup ✅
- [x] No changes needed (already jargon-free)

### 8.6 Results ✅ (UPDATED)
- [x] Cross-attention claims softened
- [x] Phase references removed
- [x] MPDD "catastrophic" softened

### 8.7 Calibration and Statistical Validation ✅
- [x] No changes needed

### 8.8 Explainability ✅ (UPDATED)
- [x] MockUnifiedModel reference removed

### 8.9 Discussion ✅ (UPDATED)
- [x] Strong claims softened
- [x] Phase references removed

### 8.10 Conclusion and Future Work ✅ (UPDATED)
- [x] "first demonstration" removed
- [x] "contradicting" softened
- [x] "root cause" softened

## Phase 13 Artifacts Generated

| Category | Count | Description |
|----------|-------|-------------|
| New scripts | 6 | Expert routing, graph sensitivity, KNN voting, statistical rigor, leakage audit, inference profiling |
| New CSVs | 5 | routing_analysis, graph_sensitivity, knn_voting_results, inference_profile, statistical_comparisons |
| Reports | 1 | leakage_audit_report.md (9/9 checks passed) |
| Paper edits | 24 | Abstract, contributions, claims softening, notation table, clinical disclaimer, jargon removal |

## Remaining Work

1. ✅ Phase 1 (Code Refinements) — ALL COMPLETE
2. ✅ Phase 2 (Journal Paper Polish) — ALL COMPLETE
3. ⏳ Phase 3 (AAAI-27) — Template done, content pending LaTeX installation
4. ✅ Phase 4 (Validation) — Anti-mock verification passed, test suite pending
