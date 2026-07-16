# Graph-Routing Improvement Review

**Date:** 2026-07-16
**Context:** The real, re-verified V0-V4 graph-routing ablation (`scripts/phase07_joint_training.py`,
`artifacts/tables/ggmoe_results.csv`) shows graph-based routing does not improve over the
non-graph MMoEEx baseline on any task, and hurts FI personality (see chat/thesis discussion,
2026-07-16). This document reviews literature on why this could happen and what to try before
concluding it's a genuine null result.

## Most likely root cause: the routing graph may carry little task-relevant signal

The KNN graph used for routing is built once, upfront, from a **freshly / randomly initialized**
`GatedLateFusion` projection (`scripts/phase06_graph.py` and `scripts/phase07_joint_training.py`,
`load_all_dataset_embeddings`) — not from the jointly-trained model's own fused representation,
and not from any pretrained/contrastive embedding. If neighbors in this space don't actually share
similar task-relevant properties (low "homophily"), the GraphSAGE router has no useful signal to
aggregate — it can only add noise, which matches the FI personality regression we observed.

**Recommended first step (cheap, diagnostic, no retraining needed):** compute label homophily /
label informativeness (Platonov et al. 2022) directly on the already-built KNN graphs from the
real V0-V4 runs — i.e., for each edge, does the neighbor share a similar depression label /
sentiment value / personality score? If homophily is near baseline/random, that is a direct,
quantitative explanation for the null result, and argues for fixing graph construction rather
than abandoning graph routing outright.

- **Platonov et al., "Characterizing Graph Datasets for Node Classification: Homophily-Heterophily
  Dichotomy and Beyond"** (arXiv:2209.06177, 2022). Defines label informativeness as a more
  reliable predictor of GNN benefit than raw homophily ratio — better diagnostic to compute here.

## If homophily is confirmed low: candidate fixes, in order of effort

1. **Build the graph from a better embedding space**, not a randomly-initialized fusion layer:
   - Use the jointly-trained model's own fused embedding after a warmup period (re-embed
     periodically during training), or
   - Contrastively pretrain the fusion embedding (e.g., SimCLR-style) before KNN construction, so
     neighbors are semantically meaningful before routing ever sees them.

2. **Replace the static KNN graph with a learned/adaptive graph structure** (Graph Structure
   Learning, GSL) — the graph is refined jointly with the task objective instead of fixed upfront:
   - **Attali et al., "Graph Rewiring in GNNs to Mitigate Over-Squashing and Over-Smoothing: A
     Survey"** (arXiv:2605.00951, 2026). Surveys structural-fix (curvature/effective-resistance)
     and feature-aware rewiring approaches — directly applicable if our fixed KNN graph is a poor
     topology for message passing across three heterogeneous datasets (DAIC/MOSEI/FI).

3. **Replace the fixed λ=0.5 log-space combination with a learned, input-dependent weight** —
   instead of a single global scalar trusting the graph router equally for every task/sample, let
   a small gating network learn how much to trust the graph router per task. This could let the
   model learn to *ignore* the graph specifically for FI (where it currently hurts) while still
   using it where it helps.
   - Precedent: dual instance-level/task-level routing in mixture-of-modality-experts systems
     (see search notes; general MoE gating literature).

4. **Expert-choice routing instead of token/sample-choice softmax gating** — could reduce expert
   collapse and improve load balancing versus the current per-task softmax gate.
   - **Zhou et al., "Mixture-of-Experts with Expert Choice Routing"** (NeurIPS 2022,
     arXiv:2202.09368).

5. **Task-affinity-aware grouping** — directly relevant to the FI-specific regression: check
   whether FI has *negative* task affinity with DAIC/MOSEI in the shared graph space, and avoid
   forcing it through the same graph-routing pathway if so.
   - **Li et al., "Boosting Multitask Learning on Graphs through Higher-Order Task Affinities"**
     (KDD 2023, DOI 10.1145/3580305.3599265).

6. **Related architecture precedent** (not a direct fix, but relevant related work for the paper):
   - **Wang et al., "Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit
     Diversity Modeling"** (NeurIPS 2023, arXiv:2304.02806) — GNN-based experts with explicit
     diversity modeling; a related but architecturally different way of combining graphs and MoE.

## Recommendation

Run the homophily/label-informativeness diagnostic first (cheap — reuses the already-built real
graphs, no retraining). If it confirms low task-relevant signal in the graph, prioritize fix #1
(better embedding space for graph construction) and #3 (learned λ) as the highest
effort-to-reward fixes, since both are compatible with the existing architecture and don't require
redesigning the router. Fixes #2, #4, #5 are larger changes, better suited to a "future work"
paragraph unless the diagnostic points specifically at over-squashing, expert collapse, or
FI-specific negative task affinity respectively.

All six references above have been added to `artifacts/references/bibliography.bib` under keys
`platonov2022characterizing`, `attali2026graphrewiring`, `li2023boosting`, `zhou2022mixture`,
`wang2023graphmoe`.
