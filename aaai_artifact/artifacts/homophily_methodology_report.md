# Homophily Diagnostic: Untrained vs. Trained Representation

**Date:** 2026-07-28

## The methodological issue

`scripts/diagnose_graph_homophily.py` (the standalone homophily diagnostic) and
`scripts/phase07_joint_training.py`'s actual graph construction (line ~1424,
`load_all_dataset_embeddings()`) both build the routing/diagnostic KNN graph over
embeddings from a **freshly-initialized, untrained `GatedLateFusion` module** —
not the trained `UnifiedMMoEEx` model's own fused representation (`self.fusion`
inside the model class, trained jointly with everything else, is a *separate*
instance). No checkpoint is ever loaded in `load_all_dataset_embeddings()`. This
is true of every V0–V4 run in this project's history, not something introduced
this session.

This means the existing homophily numbers (DAIC 0.568 vs. 0.555 random, etc.)
describe graph structure over a random projection of the raw multimodal
features, not the model's learned semantic space — and the routing graphs
actually used during V0–V4 training are built the same way.

## What we did about it

Per instruction to run the necessary experiments rather than leave this
undetermined: `analysis/homophily_trained_representation.py` recomputes the
identical diagnostic (same `edge_agreement` label-agreement/correlation logic)
using each trained non-graph MMoEEx checkpoint's own `get_fused_representation`
output, across all 6 available baseline checkpoints (original + 5 seeds) and
all 4 leakage-safe variants (V0, V1, V3, V4 — V2/transductive excluded, per
`diagnose_graph_homophily.py`'s own pre-existing note that its train/val/test
edge assignment is scrambled for transductive graphs, a separate bug).

## Result: more nuanced than either the untrained-embedding story or a clean confirmation

Aggregated across 6 checkpoints, real-edge vs. random-pairing baseline,
paired t-test on the delta:

| Variant | DAIC delta | DAIC p | MOSEI delta | MOSEI p | FI delta | FI p |
|---|---|---|---|---|---|---|
| V0 (inductive, k=10) | +0.064 | <0.0001 | +0.326 | <0.0001 | +0.500 | <0.0001 |
| V1 (split-local, k=10) | -0.012 | 0.363 | +0.687 | <0.0001 | +0.366 | <0.0001 |
| V3 (inductive, k=15) | +0.042 | <0.0001 | +0.329 | <0.0001 | +0.484 | <0.0001 |
| V4 (split-local, k=15) | -0.001 | 0.945 | +0.606 | <0.0001 | +0.348 | <0.0001 |

**MOSEI and FI show strong, highly significant real-vs-random separation in
every variant** — consistent with (and even stronger than) the untrained-
embedding version. **DAIC is graph-topology-dependent**: inductive
construction (V0, V3) shows a small but statistically real signal above
chance; split-local construction (V1, V4) does not. This is a genuinely
different, more specific finding than either "DAIC carries no signal in any
graph" (the untrained-embedding story) or a uniform confirmation.

## Reconciling with Table 1 (5-seed routing performance)

Per `analysis/routing_table1_statistics.py` (reportability-gated, SPEC-H4-03):
**no variant shows a reportable DAIC AUROC change from baseline** — not V0,
not V3, despite both showing real graph signal above chance in the trained
representation. This means the router's failure on DAIC is not simply "no
signal to route on" (that would only explain V1/V4); for V0/V3, real signal
exists in the topology but does not translate into a measurable depression-
detection benefit. FI, by contrast, shows the opposite pattern: strong real
signal in every variant, and a robust, statistically reportable *negative*
performance effect in most variants (3/5 reportable at $p<0.05$, the other 2
borderline at $p\approx0.05$-$0.06$) — the router has access to real FI signal
and using it makes FI prediction reportably worse, not merely fails to help.

## What this means for the paper's framing

The Discussion's original claim that the homophily diagnostic and the E1
construct-profile analysis are "two measurements of the same underlying
geometry" does not hold as stated: they use different representations
(untrained-fusion embedding vs. the model's own trained fused representation).
The corrected, more precise story:

1. The graph actually used to route experts in V0-V4 is built over an
   untrained, random projection of raw features (disclosed as a limitation,
   not fixed retroactively — fixing it would mean re-deriving the entire
   training pipeline's graph construction, out of scope here).
2. A supplementary diagnostic computed on the trained representation shows
   DAIC's graph-topology-dependent signal (present for inductive, absent for
   split-local) and strong MOSEI/FI signal in all variants.
3. Table 1 shows no reportable DAIC change in any variant (including the two
   with real trained-representation signal) and a robust FI degradation in
   most variants (despite strong FI signal in all of them) — the clean
   "signal absent -> no benefit, signal present -> router converts it"
   narrative does not survive 5-seed reportability-gated scrutiny. The
   defensible claim is narrower: routing does not reportably help DAIC or
   MOSEI at this sample size and seed count, and reportably *hurts* FI despite
   real recoverable signal being present.

This is a weaker, more honest characterization of the routing result than
what was previously written, consistent with treating it as supporting/
convergent material rather than the paper's central finding (which remains
the construct-bottleneck result in Section~\ref{sec:construct-bottleneck},
unaffected by any of this).
