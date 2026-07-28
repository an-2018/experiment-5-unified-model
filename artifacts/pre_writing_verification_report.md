# Pre-Writing Verification Pass

**Date:** 2026-07-27
**Purpose:** four checks requested before any manuscript prose, in risk order.

## 1. Split/sample-id consistency across the ladder — FOUND A REAL PROBLEM

The headline figure's candidate "raw text" rung (0.671) comes from
`artifacts/tables/unimodal_baselines.csv`, produced by `scripts/phase03_unimodal_baselines.py`
— an entirely separate script using a differently-configured `LogisticRegression`
(not the nested-CV pipeline used everywhere in E1), from an early, disconnected
phase of the project. Both pipelines use the same underlying DAIC split CSVs and
land on the same test n=47, but the classifier fitting protocol differs, and
provenance for the 0.671 run (exact preprocessing, model selection) is not
re-verifiable from this session. **This is the same failure class as the historical
0.671-vs-0.699 bug** the plan's own risk register names.

**Fix applied:** do not use 0.671 in any new figure. The E1 pipeline already computes
a same-protocol, same-sample_id raw-text control (`E1_text_only_auroc` in every
`phase1_gate_e1_e7_*.json`) — **0.5844, identical across all 5 checkpoints** (deterministic:
same raw features, same LR-CV fit each time). This is now the verified "raw text" rung.
Root cause of the 0.671 discrepancy could not be pinned down (ruled out feature
loading and classifier hyperparameters; see `artifacts/text_only_auroc_discrepancy_report.md`)
— documented as an unresolved provenance issue in the reproducibility appendix,
not silently replaced.

**UPDATE: with raw text corrected to 0.584, the "monotone decreasing with
construct alignment" framing does not survive.** Corrected ordering:

| Representation | Mean AUROC (5 seeds) |
|---|---|
| Random projection (12-dim) | 0.6160 ± 0.0174 |
| PCA-12 | 0.6095 ± 0.0397 |
| Raw text (768-dim, same protocol) | 0.5844 (deterministic) |
| **Construct profile (12-dim)** | **0.5238 ± 0.0394** |

Raw text now sits *below* both projections, so a clean monotone ordering across
all four is not the right framing. **The headline is the matched-dimensionality
comparison specifically**: at 12 dimensions, random ≈ PCA ≫ construct — three
projections of the identical input, identical dimensionality, identical
protocol, differing only in whether the projection is construct-aligned. That
is a controlled experiment. Raw text is reported as a separate observation
about a completely different (768-dim, unprojected) representation at this
sample size — alongside the matched-dimensionality comparison, not folded into
the same ordering.

**Important scope note for the write-up:** the "fused" 256-dim representation
(input to PCA-12/random-12/profile) is itself produced by `text_projector`, a
layer trained jointly across all 4 tasks *including DAIC's own depression loss*
— it is not a construct-naive representation in the strictest sense. The precise
claim the matched-dimensionality comparison supports is: **going from the shared
multi-task representation to the per-task construct-supervised heads
(majority-trained on MOSEI/FI, not DAIC) destroys depression-relevant variance
that survives an unsupervised or random projection of that same shared
representation.** Not "any representation of the raw signal beats construct
heads" — the comparison is specifically at the fused-representation-to-profile
step.

## 2. Statistical power — claims separated, then corrected again

See `artifacts/e1_statistical_treatment_report.md`. Original pass:
- Weak claim ("profile doesn't decode depression"): correctly described as
  underpowered at n=47 — can exclude a large effect, not a small one.
- Strong claim (inversion): sign test 5/5, p=0.0312; paired t-test across seeds
  p=0.0003; 95% CI on mean delta [-0.0781, -0.0479].

**Caught on review: that cross-seed CI only propagates training-randomness
uncertainty** (every seed shares the same 47 test subjects, so subject-sampling
error is invisible to it). Fixed with a subject-level bootstrap (2000 resamples
of the 47 test participants, delta averaged across all 5 seeds' fixed
predictions within each resample): **mean delta -0.0756, 95% CI [-0.1186,
-0.0369], entirely below zero, 99.95% of resamples negative.** This is the
correctly-powered headline statistic — it propagates both training-randomness
and subject-sampling uncertainty and the inversion survives both. Per-seed
DeLong tests against individual random-projection draws remain individually
underpowered (10/100 significant) — expected, not contradictory, at n=47.

## 3. In-domain control dimensionality — VERIFIED CORRECT, absolute numbers added

Re-checked `analysis/e1_indomain_control.py`: the random-projection baseline is
already generated at `matched_dim = len(kept)` = 2 — not 7 (pre-gate) or 12.
Apples-to-apples confirmed, no fix needed.

**Added per request:** absolute AUROCs with CIs for both arms (not just the
delta) — profile 0.513 [0.345, 0.683], matched-dim random-projection ensemble
0.639 [0.474, 0.788]. These CIs overlap substantially: neither arm is
individually well-established at n=47 with only 2 usable dimensions. This is
the thinnest evidence in the chain — corroborating (consistent sign with every
other control) but not independently decisive on its own. See
`artifacts/e1_indomain_control_report.md` for the full reframing.

## 4. MPDD 0.719 out-of-sample check — the number itself needed fixing

Checked subject-level overlap in `src/data/mpdd_loader.py`'s Young-track split:
found real leakage (subject 077 in train+val, subject 093 in val+test) caused
by splitting on row index within the segment list with no subject grouping,
despite the loader's own docstring claiming subject-independence. **Fixed the
loader** to assign whole subjects to splits (Task #10). Separately found and
fixed a pseudo-replication issue: Big-Five scores and the depression label are
both subject-level constants, so evaluating at the segment level silently
duplicated each subject 3-4x with byte-identical rows, inflating apparent CI
precision. Fixed to evaluate one row per subject.

**Corrected numbers:** Young 0.667 [0.273, 1.000] (n=13 test subjects — CI
crosses down to near-chance, essentially uninformative alone); Elderly 0.925
[0.727, 1.000] (n=13); **pooled Young+Elderly 0.819 [0.611, 0.974] (n=26) is
the number to cite** — legitimate to pool here since this arm uses only
real labels with no cross-track feature transfer involved. Even the pooled
number is framed as corroborating the clinical literature (Kotov et al. 2010),
not independently establishing construct informativeness on its own. See
`artifacts/mpdd_replication_attempt_report.md` for the full update.

## 5. PCA-12 addition (cheap, requested)

Done — see table above. PCA-12 (0.6095 ± 0.0397) lands within noise of random
projection (0.6160 ± 0.0174), both clearly above the construct profile
(0.5238 ± 0.0394). This closes the "any dimensionality reduction would do this"
alternative explanation: the effect is specific to construct-aligned supervision,
not dimensionality reduction per se.

## Status

All four original checks complete, plus a second round after review surfaced
four more issues: (1) the ladder framing had to change once 0.671 was
corrected — matched-dimensionality comparison replaces the monotone-ordering
claim; (2) the cross-seed CI was answering the wrong uncertainty question,
fixed with a subject-level bootstrap; (3) MPDD's headline number had a real
loader leakage bug plus a pseudo-replication issue, both fixed, number revised
from 0.719 to a pooled 0.819 [0.611, 0.974]; (4) the in-domain control's
absolute numbers show overlapping CIs, downgrading it from "rules out domain
shift" to "corroborates, doesn't independently decide." The core finding
(matched-dimensionality inversion, 5 seeds, degeneracy-controlled, subject-level
bootstrapped, PCA-closed) is unaffected and is now on materially firmer footing
than before this round. Ready to freeze the claim ledger next.
