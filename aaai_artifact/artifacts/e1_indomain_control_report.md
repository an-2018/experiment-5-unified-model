# In-Domain DAIC Construct-Supervision Control

**Date:** 2026-07-27
**Purpose:** rule out cross-corpus domain shift as the explanation for the E1
inversion. The original DAIC profile's sentiment/emotion dimensions come from
heads trained on MOSEI and applied to DAIC zero-shot; this control derives
sentiment/emotion supervision *directly on DAIC* and re-tests.

## Method

1. Off-the-shelf labellers (`cardiffnlp/twitter-roberta-base-sentiment-latest`,
   `j-hartmann/emotion-english-distilroberta-base`) applied to each DAIC
   participant's own transcript (Participant-only turns, chunked to fit context
   window, averaged across chunks) — one sentiment scalar + 6-way emotion
   vector per participant, all 189. Saved to `data/daic_indomain_labels.json`.
2. A RidgeCV projector trained per-dimension on DAIC's own fused text-only
   representation (the 256-dim internal representation the unified model
   computes for DAIC, train split, n=107) to predict each in-domain label.
3. **Derived-feature validity gate applied before use** (the rule the MPDD
   attempt established): each dimension's projector must clear test-set R² > 0
   on the DAIC test split before its predictions may feed the downstream
   depression-decoding comparison.

## Gate results

| Dimension | Train R² | Test R² | Gate |
|---|---|---|---|
| sentiment | 0.544 | 0.011 | **PASS** |
| anger | 0.100 | -0.128 | FAIL |
| disgust | 0.236 | -0.860 | FAIL |
| fear | 0.024 | -0.037 | FAIL |
| happiness | 0.389 | -0.016 | FAIL |
| sadness | 0.082 | -0.090 | FAIL |
| surprise | 0.220 | 0.134 | **PASS** |

**Only 2 of 7 in-domain dimensions generalize at all** (sentiment weakly,
surprise moderately) — the other 5 show the same train/test R² divergence
pattern as the MPDD personality regressor, just less extreme. This is itself
evidence for the measurement-bottleneck thesis: even with in-domain supervision
(no cross-corpus transfer at all), most individual affective dimensions are not
reliably estimable from DAIC's n=107 training set.

## E1 result on the gated in-domain profile (sentiment + surprise, 2-dim)

**Absolute AUROCs, both arms, with CIs (requested — the delta alone was hiding
whether either arm is individually well-established):**

| Arm | AUROC | 95% CI |
|---|---|---|
| In-domain profile (2-dim) | 0.513 | [0.345, 0.683] |
| Matched-dim (2) random projection (draw-ensemble) | 0.639 | [0.474, 0.788] |
| Delta | -0.108 | (mean-of-draws; ensemble-vs-profile point delta -0.126) |

**These CIs overlap substantially** (profile's upper bound 0.683 vs. RP's lower
bound 0.474). Read plainly: the profile arm is indistinguishable from chance on
its own (CI straddles 0.5), and the RP arm, while numerically higher, is not
precisely pinned down either at this n. Neither arm is independently
well-established at n=47 with only 2 usable dimensions — this is, as
anticipated, the thinnest piece of evidence in the chain. The delta's *sign* is
consistent with every other control (cross-corpus DAIC, degeneracy-controlled,
PCA-12), which is what makes it worth reporting, but it should be framed as
**corroborating, not independently decisive** — a single point of evidence
among five converging ones, not a standalone proof that domain shift is ruled
out.

## UPDATE 2026-07-28: extended to all 5 canonical seeds — the single-checkpoint result does not replicate

The paragraphs above were written from the `original` checkpoint only. Running
the identical pipeline across all 5 canonical seeds
(`analysis/e1_indomain_control.py`, `artifacts/stats/e1_indomain_control_5seed_aggregate.json`)
gives a materially different picture:

| Seed | Gated dims (of 7) | Profile AUROC | Matched-RP AUROC | Delta |
|---|---|---|---|---|
| 17 | 3 | 0.652 | 0.595 | **+0.056** |
| 42 | 1 | 0.563 | 0.604 | -0.041 |
| 1337 | 1 | 0.491 | 0.669 | -0.177 |
| 2024 | 2 | 0.658 | 0.619 | **+0.039** |
| 31415 | 2 | 0.537 | 0.600 | -0.063 |
| **mean +- std** | | 0.580 +- 0.065 | 0.617 +- 0.027 | **-0.037 +- 0.084** |

**3 of 5 seeds show the expected negative delta; 2 of 5 show a positive one.**
A sign test on this (3 negative out of 5) gives one-sided $p=0.5$ — no better
than chance. This control does **not** replicate consistently across seeds.
Also notable: which dimensions clear the generalization gate varies seed to
seed (1-3 of 7, not consistently sentiment+surprise as in the original
checkpoint) — the in-domain projector's gate outcome is itself seed-sensitive
at $n_{\text{train}}=107$.

## Corrected interpretation

**This control is inconclusive, not corroborating.** The original
single-checkpoint framing ("the inversion's sign persists in-domain... domain
shift is not supported as the explanation") overstated what one checkpoint can
show. With proper 5-seed verification, the in-domain control neither confirms
nor rules out domain shift as a contributing factor — it is simply too noisy
at this sample size ($n_{\text{test}}=47$, 1-3 usable dimensions depending on
seed) to move the needle either way. The paper's evidence against domain
shift as the primary explanation rests on the cross-corpus DAIC result itself
(5 seeds, degeneracy-controlled, subject-bootstrapped, CI entirely below
zero) and the MPDD ground-truth-vs-estimated dissociation — both of which
remain solid — not on this specific control, which should be described as
attempted-and-inconclusive rather than corroborating.

**Caveats (updated):**
- Now seed-verified (5/5 canonical seeds) — the mixed-sign result IS the
  finding, not a gap needing further verification.
- Only 1-3 of 7 in-domain dimensions pass the gate per seed, so statistical
  power is low and unstable across seeds; both arms' absolute AUROCs (see
  above) carry wide, seed-dependent uncertainty.
- Off-the-shelf labellers are themselves an imperfect proxy for "true" sentiment/
  emotion — disclosed as a limitation, same as any pseudo-labelling approach.
