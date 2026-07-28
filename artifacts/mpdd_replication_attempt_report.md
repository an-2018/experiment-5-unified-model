# MPDD Replication Attempt — Reframed as a Measurement-Bottleneck Dissociation

**UPDATE 2026-07-27 (later same day):** on review, the two numbers below are not a
blocked experiment — they're a controlled dissociation that refines the paper's
thesis. Ground-truth Big Five decodes depression (real, clinically-consistent
signal, per Kotov et al. 2010); the *audio/video-estimated* version of the same
construct doesn't generalize at all (test R² -1.2 to -2.8). The constructs are
informative; they are not recoverable from behavioral multimodal signal at this
sample size. Refined thesis:

> **The bottleneck is construct measurement, not construct validity.** Affective
> constructs carry real depression signal (MPDD ground-truth Big Five, AUROC
> 0.719). Constructs *estimated* from multimodal behavioral signal are too noisy
> to carry that signal forward, and projecting through them destroys
> depression-relevant variance that survives even a random projection of the same
> features (DAIC, all 4/5 seeds).

This is a stronger position than "affective constructs don't predict depression"
(which would put the paper against Kotov's d=1.65 and a large clinical literature)
— it agrees with psychology and localizes why the ML pipeline built on that
psychology fails anyway.

**UPDATE 2026-07-27 (later still): loader leakage fixed, numbers revised with CIs.**
`src/data/mpdd_loader.py` assigned train/val/test by row index within each
track's segment list, with no subject grouping — 2 subjects (077, 093) had
segments split across train/val or val/test despite the loader's own docstring
claiming subject-independence. Fixed to assign whole subjects to splits (see
`artifacts/pre_writing_verification_report.md` for the full diagnosis).
Separately, evaluation was corrected to one row per subject (Big-Five scores
and depression label are both subject-level constants; evaluating at the
segment level silently duplicated each subject 3-4x with byte-identical
(X, y) pairs, inflating apparent CI precision without adding information).

Ground-truth Big-Five → depression, post-fix, subject-deduplicated, with CIs:

| Track | AUROC | 95% CI | n_test (subjects) |
|---|---|---|---|
| Young | 0.667 | [0.273, 1.000] | 13 |
| Elderly | 0.925 | [0.727, 1.000] | 13 |
| **Pooled (Young+Elderly)** | **0.819** | **[0.611, 0.974]** | 26 |

Individual-track CIs are very wide at n=13 (Young's crosses down to 0.27,
essentially uninformative on its own). Pooling both tracks — legitimate here
specifically because this arm uses only real, labeled personality/depression
data with no cross-track *feature* transfer involved (unlike the audio/video
prediction arm below, where the existing benchmark report documents genuine
age-group biomarker differences) — gives a materially tighter, clearly
above-chance CI. **The pooled number (0.819, CI [0.611, 0.974]) is the correct
one to cite**, with the single-track numbers reported as supporting detail, not
as the headline. Per the original recommendation: even the pooled number
should be framed as "consistent with the clinical literature" rather than as
independently establishing construct informativeness — Kotov et al. (2010)'s
much larger meta-analytic evidence carries the actual weight of that claim;
this MPDD arm corroborates rather than proves it.

Evidence base (revised):

| Corpus | Construct source | Depression AUROC | 95% CI |
|---|---|---|---|
| MPDD (pooled) | Ground-truth Big Five | 0.819 | [0.611, 0.974] |
| MPDD (Young) | Estimated from A/V embeddings | not estimable (test R² < 0) | — |
| DAIC | Estimated by cross-corpus heads | 0.524 (mean, 5 seeds) | — |
| DAIC | Random/PCA 12-dim projection | ~0.61 (mean, 5 seeds) | — |
| DAIC | Raw text (same protocol as above) | 0.584 | — |

Two independent routes to construct-estimation failure — cross-corpus transfer
(DAIC) and data starvation (MPDD) — both destroy depression signal. These are
reported as distinct mechanisms converging on the same endpoint, not blurred
into one. (Old DAIC raw-text figure 0.671 removed — see
`artifacts/pre_writing_verification_report.md` for why it doesn't belong in
this table.)

**Decision:** MPDD-Elderly is not being tried (236 vs 184 train doesn't change the
regime, and the existing benchmark report already documents Elderly-specific
prevalence/distribution-shift problems — low payoff, and trying another split
until one works would look like exactly that). The corpus is kept; only the
inversion-replication design (train an intermediate regressor, hope it
generalizes) is dropped in favor of the ground-truth-vs-estimated dissociation
already in hand.

The discarded AUROC=0.174 number (predicted-profile decoding, built on a
regressor with test R² < 0) is retained below, clearly labeled invalid, as a
documented failure mode for the reproducibility appendix — not as a result.

---

# Original report (superseded by the reframing above)

**Date:** 2026-07-27
**Goal:** replicate the DAIC E1 finding (random projection beats a construct-supervised
profile) on MPDD-Young as an independent second corpus.

## Why the DAIC checkpoint couldn't be reused directly

The unified model's projectors are architecturally tied to RoBERTa (768-dim text),
WavLM (768-dim audio), ViT (1536-dim video). MPDD has no text modality at all and
uses different encoder families entirely (Wav2Vec 512-dim audio, OpenFace 709-dim
video). Forcing MPDD features through the DAIC-trained projectors via padding/
truncation would conflate an encoder-family mismatch with the construct-validity
question — this project's own domain-adaptation findings (`artifacts/mpdd_benchmark_report.md`,
section 7C) already document that encoder-family match matters more than domain
similarity for cross-dataset transfer, so this shortcut would have been invalid on
its own logic.

## The alternative approach, and where it broke

Planned instead: train a personality projector *native* to MPDD's own feature space
(raw audio+video -> MPDD's real Big-Five trait labels, via Ridge regression),
extract its predicted 5-dim trait profile, and compare depression decoding from
that profile against a random 5-dim projection of the same raw feature space —
the same hypothesis, adapted to what MPDD actually provides.

**This failed at the first checkpoint, not the last one.** Before trusting any
depression-decoding comparison, I checked whether the trait regressor generalizes
at all (out-of-sample R² on the held-out test set). It does not:

**UPDATE (post subject-leakage fix, `artifacts/stats/mpdd_regressor_generalization.json`):**

| Trait | Train R² (plain Ridge, 1221-dim) | Test R² |
|---|---|---|
| Extraversion | 0.940 | **-2.86** |
| Agreeableness | 0.921 | **-1.75** |
| Conscientiousness | 0.966 | **-2.05** |
| Neuroticism | 0.910 | **-2.00** |
| Openness | 0.990 | **-2.42** |

(Original pre-fix numbers, superseded: Extraversion -2.82, Agreeableness -1.82,
Conscientiousness -1.61, Neuroticism -1.16, Openness -2.33 — same conclusion,
slightly different composition after the subject-leakage fix changed which
rows fall in train vs. test.)

Massive overfitting (1221 raw features, 184 train samples). Tried PCA dimensionality
reduction (5/10/20/30/50 components) with a wide RidgeCV alpha search at each size —
test R² stayed negative for 4 of 5 traits at every setting tried; only Agreeableness
ever reached a modest positive test R² (~0.09-0.15). **Big-Five personality is
essentially not predictable from MPDD's raw Wav2Vec/OpenFace features at this sample
size, with these methods.**

An initial run (before this check) that skipped straight to the depression-decoding
comparison produced a striking-looking result (predicted-profile AUROC = 0.174,
random-projection AUROC = 0.590) — but that number is not evidence of anything,
because it's built on a profile that is itself indistinguishable from noise
out-of-sample. That result was discarded, not reported, once the R² check surfaced
the problem. (Reported here only so the failure mode is visible, not to imply a
finding.)

## What is NOT invalidated

As a side observation only (not a replication of the E1 mechanism, since it
involves no construct-supervised projection step): the *ground-truth* Big-Five
scores (real labels, not predicted) decode depression at AUROC 0.719 on this same
train/test split. Real personality and depression are correlated in this corpus —
consistent with the clinical literature (Kotov et al. 2010) — but that's a
psychometric fact about the labels, not a statement about what the audio/video
features encode, and not something the paper can use as an MPDD replication of
the inversion finding.

## Status

MPDD-Young, as currently featurized in this repo, cannot support a valid replication
of the E1 inversion with the approach tried. Options, none yet acted on:

1. **Try MPDD-Elderly** (larger train n=236 vs 184) — may or may not fix the
   generalization problem; the existing MPDD benchmark report already flags
   Elderly as having its own issues (lower/different prevalence, catastrophic
   audio/video distribution shift vs. Young).
2. **Drop the MPDD replication** and rely on the DAIC-internal multi-seed +
   degeneracy-controlled result as the evidentiary base, disclosing in the paper
   that an MPDD replication was attempted and is reported as infeasible with
   available features, rather than omitted or forced.
3. **Re-extract MPDD features differently** (e.g. a lower-capacity/more standard
   personality prediction pipeline matching what the MPDD challenge baseline
   uses) — more work, uncertain payoff given the trait signal may just not be
   there in these particular pre-extracted embeddings.

Not proceeding further on this without direction, since option 1 costs real time
for an uncertain payoff and option 3 is a bigger rebuild.
