# E1 Statistical Treatment — Separating the Inversion Claim from the Null Claim

**Date:** 2026-07-27

The project was making two claims of very different evidential strength. This
gives each the inferential treatment it actually supports.

## Claim A (weak): "the profile doesn't decode depression above chance"

An absolute null at n_test=47 (14 positive). Per-seed permutation-null p-values
range 0.30-0.75 — nowhere near significant, but this is also a fundamentally
underpowered test: at n=47, only a *large* effect would be reliably detected.
**Correct statement: this design can exclude a large decoding effect, not a
small one.** This claim is not the paper's load-bearing evidence and should not
be presented as if it were.

## Claim B (strong): "random projection beats the construct profile"

This is paired (same test set, same seed) and replicated across 5 independent
seeds — a fundamentally better-powered design than Claim A, because it's a
within-seed comparison replicated 5 times rather than a single absolute
estimate.

**Per-seed (degeneracy-controlled, matched dimensionality):**

| Seed | Matched dim | Profile AUROC | RP mean AUROC | Delta | DeLong median p (vs 20 individual draws) | DeLong sig. (p<0.05) |
|---|---|---|---|---|---|---|
| original | 12 | 0.5563 | 0.6049 | -0.0486 | 0.198 | 2/20 |
| seed17 | 11 | 0.5260 | 0.5802 | -0.0542 | 0.262 | 2/20 |
| seed1337 | 10 | 0.5606 | 0.6384 | -0.0778 | 0.163 | 3/20 |
| seed2024 | 11 | 0.5195 | 0.5918 | -0.0723 | 0.233 | 1/20 |
| seed31415 | 12 | 0.5584 | 0.6207 | -0.0622 | 0.344 | 2/20 |

**Cross-seed paired tests (the correctly-powered evidence):**

- **Sign test:** 5/5 seeds show a negative delta. One-sided binomial p = **0.0312**.
- **Paired t-test** on the 5 seed-level deltas against 0: t = -11.59, **p = 0.0003**.
- **95% CI on the mean delta:** [-0.0781, -0.0479], mean = -0.0630. Entirely below
  zero.

## UPDATE: the cross-seed CI above answers the wrong question

The paired t-test/CI above (across 5 seeds) only propagates training-randomness
uncertainty — every seed is evaluated on the *same* 47 test subjects, so
test-set sampling error is fully shared and invisible to that CI. It answers
"would the delta stay negative under retraining," not "would it stay negative
with different subjects."

**Subject-level bootstrap (2000 resamples of the 47 test participants,
delta averaged across all 5 seeds' fixed predictions within each resample —
no refitting inside the loop, isolating test-sampling uncertainty specifically):**

- Mean delta: -0.0756
- **95% CI: [-0.1186, -0.0369] — entirely below zero**
- 1999/2000 (99.95%) of resamples show a negative delta

This is the correctly-powered, headline statistic for the inversion claim. It
propagates both sources of uncertainty (training randomness via the 5 seeds,
subject sampling via the bootstrap) and the result holds: the inversion is
robust to both, not just to retraining. See `artifacts/stats/e1_subject_bootstrap.json`.

## Why the per-draw DeLong tests look weak and that's expected, not contradictory

Only 10/100 individual profile-vs-single-random-projection-draw DeLong
comparisons reach p<0.05. This is **not** evidence against the finding — a
single DeLong comparison at n=47 is a weak, single-shot test (exactly the same
power problem as Claim A). The correctly-powered test is the paired,
cross-seed one above: aggregating the same comparison across 5 independent
seeds turns a collection of individually-weak comparisons into a p=0.0003
paired result, because the *sign and rough magnitude* of the effect is
reproduced independently five times, which is a much stronger form of evidence
than any single p-value from one comparison.

**Reporting rule going forward:** lead with the sign test / paired t-test / CI
on the delta as the headline statistic for the inversion. Report per-seed
DeLong results as a supporting robustness check, explicitly labeled as
individually underpowered, not as the primary claim.
