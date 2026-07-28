# E1 Degeneracy Check (SPEC-H2-03 `no_constant_output`, applied post-hoc)

**Date:** 2026-07-27
**Trigger:** the ablation ladder previously reported FI CCC = 0.000 (a constant-output
collapse). Before treating "random projection beats the construct profile" as a
real finding, checked whether the profiles feeding E1 have the same collapse —
which would make the finding a trivial consequence of reduced effective
dimensionality rather than a construct-supervision effect.

## Step 1: per-dimension variance / correlation scan

`analysis/e1_degeneracy_check.py`, run on all 4 checkpoints (original + 3 retrained
seeds), computing per-dimension std, pairwise correlations, and PCA effective rank
on the combined DAIC train+test profiles.

**Finding:** `happiness` is near-constant (std ~0.0001-0.0002) in 3 of 4 checkpoints
(seed17, seed1337, seed2024) — consistent with the known rare-positive-class issue
in the emotion head noted in `scripts/phase05_mmoe_ex.py` ("fear only has 0.3% at
0.5 [threshold]"). `fear` is additionally degenerate in seed1337, which also shows
a near-duplicate personality pair (extraversion<->agreeableness, r=0.964). The
**original** checkpoint (the one currently backing Table 1 in `main-conference.tex`)
shows no degenerate dimensions by this test. This is itself worth noting: the
freshly retrained seeds are somewhat less healthy on this specific invariant than
the original — a real H2-class issue in `scripts/phase05_mmoe_ex.py`'s emotion loss
that should be tracked separately (rare-class emotion head instability), independent
of the construct-validity question.

## Step 2: degeneracy-controlled re-run

`analysis/e1_degeneracy_controlled_rerun.py`. For each checkpoint: drop dimensions
with train-set std < 1e-3 (label-blind, decided from train only), refit the E1
logistic-regression decoder on the remaining dimensions, and re-run the
random-projection control at the SAME reduced dimensionality (not 12) so the
comparison is apples-to-apples.

| Checkpoint | Dropped | Controlled profile AUROC | Matched-dim random projection (mean) | Delta |
|---|---|---|---|---|
| original | none | 0.5563 | 0.6049 | -0.0486 |
| seed17 | happiness | 0.5260 | 0.5802 | -0.0542 |
| seed1337 | fear, happiness | 0.5606 | 0.6384 | -0.0778 |
| seed2024 | happiness | 0.5195 | 0.5918 | -0.0723 |
| **mean +- std** | | | | **-0.0632 +- 0.0121** |

## Conclusion

The inversion (random projection of the fused embedding beats the supervised
construct profile) **survives dimensionality control** — the gap persists at
closely the same magnitude, with lower cross-seed variance than the naive 12-dim
comparison, after removing every dimension flagged as near-constant on train data
alone. This rules out "the profile just has fewer than 12 real dimensions" as the
explanation. The finding is a construct-supervision effect, not a collapsed-head
artifact.

Separately, the emotion-head rare-class collapse (happiness/fear near-constant in
3/4 freshly retrained seeds) is a real instrumentation issue worth fixing under
SPEC-H2-03 `no_constant_output` before any further training, independent of this
result — it affects the quality of E3/E6 (which use the emotion dimensions
directly) even though it doesn't explain the E1/E7 inversion.
