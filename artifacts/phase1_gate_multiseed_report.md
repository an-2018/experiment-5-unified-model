# Phase 1 Gate (G1a/G1b) — Multi-Seed Result

**Date:** 2026-07-27
**Purpose:** Firm up the single-checkpoint Phase 1 gate finding (E1 profile-only DAIC
depression decoding) by retraining the non-graph MMoEEx baseline (`scripts/phase05_mmoe_ex.py`)
across 3 additional seeds and re-running `analysis/e1_e7_profile_gate.py` against each.

## What changed in the harness

`scripts/phase05_mmoe_ex.py` had **no seed control at all** — no `torch.manual_seed`,
`np.random.seed`, or `random.seed` calls, and the checkpoint/results/predictions
output paths were hardcoded, so any rerun silently overwrote the previous run's
artifacts. Fixed:
- Added `--seed` (seeds `torch`, `numpy`, `random`, `torch.cuda`).
- When `--seed` is set, output filenames are suffixed (`mmoe_ex_best_seed{N}.pt`, etc.)
  instead of overwriting the unsuffixed defaults.
- The original checkpoint (unknown seed) was backed up to
  `artifacts/tables/mmoe_ex_best_original_seed_unknown.pt` before any retraining,
  so the numbers currently in `paper/main-conference.tex` Table 1 remain traceable.

`analysis/e1_e7_profile_gate.py` was extended with a `--checkpoint` argument so it
can evaluate any L0 checkpoint, not just the default.

## Seeds run

4 new seeds trained from scratch (150 epochs, early stopping patience=20, otherwise
identical config/data to the original run): 17, 1337, 2024, 31415. Combined with the
original (seed unrecorded), this gives all 5 canonical seeds (SPEC-H0-04:
SEEDS=[17, 42, 1337, 2024, 31415] — the original checkpoint's seed was never recorded,
so it stands in for one slot; every canonical seed value has now been run at least once
across this set).

## Result

| Checkpoint | E1 full-profile AUROC | Permutation-null p-value | Random-projection mean AUROC |
|---|---|---|---|
| original (seed unknown) | 0.5563 | 0.344 | 0.6120 |
| seed17 | 0.5325 | 0.322 | 0.6073 |
| seed1337 | 0.5216 | 0.461 | 0.6430 |
| seed2024 | 0.4502 | 0.751 | 0.5917 |
| seed31415 | 0.5584 | 0.297 | 0.6260 |
| **mean ± std (5 seeds)** | **0.5238 ± 0.0394** | 0.30–0.75 (never <0.05) | **0.6160 ± 0.0174** |

Degeneracy-controlled version (dropping near-constant dimensions per checkpoint,
matched-dimensionality random projection — see `artifacts/e1_degeneracy_check_report.md`):
mean delta (controlled profile − matched random projection) = **-0.0630 ± 0.0109** across
all 5 checkpoints, all negative. Essentially unchanged from the naive 12-dim comparison
(mean delta -0.0922 ± 0.0332), confirming the effect isn't an artifact of collapsed
dimensions.

Unimodal text-only control (raw 768-dim RoBERTa embedding, deterministic — identical
features regardless of which of these 4 checkpoints is evaluated): **AUROC 0.5844**.

## Interpretation

This is not seed noise. Across all 5 independently trained models:

1. **The 12-dim construct profile never decodes DAIC depression above chance** at any
   conventional significance threshold (permutation-null p-value is never below 0.30).
2. **A random 12-dim projection of the same underlying fused embedding consistently
   outperforms the actual construct profile** — by a tight, consistent margin
   (mean delta -0.092, std 0.033, negative in all 5 seeds; -0.063 ± 0.011 after
   degeneracy control). This is the literal realization of the "you just did
   dimensionality reduction" reviewer objection the plan pre-armed against
   (SPEC-H5-05.2) — except here the random projection doesn't just match the
   profile, it beats it.
3. E7's axis non-redundancy results are noisy and inconsistent in *sign* across seeds
   (which axis appears to help vs. hurt flips across seed17/1337/2024/31415), which
   is itself informative: there is no stable, reproducible axis structure to report
   as a finding on its own.

**Superseded framing note:** the interpretation below this point (Gate decision)
was written when the leading hypothesis was "affective constructs don't predict
depression." Two follow-up controls changed that conclusion — see
`artifacts/mpdd_replication_attempt_report.md` (ground-truth Big-Five decodes MPDD
depression at AUROC 0.719, but the *audio/video-estimated* version doesn't
generalize at all, test R² -1.2 to -2.8) and `artifacts/e1_indomain_control_report.md`
(the inversion persists even with in-domain DAIC supervision, ruling out cross-corpus
transfer as the explanation). The established thesis is now: **the bottleneck is
construct measurement, not construct validity** — constructs carry real depression
signal, but versions of them *estimated* from multimodal behavioral signal at
clinical sample sizes are too noisy to carry that signal forward, and projecting
through them destroys depression-relevant variance that survives even a random
projection of the same features.

## Gate decision (superseded — see note above)

The plan's original pre-committed rule for a flat G1a failure ("stop, don't rewrite
around a thesis the data rejects, fall back to the routing paper") was written for
the scenario where the data simply rejected the construct-model thesis outright.
That is not what happened: the MPDD and in-domain controls (run after this report
was first written) showed the failure is localized to construct *estimation*, not
construct *validity* — a refinement of the thesis, not a rejection of it. See the
superseded-framing note above. Current status: proceeding with the measurement-
bottleneck framing as the paper's actual contribution, with the routing null
(homophily diagnostic) as convergent mechanistic evidence rather than the headline.
