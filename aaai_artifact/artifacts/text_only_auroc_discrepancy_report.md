# Diagnosis: 0.671 vs 0.584 DAIC Text-Only AUROC Discrepancy

**Date:** 2026-07-27

## The two numbers

- `artifacts/tables/unimodal_baselines.csv` (from `scripts/phase03_unimodal_baselines.py`,
  run 2026-06-06): DAIC text-only AUROC = **0.671**
- E1 pipeline's same-protocol raw-text control (`analysis/e1_e7_profile_gate.py`,
  `E1_text_only_auroc`): **0.5844**, identical and deterministic across all 5
  checkpoints

## What was ruled out

1. **Feature loading path.** Both scripts read the same `text_roberta` feature
   key from the same manifest, extracting `pooled_embedding` from the same
   cached `.pt` files. Directly invoked `phase03_unimodal_baselines.py`'s own
   `build_dataset()` in this session: it returns train n=107 (30 positive),
   val n=35 (12 positive), test n=47 (14 positive) — identical composition to
   what the E1 pipeline uses. No NaNs, no duplicate rows, feature std range
   sane (0.005-0.279). Feature loading is not the source of the discrepancy.

2. **Classifier hyperparameters.** `phase03_unimodal_baselines.py`'s
   `train_and_evaluate_daic()` uses a fixed `LogisticRegression(C=1.0,
   class_weight="balanced", ...)`, not the CV-selected-C, non-balanced
   `LogisticRegressionCV` the E1 pipeline uses. Tried all four combinations
   (fixed C=1.0 / CV-selected C) x (balanced / unbalanced) on the identical
   feature arrays in this session:

   | Config | AUROC |
   |---|---|
   | C=1.0, balanced (phase03's exact config) | 0.515 |
   | C=1.0, unbalanced | 0.515 |
   | CV-selected C, unbalanced (E1's config) | 0.584 |
   | CV-selected C, balanced | 0.576 |

   **None of these four reproduce 0.671.** Hyperparameter differences explain
   the 0.515-vs-0.584 spread across configs, but not the original 0.671 value.

3. **Code changes since the original run.** `git log -p` on
   `scripts/phase03_unimodal_baselines.py` shows the `class_weight="balanced"`
   / `C=1.0` configuration has been present, unchanged, since the file's
   creation (2026-05-30) — before and after the 2026-06-06 run that produced
   0.671. The code has not changed.

## Conclusion: inconclusive, not resolved in either direction

**Directly re-running `phase03_unimodal_baselines.py`'s own `build_dataset()` +
`train_and_evaluate_daic()` — its own code, unmodified, on the feature files
currently on disk — reproduces 0.515, not 0.671.** Since the code is
unchanged and the data files' paths/mtimes are consistent with the original
run, the most likely explanation is that the underlying cached feature
tensors (`data/features/daic/*/text/roberta/*.pt`) were regenerated at some
point after the 2026-06-06 run without the change being tracked (e.g. a
re-extraction that preserved file mtimes, or an environment/library
difference in that earlier run that no longer reproduces). This could not be
confirmed further in this session — there is no manifest/feature-cache hash
history to check against (a gap that H0's checkpoint/cache SHA-256 binding,
once built, would close).

**Action taken:** 0.671 is not used anywhere in the new manuscript or
figures. Every reported number in the construct-bottleneck analysis comes
from the E1 pipeline's own same-protocol, same-sample-id computation
(0.5844 for raw text). This sentence is the reproducibility-appendix
disclosure: *an earlier project phase (Phase 3, June 2026) reported a
different DAIC text-only AUROC (0.671) than this analysis's same-protocol
recomputation (0.584); the discrepancy could not be attributed to either
feature loading or classifier configuration and remains unresolved — only
the recomputed, same-pipeline number is used in this paper.*
