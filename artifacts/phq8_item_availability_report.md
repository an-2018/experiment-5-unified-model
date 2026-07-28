# PHQ-8 Item-Level Availability Report

**Date:** 2026-07-27
**Purpose:** Day-1 gating check for E3 (profile x PHQ-8 item correlation matrix), per SPEC-H6-04 and the pre-registration protocol (Section 4 of the AAAI resubmission plan). This determines whether E3 can run on the DAIC test split or must fall back to dev.

## Source files

Located at `/home/anilson/projects/daic-first-impressions-multimodal-experiments/data/raw/daic/` (the official AVEC2017 DAIC-WOZ split label CSVs; not vendored into this repo's `data/` tree, which only holds pre-extracted feature caches).

| File | Data rows | Columns |
|---|---|---|
| `train_split_Depression_AVEC2017.csv` | 106 | `Participant_ID, PHQ8_Binary, PHQ8_Score, Gender` + 8 item columns |
| `dev_split_Depression_AVEC2017.csv` | 34 | same as train (8 item columns present) |
| `test_split_Depression_AVEC2017.csv` | 47 | `participant_ID, Gender` only — **no PHQ scores or items at all** |
| `full_test_split.csv` | 46 | `Participant_ID, PHQ_Binary, PHQ_Score, Gender` — **binary + total only, no items** |

The 8 item columns present in train/dev (`PHQ8_NoInterest, PHQ8_Depressed, PHQ8_Sleep, PHQ8_Tired, PHQ8_Appetite, PHQ8_Failure, PHQ8_Concentrating, PHQ8_Moving`) map onto the standard PHQ-8 items (anhedonia, depressed mood, sleep, fatigue, appetite, self-worth/failure, concentration, psychomotor).

This repo's `configs/dataset_contract.yaml` splits (num_train=107, num_val=35, num_test=47) correspond to train/dev/test above (small off-by-one likely a header/duplicate artifact, not investigated further here — does not affect the conclusion).

## Finding

**Item-level PHQ-8 scores exist only for train and dev/val (140 participants combined). The held-out test split (47 participants) never had item-level labels released as part of AVEC2017** — only `PHQ_Binary` and `PHQ_Score` (total). This is a property of the original AVEC2017 challenge release, not a local data issue.

## Consequence for E3 (SPEC-H6-04)

Per the plan's pre-committed fallback ("If items are only available for train/dev, run E3 on dev and say so explicitly"):

- **E3 (12 × 8 profile-item correlation matrix) must run on the DAIC dev/val split (n=34).**
- This must be explicitly labeled `exploratory`-adjacent in the sense that it is not the confirmatory test-split evaluation used for headline AUROC/CCC numbers — but per SPEC-H6-04/H-A it is still the confirmatory analysis for the item-correlation hypothesis, just on a smaller, non-test partition. The manuscript must state this limitation plainly in the Method (per SPEC-CM-03), not bury it in a footnote.
- n=34 is small for a 12×8 = 96-test BH-FDR correction (SPEC-H6-04); expect wide CIs and reduced power. This should be stated as a limitation alongside the result, and considered when interpreting non-significant cells (absence of signal at this n is weak evidence of a true null).
- E2 (sign test, SPEC-H6-03) contrasts PHQ-8 positive/negative subjects — this can still use the *total* PHQ8_Score/Binary on the full test split (all partitions have this), so E2 is unaffected.

## Action

Task-tracked as resolved; downstream H6 implementation (when built) must read dev-split CSV for E3 and assert at load time that item columns are present, failing loudly if a future data refresh silently drops them.
