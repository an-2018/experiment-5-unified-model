# Claim Ledger — Day 1 Seed Report

**Date:** 2026-07-27
**Purpose:** SPEC-H8 minimal proof-of-concept per the plan's Day 3 action item ("Build H8 against the current v2 numbers. Watch it fail. That failing test is your proof the harness works.")

## Files added

- `claims/ledger.yaml` — 4 claim entries (1 superseded/kept-as-regression-guard, 3 live)
- `claims/verify_claims.py` — evaluates `kind: assertion` entries against their source CSVs

## Result: found a live bug, not just a historical one

The plan's failure-mode table describes "every variant exceeds baseline" as something that happened in a *past* version (V0 = 0.486 < 0.493). Running the ledger against the **current** `paper/main-conference.tex` (as of commit `2f408d0`) shows this exact bug is still present today, not merely historical:

- Prose claimed: "every graph-routed variant now exceeds the non-graph baseline on... sentiment CCC (0.486--0.512 vs. 0.493)"
- Source data (`artifacts/tables/mmoe_ex_results.csv` baseline=0.4931, `artifacts/tables/ggmoe_results.csv` variants=[0.4858, 0.5086, 0.5124, 0.5081, 0.5064]): **V0 = 0.4858 is below the baseline.** 4 of 5 variants exceed it; V0 does not.

## Fix applied

- `paper/main-conference.tex`: corrected the sentence to state "on emotion AUROC every graph-routed variant exceeds the non-graph baseline... on sentiment CCC four of five variants exceed the baseline... while V0 falls slightly below it," matching the source data exactly.
- `claims/ledger.yaml`: `C-MOSEI-ALL-EXCEED` kept in the ledger but marked `SUPERSEDED`, deliberately left failing as a live regression guard — if it ever starts passing, that means the underlying artifact data changed and the manuscript text must be re-checked, not that the historical claim was right. `C-MOSEI-4-OF-5-EXCEED` added as the assertion that now actually matches the corrected prose, and passes.
- The two other headline aggregate claims in the same paragraph (all variants underperform baseline on FI CCC; all variants underperform baseline on DAIC AUROC) were checked and are both **true** against source data — no fix needed there.

## What this is not

This is a proof-of-concept, not the full H8 build (SPEC-H8-01..05). It does not yet:
- Auto-render LaTeX macros (`\ClaimXxx`) so Results contains zero hand-typed numerals (SPEC-H8-02).
- Lint for bare decimal literals (SPEC-H8-04).
- Apply the reportability rule / seed-variance floor (SPEC-H4-03) — there is currently only one seed's worth of data behind these numbers, which the harness proper (H0/H2 canonical 5-seed runs) has not yet produced.
- Bind to manifest/checkpoint hashes (SPEC-H0/H7-02).

Those are Phase 0 harness-build items, not Day-1 checks, and depend on rerunning the ablation ladder under H0-H2 with the canonical 5-seed set.

## How to run

```
python3 claims/verify_claims.py
```

Exits non-zero if any claim fails (including the intentionally-superseded regression guard, so a clean run currently means 3/4 pass with 1 expected historical failure — not yet wired into a CI gate that distinguishes "expected fail" from "blocking fail").
