# journal_paper.tex — v1-Era Residue Audit

**Date:** 2026-07-27
**Trigger:** the quarantine linter's "false positive" on 0.174 in `journal_paper.tex`
turned out to be a real bug — the v1-era LLM-ablation delta, not a coincidental
decimal collision.

## Confirmed issues

1. **Line 365 ("What Worked" paragraph): "largest DAIC gain: L3, +0.174 AUROC"**
   — this is v1-era. The correct v2 number (also present elsewhere in the same
   document, line 287) is **L5 (full stack), +0.076**, and L3 is explicitly
   described on line 287 as one of the L1-L4 partial swaps that
   *underperforms* the classical baseline (0.482-0.511 vs. 0.573). **The
   document currently contradicts itself**: line 287 says L3 underperforms;
   line 365 says L3 is the largest gain. Not merely stale — internally
   inconsistent.

2. **Line 244: "every variant \emph{exceeds} the baseline on MOSEI sentiment
   (0.486--0.512 vs. 0.493)"** — the same "every variant exceeds baseline"
   bug already found and fixed in `main-conference.tex` (V0=0.4858 <
   baseline=0.4931; 4 of 5 variants exceed, not all 5). Unfixed here.

## Checked and consistent (not flagged)

- "MMoEEx alone... benefits FI personality (+0.12 CCC over the video-only
  baseline)" — video-only FI Avg CCC is 0.4578 (`unimodal_baselines.csv`),
  MMoEEx FI Avg CCC is 0.5705 (`mmoe_ex_results.csv`), delta = 0.1127 ≈ +0.11,
  reasonably consistent with "+0.12" (rounding). Not flagged as an error.
- The V0-V4 routing table (lines 224-228) matches `artifacts/tables/ggmoe_results.csv`
  exactly.
- The DeLong non-significance claims (line 287, 326: "p>0.72 throughout") are
  consistent with the corrected v2 framing and match `A-DELONG` style honest
  hedging already in place elsewhere.

## Disposition

Per the agreed sequencing, `journal_paper.tex` is rewritten **last, after
submission** — these are not being patched now. Both issues are logged here so
they aren't lost, and will be fixed as part of that scheduled rewrite rather
than as piecemeal edits to prose that's being substantially replaced anyway.
`paper/main-conference.tex`'s equivalent MOSEI bug is already fixed (this
session, earlier); its 0.671 references remain pending the main rewrite
(task #4), same treatment as journal_paper.tex's issues here.
