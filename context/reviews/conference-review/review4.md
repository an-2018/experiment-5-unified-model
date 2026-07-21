

## 1) Abstract – scope and wording

**Current (start of abstract, compressed):**
“Westudywhether graph-guided MoE routing improves multi-modal mental-health assessment under leakage-safe graph construction, and find that it does not outperform a non-graph MMoEEx baseline on DAIC-WOZ, CMU-MOSEI, or ChaLearn First Impressions. … Across five leakage-safe graph configurations, graph routing consistently underperforms on both DAIC … and FI … and shows a small, consistent edge on MOSEI … A graph-homophily diagnostic explains this per-task split … We report this as a rigorously investigated, task-dependent negative result. Separately, cross-attention fusion does not reproduce a previously reported gain over gated fusion, and LLM-based encoders show a similarly task-dependent pattern …”[^1]

**Issues:**

- First sentence is good, but the next few lines jump between “does not outperform” (global) and then “small gain on MOSEI” (local), which is mildly confusing.
- Too many details in the abstract (bug fix, exact ranges) reduce readability.

**Suggested replacement (full abstract body):**

> “We study whether graph-guided mixture-of-experts (MoE) routing improves multimodal mental-health assessment when graphs are constructed in a leakage-safe way. We evaluate a unified model on DAIC-WOZ depression detection, CMU-MOSEI sentiment and emotion recognition, and ChaLearn First Impressions personality prediction, combining modality-specific encoders, gated late fusion, an MMoEEx expert bank, and a KNN-graph GraphSAGE router. After fixing a temperature-balanced sampling bug, our non-graph MMoEEx baseline becomes a working depression classifier (DAIC AUROC 0.573, consistent with a 0.671 unimodal text-only ablation). Against this corrected baseline, across five leakage-safe graph configurations, graph-guided routing consistently underperforms on DAIC and FI personality, while providing only a small but consistent improvement on MOSEI sentiment and emotion. A graph-homophily diagnostic explains this task-dependent split: the routing graph carries no usable depression-label signal, limiting any router’s potential benefit on DAIC, whereas it carries real signal for MOSEI and FI that is exploited in MOSEI but not in FI. A K-sensitivity sweep and a learned per-task routing weight do not recover a DAIC benefit. We report this as a rigorously investigated, task-dependent negative result. Separately, cross-attention fusion underperforms gated fusion in our setting, and LLM-based encoders yield substantial point-estimate gains on MOSEI, a DAIC gain only for a fully LLM-based stack, and consistent losses on FI, with DAIC gains remaining statistically non-significant at this sample size. We release our leakage-safe graph construction protocol and full ablation ladder.”[^1]

You can keep or trim the bug sentence if it makes the abstract too long, but the overall structure (question → setup → corrected baseline → task-dependent result → diagnostic → secondary findings) should remain.

***

## 2) Introduction – first paragraph after “This paper asks…”

**Current:**
“This paper asks whether graph-guided MoE routing improves multimodal mental-health assessment when graph construction is made leakage-safe and evaluation is performed under strict subject-independent splits. We present interpretability and calibration as supporting analyses, not co-equal contributions.”[^1]

This is good; I’d only smooth:

**Suggested replacement:**

> “This paper asks whether graph-guided MoE routing improves multimodal mental-health assessment when graph construction is made leakage-safe and evaluation is performed under strict subject-independent splits. Interpretability and calibration are treated as supporting analyses rather than co-equal contributions.”[^1]

No change in meaning; just slightly more natural.

***

## 3) Contributions paragraph – tighten and de-duplicate

**Current:**
“We evaluate a unified architecture to isolate the effect of graph-guided routing… Our contributions are fourfold. First, a leakage-safe experimental protocol for graph-based routing in multimodal clinical settings. Second, a rigorous negative result showing graph-guided routing does not beat a non-graph MMoEEx baseline here. Third, a diagnostic analysis explaining the failure via graph homophily. Fourth, a secondary result that LLM-based encoders outperform classical encoders on some tasks, though not universally.”[^1]

**Issues:**

- “Does not beat” is slightly too absolute; “in our setting” should be explicit.
- “Failure” sounds stronger than you actually show for MOSEI (where there is a small positive effect).

**Suggested replacement:**

> “We evaluate a unified architecture to isolate the effect of graph-guided routing, combining modality-specific encoders, gated late fusion, an MMoEEx expert bank, and a graph-guided router built on a leakage-safe K-nearest-neighbour similarity graph over multimodal embeddings. Our contributions are fourfold. First, we introduce a leakage-safe experimental protocol for graph-based routing in multimodal clinical settings, including inductive and split-local graph constructions under strict subject-independent splits. Second, we provide a rigorous, task-dependent negative result: in our evaluated setting, graph-guided routing does not outperform a corrected non-graph MMoEEx baseline on depression detection or personality prediction, and offers only a small, consistent gain on sentiment and emotion recognition. Third, we use a graph-homophily diagnostic to explain this pattern, showing where the routing graph carries usable label signal and where it does not. Fourth, we report a secondary, task-dependent result that LLM-based encoders outperform classical encoders on MOSEI and only partially on DAIC, while underperforming on FI.”[^1]

This keeps everything accurate but avoids overstating “failure.”

***

## 4) Method section – “Critically, m_i is not only a missing-data indicator”

This is an excellent paragraph but is dense and has one awkward cross-reference “Section and §”.[^1]

**Current core sentence:**
“Unmasking audio/video for DAIC did not improve depression detection in development (Section and §), so the released model runs DAIC as an effectively unimodal-text task inside a jointly-trained multimodal, multitask architecture.”[^1]

**Suggested replacement:**

> “Unmasking audio and video for DAIC did not improve depression detection in development (see Results), so the released model treats DAIC as an effectively unimodal text task inside a jointly trained multimodal, multitask architecture.”[^1]

You can then delete the stray “§” symbol.

***

## 5) Results – first paragraph about the sampling bug

This is important but currently a bit long and slightly informal.[^1]

**Current (compressed):**
“A earlier version … baseline never exceeded chance … because a temperature-balanced sampling bug … compute_sampling_weights() … weight was never passed … With this fixed … baseline reaches AUROC 0.573 … every graph-routed variant now underperforms it …”[^1]

**Suggested replacement:**

> “An earlier version of this ablation used a non-graph baseline that never exceeded chance on DAIC (AUROC 0.493) because of a temperature-balanced sampling bug: our `compute_sampling_weights()` function computed per-sample weights to prevent MOSEI’s ∼32k task rows from swamping DAIC’s 107 training rows, but those weights were never actually passed to the training `DataLoader`, so most optimizer steps contained no DAIC gradient at all. With this bug fixed (by using a `WeightedRandomSampler` over the same weights), the non-graph baseline reaches DAIC AUROC 0.573—a non-chance result, consistent with DAIC’s unimodal text-only ablation AUROC of 0.671—and every graph-routed variant now underperforms it on all five leakage-safe configurations.”[^1]

This reads cleaner and keeps the key numbers.

***

## 6) Results – paragraph interpreting Table 2 (homophily)

**Current key sentence:**
“DAIC’s routing graph carries no signal beyond chance, so no router could extract a benefit there regardless of the baseline’s strength; MOSEI carries real signal that the router now measurably converts into a small gain; FI carries real signal the router still fails to exploit, and instead degrades relative to the non-graph baseline.”[^1]

**Issue:** “No router could extract a benefit” is too absolute.

**Suggested replacement:**

> “DAIC’s routing graph carries almost no signal beyond chance, which strongly limits the potential benefit of any router there regardless of the baseline’s strength; MOSEI’s graph carries real signal that the router now measurably converts into a small gain; FI’s graph also carries real signal, but the router in our configuration still fails to exploit it and instead degrades relative to the non-graph baseline.”[^1]

***

## 7) LLM-encoder results – avoid “do not transfer” as a universal statement

**Current:**
“On FI personality, every single LLM level underperforms the classical baseline …—LLM-based encoders do not transfer to this task in our setting.”[^1]

**Issue:** “Do not transfer” sounds categorical.

**Suggested replacement:**

> “On FI personality, every LLM level underperforms the classical baseline (0.472–0.537 vs. 0.571)—in our setting, the LLM-based encoders we tested did not transfer effectively to this task.”[^1]

***

## 8) Calibration and statistics – clarify interpretation

**Current ending of that paragraph:**
“We recommend larger-sample replication before drawing clinical conclusions from any single LLM-level comparison, and read the point-estimate deltas reported above (Table 3) as suggestive rather than confirmed.”[^1]

This is already good; I’d only change “read” to “treat”:

> “… and treat the point-estimate deltas reported above (Table 3) as suggestive rather than confirmed.”[^1]

***

## 9) Explainability – last sentence about LLM narratives

**Current:**
“… a concrete, reproducible instance of the risk that post-hoc clinical narratives can be fluent without being grounded, requiring mitigation of cognitive biases (Wang and Redelmeier 2026), reported here as an honest limitation of the explanation pipeline rather than a resolved property of it …”[^1]

**Issue:** Long, nested; “requiring mitigation” is a bit strong.

**Suggested replacement:**

> “… a concrete, reproducible instance of the risk that post-hoc clinical narratives can be fluent without being grounded, echoing concerns about cognitive biases in medical reasoning (Wang and Redelmeier 2026). We report this as an honest limitation of the explanation pipeline rather than a resolved property of it.”[^1]

***

## 10) Conclusion – scope of the main claim

**Current last sentences:**
“For small-sample multimodal mental-health datasets with leakage-safe graphs, graph-guided routing was not beneficial in our experiments; future gains may require jointly learned graph structure, better embeddings for neighbor construction, or task-aware expert grouping.”[^1]

This is already quite good. I’d make it slightly more cautious:

> “For small-sample multimodal mental-health datasets with leakage-safe graphs, graph-guided routing was not beneficial in our experiments on DAIC, CMU-MOSEI, and ChaLearn First Impressions. Future gains may require jointly learned graph structure, better embeddings for neighbour construction, or task-aware expert grouping.”[^1]

***

If you like, I can now do the same kind of targeted rewrite for your **AAAI rebuttal text**, aligning it tightly with the revised framing in this version.

<div align="center">⁂</div>

[^1]: main-conference.pdf

