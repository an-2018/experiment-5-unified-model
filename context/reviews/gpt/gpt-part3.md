# Part 3 — Results, Figures, Tables, Statistical Analysis, and Scientific Evidence

---

# Overall Evaluation of the Experimental Results

**Score: 7.9 / 10**

This chapter contains an unusually comprehensive experimental evaluation. The breadth of experiments is a major strength, but it also creates one of the manuscript's biggest weaknesses: **the results section is too dense and difficult to navigate.**

A reviewer should be able to answer three questions after reading the results:

1. Did the proposed method improve performance?
2. Why did it improve?
3. When does it fail?

At present, the manuscript answers all three questions, but the answers are spread across numerous tables, ablation studies, and supplementary experiments. The scientific message would be much stronger if these findings were synthesized more clearly.

---

# General Assessment of the Results Section

## Strengths

The experimental evaluation is unusually extensive.

It includes:

* Multiple datasets
* Multiple tasks
* Strong baselines
* Ablation studies
* Statistical validation
* Calibration
* Domain adaptation
* Transfer learning
* Explainability
* Cross-dataset evaluation
* Negative results

This exceeds what is typically expected in a conference paper and is comparable to a journal article.

---

## Weaknesses

The section is **too descriptive**.

Many paragraphs simply restate values from tables instead of interpreting them.

Example pattern:

> V0 obtains...

> V1 obtains...

> V2 obtains...

> V3 obtains...

> V4 obtains...

This is not scientific analysis.

Instead, the discussion should identify patterns.

For example:

> Increasing neighborhood size benefits DAIC but degrades MOSEI sentiment, suggesting that larger local neighborhoods improve routing for small datasets while oversmoothing representations in larger datasets.

That conveys insight rather than numbers.

---

# Review of the Ablation Studies

One of the strongest parts of the chapter.

The progression

Baseline

↓

Fusion

↓

MoE

↓

Graph

↓

LLM

↓

Domain adaptation

is logical.

---

## However

There are **too many ablation tables**.

Some could be merged.

Current structure includes:

* Fusion
* MoE
* Graph
* MPDD
* Transfer
* LLM
* Domain adaptation
* Calibration
* XAI

Readers must continuously switch context.

---

### Recommendation

Introduce a summary figure.

For example

```text
Baseline
   │
   ▼
Fusion
 +0.11

   ▼
MoE
 +0.03

   ▼
Graph
 +0.22

   ▼
LLM
 +0.06

   ▼
Calibration
 better confidence

   ▼
XAI
 interpretability
```

One figure communicates the evolution of the architecture much more effectively than several pages of text.

---

# Review of Graph Routing Results

This is the paper's central experimental contribution.

The graph ablation compares five graph construction variants with the non-graph baseline. 

This experiment is well designed because it isolates the effect of graph construction.

---

## Strength

The manuscript correctly concludes that no single graph variant dominates all tasks. 

This is scientifically honest.

Many papers would simply highlight the best-performing variant.

Instead, the chapter discusses trade-offs.

That increases credibility.

---

## Weakness

The explanation remains qualitative.

For example

> V3 works better because K=15 provides richer neighborhood context.

That is plausible.

But it is not demonstrated.

I would recommend adding

Average neighborhood purity

Graph density

Average edge similarity

Routing entropy

These metrics would explain **why** V3 behaves differently.

---

# Review of Table 1.9 (Graph Routing Ablation)

This is one of the most important tables.

Overall quality

Very good.

---

## Improvements

### Highlight improvements relative to baseline

Instead of only bolding the best value,

include

Δ from MMoE

Example

| Variant | AUROC | Δ     |
| ------- | ----- | ----- |
| V0      | 0.71  | +0.22 |
| V3      | 0.89  | +0.40 |

Readers immediately understand effect size.

---

### Color coding

If the thesis permits,

green

better

red

worse

makes interpretation much easier.

---

### Confidence intervals

Every reported metric should include

95% CI

or

bootstrap interval.

---

# Cumulative Ablation Ladder

One of my favorite parts of the chapter.

The cumulative ladder clearly illustrates the contribution of each architectural component. 

However,

there is an unexpected issue.

---

Observe

```text
Unimodal

↓

Gated Fusion

↓

MMoE

↓

Graph
```

Performance does **not** increase monotonically.

This is scientifically interesting.

Instead of treating it as a weakness,

emphasize it.

Example

> Contrary to expectations, progressively more complex architectures do not consistently improve performance. Instead, graph routing provides the first substantial gain after several ineffective intermediate models.

This is a valuable insight.

---

# MPDD Results

These are scientifically valuable.

Especially because

Logistic Regression

beats

GGMoE. 

Many authors would omit this result.

Keeping it increases credibility.

---

## However

The discussion should go further.

Explain

why

LR wins.

Possible reasons

* low sample size

* high-dimensional embeddings

* linear separability

* overparameterization

* insufficient regularization

These hypotheses deserve discussion.

---

# Cross-Dataset Transfer

Very interesting.

Particularly the comparison

Young

↓

Elderly

and

MPDD

↓

DAIC. 

---

However,

one concern exists.

The manuscript occasionally interprets

below-random AUROC

as evidence of

age-specific biomarkers.

That conclusion is stronger than the evidence.

Alternative explanations include

* annotation differences

* acquisition protocol

* feature extraction differences

* sampling bias

* class imbalance

Therefore

rewrite

> proves biomarkers differ

to

> suggests substantial domain-specific differences.

---

# LLM Ablation

This section is well executed. 

It evaluates

Frozen

↓

LoRA

↓

Audio

↓

Video

↓

Full stack

This progression is logical.

---

## However

One question remains unanswered.

Why does

L5

not outperform

L3?

The paper notes the observation but does not investigate it.

Possible explanations include

* feature redundancy

* optimization conflict

* modality interference

* overfitting

* gradient competition

These possibilities should be discussed.

---

# Domain Adaptation Results

Scientifically excellent.

Not because the models improve.

Because they **fail**. 

Negative findings are valuable.

---

However

avoid saying

> domain adaptation does not work.

Instead

say

> Under the evaluated datasets and adaptation methods, domain adaptation did not improve performance.

This distinction is important.

---

# Calibration Results

This section is surprisingly good.

Few multimodal papers include calibration analysis. 

Including

ECE

Brier

Reliability diagrams

is commendable.

---

Recommendation

Include one visual summary.

Example

```text
Raw

↓

Temperature

↓

Platt

↓

Isotonic
```

with

ECE

displayed below.

Readers immediately understand the effect.

---

# Explainability

Excellent inclusion.

Especially because the chapter combines

SHAP

Graph explanations

Counterfactuals

Perturbation analysis. 

This is much stronger than presenting only SHAP values.

---

## However

There is a concern.

Several demonstrations appear to use

a mock unified model with predefined modality weights. 

If so,

make this explicit.

Otherwise readers may incorrectly assume

these explanations were generated

from the final trained model.

---

# Statistical Analysis

Overall quality

Very good.

Bootstrap

Permutation tests

Effect sizes

Calibration

all strengthen the paper. 

---

## One important issue

The paper reports

> no statistically significant differences among LLM variants.

This is actually an important result.

Unfortunately,

it is hidden near the end.

I would instead emphasize

> Although LLM encoders consistently improved average performance, none of the observed gains reached statistical significance because of the limited DAIC sample size.

That is a mature scientific conclusion.

---

# Review of Figures

Overall quality

8.5 /10

---

## Figure 1.3 (SHAP)

Good.

However

Feature IDs such as

audio_44

video_693

are meaningless to readers. 

Provide

actual feature names

or

appendix mapping.

---

## Figure 1.4 (LLM deltas)

Excellent idea. 

Instead of reporting absolute metrics,

showing improvements over baseline is much easier to interpret.

---

## Figure 1.5 (UMAP)

Useful visualization. 

However

avoid overinterpreting UMAP.

A distinct cluster

does **not**

prove

better representations.

Only say

> indicates that embeddings occupy different regions.

---

## Figure 1.6 (SHAP Beeswarm)

Good.

However

consider ordering modalities consistently

Text

Audio

Video

across all figures. 

---

## Figure 1.7 (Graph explanation)

Very nice visualization. 

Recommendation

Use

different colors

for

training nodes

test node

neighbors

Currently readers may need additional explanation.

---

# Numerical Consistency

I carefully examined the reported values across the available sections.

Overall consistency is good.

However, I identified several issues that should be addressed.

## 1. Overly precise metrics

Many results are reported to four decimal places (e.g., AUROC = 0.8967). 

Given the relatively small datasets (particularly DAIC), this level of precision implies a degree of certainty that is unlikely to be meaningful.

**Recommendation:** report three decimal places throughout the manuscript unless there is a specific reason to retain four.

---

## 2. "Best" results should be contextualized

Several tables compare your results against state-of-the-art values (e.g., DAIC AUROC 0.8967 vs. reported SoA 0.7800). 

Such improvements are substantial. Before claiming superiority, make it explicit that:

* evaluation protocols are identical,
* train/test splits match,
* preprocessing is comparable,
* metrics are computed in the same way.

Otherwise, reviewers may question whether the comparison is fair.

---

## 3. Mixed metrics across tasks

The manuscript alternates between AUROC, CCC, F1, and regression metrics depending on the task. While appropriate, it can make tables difficult to interpret.

**Recommendation:** add a short reminder before each major table explaining why each metric is used and what constitutes a meaningful improvement.

---

# Human Writing Review (Results)

The writing in this section would improve significantly by shifting from **describing tables** to **explaining patterns**.

### Example 1 – Table interpretation

**Current style**

> V0 achieves the best MOSEI sentiment performance, while V3 achieves the best DAIC AUROC.

**Improved**

> The graph variants reveal a clear trade-off. Smaller neighborhoods consistently benefit sentiment analysis, whereas larger neighborhoods improve depression detection. This suggests that the optimal graph structure depends on both dataset size and task characteristics.

---

### Example 2 – LLM ablation

**Current style**

> L3 achieves the highest DAIC AUROC.

**Improved**

> Replacing the classical audio encoder with CLAP produces the largest improvement in depression detection, indicating that richer audio representations contribute more than additional video features under the current training conditions.

---

### Example 3 – Domain adaptation

**Current style**

> Domain adaptation fails.

**Improved**

> None of the evaluated domain adaptation methods consistently improved performance. The results suggest that simple feature alignment is insufficient to overcome the substantial distribution differences between the clinical and non-clinical datasets considered in this study.

---

# Overall Assessment After Part 3

The experimental evaluation is **one of the strongest aspects of the chapter**. It is broad, transparent, and unusually honest in reporting both positive and negative findings. The graph-routing experiments, calibration analysis, and transfer-learning studies substantially strengthen the work.

The primary weakness is not the experiments themselves but **how they are communicated**. Many paragraphs simply restate numerical values instead of extracting broader scientific insights. By reducing numerical repetition, synthesizing results across tables, and being more cautious when interpreting causal mechanisms, the chapter would become considerably more persuasive to reviewers at high-impact venues.

In **Part 4**, I will review the **Discussion, Conclusions, References, overall scientific contribution, publication readiness, and provide a prioritized list of major and minor issues together with a final acceptance-style recommendation.
