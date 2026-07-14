# Part 4 — Discussion, Conclusions, References, Scientific Contribution, and Final Reviewer Assessment

---

# Overall Assessment of the Discussion

**Score: 8.4 / 10**

The discussion is balanced and scientifically mature. Unlike many papers that focus exclusively on positive findings, this chapter explicitly discusses failures, limitations, and unexpected outcomes. This substantially increases its credibility.

However, the discussion remains **experiment-oriented** rather than **knowledge-oriented**.

A strong discussion should answer:

> "What have we learned about multimodal learning?"

rather than

> "Which experiment worked?"

This distinction is important.

---

# 1. Review of the Discussion

The discussion is organized into

* What worked
* What did not work
* Limitations

This structure is good. 

---

## Strengths

The chapter correctly avoids claiming that every proposed component improves every task.

For example, it explicitly states that:

* Graph routing consistently helps some tasks.
* Cross-attention fails.
* Domain adaptation shows negative transfer.
* Cross-demographic transfer fails.
* LLM encoders improve several tasks but do not outperform graph routing. 

This is excellent scientific practice.

---

## Weakness 1

The discussion is still largely **section-by-section**.

Instead,

it should synthesize the broader scientific lessons.

For example

Current style

> Graph routing improved...

Better style

> Across all experiments, architectures that exploit structural relationships between samples consistently outperform architectures that only fuse modality features. This suggests that inter-sample context is more valuable than increasing model complexity.

That is a much stronger scientific conclusion.

---

## Weakness 2

The discussion rarely connects findings back to the literature.

For example

Instead of

> Cross-attention underperformed.

Say

> Unlike previous work on larger multimodal datasets, our experiments suggest that cross-attention offers limited benefits when training data are scarce, reinforcing recent observations regarding overparameterization in clinical machine learning.

That positions the work within existing research.

---

## Weakness 3

Several explanations are speculative.

Example

> Larger K provides richer neighborhood context.

Maybe.

But there is no direct evidence.

Instead

say

> A possible explanation is...

High-impact journals expect cautious language.

---

# 2. Review of the Conclusions

Current score

**7.8 /10**

Ironically,

the conclusion is weaker than the discussion.

---

## Problem 1

The conclusion summarizes experiments.

It should summarize

knowledge.

Instead of

"We evaluated..."

say

"This work demonstrates..."

---

## Problem 2

Too many details

The conclusion still contains

model variants

dataset names

ablation references

specific experiment identifiers

Readers should finish the paper remembering ideas,

not experiment IDs.

---

## Recommended structure

Paragraph 1

Research problem

Paragraph 2

Main contribution

Paragraph 3

Scientific findings

Paragraph 4

Limitations

Paragraph 5

Future work

---

# 3. Limitations

This section is surprisingly honest.

That is a strength.

However,

I would add several additional limitations.

---

## External validity

The datasets

DAIC

MOSEI

FI

represent different collection protocols.

Real clinical deployment remains untested.

---

## Generalization

The graph router is evaluated only on

KNN graphs.

Future work should evaluate

* Graph Transformers

* Dynamic graphs

* Learned graphs

---

## Foundation models

Only one LLM family is investigated.

Future work

* Qwen

* Gemma

* Llama

* Phi

could produce different conclusions.

---

## Clinical limitations

This is particularly important.

The paper should explicitly state

> The proposed system is intended as a decision-support tool and should not replace clinical diagnosis.

Nearly every clinical AI journal expects this statement.

---

# 4. Future Work

Current future work is good,

but can be strengthened.

---

I recommend grouping future work into

### Short-term

Evaluate larger datasets

Graph Transformers

Expert balancing

---

### Medium-term

Continual learning

Missing modalities

Federated learning

---

### Long-term

Clinical deployment

Prospective evaluation

Human-AI collaboration

Explainable clinical assistants

---

# 5. References Review

Overall

**7.8 /10**

---

Strengths

Good coverage.

Modern citations.

Appropriate multimodal literature.

---

Weaknesses

Some areas deserve additional references.

---

## Missing recent work

### Graph Foundation Models

Very active research area.

Should be discussed.

---

### Sparse Mixture-of-Experts

Recent MoE papers

particularly

Switch Transformer

Mixtral

DeepSeek-MoE

would strengthen positioning.

---

### Clinical Foundation Models

Recent medical LLMs

deserve mention.

---

### Parameter-efficient multimodal learning

PEFT

LoRA

QLoRA

AdapterFusion

These areas are increasingly important.

---

# 6. Novelty Assessment

This is one of the most important reviewer questions.

---

What is genuinely novel?

I believe there are three novel contributions.

---

## Contribution 1

Graph-based routing

This is the strongest contribution.

GraphSAGE

↓

expert routing

is interesting.

---

## Contribution 2

Leakage-safe graph construction

Very few papers discuss this carefully.

This deserves much greater emphasis.

---

## Contribution 3

Comprehensive empirical evaluation

This is actually a contribution.

Negative results

Calibration

Transfer

XAI

LLMs

Graph variants

Few papers evaluate this extensively.

---

What is **not** novel?

---

LLM encoders

Already known.

---

Multimodal fusion

Well established.

---

Mixture-of-Experts

Well established.

---

Graph neural networks

Well established.

---

The novelty comes from

their combination,

routing strategy,

and empirical validation.

---

# 7. Potential Research Impact

Current estimate

High.

Especially for

IEEE Transactions on Affective Computing

Information Fusion

Pattern Recognition

Neural Networks

Machine Learning with Applications

---

For

TPAMI

NeurIPS

ICML

the methodological novelty would probably need to be stronger.

---

# 8. Major Issues

These are the issues I would include in an actual review.

---

## Major Issue 1

The scientific narrative is fragmented.

Too many contributions compete for attention.

---

## Major Issue 2

Several causal explanations are not experimentally verified.

---

## Major Issue 3

Component interactions are not always isolated.

---

## Major Issue 4

The discussion reports results

instead of synthesizing knowledge.

---

## Major Issue 5

Theoretical motivation for graph routing

needs strengthening.

---

## Major Issue 6

Expert utilization

is not analyzed.

---

## Major Issue 7

Several tables need

confidence intervals

effect sizes

and

delta columns.

---

# 9. Minor Issues

* Too many abbreviations.

* Tables contain excessive precision.

* Some figure captions are descriptive rather than explanatory.

* Several paragraphs begin with identical sentence structures.

* More cautious wording is needed in causal interpretations.

* Reduce references to internal implementation artifacts in the main text.

* Add a notation table.

* Improve transitions between sections.

---

# 10. Highest-Priority Additional Experiments

If I were reviewing this for a journal, these are the experiments I would request before acceptance.

---

## Priority 1

Expert routing analysis

How often is each expert selected?

---

## Priority 2

Routing entropy

Show specialization over training.

---

## Priority 3

Graph sensitivity

Evaluate

K =

5

10

15

20

with graph density statistics.

---

## Priority 4

Graph alternatives

Compare with

GAT

Graph Transformer

Dynamic Graph

---

## Priority 5

Cross-dataset statistical testing

Not only

mean improvements,

but

paired significance tests.

---

## Priority 6

Inference cost

Report

Latency

Memory

Parameters

FLOPs

Clinical AI increasingly values efficiency.

---

## Priority 7

Error analysis

Present

successful cases

failure cases

borderline cases

This greatly strengthens discussion.

---

# Publication Readiness

## Current state

Very good PhD chapter

Needs refinement for journal publication.

---

## Journal recommendation

| Venue                                    | Recommendation                                               |
| ---------------------------------------- | ------------------------------------------------------------ |
| Machine Learning with Applications       | Accept after revisions                                       |
| Neural Networks                          | Minor–Major Revision                                         |
| Pattern Recognition                      | Major Revision                                               |
| Information Fusion                       | Major Revision                                               |
| IEEE Transactions on Affective Computing | Major Revision                                               |
| IEEE TPAMI                               | Significant revision required                                |
| NeurIPS                                  | Borderline Reject (narrative and novelty need strengthening) |
| ICML                                     | Borderline Reject                                            |
| ACL Findings                             | Major Revision                                               |

---

# Final Recommendation

If I were serving as a senior reviewer, my recommendation would be:

> **Major Revision**

**Reasoning**

The manuscript presents a technically sophisticated and impressively comprehensive study that integrates multimodal learning, mixture-of-experts, graph routing, LLM-based encoders, calibration, explainability, and transfer learning into a unified framework. The experimental breadth, reproducibility, and transparent reporting of both positive and negative results are notable strengths.

However, the paper currently tries to communicate too many contributions simultaneously, which weakens its central scientific message. The most original contribution—the graph-guided routing strategy under a leakage-safe graph construction protocol—is not emphasized strongly enough. Several causal explanations remain speculative, some experimental gains require deeper interpretation rather than numerical description, and the discussion should focus more on extracting general scientific insights than recounting individual experiments.

With a clearer narrative centered on three core contributions, stronger theoretical justification for the routing mechanism, additional analyses of expert behavior and graph properties, and a more concise presentation of the experimental evidence, this work has the potential to become a strong journal publication.

---

# Overall Scores

| Category              | Score (/10) |
| --------------------- | ----------: |
| Originality           |     **8.5** |
| Technical Soundness   |     **8.2** |
| Experimental Design   |     **8.5** |
| Statistical Rigor     |     **8.0** |
| Reproducibility       |     **9.5** |
| Writing Quality       |     **7.5** |
| Scientific Narrative  |     **7.2** |
| Figures & Tables      |     **8.3** |
| Practical Impact      |     **8.8** |
| Publication Readiness |     **8.0** |

## **Overall Rating: 8.3 / 10**

This is **well above the average PhD thesis chapter** and contains material with clear publication potential. In my view, the revisions that would have the greatest impact are **not additional experiments**, but a **reorganization of the manuscript around a single, coherent scientific story** that consistently emphasizes the graph-guided routing contribution and the empirical insights it provides. The experimental foundation is already strong; the presentation now needs to match that quality.
