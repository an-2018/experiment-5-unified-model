# Part 2 — Technical Review of the Methodology

---

# Overall Methodology Assessment

**Score: 8.3 / 10**

This is the strongest part of the chapter.

The methodology is ambitious, modular, and generally well designed. Compared with many multimodal learning papers, it provides considerably more implementation detail and a clearer progression from baseline models to increasingly sophisticated architectures.

However, from the perspective of a reviewer for venues such as **IEEE Transactions on Affective Computing, TPAMI, NeurIPS, or ACL**, there are several methodological issues that need to be addressed before publication.

The primary concern is **scientific isolation**: because multiple architectural changes are introduced simultaneously, it is sometimes difficult to determine which component is truly responsible for the observed improvements.

---

# 1. Problem Formulation

## Strengths

The paper clearly defines the problem as:

> learning a unified multimodal representation across depression, sentiment, emotion and personality.

This is well motivated.

The distinction between

* shared representations

and

* task-specific representations

is appropriate and aligns well with current multitask learning literature.

Another positive aspect is that the authors explicitly acknowledge that personality and depression are different psychological constructs rather than implying they are equivalent. 

This increases scientific credibility.

---

## Weakness

The manuscript never formally states the optimization objective of the **entire system** before presenting the architecture.

Instead, the reader discovers the model incrementally.

A stronger organization would first define

Input

↓

Output

↓

Tasks

↓

Optimization

↓

Architecture

For example:

> Given multimodal observations (X={X_t,X_a,X_v}), the objective is to jointly learn shared representations that minimize the combined losses for depression classification, sentiment regression, emotion classification, and personality prediction while maximizing cross-task generalization.

That immediately tells the reader what is being optimized.

---

# 2. Architecture Design

Overall

**9 / 10**

The architecture is logically constructed.

Each block has a clear role.

```
Encoders

↓

Projection

↓

Fusion

↓

MMoE

↓

Graph Router

↓

Task Heads
```

This hierarchy is easy to understand.

---

## Excellent design choice

The architecture is modular.

This is extremely valuable because each component can be independently replaced.

For example

RoBERTa

↓

Mistral

without changing

GraphSAGE

Similarly

GraphSAGE

↓

GAT

↓

Graph Transformer

could also be evaluated.

This modularity significantly increases future extensibility.

---

## Weakness

The architecture includes **many innovations simultaneously**.

Specifically

* modality projectors

* gated fusion

* LR-DGN

* MMoEEx

* orthogonality loss

* GraphSAGE routing

* uncertainty weighting

* graph combination

Each one changes the optimization landscape.

A reviewer naturally asks

> Which component actually creates the improvement?

Although later ablations help answer this, they do not fully isolate all interactions.

---

### Recommendation

Include a dependency diagram such as

```
Baseline

↓

+ Gating

↓

+ MoE

↓

+ Orthogonality

↓

+ Graph

↓

+ Graph routing

↓

+ LLM encoders
```

Readers immediately understand where gains originate.

---

# 3. Multimodal Fusion

This section is one of the strongest.

The paper compares

* late fusion

* LMF

* cross attention

under identical conditions.

That is excellent experimental practice.

---

## Particularly good

Instead of assuming cross-attention is better because it is fashionable,

the paper actually evaluates it.

This is precisely what reviewers want.

---

## However

The explanation

> Cross-attention fails because it has more parameters.

is **too simplistic**.

Several alternative explanations exist.

For example

* optimization instability

* insufficient regularization

* learning rate mismatch

* attention depth

* insufficient warmup

* attention head configuration

* feature scaling

* encoder freezing

Therefore

avoid stating

> root cause identified

Instead

say

> One plausible explanation is the larger parameter count, although optimization effects may also contribute.

That is scientifically stronger.

---

# 4. Low-Rank Dynamic Gating Network (LR-DGN)

This is actually one of the more interesting contributions.

However

it is surprisingly underdeveloped.

It occupies only a small subsection.

Yet it appears to be an original idea.

---

As a reviewer I would ask

Why is this not emphasized more?

Questions remain unanswered.

For example

How does LR-DGN compare with

* Squeeze-and-Excitation

* FiLM

* Dynamic Routing

* MoE gating

* Hypernetworks

There should be a clearer positioning.

---

Recommendation

Expand this subsection.

Include

* intuition

* computational complexity

* theoretical motivation

* comparison with existing dynamic gating methods

---

# 5. Mixture-of-Experts

Overall quality

Excellent.

The discussion of shared experts versus exclusive experts is well motivated. 

The orthogonality regularization is also appropriate.

---

## Missing analysis

There is no quantitative analysis of

expert utilization.

For example

How often is each expert selected?

Does one expert dominate?

Do experts collapse?

A simple histogram would answer this immediately.

Example figure

```
Expert

E1 █████████

E2 ███

E3 ████████

...

E8 ██
```

This is now standard in MoE papers.

---

## Missing entropy analysis

Measure

Routing entropy

during training.

High entropy

↓

shared computation

Low entropy

↓

specialization

This would strengthen the paper considerably.

---

# 6. Graph Construction

This is arguably the paper's most novel methodological contribution.

I particularly like the discussion of

* inductive

* split-local

* transductive

graphs. 

Very few multimodal papers discuss graph leakage this carefully.

This is a genuine contribution.

---

## However

The manuscript should explain

**why KNN?**

Why not

Radius graph

Approximate nearest neighbors

Mutual KNN

Learned graph

Adaptive graph

Graph Transformer

Dynamic graph

At present

KNN appears as an implementation choice rather than a scientific one.

---

Recommendation

Include

> We selected KNN because...

supported by literature.

---

# 7. GraphSAGE Router

Very interesting idea.

GraphSAGE is not merely another encoder.

Instead it becomes a routing mechanism.

That is novel.

---

However

there is almost no theoretical discussion.

The paper explains

how

it works

but not

why

it should work.

For example

Graph aggregation smooths representations.

Smoother representations reduce routing variance.

Lower routing variance improves expert selection.

That chain of reasoning should be explicitly stated.

---

# 8. Mathematical Formulation

Overall

8/10

The equations are readable.

Notation is mostly consistent.

---

Strengths

Notation is clean.

Projection equations are easy to follow.

Loss functions are standard.

---

Weaknesses

Some variables appear without prior definition.

Examples include

λ

σ

K

before their meaning becomes clear.

Although eventually defined, earlier definitions would improve readability.

---

Recommendation

Add

Notation Table

before Section 1.4.

Example

| Symbol | Meaning              |
| ------ | -------------------- |
| N      | Number of samples    |
| K      | Number of experts    |
| d      | Embedding dimension  |
| λ      | Graph routing weight |
| σ      | Task uncertainty     |
| G      | Graph                |

This greatly improves readability.

---

# 9. Experimental Design

This is one of the strongest aspects.

The chapter evaluates

multiple datasets

multiple modalities

multiple architectures

multiple graph settings

multiple encoders

multiple transfer settings

multiple XAI methods

Very comprehensive.

---

## However

Some experiments are not fully controlled.

Example

Fusion

↓

MMoE

↓

Graph

Sometimes multiple hyperparameters change simultaneously.

Ideally

only one variable changes between experiments.

---

# 10. Reproducibility

Outstanding.

Probably

9.5/10.

Very few papers provide

configuration files

JSON outputs

graph statistics

training protocols

loss functions

evaluation settings

artifact paths

This exceeds current publication standards.

---

## One improvement

Provide

Git commit hash

Package versions

CUDA version

Random seed list

Framework version

This makes long-term reproducibility even stronger.

---

# 11. Threats to Validity

This section is currently too short.

A stronger discussion would acknowledge

Internal validity

Small DAIC dataset

External validity

Clinical deployment remains untested

Construct validity

Personality is only an auxiliary task

Statistical validity

Limited power for DAIC

Ecological validity

Clinical interviews differ from real-world deployment

This significantly improves scientific maturity.

---

# Human Writing Review (Methodology)

Overall, the methodology is technically sound but often reads like software documentation rather than a scientific explanation.

The goal should be to explain **why** each design choice was made before explaining **how** it was implemented.

## Example 1 – Fusion

**Current style**

> Given projected modality embeddings, we apply a modality mask and compute learned gate weights...

**Improved**

> Because not every sample contains all three modalities, the model first identifies which inputs are available. It then learns how much each modality should contribute to the final representation through a set of trainable gating weights.

The mathematics can follow immediately afterward.

This sequence is much easier to read.

---

## Example 2 – Graph Routing

**Current style**

> The GraphSAGE router performs two-layer neighborhood aggregation...

**Improved**

> Samples with similar affective characteristics are likely to benefit from similar expert routing decisions. To capture this relationship, we construct a K-nearest neighbor graph and use GraphSAGE to aggregate information from neighboring samples before selecting the mixture of experts.

This provides intuition before equations.

---

## Example 3 – LR-DGN

The current subsection is concise but does not clearly motivate the method.

A more natural introduction would be:

> Cross-attention models are powerful but often require many parameters, making them difficult to train on small clinical datasets. We therefore introduce a Low-Rank Dynamic Gating Network (LR-DGN), which approximates dynamic modality interactions using a compact low-rank representation. This design reduces computational cost while preserving the ability to adapt modality weights for each sample.

---

# Overall Assessment After Part 2

From a methodological perspective, the chapter is **stronger than many recent multimodal learning papers**, particularly in its careful treatment of leakage, reproducibility, and controlled experimentation. The graph-based routing mechanism appears to be the most distinctive contribution and deserves greater emphasis throughout the manuscript.

The main improvements are not architectural but scientific: clearly isolate the effect of each component, provide stronger theoretical motivation for the graph router and LR-DGN, include analyses of expert utilization and routing behavior, and frame design choices around explicit research questions rather than implementation steps.

In **Part 3**, I will perform the most critical review: a detailed examination of the **experimental results**, including every table and figure, numerical consistency, statistical validity, and whether the reported evidence truly supports the paper's claims. This is the section where reviewers are most likely to identify acceptance or rejection issues.
