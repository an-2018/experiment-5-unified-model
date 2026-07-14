# Part 5 — Human Writing Review (Senior Journal Editor Pass)

This review focuses exclusively on the **quality of the writing**, assuming the technical content is already sound. My goal is to make the paper read like a publication in **IEEE Transactions on Affective Computing, Information Fusion, Pattern Recognition, or ACM Computing Surveys**, while keeping the language **natural, concise, and human** rather than overly polished or AI-generated.

---

# Overall Writing Assessment

## Current score

**7.4 / 10**

## After revision

**9.2–9.5 /10**

---

The writing is already technically correct.

The main issue is **style**, not grammar.

The manuscript frequently reads like:

* laboratory notebook
* technical report
* engineering documentation

rather than

* scientific argument.

This distinction is extremely important.

---

# Biggest Writing Problem

The paper often follows this pattern:

> We implemented...

> We evaluated...

> We obtained...

> We compared...

> We observed...

This makes the manuscript sound procedural.

Instead,

focus on

**ideas**

rather than

**actions**.

---

Instead of

> We evaluated five graph variants.

write

> Five graph construction strategies were evaluated to understand how neighborhood structure influences expert routing.

The second sentence explains

**why**

the experiment exists.

---

# Writing Review by Section

---

# Title

Current

> Unified Multimodal Graph-Gated MoE for Mental Health Assessment

Problems

* acronym-heavy
* generic
* difficult to remember

---

Better

> **Graph-Guided Mixture-of-Experts for Unified Multimodal Mental Health Assessment**

or

> **Learning Graph-Guided Multimodal Representations for Mental Health Assessment**

These titles emphasize the scientific contribution before the implementation.

---

# Abstract

The abstract needs the largest rewrite.

Current issues

* too many numbers
* too many abbreviations
* too many contributions
* insufficient motivation

---

Instead use

Paragraph 1

Problem

Paragraph 2

Gap

Paragraph 3

Method

Paragraph 4

Results

Paragraph 5

Impact

---

## Example

Current

> We propose...

Improved

> Mental health assessment increasingly relies on multimodal data such as language, speech, and facial behaviour. However, existing models often struggle to generalize across related tasks while remaining interpretable. We propose a graph-guided mixture-of-experts architecture that combines multimodal learning with graph-based routing to improve representation sharing across depression, sentiment, emotion, and personality prediction. Across three public datasets, the proposed approach consistently improves performance on several tasks while providing interpretable routing decisions and robust calibration. The results show that graph-based routing offers a practical alternative to increasingly complex fusion architectures, particularly in low-resource clinical settings.

Notice

only

one

headline result.

---

# Introduction

Current quality

Very good.

Needs

more storytelling.

---

Current style

> Depression affects...

> Existing methods...

> We propose...

---

Better

Problem

↓

Gap

↓

Research question

↓

Contribution

---

Example

Instead of

> Existing multimodal architectures...

write

> Although recent multimodal models have achieved promising results, most are designed for a single task and often require increasingly complex fusion mechanisms. Whether a single architecture can effectively learn across multiple affective computing tasks remains an open question.

Much more engaging.

---

# Related Work

Current problem

It summarizes papers.

---

Better

Compare

papers.

---

Instead of

Paper A

Paper B

Paper C

Paper D

Use

Theme

↓

Comparison

↓

Gap

---

For example

> Existing multimodal fusion methods can be broadly grouped into early, late, tensor-based, and attention-based approaches. Early fusion is computationally efficient but often ignores modality-specific structure, whereas attention mechanisms capture richer interactions at the cost of substantially more parameters. This trade-off motivates lightweight alternatives such as gated fusion, which our work further extends using graph-guided routing.

One paragraph replaces several descriptive ones.

---

# Methodology

Current style

Very implementation-oriented.

---

Instead of

"The GraphSAGE router..."

start with

"The intuition behind graph routing is that..."

Readers remember ideas,

not layer names.

---

# Results

This section requires the most editing.

---

Current writing

Table

↓

Paragraph repeats table

↓

Next table

↓

Paragraph repeats table

---

Instead

Use

Pattern

↓

Interpretation

↓

Explanation

---

Example

Current

> V3 achieves AUROC 0.8967.

Better

> Increasing the neighborhood size substantially improves depression detection, suggesting that richer local context helps the router identify more suitable experts for small clinical datasets.

Readers can already see

0.8967

inside the table.

Don't repeat it.

---

# Discussion

Current

Experiment summary.

---

Should become

Scientific synthesis.

---

Instead of

> Graph routing improves...

write

> Across all evaluated tasks, the experiments indicate that explicitly modelling relationships between samples is more beneficial than increasing architectural complexity. This finding suggests that future multimodal systems may benefit more from exploiting dataset structure than from deeper fusion modules.

Now you're contributing knowledge.

---

# Conclusion

Current

Too detailed.

---

Better

Five short paragraphs.

Problem

Contribution

Findings

Limitations

Future work

---

The reader should finish with

one

clear takeaway.

---

# Sentence-Level Improvements

Here are patterns I repeatedly noticed.

---

## Pattern 1

Current

> It can be observed that...

Replace

> We observe...

or

simply

> Graph routing improves...

---

## Pattern 2

Current

> It is important to note that...

Delete.

Usually unnecessary.

---

## Pattern 3

Current

> It should be mentioned that...

Delete.

---

## Pattern 4

Current

> We can clearly see...

Instead

> The results indicate...

---

## Pattern 5

Current

> In order to...

Replace

> To...

---

## Pattern 6

Current

> Due to the fact that...

Replace

> Because...

---

## Pattern 7

Current

> A large number of...

Replace

> Many...

---

## Pattern 8

Current

> The obtained results...

Replace

> The results...

---

## Pattern 9

Current

> It was found that...

Replace

> We found...

---

## Pattern 10

Current

> Demonstrates that...

Often replace with

> Suggests that...

Much safer scientifically.

---

# Words to Avoid

Instead of

utilize

use

---

facilitate

help

---

leverage

use

---

aforementioned

these

---

numerous

many

---

subsequently

then

---

therefore

so

(when appropriate)

---

paradigm

approach

---

framework

model

(unless framework is technically correct)

---

robust

Only use

robust

if statistically demonstrated.

---

significant

Use only

after significance testing.

Otherwise

substantial

or

noticeable.

---

# Academic Tone

One of the strongest improvements would be

removing marketing language.

Examples

Current

> significantly outperforms state of the art

Better

> achieves higher performance than previously reported methods under the evaluated protocol.

---

Current

> demonstrates superior capability

Better

> performs consistently well across the evaluated datasets.

---

Current

> novel and innovative

Delete.

Let reviewers decide.

---

# Repetition

The manuscript repeatedly uses

consistently

substantially

effectively

successfully

significantly

robust

Replace with more varied vocabulary.

---

# Paragraph Length

Some paragraphs exceed

250 words.

Target

120–160 words.

Readers process shorter paragraphs more easily.

---

# Figure Captions

Most captions currently explain

what the figure is.

Instead explain

why it matters.

---

Current

> Figure X shows SHAP values.

Better

> Figure X illustrates that text features consistently contribute more to depression predictions than audio or video, supporting the routing behavior observed throughout the experiments.

Captions should communicate the main takeaway.

---

# Table Captions

Avoid

> Results table.

Instead

> Comparison of graph-routing strategies. Larger neighborhoods improve depression detection, while smaller neighborhoods yield better sentiment performance.

Again,

state the conclusion.

---

# Consistency

Sometimes you write

Graph Router

Sometimes

graph routing

Sometimes

GraphSAGE Router

Sometimes

routing graph

Choose

one

terminology.

---

# Tense

Methodology

Present tense

"We construct..."

Results

Past tense

"We evaluated..."

Conclusions

Present tense

"The results indicate..."

Currently these occasionally mix.

---

# Overall Readability

The chapter would improve dramatically if you

reduced

approximately

**20–25% of the text**.

Almost every section contains repetition.

You often

1.

Introduce a result.

2.

Repeat it in the table.

3.

Repeat it below the table.

4.

Repeat it again in the discussion.

Choose

one

place for detailed explanation.

---

# My Overall Editorial Recommendation

After reviewing the entire chapter, I believe the work is **technically stronger than its current presentation suggests**. The main limitation is not the research itself but the way the story is told. At several points, the manuscript reads like a chronological record of experiments rather than a focused scientific argument.

If I were editing this for submission to a high-impact journal, I would recommend the following priorities:

1. **Reframe the paper around one central idea:** graph-guided routing for unified multimodal learning.
2. **Reduce repetition** by approximately 20–25%, especially in the Results and Discussion.
3. **Move implementation details** (phase names, artifact paths, experiment IDs) to appendices or supplementary material.
4. **Rewrite the abstract and conclusion** to emphasize scientific contributions instead of listing experiments.
5. **Strengthen the discussion** by explaining *why* the observed behaviors matter for multimodal learning and affective computing more broadly.

## Final Assessment

After reading the chapter as a whole, I would characterize it as follows:

* **Research quality:** **9.0/10**
* **Experimental quality:** **9.2/10**
* **Writing quality (current):** **7.4/10**
* **Writing quality after revision:** **9.3/10**
* **Publication potential after revision:** **High**

### One additional recommendation

Having reviewed the chapter in detail, there is one change I believe would have the greatest impact on publication success:

> **Rewrite the manuscript around a small set of explicit research questions (RQs).**

For example:

* **RQ1:** Can a unified multimodal architecture effectively learn across depression, sentiment, emotion, and personality tasks?
* **RQ2:** Does graph-guided expert routing improve performance over conventional multimodal fusion methods?
* **RQ3:** How do graph construction strategies, LLM encoders, and domain adaptation affect generalization and interpretability?

Every experiment would then answer one of these RQs. This restructuring would transform the chapter from a sequence of experiments into a coherent scientific narrative, making it much more compelling for reviewers at top journals and conferences. In my opinion, this single organizational change would do more to improve the manuscript than adding another model or another experiment.
