# Detailed Implementation Plan: Journal & AAAI-27 Revisions

This document outlines the step-by-step tasks required to implement the reviewer feedback for the *Unified Multimodal Graph-Gated MoE* paper, producing both a full-length Journal Paper and a condensed AAAI-27 Conference Paper.

## Phase 1: Experimental & Code Refinement

### Task 1.1: Expert Routing Logging
- [ ] **Action**: Implement training callbacks to log routing expert selections (weights) per sample.
- [ ] **Action**: Calculate and record routing entropy over training epochs to demonstrate specialization.
- [ ] **Validation**: Run a 1-epoch test on a small data subset. Verify that expert usage percentages and entropy values are correctly written to TensorBoard/logs.

### Task 1.2: Graph Sensitivity Sweep
- [ ] **Action**: Update the configuration parser to accept varying $K$ values ($K \in \{5, 10, 15, 20\}$).
- [ ] **Action**: Execute training sweeps for each $K$ and collect graph density/sparsity metrics alongside performance.
- [ ] **Validation**: Inspect the output logs to confirm that graph density metrics scale appropriately as $K$ increases, and that performance metrics are correctly aggregated.

### Task 1.3: Simpler Graph-Free Baseline
- [ ] **Action**: Implement a non-GNN KNN voting mechanism or a simple graph-regularized loss baseline.
- [ ] **Action**: Execute the evaluation suite for this baseline to isolate the value of the GraphSAGE router.
- [ ] **Validation**: Ensure the baseline converges properly. Compare baseline test metrics against the full GraphSAGE router to ensure a fair, apples-to-apples ablation.

### Task 1.4: Statistical Rigor & Metric Harmonization
- [ ] **Action**: Integrate paired statistical tests (DeLong for AUROC, paired bootstrap for F1) into the evaluation scripts.
- [ ] **Action**: Refactor the DAIC evaluation to compute both AUROC and F1 scores at consistent thresholds for direct SOTA comparison.
- [ ] **Validation**: Run the statistical test suite on a mocked prediction output to verify that $p$-values and confidence intervals are computed correctly without crashing.

### Task 1.5: Leakage & Bug Audit
- [ ] **Action**: Trace and document the DAIC inductive graph construction logic to guarantee no test-label leakage occurs during V3 evaluation.
- [ ] **Action**: Verify the LoRA LLM predictions to ensure independence from the full MoE predictions.
- [ ] **Validation**: Temporarily inject a known label leakage bug into the graph construction; ensure an automated test or assertion catches it.

### Task 1.6: Inference Cost Profiling
- [ ] **Action**: Add profiler hooks (e.g., using `fvcore` or `thop`) to measure FLOPs, VRAM peak memory, parameter count, and inference latency (ms/batch).
- [ ] **Validation**: Run the profiler script and check if the outputs align with expected architecture magnitudes (e.g., LLM vs. classical experts).

---

## Phase 2: Journal Paper Writing (Full-Length)

### Task 2.1: Narrative Restructuring
- [ ] **Action**: Rewrite the "Discussion" section to focus on multimodal learning insights (e.g., "What did we learn?") rather than experiment chronologies.
- [ ] **Action**: Soften claims (remove "first demonstration") and frame the cross-attention failure as "partial replication due to overparameterization".
- [ ] **Action**: Add a clinical deployment disclaimer explicitly stating the model is a decision-support tool, not a diagnostic replacement.
- [ ] **Validation**: Conduct a peer-review-style read-through to ensure a balanced, knowledge-oriented tone.

### Task 2.2: Structural Polish & Citations
- [ ] **Action**: Scrub all internal project jargon (e.g., "Experiment 5", "Phase 8").
- [ ] **Action**: Consolidate repetitive EDA figures and schematic diagrams to reduce bloat.
- [ ] **Action**: Resolve all `(?)` and `(??)` LaTeX citation placeholders.
- [ ] **Action**: Add recent literature on Graph Foundation Models, PEFT, and sparse MoE.
- [ ] **Validation**: Compile the document (`latexmk -pdf`); verify zero "undefined reference" warnings and ensure all figures are correctly linked.

---

## Phase 3: AAAI-27 Conference Paper (Condensed)

### Task 3.1: Template Formatting
- [ ] **Action**: Initialize a new LaTeX project using the official AAAI-27 author kit.
- [ ] **Action**: Set up the 7 pages (technical) + 2 pages (references) layout constraints.
- [ ] **Validation**: Ensure the skeleton document compiles cleanly with the AAAI style files without margin violations.

### Task 3.2: Content Condensation
- [ ] **Action**: Draft a new abstract ($\le 250$ words) focused entirely on graph-routing and top-level benchmarks.
- [ ] **Action**: Distill the methodology, retaining only the core architecture (GraphSAGE router, MMoEEx). Move EDA and minor ablations to an external Supplementary Material PDF.
- [ ] **Action**: Filter the Results section to include only DAIC/MOSEI/FI benchmarks and the cross-attention negative results. Exclude MPDD.
- [ ] **Action**: Condense the GraphXAIN clinical case study into a single compelling figure and one paragraph.
- [ ] **Validation**: Check page count to ensure the technical content strictly fits within 7 pages.

### Task 3.3: Double-Blind Compliance
- [ ] **Action**: Scrub author names, affiliations, and acknowledgments.
- [ ] **Action**: Remove identifying project names and replace GitHub links with anonymized versions (or state "code in supplementary").
- [ ] **Validation**: Perform a manual text search for author names, university names, and project keywords to guarantee visual anonymity.
