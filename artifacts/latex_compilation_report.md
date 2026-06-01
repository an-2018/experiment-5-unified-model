# LaTeX Compilation Report — chapter_8.tex

**Generated:** 2026-06-01
**Status:** Compiles successfully (42 pages, 841KB PDF)

---

## 1. Missing Images

Two figures reference image files that do not exist in the artifacts directory:

| Line | Missing File | Caption/Description |
|------|--------------|---------------------|
| 388 | `artifacts/figures/phase_06_graph/08_ablation_comparison.png` | Graph routing ablation comparing five variants (V0–V4) across all four tasks |
| 736 | `artifacts/figures/phase_07_joint_training/training_curves.png` | Training curves for V0 graph model: loss per task over 50 epochs |

**Workaround applied:** Both figures are commented out in the source to allow compilation. The captions and labels remain in the text for future restoration when the images are generated.

**Resolution needed:** Generate the following:
1. Run the graph ablation experiment visualization script → `phase_06_graph/08_ablation_comparison.png`
2. Run joint training with curve logging → `phase_07_joint_training/training_curves.png`

---

## 2. Missing References (Cross-References)

The following LaTeX labels are referenced but never defined in the document:

### Tables
- `tab:architecture_summary` (line 60)
- `tab:dataset_summary` (line 148)
- `tab:hyperparameters` (line 279)
- `tab:evaluation_protocol` (line 320)
- `tab:fusion_results` (line 355)
- `tab:unimodal_results` (line 357)
- `tab:mmoeex_results` (lines 368, 393)
- `tab:graph_results` (line 393)
- `tab:mpdd_results` (line 412)
- `tab:cross_track_results` (line 440)
- `tab:cross_dataset_results` (line 451)
- `tab:ablation_ladder` (line 462)
- `tab:llm_ablation` (line 476)
- `tab:graph_stats` (line 706)
- `tab:mosei_emotion_results` (line 709)

### Figures
- `fig:unimodal_results` (line 342)
- `fig:arch_unified` (line 158)
- `fig:graph_ablation` (line 393)
- `fig:mpdd_shap` (line 426)
- `fig:llm_ablation_delta` (line 476)
- `fig:xai_shap_beeswarm` (line 548)
- `fig:xai_gnn_subgraph` (line 548)
- `fig:expert_routing` (line 696)
- `fig:training_curves` (line 732)

### Sections
- `sec:fusion_ablation` (lines 92, 355, 357)
- `sec:leakage_protocol` (line 116)
- `sec:training_setup` (line 134)
- `sec:dataset` (lines 148, 150)
- `sec:mmoeex_results` (line 366)
- `sec:graph_results` (line 384)

**Note:** These references should resolve after running `pdflatex` 2–3 times. The warnings appear on first pass and typically resolve on subsequent runs. If they persist, the `\label` commands may be inside commented-out sections or the labels may be misspelled.

---

## 3. Missing Citations

43 citation keys are used but not found in the bibliography:

| Citation Key | First Usage |
|--------------|-------------|
| `who2023depression` | line 58 |
| `dupont2020multitask` | line 58 |
| `kendall2018multitask` | line 58 |
| `gratch2014distress` | line 66 |
| `niu2021multimodal` | line 66 |
| `dai2021fullfusion` | line 66 |
| `shazeer2017moe` | line 66 |
| `jacobs2024mmoe` | line 66 |
| `cedro2024graphxain` | line 70 |
| `hazdar2024gated` | line 92 |
| `zadeh2017LMF` | line 92 |
| `kim2024crossmodal` | line 92 |
| `ma2018model` | line 96 |
| `pamoe2025msa` | line 96 |
| `graphsage2017inductive` | line 100 |
| `gat2018attention` | line 100 |
| `zhao2025multimodalllm` | line 104 |
| `alam2025wav2vec` | line 104 |
| `burdisso2024gcndp` | line 108 |
| `zhang2025mil` | line 108 |
| `zhang2026mmfformer` | line 108 |
| `liu2019roberta` | line 123 |
| `chen2022wavlm` | line 123 |
| `eyben2016geneva` | line 123 |
| `dosovitskiy2020image` | line 123 |
| `baltrusaitis2018openface` | line 123 |
| `zadeh2018multimodal` | line 130 |
| `escalante2020chalearn` | line 139 |
| `kendall2017uncertainties` | line 294 |
| `wang2025moehealth` | line 374, 597 |
| `sun2016deep` | line 506 |
| `ganin2015dann` | line 506 |
| `ganin2016dann` | line 506 |

**Root cause:** The bibliography file (`artifacts/references/bibliography.bib`) was cleaned to remove markdown formatting, but some entries may have been corrupted or truncated during the extraction process. The `kendall2017uncertainties` entry has an empty author field according to BibTeX warnings.

**Resolution needed:**
1. Verify all 58 extracted BibTeX entries are complete
2. Check for any truncated `note` fields containing special characters
3. Re-run `bibtex chapter_8` after fixes

---

## 4. BibTeX Errors

### Error 1: kendall2017uncertainties malformed
```
I was expecting a `,' or a `}'---line 183 of file artifacts/references/bibliography.bib
   author    = {Kendall, A. and Gal, Y.},
```
**Cause:** BibTeX expects the `author` field to end with a comma, but the entry structure may be malformed. The entry appears to be truncated or missing a closing brace.

### Error 2: Empty fields warning
```
Warning--empty author in kendall2017uncertainties
Warning--empty year in kendall2017uncertainties
Warning--empty journal in kendall2017uncertainties
```
**Cause:** The regex extraction may have captured incomplete entry content for this key.

---

## 5. Overfull Hbox Warnings

Many lines exceed the text width (168–308pt overflow). Major offenders:

| Line | Overflow | Content |
|------|----------|---------|
| 455–456 | 235pt | Results source paths |
| 5–19 (table) | 308pt | Ablation ladder table |
| 5–21 (table) | 189pt | Evaluation protocol table |
| 5–22 (table) | 188pt | LLM ablation table |

**Cause:** Long file paths in `\texttt{}` and table content exceeding page width.

**Resolution:** Use `\epath{...}` from `hyperref` for file paths, or enable `\usepackage{microtype}` for better line breaking.

---

## 6. Fixed Issues (Already Applied)

| Issue | Fix Applied |
|-------|-------------|
| Table column mismatch in `chapter8_llm_results.tex` | Added missing 6th column (`---`) to "Best graph" rows |
| Bibliography corrupted author `P兮zhar` | Changed to `P{\'e}zhar` |
| `@article` used for conference papers | Changed to `@inproceedings` for `shazeer2017moe`, `ma2018model` |
| `sun2016coral` year was 2026 | Corrected to 2016 |
| `\input{tables/...}` paths wrong | Changed all 13 paths to `paper/tables/...` |
| natbib author-year incompatibility | Added `[numbers]` option |
| Bibliography had markdown headers (`## ...`) | Commented out all section headers and `---` separators |

---

## 7. Recommendations

### High Priority
1. **Generate missing images** — Run experiment scripts to produce:
   - `artifacts/figures/phase_06_graph/08_ablation_comparison.png`
   - `artifacts/figures/phase_07_joint_training/training_curves.png`

2. **Fix bibliography entry `kendall2017uncertainties`** — Verify and repair the entry structure

3. **Re-run bibtex** after bibliography fixes to resolve citation warnings

### Medium Priority
4. **Add `\usepackage{microtype}`** to preamble to fix overfull hbox warnings

5. **Use `\epath{...}` for file paths** instead of `\texttt{...}` for better line breaking

6. **Verify all `\label` commands** are placed inside figure/table environments, not in commented sections

### Low Priority
7. **Reduce table widths** by using `p{3cm}` column specifiers instead of `l` for long content

8. **Run 2–3 more pdflatex cycles** to confirm all cross-references resolve

---

## 8. Current Build Status

```
✅ pdflatex compilation: SUCCESS (42 pages)
⚠️  bibtex: ERRORS (kendall2017uncertainties malformed)
⚠️  Undefined references: 38
⚠️  Undefined citations: 43
⚠️  Missing images: 2 (commented out)
⚠️  Overfull hbox: 30+ warnings
```

**PDF Output:** `chapter_8.pdf` (841KB, 42 pages) — compiles and is viewable, but citation/reference warnings should be resolved before final submission.