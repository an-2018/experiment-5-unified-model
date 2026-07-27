# Spec-Driven Development & Verification Plan
## Depression as an Affective Profile — AAAI Main Track Resubmission

**Version:** 1.0
**Target venue:** AAAI Main Technical Track
**Governing principle:** No number, claim, or figure enters the manuscript unless it was produced by a harness run with a recorded manifest hash and passes an automated claim assertion.

---

## 0. Why spec-driven for a paper

The v1→v2 history of this manuscript contains four failure classes that are *engineering* failures, not scientific ones:

| Observed failure | Root cause | Structural fix |
|---|---|---|
| Baseline pinned at chance (0.493) for an entire submission cycle | Sampler weight computed but never passed to `DataLoader` | Invariant test on effective per-task gradient steps (SPEC-H1-04) |
| "Every variant exceeds baseline" while V0 = 0.486 < 0.493 | Prose numbers typed by hand | Claim ledger with quantified assertions (SPEC-H8) |
| Text-only baseline reported as 0.671 and 0.699 | Two figures generated from two runs | Single-source LaTeX macro generation (SPEC-H8-02) |
| Homophily table numerically identical before and after retraining | Diagnostic run against a stale embedding cache | Checkpoint-hash binding on derived artifacts (SPEC-H7-02) |
| MMoEEx attributed to an unrelated 2026 paper | No verification gate on citations | Citation registry with human sign-off (SPEC-H9) |

Every one of these would have been caught by a cheap automated check. The plan below builds those checks first, then runs science through them.

---

## 1. Construct model specification *(the scientific spec)*

Everything downstream is engineering. This section is the thesis the engineering exists to test, and it must be written and frozen before any confirmatory analysis runs.

### 1.1 The claim

> Depression, as observed in multimodal clinical interviews, is not well characterised by a scalar score. It is a syndrome that manifests as a *characteristic configuration* across three affective levels that differ in temporal scale: stable trait vulnerability, transient emotional state, and valence bias. No single level is sufficient; the clinically meaningful signal is in the joint configuration.

This is a claim about the *problem formulation*, not about an architecture — which is what makes it main-track shaped. It also makes accuracy secondary by construction, which matters because your accuracy numbers are weak and will stay weak.

### 1.2 The three axes

| Axis | Construct | Dimensions | Temporal scale | Supervision source | Theory anchor |
|---|---|---|---|---|---|
| **Trait** | Perceived Big Five | 5 | Stable (months–years) | ChaLearn FI | Kotov et al. 2010; HiTOP |
| **State** | Discrete emotion | 6 | Transient (seconds–minutes) | CMU-MOSEI | Clark & Watson 1991 (PA/NA) |
| **Valence** | Continuous sentiment | 1 | Utterance-level | CMU-MOSEI | RDoC Negative/Positive Valence Systems |

`SPEC-CM-01` The 12-dimensional profile schema is frozen in `configs/profile_schema.yaml`, with each dimension tagged by axis (`trait` | `state` | `valence`), so that every downstream analysis can address axis blocks by name rather than by column index. Axis membership is a first-class field, not a convention.

`SPEC-CM-02` The word **"perceived"** is mandatory in every reference to the trait axis. ChaLearn FI supplies annotator first impressions from short clips, not self-reported or clinically assessed Big Five. Kotov's effect sizes are for *measured* traits; the gap is a disclosed inferential limitation, not a silent assumption. A lint rule fails the manuscript build on the bare string "personality trait" outside a quoted citation.

### 1.3 Falsifiable hypotheses

These are what turn E1–E7 from a batch of analyses into a thesis. Each is pre-registered (Section 4) with its predicted direction and its failure narrative.

**H-A — Hierarchy.** Trait dimensions carry *vulnerability* signal; state dimensions carry *episode* signal. Prediction: the trait block correlates preferentially with the cognitive/self-evaluative PHQ-8 items (self-worth, concentration), the state and valence blocks with the mood and anhedonia items, and neither with the somatic items (sleep, appetite).
*Tested by:* E3. *If it fails:* the trait/state distinction does not survive in learned representations — report as a negative construct-validity result, which is still informative.

**H-B — Low-positive-affect specificity.** From the tripartite model, the *depression-specific* marker is diminished positive affect, not elevated negative affect (which is shared with anxiety). Prediction: positive-valence dimensions (happiness, sentiment) carry **more** depression-discriminative weight than negative dimensions (sadness, anger, fear).
*Tested by:* E6. *Why it matters:* most affective-computing work implicitly assumes sadness is the marker. If your model replicates the clinical finding instead, that is a genuine AI↔psychology bridge result and directly serves the conference theme. *If it fails:* report the discrepancy — a model that finds sadness dominant where clinical theory expects anhedonia is a finding about the modality, the corpus, or the annotation.

**H-C — Non-localizability.** If depression is a joint configuration rather than a point, then similarity in the marginal affective embedding space should *not* predict depression labels, even where it predicts the component constructs. Prediction: routing-graph homophily is near chance for depression but above chance for sentiment and personality.
*Tested by:* E8 (reframed homophily diagnostic). **You already have this result** (0.568 vs. 0.555 for DAIC; 0.249 vs. −0.006 and 0.279 vs. −0.002 for MOSEI/FI). Under this framing it stops being a post-hoc explanation of a failed router and becomes a *prediction derived from the thesis and confirmed*, which is a much stronger epistemic position. Re-run it fresh under SPEC-H7-02 so the numbers bind to the current checkpoints.

**H-D — Joint necessity.** The three axes carry non-redundant depression information. Prediction: each axis block adds discriminative power over the others.
*Tested by:* E7. **This is the thesis's actual falsification condition.** If sentiment alone matches the full 12-dim profile, "depression is a joint configuration" collapses to "depression correlates with negative sentiment", which is neither new nor interesting. Run E7 early; it is as decisive as E1.

### 1.4 Boundary conditions stated in the paper

`SPEC-CM-03` Three limits are stated in the Method, not the Limitations section, because they qualify what the results mean rather than what future work might fix: (i) perceived-trait supervision, per SPEC-CM-02; (ii) zero-shot construct transfer — the emotion and trait heads are applied to DAIC out of distribution, and your own MPDD results document that transfer is fragile; (iii) no construct ground truth exists on DAIC, which is precisely why validation runs through theory-derived signs (E2) and clinical-instrument correspondence (E3) rather than direct evaluation.

`SPEC-CM-04` No diagnostic claim is made anywhere in the manuscript. Framing is characterisation and screening support, with an explicit statement that the system is not validated for clinical use. A lint rule fails the build on "diagnose", "diagnosis", or "detect depression in patients" outside cited prior work.

---

## 2. Requirements traceability

Requirements derive from the two prior review passes. Each requirement is testable and maps to at least one harness component and one acceptance test.

### 2.1 Correctness & rigour requirements (from the v2 blocking review)

| ID | Requirement | Source | Priority |
|---|---|---|---|
| REQ-01 | No aggregate claim ("every", "all", "consistently") may be written unless asserted programmatically over the artifact | B1 | P0 |
| REQ-02 | Every headline metric reported as mean ± std over ≥5 seeds, with 95% BCa CI | B2 | P0 |
| REQ-03 | Leakage-safe and transductive variants must never be aggregated in the same claim | B1 | P0 |
| REQ-04 | Component-level collapses (FI CCC = 0.000, DAIC 0.442) must be diagnosed or the ablation ladder is withdrawn | B3 | P0 |
| REQ-05 | The DAIC text-only masking confound must be tested, not merely disclosed | B4 | P1 |
| REQ-06 | Every bibliography entry carries a resolved DOI/arXiv ID and a human verification signature | B5 | P0 |
| REQ-07 | Every reported number carries an explicit split label (train/val/test) traceable to a prediction dump | B6 | P0 |
| REQ-08 | The bug narrative is removed from Results; reproducibility notes live in the appendix | B7 | P1 |
| REQ-09 | All derived diagnostics bind to the checkpoint hash that produced them | B8 | P0 |

### 2.2 Scientific contribution requirements (from the three-construct reframe)

| ID | Requirement | Experiment | Priority |
|---|---|---|---|
| REQ-10 | Depression must be shown decodable (or not) from the 12-dim cross-construct profile alone | E1 | P0 — gating |
| REQ-11 | Construct directions must be pre-registered from literature before analysis, with cryptographic proof of ordering | E2 | P0 |
| REQ-12 | Profile dimensions must be correlated against PHQ-8 item-level scores with multiplicity control | E3 | P0 |
| REQ-13 | Per-construct contribution must be measured by leave-one-construct-out ablation | E4 | P1 |
| REQ-14 | Profile-space heterogeneity within equal PHQ-8 totals must be tested | E5 | P2 |
| REQ-15 | The graph homophily result must be reframed and re-evidenced as a construct-localizability measurement | E8 / H-C | P1 |
| REQ-16 | The tripartite low-positive-affect prediction must be tested, not assumed | E6 / H-B | P0 |
| REQ-17 | Axis blocks must be shown non-redundant, or the joint-configuration thesis is withdrawn | E7 / H-D | P0 — gating |
| REQ-18 | Profile schema must tag every dimension by axis, enabling block-level analysis by name | SPEC-CM-01 | P0 |

---

## 3. Harness architecture

Ten components. Each is independently testable, has a declared artifact contract, and refuses to emit output when its preconditions fail (fail-closed, never fail-silent).

```
                      ┌──────────────────────────┐
                      │ H0  Repro Core           │  seeds, manifests, determinism
                      └────────────┬─────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼────────┐        ┌────────▼─────────┐       ┌────────▼────────┐
│ H1 Split &     │        │ H2 Training      │       │ H10 Ablation    │
│    Leakage     │───────▶│    Runner        │◀──────│     Ladder      │
└────────────────┘        └────────┬─────────┘       └─────────────────┘
                                   │
                          ┌────────▼─────────┐
                          │ H3 Inference &   │  per-sample prediction dumps
                          │    Prediction    │  (the single source of truth)
                          └────────┬─────────┘
                                   │
     ┌──────────────┬──────────────┼──────────────┬───────────────┐
     │              │              │              │               │
┌────▼─────┐  ┌─────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐  ┌─────▼──────┐
│ H4 Stats │  │ H5 Profile │ │ H6 Psycho- │ │ H7 Graph   │  │ H9 Citation│
│          │  │  Extract   │ │   metrics  │ │ Diagnostics│  │  Integrity │
└────┬─────┘  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘  └─────┬──────┘
     └──────────────┴──────────────┼──────────────┴───────────────┘
                                   │
                          ┌────────▼─────────┐
                          │ H8 Claim Ledger  │──▶ LaTeX macros, figures, CI gate
                          └──────────────────┘
```

### H0 — Reproducibility Core

**Purpose:** make every run addressable and replayable.

**Spec:**
- `SPEC-H0-01` Every run emits `artifacts/runs/{run_id}/manifest.json` containing: git SHA (dirty flag), config hash (SHA-256 of the resolved config), seed, library versions (torch, numpy, sklearn, transformers), CUDA/driver version, GPU model, hostname, UTC start/end, and the SHA-256 of every input feature cache consumed.
- `SPEC-H0-02` `run_id = sha256(config_hash || seed)[:12]`. Deterministic and collision-checked.
- `SPEC-H0-03` Determinism enforced: `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, seeded Python/NumPy/Torch/CUDA generators, `DataLoader(worker_init_fn=..., generator=...)`.
- `SPEC-H0-04` Canonical seed set frozen: `SEEDS = [17, 42, 1337, 2024, 31415]`. Never changed mid-project; adding seeds requires appending, never replacing.
- `SPEC-H0-05` A run refuses to start if the working tree is dirty unless `--allow-dirty` is passed, and the manifest records that override.

**Acceptance test:** `test_determinism()` — the same `run_id` executed twice produces bitwise-identical prediction dumps. Fails the build if not.

### H1 — Split & Leakage Harness

**Purpose:** extend the existing 9/9 audit into a mutation-tested guarantee, and prevent the class of bug that produced the chance-level baseline.

**Spec:**
- `SPEC-H1-01` Split assignment is a pure function of subject/clip ID, materialised once into `artifacts/splits/{dataset}_splits.json`, hashed, and never recomputed at training time.
- `SPEC-H1-02` Subject-independence assertion for DAIC: `set(train_ids) ∩ set(val_ids) ∩ set(test_ids) == ∅` at the *participant* level.
- `SPEC-H1-03` Graph edge audit: for inductive and split-local modes, assert every edge's destination split label by direct lookup (never by index arithmetic). Transductive mode asserts cross-split edges *are* present (positive control).
- `SPEC-H1-04` **Sampler invariant (the 0.493 regression test).** After constructing the `DataLoader`, run one dry epoch and assert: (a) the sampler instance is `WeightedRandomSampler`, not the default; (b) each task contributes ≥ `MIN_TASK_STEP_FRACTION` (default 0.10) of optimizer steps; (c) the observed per-task sampling frequency matches the intended temperature-balanced distribution within 5% TV distance.
- `SPEC-H1-05` **Mutation testing.** A `--inject-bug` mode deliberately introduces each of: a cross-split edge, a subject overlap, a dropped sampler weight, a label shuffle. The audit must detect all four. CI fails if any injected bug goes undetected.

**Acceptance test:** `test_leakage_mutations()` — 4/4 injected bugs detected. This is the strongest single reproducibility claim you can put in the paper.

### H2 — Training Runner

**Spec:**
- `SPEC-H2-01` Configuration is declarative YAML; no experiment is launched from an edited script. Config schema validated against a Pydantic model.
- `SPEC-H2-02` A *sweep* is `(config, seed) → run_id`; the runner materialises the full cross-product and skips already-completed `run_id`s idempotently.
- `SPEC-H2-03` **Health invariants checked every N steps and recorded in the manifest:**
  - `no_dead_head`: each task head receives non-zero gradient norm in ≥ 50% of steps.
  - `no_constant_output`: `std(predictions)` per head > 1e-6 on a held-out probe batch. *(An FI CCC of exactly 0.000 is a constant-output collapse — this invariant catches REQ-04 at source.)*
  - `no_dead_modality`: for each dataset, the gated-fusion weights over *unmasked* modalities are not identically zero.
  - `loss_finite`: no NaN/Inf.
- `SPEC-H2-04` Violations are recorded, not silently tolerated. A run with a violated invariant is flagged `status: degraded` and is excluded from claim aggregation unless explicitly whitelisted with a written justification.

### H3 — Inference & Prediction Dumps

**Purpose:** eliminate REQ-07 (val/test ambiguity) structurally.

**Spec:**
- `SPEC-H3-01` Every evaluation writes `artifacts/preds/{run_id}/{dataset}/{split}/{task}.npz` with arrays: `sample_id`, `y_true`, `y_score`, and attributes `split`, `dataset`, `task`, `run_id`, `checkpoint_sha`, `routing_mask`.
- `SPEC-H3-02` **No metric is ever computed inside the training loop for reporting purposes.** All reported metrics are computed downstream from these dumps by H4. Training-time metrics exist only for early stopping.
- `SPEC-H3-03` Model selection uses `val` exclusively; every headline table reports `test`. Any table cell sourced from `val` is auto-annotated with a dagger and a footnote by H8.
- `SPEC-H3-04` Dumps are immutable. Re-running overwrites only after the previous dump is archived under its `checkpoint_sha`.

### H4 — Statistics Harness

**Spec:**
- `SPEC-H4-01` Implemented once, tested against reference values: BCa bootstrap (2000 resamples, seeded), DeLong test for correlated AUROCs, paired permutation (10 000 sign-flips) for CCC/F1/MAE, Cohen's *d*, Benjamini–Hochberg FDR.
- `SPEC-H4-02` **Seed aggregation contract:** for each (config, task, metric), report `mean ± std` over the 5 seeds *and* the pooled BCa CI. Both go in the table.
- `SPEC-H4-03` **Reportability rule (REQ-01/REQ-02).** A delta between two configurations may be described directionally in prose only if
  `|Δ| > max(CI_halfwidth_baseline, 2 × seed_std_pooled)`.
  Otherwise H8 forces the phrasing "indistinguishable at this sample size". This rule is enforced in code, not by discipline.
- `SPEC-H4-04` Every statistical test emits its own artifact `artifacts/stats/{comparison_id}.json` including the test name, statistic, p-value, effect size, n, and the input dump hashes.

**Acceptance test:** `test_stats_reference()` — BCa CI and DeLong implementations reproduce published worked examples to 4 decimal places.

### H5 — Construct Profile Extractor (E1)

**Spec:**
- `SPEC-H5-01` For every sample in every dataset, forward the shared representation through **all four** task heads in a single pass, producing a 12-dimensional profile: 5 personality traits, 6 emotion probabilities, 1 sentiment scalar. Column order frozen in `configs/profile_schema.yaml`.
- `SPEC-H5-02` Profiles are computed under each dataset's **native routing mask**, and the mask is recorded per sample. A profile computed under a different mask is a different artifact.
- `SPEC-H5-03` Output: `artifacts/profiles/{run_id}/{dataset}_{split}_profiles.parquet` with `sample_id`, 12 named columns, `routing_mask`, `checkpoint_sha`.
- `SPEC-H5-04` **E1 decoder:** L2-regularised logistic regression on the 12 profile dimensions *only*, predicting DAIC binary PHQ-8. Nested CV for the regularisation constant on train, evaluated on test. Report AUROC + BCa CI, averaged over the 5 seeds.
- `SPEC-H5-05` **Mandatory controls.** E1 is meaningless without these three, and a reviewer will demand all of them:
  1. **Label-permutation null** — 1000 permutations of `y`, report the null AUROC distribution and an empirical p-value.
  2. **Random-projection control** — the same logistic regression on a random 12-dim projection of the fused embedding, matched dimensionality, 20 draws. This answers "you just did dimensionality reduction".
  3. **Unimodal-text control** — logistic regression on the raw RoBERTa CLS embedding, to establish how much of the profile's signal is simply text.
- `SPEC-H5-06` **Masking confound probe (REQ-05).** Re-extract DAIC profiles from a model trained with audio/video unmasked, and repeat E1. If the profile decoding is unchanged, the text-only masking is not driving the construct result; if it changes materially, that becomes a reported finding.

### H6 — Psychometric Validation (E2, E3)

**Spec:**
- `SPEC-H6-01` **Pre-registration file.** `preregistration/construct_signs.yaml`, one entry per profile dimension:
  ```yaml
  - dimension: neuroticism
    predicted_direction: positive      # depressed > non-depressed
    literature_effect_size: 1.65       # Cohen's d
    source: kotov2010                  # must resolve in refs/registry.yaml
    confidence: high
  - dimension: extraversion
    predicted_direction: negative
    literature_effect_size: -1.47
    source: kotov2010
    confidence: high
  - dimension: happiness
    predicted_direction: negative
    source: clark_watson1991           # low positive affect, depression-specific
    confidence: high
  # ... all 12 dimensions, including explicit "no prediction" entries
  ```
- `SPEC-H6-02` **Temporal integrity proof.** The pre-registration file is committed to git before any E2 analysis code is run. `test_prereg_ordering()` asserts (a) the file's SHA-256 recorded in the results artifact matches the file on disk, and (b) the commit introducing `construct_signs.yaml` is an ancestor of the commit introducing `analysis/e2_sign_test.py`. State this check explicitly in the paper — it is verifiable by anyone with the repository.
- `SPEC-H6-03` **E2 analysis:** Cohen's *d* per dimension between PHQ-8 positive and negative DAIC subjects; sign match against pre-registration; binomial test on matches out of the number of dimensions carrying a prediction; report the full 12-row table including misses.
- `SPEC-H6-04` **E3 analysis:** Spearman correlation of each of the 12 profile dimensions against each of the 8 PHQ-8 item scores → a 12 × 8 matrix, with BH-FDR correction across all 96 tests at q = 0.05.
- `SPEC-H6-05` **E3 block hypothesis, pre-registered alongside E2:** affective dimensions (sentiment, happiness, sadness) load on items 1–2 (anhedonia, depressed mood) and personality/neuroticism loads on item 6 (self-worth); somatic items 3 and 5 (sleep, appetite) are predicted null. Tested as a planned contrast (mean |ρ| in predicted-signal block vs. predicted-null block, permutation test), *not* as a heatmap eyeball.

**Data dependency — verify on day 1.** E3 is the strongest experiment in the plan and depends entirely on per-item PHQ-8 availability. DAIC-WOZ ships item-level PHQ-8 in the train/dev label CSVs; confirm coverage on your test split before committing to this experiment. If items are only available for train/dev, run E3 on dev and say so explicitly.

### H7 — Graph Diagnostics

**Spec:**
- `SPEC-H7-01` Homophily diagnostic recomputed for every (variant, seed), reporting real-edge label agreement/correlation vs. a random-pairing baseline with matched degree distribution.
- `SPEC-H7-02` **Staleness guard (REQ-09).** The graph builder records the `checkpoint_sha` of the embeddings it consumed. `test_diagnostic_freshness()` asserts that the `checkpoint_sha` in every homophily artifact equals the `checkpoint_sha` of the model currently reported in Table 1. This is the check that would have caught the unchanged homophily table.
- `SPEC-H7-03` Diagnostics are reported per split with an explicit label, and leakage-safe variants are aggregated separately from transductive (REQ-03) — enforced by the aggregation function refusing mixed input.
- `SPEC-H7-04` **Reframed output (REQ-15):** the diagnostic is reported as *construct localizability* — the degree to which each construct forms coherent neighbourhoods in the shared affective space — with the routing result as a downstream consequence rather than the headline.

### H8 — Claim Ledger

**Purpose:** the component that would have prevented B1 and B8 outright. This is the highest-leverage item in the plan.

**Spec:**
- `SPEC-H8-01` `claims/ledger.yaml` registers every numeric or quantified statement in the manuscript:
  ```yaml
  - id: C-E1-AUROC
    kind: scalar
    template: "profile-only decoding reaches AUROC {value:.3f} (95% CI {ci_lo:.3f}–{ci_hi:.3f})"
    source: artifacts/stats/e1_profile_decode.json
    pointer: $.test.auroc
  - id: C-MOSEI-ALL-EXCEED
    kind: assertion
    statement: "every leakage-safe graph variant exceeds the non-graph baseline on MOSEI sentiment CCC"
    source: artifacts/tables/table1.json
    assertion: "all(v.mosei_ccc > baseline.mosei_ccc for v in variants if v.leakage_safe)"
  ```
- `SPEC-H8-02` A build step renders the ledger to `paper/generated_macros.tex`. **The manuscript contains zero hand-typed numerals in Results.** `\ClaimEOneAuroc` replaces `0.712`.
- `SPEC-H8-03` `verify_claims.py` evaluates every `kind: assertion` entry against its artifact. A false assertion fails CI with the offending values printed. *Applied to v2, `C-MOSEI-ALL-EXCEED` fails immediately: V0 = 0.486 < 0.493.*
- `SPEC-H8-04` A linter scans the `.tex` source for bare decimal literals in Results/Abstract sections and fails the build on any that are not ledger macros.
- `SPEC-H8-05` The reportability rule (SPEC-H4-03) is applied at render time: a claim whose delta fails the threshold renders as "indistinguishable" phrasing automatically.

### H9 — Citation Integrity

**Spec:**
- `SPEC-H9-01` `refs/registry.yaml`, one entry per bibliography key:
  ```yaml
  - key: aoki2021mmoeex
    doi: 10.1109/TCBB.2022.3175456
    title: "Heterogeneous Multi-task Learning with Expert Diversity"
    authors: [Aoki, Tung, Oliveira]
    venue: "BIOKDD 2021 / IEEE-ACM TCBB 2022"
    supports_claim: "MMoEEx expert exclusivity regularisation — our expert bank baseline"
    located_at: "Sec. III-B, Eq. 4"
    verified_by: AM
    verified_date: 2026-08-02
  ```
- `SPEC-H9-02` Automated resolution: every DOI/arXiv ID resolved against Crossref/arXiv APIs; fuzzy-match title, first author, and year. Mismatch → build failure. *(Run this outside the sandbox; the container's egress allowlist does not include Crossref.)*
- `SPEC-H9-03` **Human gate:** `verified_by` and `located_at` cannot be auto-populated. An entry without them fails the build. `located_at` forces you to have actually opened the paper and found the claim.
- `SPEC-H9-04` **Known corrections to apply immediately:**
  - MMoEEx → Aoki, Tung & Oliveira (BIOKDD 2021 / IEEE-ACM TCBB 2022). The current attribution to Deng, Liu & Yang (2026, *Scientific Reports*, vocal metaverse classrooms) is a real paper miscredited for your core component.
  - Re-add Vyalla et al. (2026), PsyGAT — the closest competing graph model on DAIC, currently demoted to supplementary.
  - Add the psychology anchors: Clark & Watson (1991); Kotov, Gámez, Schmidt & Watson (2010), *Psych. Bulletin* 136(5):768–821; Insel et al. (2010) RDoC; Rottenberg / Bylsma et al. (2008).
  - Re-verify every remaining 2026 entry against its PDF.

### H10 — Ablation Ladder Harness

**Spec:**
- `SPEC-H10-01` Each ladder row is a config differing from its predecessor in exactly one component; the runner asserts the config diff has cardinality 1.
- `SPEC-H10-02` Every row runs across the 5 seeds with H2 health invariants active. **Rows that violate `no_constant_output` are reported as instrumented failures with the diagnosis, not as metric values** — this is the fix for REQ-04. An FI CCC of 0.000 goes into the paper as "head collapse, diagnosed as X", or the row is fixed and re-run.
- `SPEC-H10-03` **E4 leave-one-construct-out:** configs `{dep}`, `{dep,emo,sent}`, `{dep,pers,sent}`, `{dep,pers,emo}`, `{all}` at matched parameter count (assert capacity parity within 2%), 5 seeds each.
- `SPEC-H10-04` The ladder must include the unimodal text-only row. If it remains the strongest DAIC configuration, that is reported as a headline finding in the abstract, not buried in supplementary Table 2.

---

### H11 — Construct Hypothesis Tests (E5–E8)

**Purpose:** test the thesis of Section 1 directly, rather than inferring it from E1–E4. All four run on the profile artifacts produced by H5 and are inference-only.

**E6 — Tripartite specificity (H-B).**
- `SPEC-H11-01` Partition the 12 dimensions into a positive-affect block (happiness, sentiment) and a negative-affect block (sadness, anger, fear, disgust). Fit the E1 logistic regression separately on each block; compare test AUROC with a paired bootstrap over the 5 seeds.
- `SPEC-H11-02` Report standardised coefficients from the full 12-dim model so the per-dimension contribution is legible, with sign checked against the E2 pre-registration.
- **Success:** positive-affect block ≥ negative-affect block, replicating Clark & Watson in a multimodal ML setting. **Failure is still reportable** as a discrepancy between clinical theory and what the modalities actually carry.

**E7 — Axis non-redundancy (H-D). *This is the thesis's falsification test — run it in Phase 1 alongside E1.***
- `SPEC-H11-03` Nested model comparison across the three axis blocks: fit logistic regressions on {trait}, {state}, {valence}, all three pairs, and the full set. Report test AUROC + BCa CI for all seven.
- `SPEC-H11-04` Incremental contribution: for each axis, the delta from removing it from the full model, with a paired permutation test and the SPEC-H4-03 reportability rule applied.
- **Success:** every axis contributes a delta exceeding the seed-variance floor.
- **Failure:** if {valence} alone matches the full profile, the thesis reduces to "depression correlates with negative sentiment". Pre-committed fallback: retitle around trait–state *dissociation* (E3/H-A) rather than joint configuration, and report the redundancy finding honestly as the main negative result.

**E5 — Profile-space subtyping (optional, P2).**
- `SPEC-H11-05` Cluster DAIC subjects in the 12-dim profile space (k selected by silhouette on validation, k ∈ {2..5}), then test whether clusters differ in PHQ-8 *item* patterns while controlling for PHQ-8 *total* — i.e. stratify to subjects within a narrow total-score band and test item-profile differences by permutation.
- **Success:** equal-total subjects occupy distinct profile regions with distinct symptom compositions. This is the clinically meaningful version of the interpretability claim: heterogeneity invisible to the scalar score.
- Given DAIC's n, treat as exploratory and label it so; do not report cluster statistics as confirmatory.

**E8 — Construct localizability (H-C, reframed homophily).**
- `SPEC-H11-06` Recompute homophily per construct under SPEC-H7-02 freshness binding, reporting real-edge agreement against a degree-matched random-pairing null, per split, leakage-safe variants aggregated separately from transductive.
- `SPEC-H11-07` Report as a *prediction test*, not a diagnostic: H-C predicts near-chance homophily for depression and above-chance for sentiment and personality. State the prediction before the numbers in the manuscript.
- This is the section where the routing work lives in the new paper — roughly one subsection, framed as a measurement of construct geometry rather than an architecture ablation.

---

## 4. Pre-registration protocol

Two files are frozen before any confirmatory analysis, and their git ordering is asserted in CI (SPEC-H6-02):

1. `preregistration/construct_model.yaml` — hypotheses H-A through H-D from Section 1.3, each with its predicted direction, its acceptance criterion, and **its pre-committed failure narrative**. Writing the failure narrative in advance is what stops a null result becoming spin under deadline pressure.
2. `preregistration/construct_signs.yaml` — E2 directional predictions with citations.
3. `preregistration/block_hypothesis.yaml` — E3 predicted signal/null blocks in the 12 × 8 matrix.

**Exploratory work is permitted and expected**, but must be run on the *validation* split and labelled `exploratory` in every artifact. Confirmatory analyses touch `test` exactly once. This distinction is stated in the paper and is a large part of what will differentiate it from typical affective-computing submissions.

---

## 5. Execution phases and decision gates

Each phase ends in a gate with a pre-specified decision, including what gets written if the gate fails. Pre-committing the failure narrative is what stops a null result from turning into spin under deadline pressure.

### Phase 0 — Harness foundation *(no science)*
Build H0, H1, H3, H4, H8, H9. Port existing leakage audit into H1 and add mutation tests.
**Exit criteria:** `test_determinism` passes; 4/4 injected bugs detected; H8 evaluates the v2 ledger and correctly *fails* `C-MOSEI-ALL-EXCEED`; H9 flags the MMoEEx miscitation.
**Estimate:** 5–8 working days.

### Phase 1 — Preview run *(cheap, high information)*
Run H5/H6/H11 against your **existing checkpoints** before any retraining. E1, E2, E3, E6, E7 are all inference-only.
**Gate G1a — decoding:** Does profile-only decoding beat the permutation null and both controls (E1)?
**Gate G1b — non-redundancy:** Does each axis block contribute beyond the others (E7)?
- **Both pass** → the joint-configuration thesis is viable; proceed to Phase 2.
- **G1a passes, G1b fails** → the profile decodes but is carried by one axis. Retitle around trait–state dissociation; the paper survives in weakened form.
- **G1a fails** → stop. Do not rewrite around a thesis the data rejects. Fall back to the routing paper with the rigour fixes, and target TMLR.

**Estimate:** 2–3 days. This is deliberately the earliest gate because it is the cheapest way to learn whether the whole reframe is real.

### Phase 2 — Confirmatory training
Full sweep: 6 routing configs + 7 ladder rows + 5 E4 configs = 18 configs × 5 seeds = **90 runs**. Budget GPU-hours accordingly and run the sweep idempotently so interruptions are free.
**Gate G2:** Do the H2 health invariants pass on every row? Any row that collapses must be diagnosed before it appears in a table.

### Phase 3 — Confirmatory analysis
E1 (final), E2 sign test, E3 item matrix, E4 construct ablation, E6 tripartite specificity, E7 non-redundancy, E8 localizability.
**Gate G3:** ≥ 8/12 pre-registered signs correct at *p* < 0.05 binomial → construct validity claim is supported. Below that, report the sign table honestly as a partial result and reframe the contribution around E1 + E3 only.

### Phase 4 — Optional depth
E5 profile-space subtyping (exploratory, labelled as such). Include only if Phases 1–3 leave page budget.

### Phase 5 — Manuscript assembly
Rebuild the paper against the claim ledger. Delete Figure 1 (software pipeline diagram). Redraw Figure 2 as a clean schematic. Move domain adaptation, MPDD, calibration, and LLM ablations to supplementary. Remove the bug narrative from Results (REQ-08).
**Gate G4:** `verify_claims.py` green; bare-numeral linter green; citation registry 100% human-verified; AAAI reproducibility checklist complete.

---

## 6. Verification levels

| Level | What it checks | Examples | Runs |
|---|---|---|---|
| **L0 Unit** | Pure functions | BCa CI vs. reference values, DeLong vs. published example, KNN graph construction, CCC formula | Every commit |
| **L1 Contract** | Artifact schemas | Manifest completeness, `.npz` field presence, parquet column order matches `profile_schema.yaml` | Every commit |
| **L2 Invariant** | System properties | Determinism, subject-independence, sampler correctness, no dead head, no constant output, mutation detection | Every commit |
| **L3 Statistical** | Inference validity | Permutation nulls, random-projection controls, seed-variance floor, FDR correction applied | Per analysis run |
| **L4 Claim** | Paper ↔ artifact | Ledger assertions, bare-numeral lint, split-label annotation, leakage-safe/transductive separation | Pre-submission, every build |
| **L5 Scientific** | Pre-registration | Sign-table binomial, block-hypothesis contrast, git ordering proof | Once, confirmatory |

**CI policy:** L0–L2 on every push; L3–L4 nightly and as a pre-submission gate; L5 exactly once, with the run recorded.

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| E1 lands at chance → reframe invalid | Medium | Critical | Gate G1a runs in Phase 1 on existing checkpoints, before any rewriting cost is sunk |
| One axis dominates → joint-configuration thesis collapses to "sentiment correlates with depression" | **Medium-High** | Critical | E7 is co-gating at G1b with a pre-committed fallback title and framing |
| H-B fails (sadness dominates over anhedonia) | Medium | Low | Reportable either way; a theory–model discrepancy is a publishable observation |
| PHQ-8 item scores unavailable for the test split | Medium | High | Verify day 1; fall back to dev-split E3 with explicit disclosure |
| Apparent-personality validity attacked in review | **High** | High | Pre-empt in the Method: consistently write "perceived personality", devote a paragraph to what perceived traits can and cannot support, and cite Kotov's effect sizes as *measured*-trait references with the gap acknowledged |
| Reviewer: "12-dim LR is just dimensionality reduction" | High | High | SPEC-H5-05 random-projection and text-only controls exist precisely for this |
| MOSEI→DAIC domain shift invalidates zero-shot profiles | Medium | Medium | Report explicitly; your own MPDD/DA results already document transfer fragility — cite them as a known boundary |
| 90-run sweep exceeds compute budget | Medium | Medium | Prioritise: routing configs and E4 at 5 seeds; ladder rows at 3 seeds with the reduction disclosed |
| Seed variance swamps all routing deltas | **High** | Low | This is a *result*, not a failure — it is the honest answer to B2 and strengthens the paper |
| Deadline pressure → skip gates | Medium | Critical | Gates are CI-enforced; skipping requires an explicit override recorded in the manifest |

---

## 8. Deliverable map

| Paper element | Produced by | Verified by |
|---|---|---|
| Table 1 (routing, 5 seeds, CIs) | H2 → H3 → H4 | L4 claim assertions, SPEC-H4-03 |
| Table 2 (construct localizability) | H7 | SPEC-H7-02 freshness guard |
| Table 3 (E2 sign table, 12 rows) | H6 | L5 pre-registration ordering |
| Figure: 12 × 8 PHQ-8 item matrix | H6 | BH-FDR, block contrast |
| Table 4 (E4 construct ablation) | H10 | Capacity parity assertion |
| Figure: E1 decoding + 3 controls | H5 | Permutation null, control arms |
| Table: E7 axis non-redundancy (7 models) | H11 | Paired permutation, SPEC-H4-03 |
| Figure: E6 positive- vs negative-affect blocks | H11 | Paired bootstrap over seeds |
| Table/Figure: E8 construct localizability | H11 | Freshness binding, degree-matched null |
| Construct model + hypotheses (Sec. 1) | SPEC-CM | Pre-registration ordering proof |
| Reproducibility appendix | H0 + H1 | Mutation test results, 4/4 |
| Bibliography | H9 | 100% human-verified with `located_at` |

---

## 9. First week, concretely

1. **Day 1** — Verify PHQ-8 item availability on DAIC splits. This single fact determines whether E3 (your strongest experiment) is feasible. Do nothing else until you know.
2. **Day 1–2** — Write `refs/registry.yaml`; fix the MMoEEx attribution; re-add PsyGAT; add the four psychology anchors.
3. **Day 2–3** — Build H8 against the *current* v2 numbers. Watch `C-MOSEI-ALL-EXCEED` fail. That failing test is your proof the harness works.
4. **Day 3–4** — Write and commit `preregistration/construct_signs.yaml`. Commit it *before* writing any analysis code, so the git ordering proof is genuine.
5. **Day 3–4** — Write and commit `preregistration/construct_model.yaml` (H-A…H-D with failure narratives) alongside the sign table.
6. **Day 4–5** — H5 profile extraction against existing checkpoints; run E1 with all three controls **and E7 axis non-redundancy**. These two together decide the paper.
7. **End of week** — Gate G1a/G1b decision. You will know whether you have an AAAI paper or a TMLR paper, at a cost of one week and near-zero GPU time.

---

*The plan is deliberately front-loaded with cheap, decisive tests. Everything expensive happens after Gate G1, and nothing is written until the harness can prove it.*
