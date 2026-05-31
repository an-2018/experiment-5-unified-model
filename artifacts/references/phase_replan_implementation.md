# Implementation Plan: Phases 1-5 Re-execution with MOSEI Emotion + DAIC Fixes

## Context

Phase 5 QA revealed two critical issues:
1. **MOSEI Emotion = 0 samples**: `mosei_senti_data.pkl` only has sentiment (1-dim), but `mosei.hdf5` has 7-dim labels (sentiment + 6 emotions) at segment level
2. **DAIC Regression**: Joint training dropped AUROC from 0.6991 → 0.5145 (near random) because 107-sample DAIC is overwhelmed by 16,265-sample MOSEI

## Changes Summary

| Phase | Change | Rationale |
|-------|--------|-----------|
| Phase 1 | Add class imbalance plots for DAIC; Add MOSEI emotion distribution | Visualization-first; missing EDA content |
| Phase 2 | Extract MOSEI 7-dim labels from HDF5, rebuild manifest | Enable emotion task training |
| Phase 3 | Rerun unimodal baselines; document SoA for DAIC | Establish new baselines with correct data |
| Phase 4 | Add MOSEI emotion to fusion baselines | Validate fusion works with 7-dim labels |
| Phase 5 | Isolate DAIC experts + add MOSEI emotion | Fix regression + enable emotion training |

---

## PHASE 1: EDA Enhancement

### 1.1 Add DAIC Class Imbalance Plots
**File**: `scripts/phase01_eda.py`

Add new figure `09_daic_class_imbalance.png`:
- Stacked bar chart: train/val/test splits showing depressed vs non-depressed counts
- Pie chart: overall class distribution
- Table: exact numbers (already documented in `artifacts/references/daic_class_balance.md`)
- Title: "DAIC Class Imbalance Analysis — Mild (1:2.4), No SMOTE Needed"
- Legend: "Clinical threshold PHQ-8 ≥ 10"
- Color: green (non-depressed), red (depressed)

**Output**: `artifacts/figures/phase_01_eda/09_daic_class_imbalance.png`

### 1.2 Add MOSEI Emotion Distribution Plot
**File**: `scripts/phase01_eda.py`

Add new figure `10_mosei_emotion_distribution.png`:
- 6 subplots (2×3): one histogram per emotion (happiness, sadness, anger, fear, disgust, surprise)
- Per-emotion value distribution (0, 1, 2, 3 Likert scale counts)
- Include Krippendorff alpha values as text annotation
- Title: "MOSEI Emotion Label Distribution (from HDF5 All Labels)"
- Note: "Fear (α=0.02) and Surprise (α=0.09) are unreliable — consider excluding"

**Output**: `artifacts/figures/phase_01_eda/10_mosei_emotion_distribution.png`

### 1.3 Verification
- Run `uv run python scripts/phase01_eda.py`
- Verify both new figures exist and are non-empty
- QA validate: check axes labeled, titles present, data visible

---

## PHASE 2: MOSEI Emotion Label Extraction

### 2.1 Create MOSEI Emotion Label Extractor
**New file**: `scripts/phase02_extract_mosei_emotions.py`

Steps:
1. Load `data/mosei/mosei.hdf5` → `All Labels` group
2. Load `data/mosei/mosei.hdf5` → `words` group for timestamp mapping
3. For each video in All Labels:
   - Parse segment indices (e.g., `-3g5yACwYnA[0]`, `-3g5yACwYnA[1]`)
   - Get 7-dim label vector per segment from `All Labels/video_id[seg_idx]/features`
   - Get word intervals from `words/video_id[seg_idx]/intervals` to map segments to utterances
4. Aggregate segment-level labels → utterance-level (by timestamp overlap with words)
5. Build mapping: `utterance_id → {sentiment: float, happiness: float, sadness: float, anger: float, fear: float, disgust: float, surprise: float}`

**Note**: Due to noisy inter-annotator reliability (4/6 emotions have alpha < 0.2), extract all 6 but document their limitations.

**Output**: `data/mosei/mosei_emotion_labels.json` (or add to existing pickle)

### 2.2 Verify Extraction
- Check that emotion labels exist for ≥ 80% of MOSEI utterances
- Verify each emotion dimension has non-zero values
- Compare segment count vs utterance count (expected ~7 segments per video)
- Print statistics: how many samples have each emotion label non-zero

### 2.3 Update Manifest
**File**: `scripts/phase02_preprocess.py` (modify)

Add emotion labels to manifest entries:
```json
{
  "id": "mosei_train_00000",
  "dataset": "mosei",
  "split": "train",
  "labels": {
    "sentiment": 1.0,
    "happiness": 0.67,
    "sadness": 0.0,
    "anger": 0.0,
    "fear": 0.0,
    "disgust": 0.0,
    "surprise": 0.0
  }
}
```

Rebuild `data/features/manifest.json` with emotion labels included.

### 2.4 Update Dataset Contract
**File**: `configs/dataset_contract.yaml`

Add MOSEI emotion labels section:
```yaml
mosei:
  labels:
    sentiment:
      type: continuous
      range: [-3, 3]
    emotion_happiness:
      type: continuous
      range: [0, 3]
      reliability: alpha=0.41
    emotion_sadness:
      type: continuous
      range: [0, 3]
      reliability: alpha=0.12  # LOW
    # ... (document all 6 with reliability values)
```

### 2.5 Verification
- Run extraction script, verify output JSON exists
- Check manifest has emotion labels for MOSEI samples
- Run QA: validate all 7 label dimensions present

---

## PHASE 3: Unimodal Baselines (Re-run)

### 3.1 Rerun with Correct MOSEI Labels
**File**: `scripts/phase03_unimodal_baselines.py` (existing)

No structural changes needed — just re-run after Phase 2 fix ensures MOSEI emotion labels are in manifest.

### 3.2 Expected Changes
- MOSEI emotion task will now show results (previously showed N/A)
- May need to handle emotion multi-label (6 classes) differently from sentiment (regression)

### 3.3 Document SoA Improvements
**New file**: `artifacts/references/daic_soa_improvements.md` (already saved)

Update `artifacts/references/soa_sources.md` to include:
- Multi-instance learning (MIL) for DAIC: F1=0.88
- LLM-empowered structural graph: F1=0.85
- BiLSTM + adaptive pooling: F1=0.85
- Weighted BCE loss for class imbalance (no SMOTE)
- Therapist prompt bias (F1=0.88 Ellie-only vs 0.72 participant-only)

### 3.4 Verification
- Run `uv run python scripts/phase03_unimodal_baselines.py`
- Verify new CSV has emotion results for MOSEI
- QA validate: check all unimodal baselines still pass

---

## PHASE 4: Fusion Baselines (Re-run)

### 4.1 Add MOSEI Emotion Task
**File**: `scripts/phase04_fusion.py` (existing, 1461 lines)

Add MOSEI emotion to the fusion training loop:
1. Load emotion labels from manifest (6 emotions, 0-3 each)
2. Add emotion head to GatedLateFusion model
3. Train with BCE loss for multi-label emotion prediction
4. Evaluate with AUROC per emotion (or average across 6)

### 4.2 Model Update
**File**: `src/models/fusion.py`

Add `EmotionMultiLabelHead` to GatedLateFusion:
- Input: fused multimodal features (256-dim)
- Output: 6 logits (one per emotion)
- Loss: BCEWithLogitsLoss (multi-label)
- Metric: AUROC per emotion, average AUROC across 6

### 4.3 Verification
- Run `uv run python scripts/phase04_fusion.py --epochs 50`
- Verify emotion task has non-zero metrics
- QA validate: check GatedLateFusion handles 4 tasks (DAIC, MOSEI sentiment, MOSEI emotion, FI)

---

## PHASE 5: MMoEEx Joint Training (Critical Fix + MOSEI Emotion)

### 5.1 DAIC Expert Isolation (CRITICAL FIX)
**File**: `src/models/unified_moe.py`

Current architecture shares all 8 experts across all tasks. For DAIC (107 samples), this causes severe overfitting to MOSEI patterns.

**Fix**: Add `expert_isolation` flag to MMoEEx:
- When `expert_isolation=True`, DAIC task uses dedicated experts (not shared with MOSEI/FI)
- MOSEI and FI continue sharing (large datasets, robust to sharing)
- Implementation: add `task_to_expert_map` parameter
  - DAIC → experts 0-1 (isolated)
  - MOSEI sentiment + emotion → experts 2-5 (shared)
  - FI personality → experts 6-7 (shared)

### 5.2 Add MOSEI Emotion Task
**File**: `scripts/phase05_mmoe_ex.py`

Add Task 2 (MOSEI emotion):
- 6-binary labels (one per emotion)
- BCEWithLogitsLoss
- AUROC per emotion, average across 6
- Route MOSEI emotion samples to same GatedLateFusion as MOSEI sentiment

### 5.3 Training Configuration
```python
config = {
    "daic_routing": "text_only",      # Keep isolated (CRITICAL)
    "daic_experts": [0, 1],           # Isolated from MOSEI/FI
    "mosei_routing": "multimodal",    # GatedLateFusion
    "mosei_sentiment_experts": [2, 3],  # Shared with emotion
    "mosei_emotion_experts": [2, 3],    # Same as sentiment
    "fi_routing": "video_only",       # Keep isolated
    "fi_experts": [4, 5],             # Separate from MOSEI
    "shared_experts": [6, 7],         # Global shared (optional)
    "temperature": 3.0,               # Stronger upweighting for DAIC
    "patience": 20,                   # More patience for small dataset
}
```

### 5.4 Expected Results After Fix
| Dataset | Before (Phase 5 v1) | After Fix (Phase 5 v2) |
|---------|---------------------|----------------------|
| DAIC AUROC | 0.5145 | Target: ≥ 0.70 (Phase 3 text baseline) |
| MOSEI Sentiment CCC | 0.4898 | Target: ≥ 0.52 |
| MOSEI Emotion AUC | N/A | Target: ≥ 0.60 (if labels reliable) |
| FI Avg CCC | 0.5620 | Target: ≥ 0.56 (maintain) |

### 5.5 Verification
- Run `uv run python scripts/phase05_mmoe_ex.py --epochs 150`
- Compare DAIC AUROC before vs after (should improve significantly)
- QA validate: check expert routing distribution, loss curves, metric improvements

---

## QA Validation Checklist

After each phase:
1. **Phase 1**: Verify 2 new EDA figures exist, labeled, non-empty
2. **Phase 2**: Verify MOSEI emotion labels in manifest for ≥80% samples; verify 7-dim labels
3. **Phase 3**: Verify emotion task shows results; all unimodal baselines still pass
4. **Phase 4**: Verify GatedLateFusion handles 4 tasks; emotion AUC non-zero
5. **Phase 5**: Verify DAIC AUROC ≥ 0.70; MOSEI emotion AUC non-zero; expert isolation visible in routing

After all phases complete:
- Run full QA validator
- Compare against Phase 3-5 v1 results
- Document improvements in `artifacts/references/phase_regression_analysis.md`
- Decision: proceed to Phase 6 or iterate further

---

## Timeline
- Phase 1 EDA: ~15 min (script update + run)
- Phase 2 MOSEI extraction: ~30 min (HDF5 parsing + manifest rebuild)
- Phase 3 baselines: ~2 hours (GPU training)
- Phase 4 fusion: ~3 hours (GPU training)
- Phase 5 MMoEEx: ~4 hours (150 epochs)
- QA validation: ~1 hour (review + report)

**Total**: ~11 hours (can run overnight)

---

## Critical Decisions Needed Before Phase 5

1. **MOSEI emotion reliability**: Use all 6 emotions (noisy) or only happiness (alpha=0.41)?
2. **Expert isolation strategy**: Full isolation (DAIC separate tower) vs partial (temperature=3.0)?
3. **Training duration**: Keep 150 epochs or increase for DAIC (slow learner)?

Recommend: Use happiness only (most reliable), full DAIC isolation, 200 epochs with patience=25.