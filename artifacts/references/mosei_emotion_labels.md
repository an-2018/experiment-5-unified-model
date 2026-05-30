# MOSEI Emotion Labels — Data Availability Note

## Current State
- **Source**: `data/mosei/mosei.hdf5` / `All Labels` group
- **Format**: 23,248 segment entries × 7-dim feature vector
- **Available in HDF5 since 2018**: Yes (original CMU-MOSEI dataset)
- **In pickle (`mosei_senti_data.pkl`)**: NO — only sentiment (1-dim) was extracted, 6 emotions discarded

## The 7 Dimensions

| Index | Label | Range | Description |
|-------|-------|-------|-------------|
| 0 | Sentiment | -3 to +3 (7 values) | Already in pickle |
| 1 | Happiness | 0-3 | Ekman basic emotion |
| 2 | Sadness | 0-3 | Ekman basic emotion |
| 3 | Anger | 0-3 | Ekman basic emotion |
| 4 | Fear | 0-3 | Ekman basic emotion |
| 5 | Disgust | 0-3 | Ekman basic emotion |
| 6 | Surprise | 0-3 | Ekman basic emotion |

## Inter-Annotator Reliability (Krippendorff's Alpha)

| Label | Alpha | Quality |
|-------|-------|---------|
| Sentiment | 0.53 | Moderate |
| Happiness | 0.41 | Moderate |
| Disgust | 0.21 | Low |
| Anger | 0.18 | Very Low |
| Sadness | 0.12 | Very Low |
| Surprise | 0.09 | Very Low |
| Fear | 0.02 | Near Zero |

**Implication**: Emotion labels are noisy. Happiness is most reliable; fear/surprise are nearly unusable.

## Extraction Plan

The HDF5 has segment-level labels (per video segment), not utterance-level.
Need to:
1. Map `All Labels/video_id[segment_idx]` → word timestamps via `words/video_id[segment_idx]/intervals`
2. Aggregate segment-level multi-hot emotions (0-3 per emotion) → utterance-level labels
3. Build new pickle or JSON manifest with 7-dim labels per utterance

## Current Implementation

`src/data/mosei_loader.py` has `raise NotImplementedError` at line 34 for emotion loading.
The dataset contract (`configs/dataset_contract.yaml`) only specifies `sentiment` as MOSEI label.

## Recommendation

- **Option A**: Extract emotion labels properly in Phase 2 re-run
- **Option B**: Document as "emotion labels available but not extracted" and defer to Phase 8/9 (LLM-derived emotion analysis)
- **Option C**: Use only Happiness (most reliable, alpha=0.41) and drop other 5 emotions

Given the noise (alpha < 0.2 for 4/6 emotions), Option B or C is most scientifically honest.

## References
- Zadeh et al. ACL 2018: "Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph"
- CMU-MOSEI paper: http://dx.doi.org/10.18653/v1/P18-1208