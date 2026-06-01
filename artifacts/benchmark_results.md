# MPDD Benchmark Results - Final

## Dataset
- **Track**: MPDD-Young
- **Train**: 184 samples (54.3% depressed)
- **Val**: 39 samples (51.3% depressed)
- **Test**: 41 samples (22.0% depressed)
- **Features**: Wav2Vec2 (512d) + OpenFace (709d) = 1221 total

## Results Summary

### Methods Comparison

| Method | Val AUROC | Test AUROC | Notes |
|--------|-----------|------------|-------|
| Logistic Regression | 0.926 | **0.674** | Best performer (C=0.01) |
| GGMoE (no graph) | 0.559 | - | |
| GGMoE (with graph) | 0.545 | - | Batch-level KNN graph |
| Simple MLP | 0.512 | - | Not learning |

### Cross-Track Validation Results

| Evaluation | AUROC | Notes |
|------------|-------|-------|
| Young Val (within-track) | 0.926 | Training on Young |
| Cross-track (Young→Elderly) | 0.395 | **NEGATIVE TRANSFER** |
| Within Elderly (70/15 split) | 0.277 | Elderly alone also fails |

**Key Finding: Cross-track transfer FAILS - model trained on Young performs WORSE than random on Elderly**

## XAI Analysis (SHAP on Logistic Regression)

### Audio vs Video Importance
- **Audio total**: 0.141 (51.0%)
- **Video total**: 0.135 (49.0%)
- **Conclusion**: Essentially balanced in Young track

### Top 10 Important Features (Young Track)
1. audio_44: 0.0180 (most important single feature)
2. video_693: 0.0172
3. video_657: 0.0171
4. audio_388: 0.0086
5. audio_226: 0.0080
6. audio_433: 0.0080
7. video_293: 0.0077
8. video_652: 0.0076
9. video_684: 0.0073
10. video_680: 0.0072

### audio_44 Investigation
- **Young track**: Depressed subjects have LOWER audio_44 (diff: -0.024)
- **Elderly track**: Same direction but smaller difference (diff: -0.014)
- **NOT predictive in Elderly**: AUC=0.468 (below random)
- **Distribution shift**: audio_44 mean shifts from 0.040 (Young) to 0.162 (Elderly)
- **Not in Elderly top 20**: coef=0.0099 (negligible in Elderly model)

### Cross-Track Distribution Shift Analysis
- **Video shift**: 441.48 mean absolute difference (17,554x larger than audio)
- **Audio shift**: 0.025 mean absolute difference
- **Conclusion**: Video features have catastrophic distribution shift between tracks

### Feature Overlap Between Tracks
- Young top features: audio_44, audio_388, audio_226, audio_433...
- Elderly top features: video_186, video_144, video_150, video_170...
- **Overlap**: Essentially NONE for top features

## Key Insights

1. **Signal exists**: LR achieves test AUC=0.674 on Young, proving features contain depression-related information

2. **Neural networks underperform**: Small dataset (n=184) makes LR with L2 regularization more effective

3. **Graph routing not helpful**: Batch-level fully-connected graph doesn't improve GGMoE

4. **Cross-track transfer fails**: Age group difference causes severe distribution shift
   - Video features shift 17,554x more than audio
   - Top features don't overlap between tracks
   - Model trained on Young performs worse than random on Elderly

5. **audio_44 is track-specific**: Most important feature in Young is NOT predictive in Elderly

6. **Domain adaptation needed**: Future work should focus on Elderly-specific features or domain adaptation

## Implementation Artifacts

### Data
- `src/data/mpdd_loader.py` - MPDD data loader with subject-independent splits

### Benchmarks
- `scripts/benchmark_mpdd_simple.py` - LogisticRegression baseline
- `scripts/benchmark_ggmoe.py` - GGMoE with graph routing option
- `scripts/xai_analysis.py` - SHAP-based XAI analysis
- `scripts/cross_track_validation.py` - Cross-track validation
- `scripts/investigate_audio44.py` - audio_44 feature investigation

### Outputs
- `artifacts/figures/xai_analysis/feature_importance.png` - Top-30 feature bar chart
- `artifacts/figures/xai_analysis/xai_results.json` - Full SHAP results
- `artifacts/figures/cross_track_validation/cross_track_comparison.png` - Cross-track comparison
- `artifacts/figures/cross_track_validation/audio_44_analysis.png` - audio_44 distribution analysis
- `artifacts/figures/cross_track_validation/cross_track_results.json` - Cross-track metrics

## Recommendations

1. **For Young track**: Use LR with C=0.01, audio_44 is a potential biomarker
2. **For Elderly track**: Need Elderly-specific model or domain adaptation
3. **Generalization**: Current model does NOT generalize across age groups

## Date
2025-06-01