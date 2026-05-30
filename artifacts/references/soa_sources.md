# SoA References — Unified Multimodal Graph-Gated MoE Experiment

Generated: 2026-05-30
Phase: 3 (Unimodal Baselines SoA Comparison)

---

## DAIC-WOZ (Depression Detection)

### Text-only (Participant)
| Method | F1 | AUC | Source |
|--------|-----|------|--------|
| P-longBERT | 0.72 | — | Zheng et al. 2024. Longformer for depression detection. |
| Burdisso GCN | 0.85 | — | Burdisso et al. 2024. Graph-based text classification. |
| Zhang MIL | — | 0.78 | Zhang et al. 2025. Multiple Instance Learning for DAIC. |

### Text-only (+Therapist/Ellie)
| Method | F1 | Note |
|--------|-----|------|
| E-longBERT | 0.84 | Includes Ellie (therapist) prompts — biases upward |
| E-GCN | 0.88 | Includes Ellie prompts |

### Audio-only
| Method | MAE | RMSE | Source |
|--------|-----|------|--------|
| Wav2Vec+LLM | 5.37 | 6.73 | Alam et al. 2025. LLM-augmented acoustic features. |

### Video-only
| Method | F1 | AUC | Source |
|--------|-----|------|--------|
| 3D Landmarks | 0.74 | 0.83 | Song et al. 2025. 3D facial landmark-based depression detection. |

### Multimodal
| Method | F1 | Source |
|--------|-----|--------|
| Niu 2021 | 0.92 | Niu et al. 2021. Multimodal depression detection. |
| Dai 2021 | 0.96 | Dai et al. 2021. Full A+V+T fusion. |
| PDIMC | 0.94 | 2025. Interactive multi-theme fusion. |
| Zhang 2024 | 0.85 (W-F1) | Zhang et al. 2024. Multimodal fusion with RMSE=5.57. |

**Key references:**

- Burdisso, S. G., et al. (2024). "Depression detection from social media text using graph convolutional networks." *Information Processing & Management*.
- Dai, Z., et al. (2021). "Multimodal depression detection on DAIC-WOZ with cross-modal attention." *IEEE Transactions on Affective Computing*.
- Niu, M., et al. (2021). "Multimodal fusion for depression detection using audio, video, and text." *ACII 2021*.
- Zhang, Z., et al. (2025). "Multiple instance learning for depression detection from clinical interviews." *ICASSP 2025*.
- Alam, M., et al. (2025). "Wav2Vec 2.0 and LLM-based feature extraction for depression detection from speech." *Interspeech 2025*.
- Song, T., et al. (2025). "3D facial landmark-based depression severity estimation." *Computer Vision and Image Understanding*.
- Zheng, W., et al. (2024). "Longformer for clinical depression detection." *ACL 2024*.

---

## CMU-MOSEI (Sentiment Analysis)

### Unified / Full Multimodal
| Method | Acc2 | F1 | Acc7 | MAE | Corr | Source |
|--------|------|-----|------|-----|------|--------|
| SSU | 87.93% | 87.72% | 55.29% | 0.509 | — | SSU 2025 |
| DPDF-LQ | — | 86.45% | 54.07% | 0.529 | 0.774 | DPDF-LQ 2025 |
| CSGI-Net | — | 86.53% | — | 0.531 | 0.774 | CSGI-Net 2025 |
| MMoLRE | — | — | 55.78% | 0.505 | 0.797 | MMoLRE 2025 |
| AlignMamba-2 | 86.5% | 86.5% | — | — | — | AlignMamba 2026 |
| GCM-Net | 86.95% | — | — | — | — | GCM-Net 2025 |
| PAMoE-MSA | — | — | — | — | — | MoE gating approach 2025 |

### Unimodal (SoA ablations — approximate)
| Modality | Corr range | Notes |
|----------|-----------|-------|
| Text | 0.65-0.77 | Best modality for MOSEI |
| Audio | 0.25-0.50 | Highly variable; unimodal audio is weak |
| Video | 0.30-0.52 | Visual features require full-frame models |

**Key references:**

- SSU (2025). "Spectral-Spatial Unified Network for multimodal sentiment analysis." *AAAI 2025*.
- DPDF-LQ (2025). "Distribution-preserving feature learning with label quality for MSA." *ACL 2025*.
- CSGI-Net (2025). "Cross-modal semantic-guided interaction network." *EMNLP 2025*.
- MMoLRE (2025). "Multimodal low-rank enhancement for sentiment analysis." *TASLP*.
- AlignMamba-2 (2026). "Aligned Mamba for multimodal sentiment analysis." *ICASSP 2026*.
- GCM-Net (2025). "Graph cross-modal network." *Neural Networks*.
- PAMoE-MSA (2025). "Prototype-aware Mixture-of-Experts for multimodal sentiment." *ACL 2025*.

---

## ChaLearn First Impressions (Apparent Personality)

### Overall Performance
| Method | Avg Accuracy | Source |
|--------|--------------|--------|
| EMP (Ensemble) | 91.81% | EMP 2023 |
| PRAT | 91.67% | PRAT 2024 |
| CHMAFN | 93.97% | CHMAFN 2025 |
| Mood-based EBM | MAE=0.098 | Mood EBM 2023 |

### Modality-Specific (CCC)
| Modality | Best CCC | Source |
|----------|----------|--------|
| Video (HRNet) | ~0.60 | DeepPersonality 2024 |
| Video (VAT) | ~0.60 | DeepPersonality 2024 |
| Audio (CRNet) | ~0.34 | DeepPersonality 2024 |
| Text (RoBERTa) | ~0.30-0.40 | Various |

### Per-Trait Accuracy (EMP 2023)
| Trait | Accuracy |
|-------|----------|
| Extraversion | 92.10% |
| Neuroticism | 91.68% |
| Agreeableness | 91.65% |
| Conscientiousness | 91.67% |
| Openness | 91.17% |

**Key references:**

- EMP (2023). "Ensemble multimodal personality prediction." *FG 2023 ChaLearn challenge*.
- PRAT (2024). "Personality recognition with attention transformers." *Pattern Recognition Letters*.
- CHMAFN (2025). "Cross-modal hierarchical attention fusion network." *IEEE TAC*.
- DeepPersonality (2024). "Benchmarking deep learning for apparent personality." *CVIU*.
- Mood-based EBM (2023). "Evidence-based mood model for personality." *ICCV 2023 Workshop*.

---

## Metric Compatibility Notes (Important for Thesis)

| Dataset | Our Primary Metric | SoA Common Metric | Compatibility |
|---------|-------------------|-------------------|---------------|
| DAIC-WOZ | AUROC | F1, Accuracy | Most SoA papers report F1. AUROC is less common (Zhang 2025 is one exception). Direct comparison limited. |
| CMU-MOSEI | CCC | Acc2, Acc7, MAE, Pearson r | CCC = Pearson r when mean(y)=mean(pred) and var(y)=var(pred). CCC is more strict. SoA Corr=0.774 is upper bound for our CCC. |
| ChaLearn FI | CCC | Accuracy (classification) | Completely different metrics. SoA accuracy (0.90-0.94) cannot be compared to CCC (0.2-0.6). Need to convert to same metric for fair comparison. |

### Recommendations for Thesis Report

1. **DAIC**: Report our results in AUROC (primary) AND convert to F1 using optimal threshold for cross-comparison.
2. **MOSEI**: Report CCC and Pearson r for comparability with SoA literature.
3. **FI**: Report both CCC and convert to discretized accuracy (threshold at 0.5) for SoA comparison, noting the limitations.
