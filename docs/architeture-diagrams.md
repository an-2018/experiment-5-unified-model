Here are two Mermaid diagrams you can drop straight into your thesis / docs.

***

## Unified architecture diagram

This shows datasets → encoders → fusion → experts + GG‑MoE router → task heads, with domain adaptation and XAI around it. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/36215687/1a7d79d8-bdaa-4f89-ac73-642f883e9dff/detailed-plan-2.md)

```mermaid
flowchart LR
    %% Datasets
    subgraph Datasets
        DAIC["DAIC-WOZ<br>(depression)"]
        MOSEI["CMU-MOSEI<br>(sentiment+emotions)"]
        FI["ChaLearn FI<br>(apparent personality)"]
    end

    %% Modalities
    DAIC -->|audio, video, text| Encoders
    MOSEI -->|audio, video, text| Encoders
    FI -->|audio, video, text?| Encoders

    subgraph Encoders["Modality-specific encoders"]
        AEnc["Audio encoder<br>(Wav2Vec/HuBERT/WavLM)"]
        VEnc["Video encoder<br>(OpenFace AUs + ViT/3D-ResNet)"]
        TEnc["Text encoder<br>(DistilBERT/RoBERTa/LLM)"]
    end

    AEnc --> Fusion
    VEnc --> Fusion
    TEnc --> Fusion

    subgraph Fusion["Low-Rank / Gated Late Fusion"]
        FusionCore["LMF / gated fusion<br>+ cross-modal attention"]
    end

    FusionCore --> H[Shared fused embedding h_i]

    %% Domain adaptation on shared space
    subgraph DomainAdapt["Domain adaptation (optional)"]
        MMD[MMD]
        CORAL[Deep CORAL]
        DANN["DANN + GRL"]
    end

    H --> MMD
    H --> CORAL
    H --> DANN
    MMD --> H
    CORAL --> H
    DANN --> H

    %% Experts + router
    subgraph MoE["MMoEEx + GG-MoE"]
        H --> Experts["Bank of K experts<br>(MLPs / small transformers)"]
        H --> Gates["Task-specific gates g_t(h_i)"]
    end

    %% Graph-level routing
    subgraph GraphRouter["Graph router over KNN graph"]
        H_all["All h_i<br>(train/val/test)"]
        H_all --> KNN["KNN graph<br>(nodes=samples,<br>edges=similarity)"]
        KNN --> GraphSAGE["GraphSAGE / GAT router"]
        GraphSAGE --> R_all["Routing weights r_i"]
    end

    %% Combine gates + router
    R_all -->|batch indices| Combine["Combine g_t(h_i)<br>+ r_i in log-space"]
    Experts --> Combine
    Combine --> U_dep[Depression task embedding u_dep]
    Combine --> U_sent[Sentiment task embedding u_sent]
    Combine --> U_emo[Emotion task embedding u_emo]
    Combine --> U_pers[Personality task embedding u_pers]

    %% Heads
    subgraph Heads["Task-specific heads"]
        U_dep --> DepHead["Depression head<br>(binary + PHQ-8)"]
        U_sent --> SentHead[Sentiment regression]
        U_emo --> EmoHead[Multi-label emotions]
        U_pers --> PersHead[Big-Five regression]
    end

    %% Loss + uncertainty weighting
    subgraph Losses["Multitask loss"]
        DepHead --> L_dep[Depression loss]
        SentHead --> L_sent[Sentiment loss]
        EmoHead --> L_emo[Emotion loss]
        PersHead --> L_pers[Personality loss]
        L_dep --> Unc["Homoscedastic<br>uncertainty weighting"]
        L_sent --> Unc
        L_emo --> Unc
        L_pers --> Unc
        Unc --> L_total["Total loss<br>+ exclusivity reg + weight decay"]
    end

    %% XAI + narratives
    subgraph XAI["Explainable AI + Narratives"]
        H --> SHAP["SHAP / IG<br>(modality + feature importance)"]
        GraphSAGE --> GNNExp["GNNExplainer / PGExplainer<br>(subgraphs)"]
        SHAP --> XAIN["GraphXAIN / LLM<br>narrative generation"]
        GNNExp --> XAIN
    end
```

***

## End-to-end process / flow diagram

This shows the process from data prep → baseline models → unified GG‑MoE training → domain adaptation → evaluation → XAI and thesis integration. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/36215687/e7edca1f-d72d-4d67-bdd5-3740325f5e48/deep-research-report-2.md)

```mermaid
flowchart TD
    %% Data + setup
    A["Phase 0:<br>Data loading & preprocessing"] --> B
    subgraph A_details[ ]
        A1["Implement DAIC/MOSEI/FI loaders<br>(utterance/clip level, no leakage)"]
        A2["Extract audio/video/text features<br>via encoders (or on-the-fly)"]
        A3["Standardise labels & masks<br>(depression, sentiment, emotions, Big-Five)"]
    end

    %% Baselines
    B["Phase 1:<br>Unimodal & fusion baselines"] --> C
    subgraph B_details[ ]
        B1["Train unimodal baselines<br>(audio-only, text-only, video-only)"]
        B2["Train per-dataset late-fusion<br>DAIC, MOSEI, FI"]
        B3["Train per-dataset MMoEEx<br>(no graph, no multitask across datasets)"]
    end

    %% Unified backbone
    C["Phase 2:<br>Unified backbone (per dataset)"] --> D
    subgraph C_details[ ]
        C1["Share encoders + fusion layer<br>across datasets"]
        C2["Train MMoEEx heads<br>per dataset (dep/sent/emo/pers)"]
        C3["Verify multi-task convergence<br>& metrics vs baselines"]
    end

    %% Graph + GG-MoE
    D["Phase 3:<br>Add KNN graph + GraphSAGE router"] --> E
    subgraph D_details[ ]
        D1["Compute h_i for all samples<br>per split"]
        D2["Build KNN graphs<br>(train/val/test separately)"]
        D3["Train GraphSAGE router<br>(r_i over experts)"]
        D4["Integrate GG-MoE:<br>combine r_i with g_t(h_i)"]
    end

    %% Joint multitask training
    E["Phase 4:<br>Joint multitask GG-MoE training"] --> F
    subgraph E_details[ ]
        E1["Mixed DAIC/MOSEI/FI batches<br>with task masks"]
        E2["Forward: encoders → fusion → experts →<br>router → task heads"]
        E3["Compute per-task losses<br>+ uncertainty-weighted sum"]
        E4["Backprop + periodic<br>rebuild of h_i & KNN graph"]
    end

    %% Domain adaptation & ablations
    F["Phase 5:<br>Domain adaptation + ablations"] --> G
    subgraph F_details[ ]
        F1["Add MMD + CORAL + DANN<br>on shared representation h"]
        F2["Run FI→DAIC, MOSEI→DAIC<br>adaptation experiments"]
        F3["Ablations:<br>- no graph<br>- no multitask<br>- modality drops<br>- no DA"]
    end

    %% Evaluation & statistics
    G["Phase 6:<br>Metrics, calibration, statistics"] --> H
    subgraph G_details[ ]
        G1["Compute metrics per task<br>(AUROC, AUPRC, F1, MAE, CCC,...)"]
        G2["Calibration: Brier, ECE,<br>reliability plots"]
        G3["Stats: BCa bootstrap CIs,<br>DeLong tests, permutation tests"]
        G4["Parallel coordinate plots<br>for hyperparam/ablation trade-offs"]
    end

    %% XAI & narratives
    H["Phase 7:<br>XAI & narrative explanations"] --> I
    subgraph H_details[ ]
        H1["SHAP / IG / Grad-CAM<br>per modality & task"]
        H2["GNNExplainer/PGExplainer<br>for graph routing subgraphs"]
        H3["GraphXAIN LLM narratives<br>for selected cases"]
        H4["Routing heatmaps + entropy<br>(modalities, experts, datasets)"]
    end

    %% Thesis integration
    I["Phase 8:<br>Thesis chapter & report"] 
    subgraph I_details[ ]
        I1["paper/experiment_5_unified_model.tex:<br>methods, results, XAI, discussion"]
        I2["Figures: metrics + CIs,<br>subgraphs, SHAP, routing heatmaps"]
        I3["Discussion of:<br>helpful vs harmful multitask transfer,<br>role of graph & LLM layers"]
    end
```