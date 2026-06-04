# Exhaustive Codebase Analysis: Experiment 5 Unified Model

## ML Models Architecture Diagram
The following diagram documents the core machine learning architecture as explicitly implemented and instantiated by the `JointTrainingPipeline` in `scripts/phase07_joint_training.py` and defined across `src/models/`.

```mermaid
graph TD
    %% Input Features and Masks
    TextIn["Text Features (e.g., RoBERTa)<br>Shape: (Batch, 768)"]
    AudioIn["Audio Features (e.g., WavLM)<br>Shape: (Batch, 768)"]
    VideoIn["Video Features (e.g., ViT)<br>Shape: (Batch, 1536)"]
    Mask["Modality Mask<br>Shape: (Batch, 3)"]
    TaskID["Task ID (0 to 3)"]
    EdgeIndex["Graph Edge Index<br>Shape: (2, Num_Edges)"]

    %% JointTrainingPipeline Input Routing
    subgraph JointTrainingPipelineRouting [JointTrainingPipeline Routing]
        CondText{"routing == 'text_only'<br>(DAIC Depression)"}
        CondVideo{"routing == 'video_only'<br>(FI Personality)"}
        CondMulti{"routing == 'multimodal'<br>(MOSEI Sentiment & Emotion)"}
    end

    TextIn --> CondText & CondMulti
    AudioIn --> CondMulti
    VideoIn --> CondVideo & CondMulti

    %% Unimodal Projectors (JointTrainingPipeline level)
    subgraph ModalityProjectors [Modality Projectors Unimodal Routing]
        TextProj["text_proj<br>Sequential(Linear(768,256), LayerNorm, GELU)"]
        VideoProj["video_proj<br>Sequential(Linear(1536,256), LayerNorm, GELU)"]
    end

    CondText --> TextProj
    CondVideo --> VideoProj

    %% GatedLateFusion (Multimodal Routing)
    subgraph GatedLateFusionNode [GatedLateFusion]
        GLF_TProj["text_proj<br>ModalityProjector(768, 256)"]
        GLF_AProj["audio_proj<br>ModalityProjector(768, 256)"]
        GLF_VProj["video_proj<br>ModalityProjector(1536, 256)"]
        GLF_TGate["text_gate<br>Sequential(Linear(256,256), Sigmoid)"]
        GLF_AGate["audio_gate<br>Sequential(Linear(256,256), Sigmoid)"]
        GLF_VGate["video_gate<br>Sequential(Linear(256,256), Sigmoid)"]
        GLF_Sum["Gate-Weighted Sum<br>t_g*t + a_g*a + v_g*v<br>Shape: (Batch, 256)"]

        CondMulti --> GLF_TProj & GLF_AProj & GLF_VProj
        Mask -.->|Zeros missing modalities| GLF_TGate & GLF_AGate & GLF_VGate

        GLF_TProj --> GLF_TGate
        GLF_AProj --> GLF_AGate
        GLF_VProj --> GLF_VGate

        GLF_TProj & GLF_TGate --> GLF_Sum
        GLF_AProj & GLF_AGate --> GLF_Sum
        GLF_VProj & GLF_VGate --> GLF_Sum
    end

    %% Fused Representation
    FusedFeat["Fused Representation 'h'<br>Shape: (Batch, 256)"]
    TextProj --> FusedFeat
    VideoProj --> FusedFeat
    GLF_Sum --> FusedFeat

    %% Graph-Gated MMoEEx
    subgraph MMoEEx [Graph-Gated MMoEEx Multi-Task Multi-Expert]
        TaskGate["Task-Specific Gate (per task_id)<br>Linear(256 -> 8)"]
        GraphRouter["GraphSAGE / GAT Router<br>Shape: (Batch, 8)"]
        CombinedGate["Routing Weights<br>Softmax(log(TaskGate) + 0.5 * log(GraphRouter))<br>Shape: (Batch, 8)"]

        FusedFeat & TaskID --> TaskGate
        FusedFeat & EdgeIndex --> GraphRouter
        TaskGate & GraphRouter --> CombinedGate

        subgraph Experts [Experts 8 Instances]
            ExpNet["Expert Network<br>Sequential(Linear(256,256), GELU, Dropout, Linear(256,256))"]
            ExpSkip["Skip Connection<br>Identity()"]
            ExpSum["Expert Output Sum"]
            ExpNet & ExpSkip --> ExpSum
        end

        FusedFeat --> ExpNet & ExpSkip
        CombinedGate & ExpSum --> Mixture["Weighted Expert Mixture<br>Shape: (Batch, 256)"]
    end

    %% Task Heads
    subgraph TaskHeads [Task Heads]
        DepHead["DepressionHead (Task 0)<br>Sequential(Linear(256,128), ReLU, Dropout, Linear(128,1))"]
        SentHead["SentimentHead (Task 1)<br>Sequential(Linear(256,128), ReLU, Dropout, Linear(128,1))"]
        EmoHead["EmotionMultiLabelHead (Task 2)<br>Sequential(Linear(256,128), ReLU, Dropout, Linear(128,6))"]
        PersHead["PersonalityHead (Task 3)<br>5 x Sequential(Linear(256,64), ReLU, Dropout, Linear(64,1))"]
    end

    Mixture --> DepHead
    Mixture --> SentHead
    Mixture --> EmoHead
    Mixture --> PersHead

    DepOut["Depression Score<br>Shape: (Batch, 1)"]
    SentOut["Sentiment Score<br>Shape: (Batch, 1)"]
    EmoOut["Emotion Logits<br>Shape: (Batch, 6)"]
    PersOut["Personality Scores<br>Shape: (Batch, 5)"]

    DepHead --> DepOut
    SentHead --> SentOut
    EmoHead --> EmoOut
    PersHead --> PersOut
```

## Experiment and Process Flow Diagram
The following diagram outlines the full sequential experiment execution flow configured in `scripts/run_full_pipeline.py`, detailing the `phase07_joint_training.py` data flow and monitoring strategy.

```mermaid
graph TD
    %% Initial Datasets
    subgraph RawDatasets [Raw Datasets]
        DAIC["DAIC-WOZ Dataset<br>(Depression)"]
        MOSEI["CMU-MOSEI Dataset<br>(Sentiment, Emotion)"]
        FI["ChaLearn FI Dataset<br>(Personality)"]
    end

    %% Early Phases
    Phase1["Phase 1: Exploratory Data Analysis<br>(phase01_eda.py)"]
    Phase2["Phase 2: Preprocessing & Feature Extraction<br>(phase02_preprocess.py)"]
    
    DAIC & MOSEI & FI --> Phase1
    Phase1 --> Phase2
    Phase2 --> FeatCache["Feature Cache Directory<br>Text (RoBERTa)<br>Audio (WavLM/eGeMAPS)<br>Video (ViT/OpenFace)"]

    %% Phase 6
    Phase6["Phase 6: Graph Construction<br>(phase06_graph.py)"]
    FeatCache --> Phase6
    Phase6 --> GraphEdges["KNN Graph Edges<br>(Split-local, Inductive, or Transductive)"]

    %% Phase 7 Execution Loop
    subgraph Phase7 [Phase 7 Joint Multitask Training]
        DataLoader["GraphEnhancedDataset DataLoader<br>Batch Size: 32 (Temperature=3.0)<br>Outputs: Text (32, 768), Audio (32, 768),<br>Video (32, 1536), Mask (32, 3)"]
        
        subgraph InitializationBlock [Initialization]
            ModelInit["Instantiate JointTrainingPipeline<br>Dims: Hidden=256, Experts=8, Tasks=4<br>(Projectors initially Frozen for 20 Epochs)"]
            OptInit["Initialize Optimizer<br>(AdamW lr=3e-4 + CosineAnnealingLR)"]
        end

        subgraph EpochTrainingLoop [Epoch Training Loop Max 150 Epochs]
            ForwardPass["Forward Pass<br>Inputs: Features (32, D) -> Fused (32, 256)<br>Outputs: Task Predictions (32, 1 or 6)"]
            LossCalc["Compute Uncertainty-Weighted Joint Loss"]
            OptStep["Optimizer Step"]
            
            ForwardPass --> LossCalc --> OptStep
        end
        
        subgraph CallbacksMonitors [Callbacks and Monitors]
            NegMonitor["Negative Transfer Monitor<br>(Alert if task metric < 95% of Isolated Baseline)"]
            Unfreeze["Progressive Unfreezing<br>(Unfreeze top 2 projector layers after Epoch 20)"]
        end

        DataLoader & ModelInit & OptInit --> EpochTrainingLoop
        OptStep --> NegMonitor
        OptStep --> Unfreeze
    end

    GraphEdges & FeatCache --> Phase7

    %% Subsequent Analytics Phases
    Phase34["Phase 3 & 4: Unimodal & Fusion Baselines"]
    Phase8["Phase 8: LLM Modality Ablations"]
    Phase9["Phase 9: Domain Adaptation"]
    Phase10["Phase 10: Calibration & Validation"]
    Phase11["Phase 11: XAI (SHAP, GNNExplainer)"]
    Phase12["Phase 12: Thesis Export"]

    FeatCache --> Phase34
    Phase7 --> Phase8 & Phase9 & Phase10 & Phase11
    Phase8 & Phase9 & Phase10 & Phase11 --> Phase12
```

## Excluded Components and Justifications
The following components were explicitly analyzed but excluded from the formal diagrams to maintain strict adherence to the explicit, executed architecture of the core Phase 7 pipeline:
1. **`src/models/llm_encoders.py`**: This file contains only Abstract Base Classes (e.g., `LLMTextEncoder`, `TeacherFeatureExtractor`). These are abstractions meant for Phase 8 ablation testing and are not concrete layers instantiated in the primary joint model's forward pass.
2. **Alternative Fusion Models (`CrossAttentionFusion`, `LowRankMultimodalFusion`, `LowRankGatingNetwork` in `src/models/fusion.py`)**: While fully implemented, the main `JointTrainingPipeline` relies strictly on `GatedLateFusion`. The alternative methods represent modular swaps, primarily utilized in Phase 4 baselines, and their inclusion would misrepresent the active Phase 7 architecture.
3. **Low-level Loaders (`src/data/mpdd_loader.py`, `src/data/daic_loader.py`)**: These are abstracted beneath the `MultimodalDataset` and the `GraphEnhancedDataset`. The diagrams focus on the cached feature flow and the unified PyG dataset used in training.
4. **Evaluation and Training Utilities (`src/training/calibration.py`, `src/training/losses.py`, `src/evaluation/*`)**: Excluded from the NN architecture diagram as they compute metrics, apply post-hoc calibration (Phase 10), or calculate losses outside the bounds of the tensor shape transformations occurring within the model's direct forward pass.
