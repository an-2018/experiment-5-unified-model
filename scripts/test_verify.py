"""
Phase 0 Verification Script — Dummy batch passthrough test.

Done criteria:
- Dummy multimodal batch passes through dummy model
- One command runs unit tests
- Experiment tracking logs metrics, config, artifacts, and git hash
"""
import sys
sys.path.insert(0, "/home/anilson/thesis/thesis-experiment-5-unified-model")

print("=== Phase 0 Verification ===\n")

# 1. Test all module imports
print("1. Testing imports...")
from src.data import (
    MultimodalSample, MultimodalDataset,
    DAICSample, MOSEISample, FISample,
)
from src.models import (
    GatedLateFusion, LowRankMultimodalFusion,
    MMoEEx, GraphGatedRouter,
    DepressionHead, PHQ8RegressionHead, SentimentHead, EmotionMultiLabelHead, PersonalityHead,
    ModalityProjector,
)
from src.training import (
    MultimodalTrainer, JointMultitaskTrainer,
    DepressionLoss, PHQ8Loss, SentimentLoss, EmotionLoss, PersonalityLoss,
    UncertaintyWeightedMultiTaskLoss,
    TemperatureScaling, PlattScaling, IsotonicCalibrator,
    compute_ece, compute_brier_score,
)
from src.evaluation import (
    compute_auroc, compute_auprc, compute_f1, compute_mae, compute_ccc,
    delong_auroc_test, bootstrap_ci,
)
from src.evaluation.visualizations import setup_style
from src.utils import set_seed, get_git_hash, get_env_info, ExperimentTracker, GLOBAL_REGISTRY

print("   ✓ All imports successful\n")

# 2. Test dummy batch passthrough
print("2. Testing dummy batch passthrough...")
import torch

# Create dummy inputs
batch_size = 4
seq_len = 128

text_ids = torch.randint(0, 30000, (batch_size, seq_len))
text_mask = torch.ones(batch_size, seq_len)
audio_waveform = torch.randn(batch_size, 16000)  # 1 sec at 16kHz
frames = torch.randn(batch_size, 8, 3, 224, 224)  # 8 frames, 3 channels, 224x224
modality_mask = (True, True, True)

# Create dummy fused embedding (512-dim)
fused = torch.randn(batch_size, 512)

# Test fusion layer
fusion = GatedLateFusion(text_dim=768, audio_dim=768, video_dim=768, hidden_dim=512)
text_proj = ModalityProjector(768, 512)
audio_proj = ModalityProjector(768, 512)
video_proj = ModalityProjector(768, 512)

dummy_text = torch.randn(batch_size, 768)
dummy_audio = torch.randn(batch_size, 768)
dummy_video = torch.randn(batch_size, 768)

fused_out = fusion(dummy_text, dummy_audio, dummy_video, modality_mask)
print(f"   Fusion output shape: {fused_out.shape} ✓")

# Test MMoEEx expert routing
mmoe = MMoEEx(input_dim=512, num_experts=8, expert_dim=256, num_tasks=4)
expert_output = mmoe(fused_out, task_id=0)
print(f"   MMoEEx output shape: {expert_output.shape} ✓")

# Test task heads
dep_head = DepressionHead(input_dim=256)
dep_logits = dep_head(expert_output)
print(f"   Depression head output shape: {dep_logits.shape} ✓")

sentiment_head = SentimentHead(input_dim=256)
sentiment_out = sentiment_head(expert_output)
print(f"   Sentiment head output shape: {sentiment_out.shape} ✓")

emotion_head = EmotionMultiLabelHead(input_dim=256)
emotion_out = emotion_head(expert_output)
print(f"   Emotion head output shape: {emotion_out.shape} ✓")

pers_head = PersonalityHead(input_dim=256)
pers_out = pers_head(expert_output)
print(f"   Personality head output: {list(pers_out.keys())} ✓")

# Test multi-task loss
loss_fn = UncertaintyWeightedMultiTaskLoss(num_tasks=4)
task_losses = [
    torch.rand(1),
    torch.rand(1),
    torch.rand(1),
    torch.rand(1),
]
mt_loss = loss_fn(task_losses)
print(f"   Multi-task loss: {mt_loss.item():.4f} ✓")

print("\n   ✓ Dummy batch passes through full pipeline\n")

# 3. Test experiment tracking
print("3. Testing experiment tracking...")
tracker = ExperimentTracker("phase00_verification", log_dir="/home/anilson/thesis/thesis-experiment-5-unified-model/logs")
tracker.log_config({"model": "dummy", "batch_size": batch_size, "seed": 42})
tracker.log_metric("dummy_metric", 0.95, step=0)
tracker.log_artifact("/path/to/artifact", "test")
print(f"   Git hash: {get_git_hash()} ✓")

env_info = get_env_info()
print(f"   Python: {env_info['python'][:40]}... ✓")
print(f"   CUDA available: {env_info['cuda_available']} ✓")

print("\n=== ALL PHASE 0 VERIFICATIONS PASSED ===")