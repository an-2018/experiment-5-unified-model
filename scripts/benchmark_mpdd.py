"""MPDD Benchmark Script - Unified GGMoE on MPDD Dataset

This script benchmarks the Graph-Gated Mixture of Experts (GGMoE) architecture
on the MPDD (Multimodal Personality-aware Depression Detection) dataset.

Tasks:
- Depression binary classification (primary)
- PHQ-9 severity regression (auxiliary)
- Big Five personality prediction (auxiliary)

Reference: Fu et al., "The First MPDD Challenge", ACM MM 2025
"""
import os
import sys
import json
import zipfile
from pathlib import Path
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    accuracy_score, mean_squared_error, mean_absolute_error
)
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.mpdd_loader import load_mpdd, MPDDSample
from models.encoders import ModalityProjector
from models.fusion import GatedLateFusion, LowRankMultimodalFusion


@dataclass
class MPDDTaskConfig:
    """MPDD task configuration."""
    # Task IDs
    DEPRESSION_CLF = 0      # Binary depression classification
    PHQ_REGRESSION = 1      # PHQ-9 score regression
    PERSONALITY_REGRESSION = 2  # Big Five personality regression
    
    num_tasks = 3
    
    # Task weights for multi-task learning
    task_names = ["depression_clf", "phq_regression", "personality_regression"]
    
    # Which samples have which tasks (MPDD has all tasks)
    task_available = {"depression_clf": True, "phq_regression": True, "personality_regression": True}


class MPDDMPDDataset(Dataset):
    """PyTorch Dataset for MPDD with lazy feature loading from zip."""
    
    def __init__(self, samples: List[MPDDSample], data_dir: str, track: str = "young"):
        self.samples = samples
        self.data_dir = Path(data_dir)
        self.track = track
        self.zip_path = self.data_dir / f"MPDD-{track.capitalize()}.zip"
        
        # Pre-load labels for all samples
        self.labels = {
            "depression_binary": torch.tensor([s.depression_binary for s in samples], dtype=torch.float32),
            "phq9_score": torch.tensor([s.phq9_score if s.phq9_score else 0.0 for s in samples], dtype=torch.float32),
            "personality_scores": torch.tensor([
                [s.personality_scores.get(trait, 0) for trait in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]]
                for s in samples
            ], dtype=torch.float32),
        }
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load audio features from zip
        audio_path = sample.audio_feature_path
        if audio_path and self.zip_path.exists():
            try:
                with zipfile.ZipFile(self.zip_path, 'r') as zf:
                    audio_data = zf.read(audio_path)
                    audio_feat = torch.from_numpy(np.load(BytesIO(audio_data))).float()
            except Exception as e:
                # Fallback to zeros if loading fails
                audio_feat = torch.zeros(15, 512, dtype=torch.float32)
        else:
            audio_feat = torch.zeros(15, 512, dtype=torch.float32)
        
        # Load video features from zip
        video_path = sample.video_feature_path
        if video_path and self.zip_path.exists():
            try:
                with zipfile.ZipFile(self.zip_path, 'r') as zf:
                    video_data = zf.read(video_path)
                    video_feat = torch.from_numpy(np.load(BytesIO(video_data))).float()
            except Exception as e:
                video_feat = torch.zeros(15, 709, dtype=torch.float32)
        else:
            video_feat = torch.zeros(15, 709, dtype=torch.float32)
        
        return {
            "sample_id": sample.sample_id,
            "subject_id": sample.subject_id,
            "audio": audio_feat,  # (15, 512)
            "video": video_feat,  # (15, 709)
            "depression_binary": self.labels["depression_binary"][idx],
            "phq9_score": self.labels["phq9_score"][idx],
            "personality_scores": self.labels["personality_scores"][idx],
            "modality_mask": torch.tensor([False, True, True]),  # No text
            "task_mask": torch.tensor([True, True, True]),  # All tasks available
            "task_ids": torch.tensor([0, 1, 2]),  # All tasks - we run all three
        }


class MPDDFeatureProjector(nn.Module):
    """Project MPDD audio/video features to common embedding dimension."""
    
    def __init__(self, audio_dim: int = 512, video_dim: int = 709, output_dim: int = 512):
        super().__init__()
        self.audio_projector = ModalityProjector(audio_dim, output_dim)
        self.video_projector = ModalityProjector(video_dim, output_dim)
        
    def forward(self, audio: torch.Tensor, video: torch.Tensor, modality_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project audio and video features.
        
        Args:
            audio: (batch, 15, 512)
            video: (batch, 15, 709)
            modality_mask: (batch, 3) - True if modality available
            
        Returns:
            audio_proj, video_proj: (batch, output_dim)
        """
        batch_size = audio.size(0)
        
        # Temporal pooling: mean over time dimension
        audio_pooled = audio.mean(dim=1)  # (batch, 512)
        video_pooled = video.mean(dim=1)  # (batch, 709)
        
        audio_proj = self.audio_projector(audio_pooled)  # (batch, output_dim)
        video_proj = self.video_projector(video_pooled)  # (batch, output_dim)
        
        return audio_proj, video_proj


class MPDDGGMoE(nn.Module):
    """GGMoE model for MPDD benchmark."""
    
    def __init__(
        self,
        input_dim: int = 512,
        num_experts: int = 8,
        expert_dim: int = 256,
        num_tasks: int = 3,
        use_graph: bool = True,
        graph_router_type: str = "graphsage",
    ):
        super().__init__()
        
        # Modality projection
        self.modality_projector = MPDDFeatureProjector()
        
        # MMoEEx backbone
        from models.unified_moe import MMoEEx
        self.mmoe = MMoEEx(
            input_dim=input_dim,
            num_experts=num_experts,
            expert_dim=expert_dim,
            num_tasks=num_tasks,
            num_shared=2,
            expert_isolation=True,
            graph_router_type=graph_router_type if use_graph else None,
        )
        
        # Learnable uncertainty weights for multi-task loss
        self.log_task_weights = nn.Parameter(torch.zeros(num_tasks))
        
        # Simple fusion: concatenate audio+video, project to input_dim
        # (Using simpler approach since GatedLateFusion expects 3 modalities)
        self.fusion_proj = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        modality_mask: torch.Tensor,
        task_ids: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            audio: (batch, 15, 512)
            video: (batch, 15, 709)
            modality_mask: (batch, 3) - but text is always False for MPDD
            task_ids: (batch,) - which task (0, 1, 2) to compute for each sample
            edge_index: optional graph edges
            
        Returns:
            Dict with logits, losses, etc.
        """
        # Project modalities
        audio_proj, video_proj = self.modality_projector(audio, video, modality_mask)
        
        # Simple concat fusion for audio+video only
        fused = self.fusion_proj(torch.cat([audio_proj, video_proj], dim=-1))  # (batch, 512)
        
        # GGMoE forward
        if self.mmoe.graphsage_router is not None or self.mmoe.gat_router is not None:
            out, routing_weights = self.mmoe.forward_ggmoe(fused, task_ids, edge_index)
        else:
            # Standard MMoE without graph
            routing_weights = self.mmoe.get_routing_weights(fused)  # (num_tasks, batch, num_experts)
            # Use task-specific forward
            task_id = task_ids[0].item() if task_ids.numel() == 1 else task_ids
            expert_out = self.mmoe.forward(fused, task_id)
            out = self.mmoe.task_heads[task_id](expert_out)
        
        return {
            "logits": out,
            "routing_weights": routing_weights,
            "fused_embedding": fused,
        }


class MPDDLightningModule(pl.LightningModule):
    """Lightning module for MPDD GGMoE training."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        self.model = MPDDGGMoE(
            input_dim=config.get("input_dim", 512),
            num_experts=config.get("num_experts", 8),
            expert_dim=config.get("expert_dim", 256),
            num_tasks=3,
            use_graph=config.get("use_graph", True),
            graph_router_type=config.get("graph_router_type", "graphsage"),
        )
        
        # Task-specific heads
        self.depression_head = nn.Linear(config.get("expert_dim", 256), 1)
        self.phq_head = nn.Linear(config.get("expert_dim", 256), 1)
        self.personality_head = nn.Linear(config.get("expert_dim", 256), 5)
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        # Project modalities
        audio_proj, video_proj = self.model.modality_projector(
            batch["audio"], batch["video"], batch["modality_mask"]
        )
        
        # Simple concat fusion
        fused = self.model.fusion_proj(torch.cat([audio_proj, video_proj], dim=-1))
        
        # Get routing weights (all tasks)
        routing_weights = self.model.mmoe.get_routing_weights(fused)
        
        # For multi-task, we run joint_forward to get all task outputs
        task_outputs = self.model.mmoe.joint_forward(fused)  # dict of task_id -> (batch, expert_dim)
        
        # Apply task-specific heads
        outputs = {}
        for task_id, expert_out in task_outputs.items():
            if task_id == 0:  # Depression
                outputs["depression_logit"] = self.depression_head(expert_out)
            elif task_id == 1:  # PHQ
                outputs["phq_logit"] = self.phq_head(expert_out)
            elif task_id == 2:  # Personality
                outputs["personality_logit"] = self.personality_head(expert_out)
        
        outputs["routing_weights"] = routing_weights
        outputs["fused_embedding"] = fused
        
        return outputs
    
    def test_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Test step with comprehensive metrics."""
        outputs = self.forward(batch)
        total_loss, task_losses = self.compute_loss(batch, outputs)
        
        # Get predictions
        dep_preds = torch.sigmoid(outputs["depression_logit"]).squeeze()
        phq_preds = outputs["phq_logit"].squeeze()
        pers_preds = outputs["personality_logit"]
        
        y_dep = batch["depression_binary"].cpu().numpy()
        y_phq = batch["phq9_score"].cpu().numpy()
        y_pers = batch["personality_scores"].cpu().numpy()
        
        pred_dep = dep_preds.detach().cpu().numpy()
        pred_phq = phq_preds.detach().cpu().numpy()
        pred_pers = pers_preds.detach().cpu().numpy()
        
        # Compute metrics
        metrics = {}
        metrics["depression_auc"] = roc_auc_score(y_dep, pred_dep)
        metrics["depression_auprc"] = average_precision_score(y_dep, pred_dep)
        metrics["depression_f1"] = f1_score(y_dep, (pred_dep > 0.5).astype(int))
        metrics["depression_acc"] = accuracy_score(y_dep, (pred_dep > 0.5).astype(int))
        
        metrics["phq_mae"] = mean_absolute_error(y_phq, pred_phq)
        metrics["phq_rmse"] = np.sqrt(mean_squared_error(y_phq, pred_phq))
        metrics["phq_pearson"] = pearsonr(y_phq, pred_phq)[0]
        
        # Personality (average across traits)
        pers_maes = [mean_absolute_error(y_pers[:, i], pred_pers[:, i]) for i in(0, 1, 2, 3, 4)]
        metrics["personality_mean_mae"] = np.mean(pers_maes)
        
        # Log all metrics
        for name, value in metrics.items():
            self.log(f"test/{name}", value, prog_bar=True)
        
        return metrics
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch)
        total_loss, task_losses = self.compute_loss(batch, outputs)
        
        # Log
        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/depression_loss", task_losses[0], prog_bar=False)
        self.log("train/phq_loss", task_losses[1], prog_bar=False)
        self.log("train/personality_loss", task_losses[2], prog_bar=False)
        
        return total_loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        outputs = self.forward(batch)
        total_loss, task_losses = self.compute_loss(batch, outputs)
        
        # Compute depression AUC
        dep_preds = torch.sigmoid(outputs["depression_logit"]).squeeze()
        depression_auc = roc_auc_score(
            batch["depression_binary"].cpu().numpy(),
            dep_preds.detach().cpu().numpy()
        )
        
        self.log("val/total_loss", total_loss)
        self.log("val/depression_auc", depression_auc, prog_bar=True)
        
        return {"val_loss": total_loss, "depression_auc": depression_auc}
    
    def compute_loss(self, batch: Dict, outputs: Dict) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Compute multi-task loss with uncertainty weighting."""
        task_weights = torch.exp(-self.model.log_task_weights)
        
        # Task 0: Depression binary classification
        dep_logit = outputs["depression_logit"].squeeze(-1)
        dep_loss = self.bce_loss(dep_logit, batch["depression_binary"])
        
        # Task 1: PHQ-9 regression
        phq_logit = outputs["phq_logit"].squeeze(-1)
        phq_loss = self.mse_loss(phq_logit, batch["phq9_score"])
        
        # Task 2: Personality regression (MSE)
        pers_logit = outputs["personality_logit"]
        pers_loss = self.mse_loss(pers_logit, batch["personality_scores"])
        
        task_losses = [dep_loss, phq_loss, pers_loss]
        
        # Weighted sum
        total_loss = sum(w * l for w, l in zip(task_weights, task_losses))
        
        return total_loss, task_losses
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.config.get("lr", 1e-3), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def run_benchmark(
    data_dir: str,
    track: str = "young",
    num_epochs: int = 50,
    batch_size: int = 32,
    use_graph: bool = True,
    seed: int = 42,
) -> Dict:
    """Run the full MPDD benchmark."""
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"MPDD GGMoE Benchmark")
    print(f"Track: {track}, Graph: {use_graph}, Epochs: {num_epochs}")
    print(f"{'='*60}\n")
    
    # Load MPDD data
    loader = load_mpdd(data_dir, track=track, split=None)
    all_samples = loader.samples
    
    # Split by train/val/test
    train_samples = [s for s in all_samples if s.split == "train"]
    val_samples = [s for s in all_samples if s.split == "val"]
    test_samples = [s for s in all_samples if s.split == "test"]
    
    print(f"Samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    
    # Create datasets
    train_dataset = MPDDMPDDataset(train_samples, data_dir, track)
    val_dataset = MPDDMPDDataset(val_samples, data_dir, track)
    test_dataset = MPDDMPDDataset(test_samples, data_dir, track)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model config
    config = {
        "input_dim": 512,
        "num_experts": 8,
        "expert_dim": 256,
        "use_graph": use_graph,
        "graph_router_type": "graphsage" if use_graph else None,
        "lr": 1e-3,
        "batch_size": batch_size,
    }
    
    # Create model
    model = MPDDLightningModule(config)
    
    # Create trainer
    trainer = Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=pl.loggers.CSVLogger("artifacts/logs", name=f"mpdd_{track}"),
        callbacks=[pl.callbacks.ModelCheckpoint(
            dirpath=f"artifacts/checkpoints/mpdd_{track}",
            monitor="val/depression_auc",
            mode="max",
            save_top_k=1,
        )],
        enable_progress_bar=True,
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = trainer.test(model, test_loader)
    
    return results


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> Dict:
    """Compute comprehensive metrics for a task."""
    metrics = {}
    
    if task == "depression_binary":
        metrics["auroc"] = roc_auc_score(y_true, y_pred)
        metrics["auprc"] = average_precision_score(y_true, y_pred)
        metrics["f1"] = f1_score(y_true, (y_pred > 0.5).astype(int))
        metrics["accuracy"] = accuracy_score(y_true, (y_pred > 0.5).astype(int))
        # Brier score
        metrics["brier"] = np.mean((y_pred - y_true) ** 2)
        
    elif task == "phq_regression":
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["pearson"] = pearsonr(y_true, y_pred)[0]
        metrics["spearman"] = spearmanr(y_true, y_pred)[0]
        
    elif task == "personality_regression":
        # Per-trait metrics
        for i, trait in enumerate(["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]):
            metrics[f"{trait}_mae"] = mean_absolute_error(y_true[:, i], y_pred[:, i])
            metrics[f"{trait}_ccc"] = compute_ccc(y_true[:, i], y_pred[:, i])
        # Overall
        metrics["mean_mae"] = np.mean([metrics[f"{t}_mae"] for t in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]])
        metrics["mean_ccc"] = np.mean([metrics[f"{t}_ccc"] for t in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]])
    
    return metrics


def compute_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Concordance Correlation Coefficient."""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw/mpdd")
    parser.add_argument("--track", type=str, default="young", choices=["young", "elderly", "both"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_graph", action="store_true", default=True)
    parser.add_argument("--no_graph", dest="use_graph", action="store_false")
    args = parser.parse_args()
    
    results = run_benchmark(
        data_dir=args.data_dir,
        track=args.track,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        use_graph=args.use_graph,
    )
    
    print("\n" + "="*60)
    print("Benchmark Results:")
    print(json.dumps(results, indent=2))
    print("="*60)