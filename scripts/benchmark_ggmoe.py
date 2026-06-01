"""GGMoE with Graph Routing for MPDD

Uses the working logistic regression features but with GGMoE architecture.
"""
import sys, json, torch, torch.nn as nn
from pathlib import Path
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import zipfile
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

sys.path.insert(0, 'src')
from data.mpdd_loader import load_mpdd
from models.gnn_router import GraphSAGERouter


class MPDDGraphDataset(Dataset):
    """Dataset with graph connectivity for GGMoE."""
    
    def __init__(self, samples, data_dir, track):
        self.samples = samples
        self.data_dir = Path(data_dir)
        self.track = track
        self.zip_path = self.data_dir / f"MPDD-{track.capitalize()}.zip"
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                audio_data = zf.read(s.audio_feature_path)
                audio = np.load(BytesIO(audio_data)).mean(axis=0)
                
                video_data = zf.read(s.video_feature_path)
                video = np.load(BytesIO(video_data)).mean(axis=0)
                
                feat = np.concatenate([audio, video]).astype(np.float32)
        except:
            feat = np.zeros(1221, dtype=np.float32)
        
        return {
            "feat": torch.from_numpy(feat),
            "label": torch.tensor(s.depression_binary, dtype=torch.float32),
            "sample_id": s.sample_id,
            "idx": idx,
        }


class GGMoEDepressionModel(nn.Module):
    """GGMoE model for depression classification with graph routing."""
    
    def __init__(self, input_dim=1221, hidden_dim=256, num_experts=4, use_graph=True):
        super().__init__()
        self.use_graph = use_graph
        
        # Feature projection
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # MMoE-style experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            for _ in range(num_experts)
        ])
        
        # Task-specific gates
        self.gate = nn.Linear(hidden_dim, num_experts)
        
        # Graph router (optional)
        if use_graph:
            self.graph_router = GraphSAGERouter(
                in_dim=hidden_dim,
                hidden_dim=64,
                out_dim=num_experts
            )
        
        # Output head
        self.head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x, edge_index=None):
        # Encode features
        h = self.encoder(x)  # (batch, hidden)
        
        # Get expert outputs
        expert_outputs = torch.stack([exp(h) for exp in self.experts], dim=1)  # (batch, num_experts, hidden/2)
        
        # Gate weights
        gate_weights = torch.softmax(self.gate(h), dim=-1)  # (batch, num_experts)
        
        # Graph routing (if enabled and edge_index provided)
        if self.use_graph and edge_index is not None and len(edge_index) > 0:
            graph_weights = self.graph_router(h, edge_index)  # (batch, num_experts)
            # Combine gate and graph weights
            combined_weights = gate_weights * 0.7 + graph_weights * 0.3
            combined_weights = torch.softmax(torch.log(gate_weights + 1e-8) + 0.3 * torch.log(graph_weights + 1e-8), dim=-1)
        else:
            combined_weights = gate_weights
        
        # Weighted combination of experts
        combined = (combined_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # (batch, hidden/2)
        
        return self.head(combined).squeeze(-1)


class GGMoELightning(pl.LightningModule):
    def __init__(self, input_dim=1221, hidden_dim=256, num_experts=4, use_graph=True, lr=1e-4):
        super().__init__()
        self.model = GGMoEDepressionModel(input_dim, hidden_dim, num_experts, use_graph)
        self.lr = lr
        self.save_hyperparameters()
        
    def forward(self, batch, edge_index=None):
        return self.model(batch["feat"], edge_index)
    
    def training_step(self, batch, idx):
        # Build simple KNN graph within batch
        edge_index = self.build_batch_graph(batch)
        
        logits = self(batch, edge_index)
        loss = nn.BCEWithLogitsLoss()(logits, batch["label"])
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def build_batch_graph(self, batch):
        """Build simple graph within batch (fully connected for small batches)."""
        batch_size = batch["feat"].size(0)
        device = batch["feat"].device
        
        if batch_size < 2:
            return torch.tensor([[0], [0]], dtype=torch.long, device=device)
        
        # Simple fully connected within batch
        edges = []
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    edges.append([i, j])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long, device=device).t()
        return torch.tensor([[0], [0]], dtype=torch.long, device=device)
    
    def validation_step(self, batch, idx):
        logits = self(batch, None)
        loss = nn.BCEWithLogitsLoss()(logits, batch["label"])
        
        probs = torch.sigmoid(logits)
        labels = batch["label"]
        
        if labels.sum() > 0 and labels.sum() < len(labels):
            auc = roc_auc_score(labels.cpu().numpy(), probs.detach().cpu().numpy())
            self.log("val_auc", auc, prog_bar=True, sync_dist=True)
        
        self.log("val_loss", loss, sync_dist=True)
        return loss
    
    def test_step(self, batch, idx):
        logits = self(batch, None)
        probs = torch.sigmoid(logits)
        
        labels = batch["label"].cpu().numpy()
        preds = probs.detach().cpu().numpy()
        
        # Compute metrics
        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, preds)
            print(f"Batch {idx}: AUC={auc:.3f}")
        
        return {
            "label": labels,
            "pred": preds,
        }
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def run_ggmoe_benchmark(data_dir, track, epochs=100, batch_size=16, use_graph=True):
    print(f"\n{'='*60}")
    print(f"GGMoE with Graph Routing - {track} track")
    print(f"Graph enabled: {use_graph}, Epochs: {epochs}")
    print(f"{'='*60}")
    
    loader = load_mpdd(data_dir, track=track, split=None)
    train_samples = [s for s in loader.samples if s.split == "train"]
    val_samples = [s for s in loader.samples if s.split == "val"]
    test_samples = [s for s in loader.samples if s.split == "test"]
    
    print(f"Data: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    train_ds = MPDDGraphDataset(train_samples, data_dir, track)
    val_ds = MPDDGraphDataset(val_samples, data_dir, track)
    test_ds = MPDDGraphDataset(test_samples, data_dir, track)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)
    
    model = GGMoELightning(use_graph=use_graph, lr=1e-4)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=pl.loggers.CSVLogger("artifacts/logs", name=f"ggmoe_graph{use_graph}"),
        enable_progress_bar=True,
    )
    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="young")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no_graph", action="store_true")
    args = parser.parse_args()
    
    run_ggmoe_benchmark(
        "data/raw/mpdd", 
        args.track, 
        args.epochs, 
        args.batch_size,
        use_graph=not args.no_graph
    )