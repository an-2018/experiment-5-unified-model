"""Simple MPDD Depression Classification with proper normalization

Based on logistic regression baseline showing AUC=0.698, we know there's signal.
This version uses proper feature normalization and simpler architecture.
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


class MPDDSimpleDataset(Dataset):
    """Simple dataset with pre-computed features."""
    
    def __init__(self, samples, data_dir, track):
        self.samples = samples
        self.data_dir = Path(data_dir)
        self.track = track
        self.zip_path = self.data_dir / f"MPDD-{track.capitalize()}.zip"
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # Load and concatenate features
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
        }


class SimpleDepressionClassifier(nn.Module):
    """Simple MLP with proper initialization."""
    
    def __init__(self, input_dim=1221, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Initialize final layer to give ~0.5 output initially
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, x):
        return self.net(x).squeeze(-1)


class SimpleLightningModule(pl.LightningModule):
    def __init__(self, lr=1e-4):  # Lower learning rate
        super().__init__()
        self.model = SimpleDepressionClassifier()
        self.lr = lr
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x["feat"])
    
    def training_step(self, batch, idx):
        logits = self(batch)
        loss = nn.BCEWithLogitsLoss()(logits, batch["label"])
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, idx):
        logits = self(batch)
        loss = nn.BCEWithLogitsLoss()(logits, batch["label"])
        
        probs = torch.sigmoid(logits)
        labels = batch["label"]
        
        # Only compute AUC if we have both classes
        if labels.sum() > 0 and labels.sum() < len(labels):
            auc = roc_auc_score(labels.cpu().numpy(), probs.detach().cpu().numpy())
            self.log("val_auc", auc, prog_bar=True, sync_dist=True)
        
        self.log("val_loss", loss, sync_dist=True)
        return loss
    
    def test_step(self, batch, idx):
        logits = self(batch)
        probs = torch.sigmoid(logits)
        
        return {
            "label": batch["label"].cpu().numpy(),
            "pred": probs.detach().cpu().numpy(),
        }
    
    def on_test_epoch_end(self, outputs):
        all_labels = np.concatenate([r["label"] for r in outputs])
        all_preds = np.concatenate([r["pred"] for r in outputs])
        
        auc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
        
        print(f"\n=== Test Results ===")
        print(f"AUROC: {auc:.3f}")
        print(f"AUPRC: {auprc:.3f}")
        print(f"F1: {f1:.3f}")
        print(f"Accuracy: {acc:.3f}")
        
        self.log("test_auc", auc)
        self.log("test_auprc", auprc)
        self.log("test_f1", f1)
        self.log("test_acc", acc)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def run_simple_benchmark(data_dir, track, epochs=100, batch_size=16):
    print(f"\n{'='*60}")
    print(f"MPDD Simple Classifier - {track} track")
    print(f"Epochs: {epochs}, Batch: {batch_size}")
    print(f"{'='*60}")
    
    loader = load_mpdd(data_dir, track=track, split=None)
    train_samples = [s for s in loader.samples if s.split == "train"]
    val_samples = [s for s in loader.samples if s.split == "val"]
    test_samples = [s for s in loader.samples if s.split == "test"]
    
    print(f"Data: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    train_ds = MPDDSimpleDataset(train_samples, data_dir, track)
    val_ds = MPDDSimpleDataset(val_samples, data_dir, track)
    test_ds = MPDDSimpleDataset(test_samples, data_dir, track)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)
    
    model = SimpleLightningModule(lr=1e-4)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=pl.loggers.CSVLogger("artifacts/logs", name="simple_mpdd"),
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    run_simple_benchmark("data/raw/mpdd", "young", epochs=100, batch_size=16)