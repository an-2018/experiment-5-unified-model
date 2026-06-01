"""MPDD Depression-Focused Benchmark

Single task: Binary depression classification
No multi-task complications
"""
import sys, json, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import numpy as np

sys.path.insert(0, 'src')
from data.mpdd_loader import load_mpdd, MPDDSample


class MPDDDepressionDataset(Dataset):
    def __init__(self, samples, data_dir, track):
        self.samples = samples
        self.data_dir = Path(data_dir)
        self.track = track
        self.zip_path = self.data_dir / f"MPDD-{track.capitalize()}.zip"
        self.labels = torch.tensor([s.depression_binary for s in samples], dtype=torch.float32)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio features
        audio_feat = torch.zeros(15, 512, dtype=torch.float32)
        if self.zip_path.exists():
            try:
                with zipfile.ZipFile(self.zip_path, 'r') as zf:
                    audio_data = zf.read(sample.audio_feature_path)
                    audio_feat = torch.from_numpy(np.load(bytes(audio_data))).float()
            except:
                pass
        
        # Load video features
        video_feat = torch.zeros(15, 709, dtype=torch.float32)
        if self.zip_path.exists():
            try:
                with zipfile.ZipFile(self.zip_path, 'r') as zf:
                    video_data = zf.read(sample.video_feature_path)
                    video_feat = torch.from_numpy(np.load(bytes(video_data))).float()
            except:
                pass
        
        return {
            "audio": audio_feat.mean(dim=0),  # Pool to (512,)
            "video": video_feat.mean(dim=0),  # Pool to (709,)
            "label": self.labels[idx],
            "sample_id": sample.sample_id,
        }


class DepressionClassifier(nn.Module):
    """Simple audio+video classifier for depression detection."""
    
    def __init__(self, audio_dim=512, video_dim=709, hidden_dim=256):
        super().__init__()
        self.audio_fc = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.video_fc = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, audio, video):
        audio_emb = self.audio_fc(audio)
        video_emb = self.video_fc(video)
        fused = self.fusion(torch.cat([audio_emb, video_emb], dim=-1))
        return self.classifier(fused).squeeze(-1)


class DepressionLightningModule(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = DepressionClassifier()
        self.lr = lr
        self.save_hyperparameters()
        
    def forward(self, batch):
        return self.model(batch["audio"], batch["video"])
    
    def training_step(self, batch, idx):
        logits = self(batch)
        loss = nn.BCEWithLogitsLoss()(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, idx):
        logits = self(batch)
        loss = nn.BCEWithLogitsLoss()(logits, batch["label"])
        preds = torch.sigmoid(logits)
        auc = roc_auc_score(batch["label"].cpu().numpy(), preds.detach().cpu().numpy())
        self.log("val_loss", loss)
        self.log("val_auc", auc, prog_bar=True)
        return loss
    
    def test_step(self, batch, idx):
        logits = self(batch)
        preds = torch.sigmoid(logits)
        labels = batch["label"].cpu().numpy()
        pred_np = preds.detach().cpu().numpy()
        
        return {
            "label": labels,
            "pred": pred_np,
        }
    
    def on_test_epoch_end(self, outputs):
        # Newer Lightning uses 'outputs' not 'results'
        all_labels = np.concatenate([r["label"] for r in outputs])
        all_preds = np.concatenate([r["pred"] for r in outputs])
        
        auc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
        
        self.log("test_auc", auc)
        self.log("test_auprc", auprc)
        self.log("test_f1", f1)
        self.log("test_acc", acc)
        print(f"\nTest Results: AUC={auc:.3f}, AUPRC={auprc:.3f}, F1={f1:.3f}, Acc={acc:.3f}")
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)


def run_depression_benchmark(data_dir, track, epochs=50, batch_size=16):
    print(f"\n{'='*60}")
    print(f"MPDD Depression Classification Benchmark")
    print(f"Track: {track}, Epochs: {epochs}, Batch: {batch_size}")
    print(f"{'='*60}")
    
    loader = load_mpdd(data_dir, track=track, split=None)
    train_samples = [s for s in loader.samples if s.split == "train"]
    val_samples = [s for s in loader.samples if s.split == "val"]
    test_samples = [s for s in loader.samples if s.split == "test"]
    
    print(f"Samples: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    train_ds = MPDDDepressionDataset(train_samples, data_dir, track)
    val_ds = MPDDDepressionDataset(val_samples, data_dir, track)
    test_ds = MPDDDepressionDataset(test_samples, data_dir, track)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    model = DepressionLightningModule()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=None,
        enable_progress_bar=True,
    )
    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    import zipfile
    from io import BytesIO
    
    run_depression_benchmark("data/raw/mpdd", "young", epochs=50, batch_size=16)