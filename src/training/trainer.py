"""PyTorch Lightning trainer module with multi-task support and experiment tracking."""
import torch
import pytorch_lightning as pl


class MultimodalTrainer(pl.LightningModule):
    """Lightning module wrapping the full unified model pipeline."""

    def __init__(self, model, tasks: list[str], learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.learning_rate = learning_rate
        # Logging
        self.save_hyperparameters()

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        return self.model(batch)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch)
        loss = outputs["total_loss"]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch)
        loss = outputs["total_loss"]
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]


class JointMultitaskTrainer(MultimodalTrainer):
    """Extended trainer with uncertainty-weighted multi-task loss and dataset balancing."""

    def __init__(self, model, tasks: list[str], learning_rate: float = 1e-3, temperature: float = 1.0):
        super().__init__(model, tasks, learning_rate)
        self.temperature = temperature

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch)
        total_loss = outputs["total_loss"]
        task_losses = outputs["task_losses"]

        # Log individual task losses
        for task_name, task_loss in zip(self.tasks, task_losses):
            self.log(f"train/{task_name}_loss", task_loss, prog_bar=False)

        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss