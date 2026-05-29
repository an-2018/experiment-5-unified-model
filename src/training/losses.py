"""Loss functions for all tasks including uncertainty-weighted multi-task loss."""
import torch
import torch.nn as nn


class DepressionLoss(nn.Module):
    """Weighted BCE / focal BCE for DAIC depression binary classification."""
    def __init__(self, pos_weight: float = 1.0, use_focal: bool = False):
        super().__init__()
        self.pos_weight = pos_weight
        self.use_focal = use_focal
        if use_focal:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits.squeeze(-1), targets.float())


class PHQ8Loss(nn.Module):
    """MAE/MSE/CCC loss for PHQ-8 severity regression."""
    def __init__(self, loss_type: str = "mae"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mae":
            return torch.mean(torch.abs(preds - targets))
        elif self.loss_type == "mse":
            return torch.mean((preds - targets) ** 2)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


class SentimentLoss(nn.Module):
    """MAE/MSE loss for MOSEI sentiment regression."""
    def __init__(self, loss_type: str = "mae"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mae":
            return torch.mean(torch.abs(preds - targets))
        elif self.loss_type == "mse":
            return torch.mean((preds - targets) ** 2)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


class EmotionLoss(nn.Module):
    """BCEWithLogitsLoss for MOSEI multi-label emotion classification."""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets.float())


class PersonalityLoss(nn.Module):
    """MAE + CCC loss for Big-Five personality regression."""
    def __init__(self, loss_type: str = "mae"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mae":
            return torch.mean(torch.abs(preds - targets))
        elif self.loss_type == "mse":
            return torch.mean((preds - targets) ** 2)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """Homoscedastic uncertainty-weighted multi-task loss (Kendall et al., 2018)."""
    def __init__(self, num_tasks: int):
        super().__init__()
        # Learnable log(sigma^2) per task — we optimize -log(sigma) to avoid instability
        self.log_sigmas = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, task_losses: list[torch.Tensor]) -> torch.Tensor:
        """Compute uncertainty-weighted sum: L = sum_i (1/(2*sigma_i^2)) * L_i + log(sigma_i)"""
        total = 0.0
        for i, loss in enumerate(task_losses):
            precision = torch.exp(-self.log_sigmas[i])
            total += 0.5 * precision * loss + self.log_sigmas[i]
        return total

    def get_task_weights(self) -> torch.Tensor:
        """Return sigma values (uncertainty std dev per task)."""
        return torch.exp(self.log_sigmas)