"""Task-specific heads for all four tasks.

Task 0: DAIC depression binary classification
Task 1: MOSEI sentiment regression
Task 2: MOSEI emotion multi-label classification
Task 3: FI Big-Five personality regression
"""
import torch
import torch.nn as nn


class DepressionHead(nn.Module):
    """Binary classifier for DAIC depression (PHQ-8 >= 10)."""
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PHQ8RegressionHead(nn.Module):
    """Regressor for DAIC PHQ-8 score (0–27)."""
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SentimentHead(nn.Module):
    """Regressor for MOSEI sentiment score [-3, 3]."""
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmotionMultiLabelHead(nn.Module):
    """Multi-label classifier for 6 MOSEI emotions."""
    EMOTION_LABELS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(self.EMOTION_LABELS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PersonalityHead(nn.Module):
    """5-head regressor for Big-Five apparent personality traits."""
    TRAIT_NAMES = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.heads = nn.ModuleDict({
            trait: nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )
            for trait in self.TRAIT_NAMES
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {trait: head(x) for trait, head in self.heads.items()}