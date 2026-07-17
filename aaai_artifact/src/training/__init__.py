"""Training module exports."""
from .trainer import MultimodalTrainer, JointMultitaskTrainer
from .losses import (
    DepressionLoss, PHQ8Loss, SentimentLoss, EmotionLoss,
    PersonalityLoss, UncertaintyWeightedMultiTaskLoss,
)
from .sampler import TemperatureBalancedSampler, TaskBalancedSampler
from .calibration import TemperatureScaling, PlattScaling, IsotonicCalibrator, compute_ece, compute_brier_score
from .domain_adaptation import (
    CORALLoss, MMDLoss, GradientReversalLayer, DANNLoss,
    DomainDiscriminator, DomainAdaptationLoss,
)

__all__ = [
    "MultimodalTrainer", "JointMultitaskTrainer",
    "DepressionLoss", "PHQ8Loss", "SentimentLoss", "EmotionLoss", "PersonalityLoss",
    "UncertaintyWeightedMultiTaskLoss",
    "TemperatureBalancedSampler", "TaskBalancedSampler",
    "TemperatureScaling", "PlattScaling", "IsotonicCalibrator", "compute_ece", "compute_brier_score",
    "CORALLoss", "MMDLoss", "GradientReversalLayer", "DANNLoss",
    "DomainDiscriminator", "DomainAdaptationLoss",
]