"""Unimodal encoders: text (RoBERTa/DistilBERT), audio (WavLM/HuBERT), video (OpenFace/ViT)."""
from abc import ABC, abstractmethod
import torch


class TextEncoder(ABC):
    """Base class for text encoders."""

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns [batch, hidden_dim] text embeddings."""
        ...


class AudioEncoder(ABC):
    """Base class for audio encoders."""

    @abstractmethod
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Returns [batch, hidden_dim] audio embeddings."""
        ...


class VideoEncoder(ABC):
    """Base class for video encoders."""

    @abstractmethod
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Returns [batch, hidden_dim] video embeddings."""
        ...


class RoBERTaEncoder(TextEncoder):
    """RoBERTa-base text encoder."""
    ...


class DistilBERTEncoder(TextEncoder):
    """DistilBERT text encoder."""
    ...


class WavLMEncoder(AudioEncoder):
    """WavLM audio encoder."""
    ...


class OpenFaceEncoder(VideoEncoder):
    """OpenFace action unit encoder."""
    ...


class ViTEncoder(VideoEncoder):
    """Vision Transformer frame encoder."""
    ...


# Projection layer to common embedding dimension
class ModalityProjector(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.LayerNorm(output_dim),
            torch.nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)