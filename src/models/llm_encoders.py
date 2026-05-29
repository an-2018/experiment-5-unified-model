"""LLM-based encoders for ablation studies (L0–L9 ablation matrix).

LLM-generated features are DERIVED FEATURES, not ground truth.
Keep LLM features strictly separated from dataset labels in reporting.
"""
from abc import ABC, abstractmethod
import torch


class LLMTextEncoder(ABC):
    """LLM text encoder (Mistral-LoRA, frozen Mistral, etc.)."""

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        ...


class LLMAudioEncoder(ABC):
    """LLM audio encoder (Qwen2-Audio-style)."""

    @abstractmethod
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        ...


class LLMVideoEncoder(ABC):
    """LLM video encoder (Qwen2.5-VL / LLaVA-OneVision-style)."""

    @abstractmethod
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        ...


class TeacherFeatureExtractor(ABC):
    """Generate structured LLM teacher descriptors (e.g., 'flattened prosody', 'visible agitation')."""

    @abstractmethod
    def extract(self, text: str = None, audio_feat: torch.Tensor = None, video_feat: torch.Tensor = None) -> dict[str, str]:
        """Returns a dict of textual descriptors. Treat as derived features only."""
        ...


class GraphXAINNarrator(ABC):
    """Map GNN subgraphs + SHAP values into LLM-generated narrative explanations."""

    @abstractmethod
    def narrate(
        self,
        subgraph_edges: torch.Tensor,
        shap_values: dict[str, float],
        sample_context: dict,
    ) -> str:
        """Returns a human-readable explanation string."""
        ...