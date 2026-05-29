"""Multimodal fusion layers: Gated Late Fusion, Low-Rank Multimodal Fusion (LMF).

Handle missing modalities via modality_mask — ignore missing rather than zero-fill.
"""
import torch
from .encoders import ModalityProjector


class GatedLateFusion(torch.nn.Module):
    """Gated late fusion with modality dropout stability.

    Each modality is first projected to hidden_dim, then a gate is computed on the projected representation.
    The gate-weighted projected features are concatenated and fused.
    """

    def __init__(self, text_dim: int, audio_dim: int, video_dim: int, hidden_dim: int = 512):
        super().__init__()
        # Project each modality to hidden_dim
        self.text_proj = ModalityProjector(text_dim, hidden_dim)
        self.audio_proj = ModalityProjector(audio_dim, hidden_dim)
        self.video_proj = ModalityProjector(video_dim, hidden_dim)
        # Gates operate on projected representations
        self.text_gate = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Sigmoid())
        self.audio_gate = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Sigmoid())
        self.video_gate = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Sigmoid())
        # Final fusion projection
        self.fusion = torch.nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        modality_mask: tuple[bool, bool, bool],
    ) -> torch.Tensor:
        """Fuse modalities with gate-weighted contributions. Missing modalities contribute zero."""
        # Project each modality
        t = self.text_proj(text_feat)
        a = self.audio_proj(audio_feat)
        v = self.video_proj(video_feat)

        # Zero out missing modalities
        if not modality_mask[0]:
            t = torch.zeros_like(t)
        if not modality_mask[1]:
            a = torch.zeros_like(a)
        if not modality_mask[2]:
            v = torch.zeros_like(v)

        # Gate-weighted contributions
        t_g = self.text_gate(t)
        a_g = self.audio_gate(a)
        v_g = self.video_gate(v)

        fused = torch.cat([t_g * t, a_g * a, v_g * v], dim=-1)
        return self.fusion(fused)


class LowRankMultimodalFusion(torch.nn.Module):
    """Low-Rank Multimodal Fusion (LMF) for parameter efficiency."""

    def __init__(self, text_dim: int, audio_dim: int, video_dim: int, hidden_dim: int = 512, rank: int = 16):
        super().__init__()
        self.text_proj = torch.nn.Parameter(torch.randn(text_dim, hidden_dim // rank))
        self.audio_proj = torch.nn.Parameter(torch.randn(audio_dim, hidden_dim // rank))
        self.video_proj = torch.nn.Parameter(torch.randn(video_dim, hidden_dim // rank))
        self.output_proj = torch.nn.Parameter(torch.randn(hidden_dim // rank, hidden_dim))

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        modality_mask: tuple[bool, bool, bool],
    ) -> torch.Tensor:
        t = text_feat if modality_mask[0] else torch.zeros_like(text_feat)
        a = audio_feat if modality_mask[1] else torch.zeros_like(audio_feat)
        v = video_feat if modality_mask[2] else torch.zeros_like(video_feat)

        # Low-rank element-wise product then projected
        fused = torch.cat([
            t @ self.text_proj,
            a @ self.audio_proj,
            v @ self.video_proj,
        ], dim=-1) @ self.output_proj
        return fused