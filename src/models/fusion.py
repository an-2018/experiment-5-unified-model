"""Multimodal fusion layers: Gated Late Fusion, Low-Rank Multimodal Fusion (LMF).

Handle missing modalities via modality_mask — ignore missing rather than zero-fill.
The mask is a tuple[bool, bool, bool] of (text_available, audio_available, video_available).
"""
import torch
import torch.nn as nn
from .encoders import ModalityProjector


class GatedLateFusion(nn.Module):
    """Gated late fusion with proper mask-based modality handling.

    Each modality is first projected to hidden_dim, then a gate is computed on the projected
    representation. The gate-weighted projected features are summed (not concatenated).
    Missing modalities are handled by zeroing their gate weights, NOT by zeroing projected
    features before the gate computation — this allows the gate network to receive real
    feature values during training and learn to suppress missing modalities naturally.

    Architecture:
        text_feat  → text_proj → t → text_gate(t) → t_g  → t_g * t (masked)
        audio_feat → audio_proj → a → audio_gate(a) → a_g  → a_g * a (masked)
        video_feat → video_proj → v → video_gate(v) → v_g  → v_g * v (masked)
        fused = t_g*t + a_g*a + v_g*v  [if mask component is False, gate → 0]
    """

    def __init__(self, text_dim: int, audio_dim: int, video_dim: int, hidden_dim: int = 512):
        super().__init__()
        # Project each modality to hidden_dim
        self.text_proj = ModalityProjector(text_dim, hidden_dim)
        self.audio_proj = ModalityProjector(audio_dim, hidden_dim)
        self.video_proj = ModalityProjector(video_dim, hidden_dim)

        # Gates operate on projected representations (receive real projected values for all modalities)
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.audio_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.video_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        modality_mask: tuple[bool, bool, bool],
    ) -> torch.Tensor:
        """
        Args:
            text_feat: [batch, text_dim]
            audio_feat: [batch, audio_dim]
            video_feat: [batch, video_dim]
            modality_mask: (text_available, audio_available, video_available)

        Returns:
            fused: [batch, hidden_dim] — sum of gate-weighted projected features
        """
        # Project all modalities (even missing ones — gate learns on real projections)
        t = self.text_proj(text_feat)
        a = self.audio_proj(audio_feat)
        v = self.video_proj(video_feat)

        # Compute gates on projected representations
        t_g = self.text_gate(t)  # [batch, hidden_dim]
        a_g = self.audio_gate(a)
        v_g = self.video_gate(v)

        # Zero gate for missing modalities (explicit mask — safe fallback)
        # Gates naturally learn to suppress missing data; explicit mask ensures correctness
        if not modality_mask[0]:
            t_g = torch.zeros_like(t_g)
        if not modality_mask[1]:
            a_g = torch.zeros_like(a_g)
        if not modality_mask[2]:
            v_g = torch.zeros_like(v_g)

        # Gate-weighted sum (NOT concat — allows variable modality count)
        fused = t_g * t + a_g * a + v_g * v
        return fused

    def param_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LowRankMultimodalFusion(nn.Module):
    """Low-Rank Multimodal Fusion (LMF) — proper factorized bilinear fusion.

    Reference: Liu et al. "Efficient Low-Rank Multimodal Fusion with Modality-Specific Factors"
    (ICML 2018). This implementation uses modality-specific low-rank factor matrices to
    compute bilinear interactions between modality pairs, then sums across all pairs.

    The fusion output is:
        h = sum_{i,j} (U_i ⊙ V_j)^T x_i  [when modality j is interacted with i]
    Simplified vectorized form:
        h = W · (P_text ⊙ P_audio ⊙ P_video)  [projected factor product]

    For three modalities, we compute all pairwise interactions plus self-interactions.
    Missing modalities are handled by zeroing the corresponding factor rows in the
    output computation.
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        video_dim: int,
        hidden_dim: int = 512,
        rank: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank

        # Low-rank factor matrices for each modality
        # Each maps input dim → rank (factorized tensor dimensions)
        self.text_factor = nn.Parameter(torch.randn(text_dim, rank))
        self.audio_factor = nn.Parameter(torch.randn(audio_dim, rank))
        self.video_factor = nn.Parameter(torch.randn(video_dim, rank))

        # Cross-modality interaction factors (rank × rank each)
        # We compute all 6 pairwise interactions (text×audio, text×video, audio×video)
        # plus 3 self-interactions = 6 interaction terms
        self.ta_interact = nn.Parameter(torch.randn(rank, rank))   # text × audio
        self.tv_interact = nn.Parameter(torch.randn(rank, rank))   # text × video
        self.av_interact = nn.Parameter(torch.randn(rank, rank))   # audio × video

        # Output projection: sum of all interactions → hidden_dim
        # Interaction terms are computed as: factor_i^T @ interact_ij @ factor_j @ x
        # Each interaction produces a rank-dim vector; 6 interactions → 6*rank dims → project to hidden
        self.output_proj = nn.Sequential(
            nn.Linear(rank * 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Initialize factors with small random weights
        nn.init.normal_(self.text_factor, 0, 0.02)
        nn.init.normal_(self.audio_factor, 0, 0.02)
        nn.init.normal_(self.video_factor, 0, 0.02)
        nn.init.normal_(self.ta_interact, 0, 0.02)
        nn.init.normal_(self.tv_interact, 0, 0.02)
        nn.init.normal_(self.av_interact, 0, 0.02)

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        modality_mask: tuple[bool, bool, bool],
    ) -> torch.Tensor:
        """
        Args:
            text_feat: [batch, text_dim]
            audio_feat: [batch, audio_dim]
            video_feat: [batch, video_dim]
            modality_mask: (text_available, audio_available, video_available)

        Returns:
            fused: [batch, hidden_dim]
        """
        # Compute modality-specific low-rank factors (modality embeddings)
        t_f = text_feat @ self.text_factor       # [batch, rank]
        a_f = audio_feat @ self.audio_factor
        v_f = video_feat @ self.video_factor

        # Always produce 6 interaction terms (rank × rank each)
        # Use zeros for missing modality interactions — the mask gates them
        zero_tensor = torch.zeros(text_feat.shape[0], self.rank, device=text_feat.device, dtype=text_feat.dtype)

        # Cross-modal interactions: element-wise product of modality factors
        ta = t_f * a_f if (modality_mask[0] and modality_mask[1]) else zero_tensor
        tv = t_f * v_f if (modality_mask[0] and modality_mask[2]) else zero_tensor
        av = a_f * v_f if (modality_mask[1] and modality_mask[2]) else zero_tensor

        # Project through interaction matrices
        ta_proj = ta @ self.ta_interact  # [batch, rank]
        tv_proj = tv @ self.tv_interact
        av_proj = av @ self.av_interact

        # Self-contributions (direct factor embeddings)
        t_self = t_f if modality_mask[0] else zero_tensor
        a_self = a_f if modality_mask[1] else zero_tensor
        v_self = v_f if modality_mask[2] else zero_tensor

        # Stack all 6 terms: [text×audio, text×video, audio×video, text, audio, video]
        stacked = torch.cat([ta_proj, tv_proj, av_proj, t_self, a_self, v_self], dim=-1)  # [batch, 6*rank]
        fused = self.output_proj(stacked)
        return fused

    def param_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)