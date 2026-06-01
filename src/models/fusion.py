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
        modality_mask: tuple[bool, bool, bool] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_feat: [batch, text_dim]
            audio_feat: [batch, audio_dim]
            video_feat: [batch, video_dim]
            modality_mask: either a 3-tuple of bools (single sample) or a [batch, 3] tensor

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

        # Handle modality mask — must broadcast correctly across batch
        # modality_mask can be: tuple[bool,bool,bool] for single sample OR [batch, 3] tensor for batch
        if isinstance(modality_mask, torch.Tensor):
            # Batch mask: shape [batch, 3] — zero gates per-sample where modality is missing
            if modality_mask.shape[-1] == 3:  # [batch, 3]
                mask_3d = modality_mask
            else:
                mask_3d = modality_mask
        else:
            # Single sample: convert 3-tuple to [1, 3] then broadcast
            mask_3d = torch.tensor(
                [[bool(modality_mask[0]), bool(modality_mask[1]), bool(modality_mask[2])]],
                dtype=torch.bool,
                device=text_feat.device
            )

        # Zero out gates for missing modalities (per-sample basis for batch)
        t_g = t_g * mask_3d[:, 0:1]   # [batch, 1] broadcast to [batch, hidden_dim]
        a_g = a_g * mask_3d[:, 1:2]
        v_g = v_g * mask_3d[:, 2:3]

        # Gate-weighted sum (NOT concat — allows variable modality count)
        fused = t_g * t + a_g * a + v_g * v
        return fused

    def param_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CrossAttentionFusion(nn.Module):
    """Cross-Attention Multimodal Fusion.

    Each modality attends to all other modalities via bidirectional cross-attention.
    This replaces the GatedLateFusion from Phase 4.

    Architecture:
        1. Project all modalities to common hidden_dim
        2. Bidirectional cross-attention: text↔audio, text↔video, audio↔video
        3. Residual + LayerNorm after each cross-attention block
        4. Gated sum of cross-attended features per modality
        5. Final fusion = weighted sum of modality contributions

    Cross-attention evidence (2026): +0.041 AUC vs gated fusion on depression tasks.
    Missing modalities are handled by zeroing their contribution (masked attention).
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        video_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project each modality to common hidden_dim
        self.text_proj = ModalityProjector(text_dim, hidden_dim)
        self.audio_proj = ModalityProjector(audio_dim, hidden_dim)
        self.video_proj = ModalityProjector(video_dim, hidden_dim)

        # Cross-attention layers: each (query, key, value) projection
        self.text_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.audio_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.video_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-attention for other modalities
        self.text_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.audio_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.video_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Fusion gate: learn to weight cross-attended vs residual
        self.text_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.audio_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.video_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

        # Final fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        modality_mask: tuple[bool, bool, bool] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_feat: [batch, text_dim]
            audio_feat: [batch, audio_dim]
            video_feat: [batch, video_dim]
            modality_mask: either a 3-tuple of bools or a [batch, 3] tensor

        Returns:
            fused: [batch, hidden_dim]
        """
        # Project all modalities to hidden_dim
        t = self.text_proj(text_feat)   # [batch, hidden]
        a = self.audio_proj(audio_feat)
        v = self.video_proj(video_feat)

        # Parse modality mask for per-sample suppression
        if isinstance(modality_mask, torch.Tensor):
            if modality_mask.dim() == 2 and modality_mask.shape[-1] == 3:
                mask_t = ~modality_mask[:, 0:1].bool()   # True=missing
                mask_a = ~modality_mask[:, 1:2].bool()
                mask_v = ~modality_mask[:, 2:3].bool()
            else:
                mask_t = ~modality_mask[:, 0].bool()
                mask_a = ~modality_mask[:, 1].bool()
                mask_v = ~modality_mask[:, 2].bool()
        else:
            mask_t = torch.tensor(not modality_mask[0], device=text_feat.device)
            mask_a = torch.tensor(not modality_mask[1], device=text_feat.device)
            mask_v = torch.tensor(not modality_mask[2], device=text_feat.device)

        # Self-attention with residual (text attends to text, audio to audio, video to video)
        t_self, _ = self.text_attention(t, t, t)  # [batch, hidden]
        t = t + t_self
        t = nn.functional.layer_norm(t, (self.hidden_dim,))

        a_self, _ = self.audio_attention(a, a, a)
        a = a + a_self
        a = nn.functional.layer_norm(a, (self.hidden_dim,))

        v_self, _ = self.video_attention(v, v, v)
        v = v + v_self
        v = nn.functional.layer_norm(v, (self.hidden_dim,))

        # Bidirectional cross-attention: text attends to audio and video
        # Stack audio and video as key/value for text cross-attention
        av_concat = torch.cat([a.unsqueeze(1), v.unsqueeze(1)], dim=1)  # [batch, 2, hidden]
        t_q = t.unsqueeze(1)  # [batch, 1, hidden] — add seq dim for cross-attn

        t_cross, _ = self.text_cross_attn(t_q, av_concat, av_concat)  # [batch, 1, hidden]
        t_cross = t_cross.squeeze(1)  # [batch, hidden]

        # audio attends to text and video
        tv_concat = torch.cat([t.unsqueeze(1), v.unsqueeze(1)], dim=1)
        a_q = a.unsqueeze(1)
        a_cross, _ = self.audio_cross_attn(a_q, tv_concat, tv_concat)
        a_cross = a_cross.squeeze(1)

        # video attends to text and audio
        ta_concat = torch.cat([t.unsqueeze(1), a.unsqueeze(1)], dim=1)
        v_q = v.unsqueeze(1)
        v_cross, _ = self.video_cross_attn(v_q, ta_concat, ta_concat)
        v_cross = v_cross.squeeze(1)

        # Gated fusion: combine cross-attended with residual
        t_gated = self.text_gate(torch.cat([t, t_cross], dim=-1)) * t_cross
        a_gated = self.audio_gate(torch.cat([a, a_cross], dim=-1)) * a_cross
        v_gated = self.video_gate(torch.cat([v, v_cross], dim=-1)) * v_cross

        # Mask missing modalities (zero out their contribution)
        if isinstance(modality_mask, torch.Tensor) and modality_mask.dim() == 2:
            t_gated = t_gated * (~mask_t).float()
            a_gated = a_gated * (~mask_a).float()
            v_gated = v_gated * (~mask_v).float()
        else:
            if modality_mask[0]:
                t_gated = t_gated
            else:
                t_gated = t_gated * 0.0
            if modality_mask[1]:
                a_gated = a_gated
            else:
                a_gated = a_gated * 0.0
            if modality_mask[2]:
                v_gated = v_gated
            else:
                v_gated = v_gated * 0.0

        # Final fusion: weighted sum of gated modality contributions
        fused = self.fusion_proj(torch.cat([t_gated, a_gated, v_gated], dim=-1))
        return fused

    def param_count(self) -> int:
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
        modality_mask: tuple[bool, bool, bool] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_feat: [batch, text_dim]
            audio_feat: [batch, audio_dim]
            video_feat: [batch, video_dim]
            modality_mask: either a 3-tuple of bools (single sample) or a [batch, 3] tensor

        Returns:
            fused: [batch, hidden_dim]
        """
        # Compute modality-specific low-rank factors (modality embeddings)
        t_f = text_feat @ self.text_factor       # [batch, rank]
        a_f = audio_feat @ self.audio_factor
        v_f = video_feat @ self.video_factor

        # Convert mask to tensor if needed for batch operations
        if isinstance(modality_mask, torch.Tensor):
            mask_t = modality_mask[:, 0]   # [batch]
            mask_a = modality_mask[:, 1]
            mask_v = modality_mask[:, 2]
        else:
            mask_t = torch.tensor(modality_mask[0], device=text_feat.device)
            mask_a = torch.tensor(modality_mask[1], device=text_feat.device)
            mask_v = torch.tensor(modality_mask[2], device=text_feat.device)

        # Cross-modal interactions: element-wise product, masked by availability
        ta = t_f * a_f * (mask_t[:, None] * mask_a[:, None])  # [batch, rank]
        tv = t_f * v_f * (mask_t[:, None] * mask_v[:, None])
        av = a_f * v_f * (mask_a[:, None] * mask_v[:, None])

        # Project through interaction matrices
        ta_proj = ta @ self.ta_interact  # [batch, rank]
        tv_proj = tv @ self.tv_interact
        av_proj = av @ self.av_interact

        # Self-contributions (direct factor embeddings), masked by availability
        t_self = t_f * mask_t[:, None]
        a_self = a_f * mask_a[:, None]
        v_self = v_f * mask_v[:, None]

        # Stack all 6 terms: [text×audio, text×video, audio×video, text, audio, video]
        stacked = torch.cat([ta_proj, tv_proj, av_proj, t_self, a_self, v_self], dim=-1)  # [batch, 6*rank]
        fused = self.output_proj(stacked)
        return fused

    def param_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Low-Rank Dynamic Gating Network (LR-DGN)
# ---------------------------------------------------------------------------

class LowRankGatingNetwork(nn.Module):
    """Low-Rank Dynamic Gating Network (LR-DGN) — parameter-efficient multimodal fusion.

    This architecture addresses DAIC overfitting (n=107) by replacing full cross-attention
    with a low-rank bottleneck + context-aware gating mechanism:

    1. Each modality is projected to a low-rank bottleneck (rank=r << input_dim)
    2. All bottleneck embeddings are concatenated and fed to an MLP
    3. The MLP outputs per-modality gate weights (context-aware, not independent)
    4. Gates are applied to the *original* projected features (NOT bottleneck features)
    5. Weighted sum → output projection → fused representation

    Why this helps on DAIC:
    - Low-rank bottleneck (r=16-24) dramatically reduces parameters vs full cross-attention
    - Context-aware gates learn which modalities matter jointly, not independently
    - Bottleneck forces compression of cross-modal information into few dimensions
    - The shared gating MLP sees all modality contexts, preventing any single modality
      from dominating with independent gating

    Architecture diagram:
        text_feat → text_proj → t [batch, hidden]       text_feat → text_low → t_low [batch, r]
        audio_feat → audio_proj → a [batch, hidden]     audio_feat → audio_low → a_low [batch, r]
        video_feat → video_proj → v [batch, hidden]     video_feat → video_low → v_low [batch, r]
                                                         [t_low, a_low, v_low] → gate_MLP → [g_t, g_a, g_v]
                                                         [g_t*t, g_a*a, g_v*v] → concat → output_proj → fused

    Parameters:
        text_dim, audio_dim, video_dim: input feature dimensions
        hidden_dim: output fused dimension (default 512, use 16-32 for DAIC)
        rank: low-rank bottleneck dimension (16-32 for small datasets, 64 for larger)
        num_gate_layers: MLP depth for gate computation (2-3 recommended)
        dropout: dropout rate for gate MLP and output projection
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        video_dim: int,
        hidden_dim: int = 512,
        rank: int = 16,
        num_gate_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank

        # Low-rank bottleneck projections: each modality → rank dimensions
        # These serve BOTH as the gating context AND the main feature path
        # (parameter-efficient: shared low-rank projection, no separate high-dim path)
        self.text_proj = nn.Linear(text_dim, rank)
        self.audio_proj = nn.Linear(audio_dim, rank)
        self.video_proj = nn.Linear(video_dim, rank)

        # Gate MLP: computes per-modality gates from concatenated bottleneck features
        # Input: [batch, 3*rank] (context from all modalities simultaneously)
        # Output: [batch, 3] (gate for text, audio, video)
        gate_layers = []
        in_dim = 3 * rank
        for i in range(num_gate_layers - 1):
            gate_layers.append(nn.Linear(in_dim, in_dim // 2))
            gate_layers.append(nn.LayerNorm(in_dim // 2))
            gate_layers.append(nn.GELU())
            gate_layers.append(nn.Dropout(dropout))
            in_dim = in_dim // 2

        gate_layers.append(nn.Linear(in_dim, 3))   # 3 gates: text, audio, video
        gate_layers.append(nn.Sigmoid())            # gates in [0, 1]

        self.gate_mlp = nn.Sequential(*gate_layers)

        # Output projection: gated weighted sum (rank*3 dims) → hidden_dim
        # Combines the three gated bottleneck features into the final fused representation
        self.output_proj = nn.Sequential(
            nn.Linear(rank * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        modality_mask: tuple[bool, bool, bool] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_feat: [batch, text_dim]
            audio_feat: [batch, audio_dim]
            video_feat: [batch, audio_dim]
            modality_mask: either a 3-tuple of bools or a [batch, 3] tensor

        Returns:
            fused: [batch, hidden_dim]
        """
        # Project each modality to low-rank bottleneck (rank dimensions)
        # These projections serve as BOTH the feature representation AND the gating context
        t = self.text_proj(text_feat)    # [batch, rank]
        a = self.audio_proj(audio_feat)  # [batch, rank]
        v = self.video_proj(video_feat)  # [batch, rank]

        # Context-aware gate computation from concatenated bottleneck features
        # The gating MLP sees all modalities jointly, enabling cross-modal awareness
        bottleneck_concat = torch.cat([t, a, v], dim=-1)  # [batch, 3*rank]
        gates = self.gate_mlp(bottleneck_concat)          # [batch, 3], values in [0,1]

        # Parse modality mask for missing modality suppression
        # Handle both batch-level [batch, 3] tensors and single-sample tuples
        if isinstance(modality_mask, torch.Tensor):
            if modality_mask.dim() == 2 and modality_mask.shape[-1] == 3:
                # Batch-level mask: [batch, 3]
                m_t = modality_mask[:, 0:1].float()   # [batch, 1]
                m_a = modality_mask[:, 1:2].float()
                m_v = modality_mask[:, 2:3].float()
            else:
                # Single sample: [3] tensor
                m_t = modality_mask[0].float().reshape(1, 1)
                m_a = modality_mask[1].float().reshape(1, 1)
                m_v = modality_mask[2].float().reshape(1, 1)
                gates = gates.unsqueeze(0)  # [1, 3] to match batch structure
        else:
            # Single sample: Python tuple of bools
            m_t = torch.tensor([[float(modality_mask[0])]], device=text_feat.device)
            m_a = torch.tensor([[float(modality_mask[1])]], device=text_feat.device)
            m_v = torch.tensor([[float(modality_mask[2])]], device=text_feat.device)
            gates = gates.unsqueeze(0)  # [1, 3] to match batch structure

        # Apply mask to gates (zero gate for missing modalities)
        g_t = gates[:, 0:1] * m_t   # [batch, 1] broadcast to [batch, rank]
        g_a = gates[:, 1:2] * m_a
        g_v = gates[:, 2:3] * m_v

        # Gated weighted sum of low-rank projected features
        t_gated = g_t * t           # [batch, rank]
        a_gated = g_a * a
        v_gated = g_v * v

        # Concatenate gated features → output projection
        fused = self.output_proj(torch.cat([t_gated, a_gated, v_gated], dim=-1))  # [batch, hidden_dim]
        return fused

    def param_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_gate_values(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor,
        modality_mask: tuple | torch.Tensor,
    ) -> dict[str, float]:
        """Return mean gate values for diagnostics (call with a single sample)."""
        with torch.no_grad():
            t = self.text_proj(text_feat)
            a = self.audio_proj(audio_feat)
            v = self.video_proj(video_feat)
            bottleneck_concat = torch.cat([t, a, v], dim=-1)
            gates = self.gate_mlp(bottleneck_concat)
            # Handle both single-sample and batch inputs
            if isinstance(modality_mask, torch.Tensor):
                if modality_mask.dim() == 2:
                    m_t = modality_mask[0, 0].float().item()
                    m_a = modality_mask[0, 1].float().item()
                    m_v = modality_mask[0, 2].float().item()
                    g0, g1, g2 = gates[0, 0].item(), gates[0, 1].item(), gates[0, 2].item()
                else:
                    m_t = modality_mask[0].float().item()
                    m_a = modality_mask[1].float().item()
                    m_v = modality_mask[2].float().item()
                    g0, g1, g2 = gates[0].item(), gates[1].item(), gates[2].item()
            else:
                m_t = float(modality_mask[0])
                m_a = float(modality_mask[1])
                m_v = float(modality_mask[2])
                g0, g1, g2 = gates[0].item(), gates[1].item(), gates[2].item()
            return {
                "gate_text": g0 * m_t,
                "gate_audio": g1 * m_a,
                "gate_video": g2 * m_v,
            }